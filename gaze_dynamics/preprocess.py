"""Gaze cleaning / interval primitives.

Ported from ``GazePreprocessor_swz`` with the numeric logic preserved. Three
clear crashers from the original were fixed (each flagged with FIX:):
  - ``clean_data`` called ``self.find_between`` which is a module function, not
    a method  -> would AttributeError.
  - ``mask_blinks`` called ``_find_between`` which is undefined -> NameError.
  - ``find_compare`` used the removed ``np.in1d`` -> use ``np.isin``.
"""

import numpy as np


class GazePreprocessor:
    def __init__(self, screen_res=(1920, 1080), smooth_box=3):
        self.screen_width, self.screen_height = screen_res
        self.smooth_box = smooth_box

    def clean_data(self, pts_head, blink_stamp, s2f_offset, tb, tl, range_lim):
        """Remove blinks/outliers and smooth a raw iris+head track."""
        blinks_adj = blink_stamp * 3 / 100 - s2f_offset
        frame_stamp = np.arange(pts_head.shape[1])

        frame_stamp = mask_blinks(blinks_adj, frame_stamp, tb, tl)
        if frame_stamp is None:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        valid_pts = pts_head[:, pts_head[0] > 0]
        if valid_pts.shape[1] > 0:
            center = np.mean(valid_pts, axis=1).reshape(4, 1)
            dist_sq = (pts_head[0] - center[0]) ** 2 + (pts_head[1] - center[1]) ** 2
            frame_stamp[dist_sq > range_lim] = -1
            if np.any(dist_sq > range_lim):
                print('Lost tracking at:', np.where(dist_sq > range_lim)[0])

        mask = frame_stamp >= 0
        phf_nb = pts_head[:, mask]
        fs_nb = frame_stamp[mask]
        print('Before remove: ', pts_head.shape, 'After remove: ', phf_nb.shape)

        if phf_nb.shape[1] == 0:
            pts_sm = phf_nb[:2]
        else:
            pts_sm = np.stack([self.smooth(phf_nb[0]), self.smooth(phf_nb[1])])

        low_bound = 8 if tl in [160, 250] else 38
        start_t, end_t = find_between(fs_nb, fs_nb, low_bound, tl)  # FIX: was self.find_between

        print('Final: ', phf_nb[:, start_t:end_t].shape,
              'start: ', fs_nb[start_t], 'end: ', fs_nb[end_t - 1])
        return fs_nb[start_t:end_t], phf_nb[:, start_t:end_t], pts_sm[:, start_t:end_t]

    def smooth(self, y):
        if y.shape[0] < self.smooth_box:
            return y
        box = np.ones(self.smooth_box) / self.smooth_box
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth[0], y_smooth[-1] = y[0], y[-1]
        return y_smooth

    def find_compare(self, xyt, s2f, tb, f_1000, sacc):
        stps = (sacc + tb + s2f) * 100 / 3
        t_1000 = np.round((f_1000 + tb + s2f) * 100 / 6) * 2
        t_con = np.isin(xyt[2], t_1000)  # FIX: np.in1d -> np.isin
        xy_compare = xyt[:2, t_con]
        xy_compare[0] = np.clip(xy_compare[0], 0, self.screen_width)
        xy_compare[1] = np.clip(xy_compare[1], 0, self.screen_height)
        return stps, xy_compare

    def limit_to_screen(self, data):
        data[0] = np.clip(data[0], 0, self.screen_width)
        data[1] = np.clip(data[1], 0, self.screen_height)
        return data


def find_between(arr1, arr2, left, right):
    start = np.searchsorted(arr1, left, 'left')
    end = np.searchsorted(arr2, right, 'right')
    return start, end


def mask_blinks(blink_stamp, frame_stamp, start, gap):
    start_idx, end_idx = find_between(blink_stamp[:, 1], blink_stamp[:, 0],
                                      start, start + gap)  # FIX: _find_between -> find_between
    for s, e in blink_stamp[start_idx:end_idx]:
        frame_start, frame_end = find_between(frame_stamp, frame_stamp,
                                              s - start, e - start)  # FIX
        frame_stamp[frame_start:frame_end] = -1
    if not np.sum(frame_stamp >= 0):
        print('Warning: EyeLink not stable. Cannot find a match frame!')
        return None
    return frame_stamp


def find_overlap(list1, list2):
    """Overlapping intervals between two sorted ``[start, end]`` lists.

    Returns ``(overlaps, mask)`` where mask marks, over the concatenated
    duration of ``list2``, which samples fall inside an overlap.
    """
    mask = np.zeros(np.sum(list2[:, 1] - list2[:, 0]), dtype=int)
    overlaps = []
    i, j, current_idx = 0, 0, 0
    n1, n2 = len(list1), len(list2)

    while i < n1 and j < n2:
        start1, end1 = list1[i]
        start2, end2 = list2[j]
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        if inter_start <= inter_end:
            overlaps.append([inter_start, inter_end])
            rel_start = current_idx + (int(inter_start) - start2)
            rel_end = current_idx + (int(inter_end) - start2)
            if int(inter_start) < inter_start:
                rel_start += 1
            mask[rel_start:rel_end] = 1
        if end1 < end2:
            i += 1
        else:
            j += 1
            current_idx += end2 - start2
    return np.array(overlaps), mask


def calculate_synchronization(gt_blinks, video_blinks, sample_rate=1000):
    """Align video blinks (frames) to ASCII GT blinks; return the s2f offset."""
    if len(gt_blinks) == 0:
        print("No ground truth blinks found in ASCII.")
        return None
    video_blinks[:, 1] += video_blinks[:, 0]
    shift = gt_blinks[:, 0] * 30 / sample_rate - video_blinks[0, 0]
    s2f_estimates = []
    for i in range(min(30, len(gt_blinks))):
        overlap, _ = find_overlap(gt_blinks * 30 / sample_rate - shift[i], video_blinks)
        s2f_estimates.append(np.sum(overlap[:, 1] - overlap[:, 0]))
    final_s2f = shift[np.argmax(s2f_estimates)]
    print(f"Calculated Synchronization Offset (s2f): {final_s2f:.2f}")
    return final_s2f


def pad_frames_with_rules(frames, x_values, total_length):
    """Pad a frame/value sequence to span [0, total_length): interpolate internal
    gaps, fill the ends with the mean."""
    if len(frames) != len(x_values):
        raise ValueError("frames and x_values must have same length")
    mean_x = sum(x_values) / len(x_values)
    padded_frames, padded_x = [], []

    first_frame = frames[0]
    if first_frame > 0:
        for f in range(0, first_frame):
            padded_frames.append(f)
            padded_x.append(mean_x)

    padded_frames.append(frames[0])
    padded_x.append(x_values[0])
    for i in range(1, len(frames)):
        f_prev, f_curr = frames[i - 1], frames[i]
        x_prev, x_curr = x_values[i - 1], x_values[i]
        gap = f_curr - f_prev
        if gap > 1:
            step = (x_curr - x_prev) / gap
            for k in range(1, gap):
                padded_frames.append(f_prev + k)
                padded_x.append(x_prev + step * k)
        padded_frames.append(f_curr)
        padded_x.append(x_curr)

    last_frame = frames[-1]
    if last_frame < total_length - 1:
        for f in range(last_frame + 1, total_length):
            padded_frames.append(f)
            padded_x.append(mean_x)
    return padded_frames, padded_x


def shift_array(arr, shift, length):
    """Place ``arr`` into a NaN canvas of ``length`` at offset ``shift`` (clipped)."""
    res = np.full(length, np.nan)
    src_start, src_end = 0, len(arr)
    dst_start, dst_end = shift, shift + len(arr)
    if dst_start < 0:
        src_start = -dst_start
        dst_start = 0
    if dst_end > length:
        diff = dst_end - length
        dst_end = length
        src_end -= diff
    if src_end > src_start and dst_end > dst_start:
        res[dst_start:dst_end] = arr[src_start:src_end]
    return res
