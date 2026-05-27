"""Event segmentation: I-VT saccade/fixation detection and blink categorization.

``detect_events`` is lifted out of ``SaccadeAnalyzer`` so saccade and heatmap
analyzers can share one detector. ``clean_blink_data`` comes from ``BlinkSync``.
"""

import numpy as np

from .metrics import calculate_velocity


def detect_events(xy_data, sample_rate, vel_thresh):
    """I-VT classification of a gaze track into saccades and fixations.

    Returns ``(saccades, fixations, velocity)`` where saccades/fixations are
    ``[k, 2]`` arrays of ``[start, end]`` sample indices.
    """
    vel = calculate_velocity(sample_rate, xy_data)
    is_saccade = vel > vel_thresh
    diff = np.diff(is_saccade.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if is_saccade[0]:
        starts = np.insert(starts, 0, 0)
    if is_saccade[-1]:
        ends = np.append(ends, len(vel))
    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    if len(ends) > len(starts):
        ends = ends[:len(starts)]

    saccades = list(zip(starts, ends))
    fixations = []
    last_end = 0
    for s, e in saccades:
        if s > last_end:
            fixations.append((last_end, s))
        last_end = e
    if last_end < len(vel):
        fixations.append((last_end, len(vel)))
    return np.array(saccades), np.array(fixations), vel


def clean_blink_data(events, lower, gap=None):
    """Categorize blink ``[start, duration]`` events into single / long / gap masks.

    FIX: the original left ``gaps_mask`` undefined when ``len(events) <= 1`` and
    then printed it -> NameError; it is now initialized empty.
    """
    single_mask = np.where(events[:, 1] <= 1)[0]

    if gap is not None:
        if gap <= 0:
            print('Error: Gap is not positive!')
            return
        mask = (events[:, 1] >= lower) & (events[:, 1] < lower + gap)
    else:
        mask = (events[:, 1] >= lower)
    long_mask = np.where(mask)[0]

    event_ends = events[:, 0] + events[:, 1]
    if len(events) > 1:
        gaps = events[1:, 0] - event_ends[:-1]
        gaps_mask = np.where(gaps <= 1)[0]
    else:
        gaps_mask = np.array([], dtype=int)  # FIX: was undefined

    print(f"\nFound: \t{len(events)} total events.")
    print(f"   - \t{len(single_mask)} Single Frame Events")
    print(f"   - \t{len(long_mask)} Long Frame Events (> {lower} frames)")
    print(f"   - \t{len(gaps_mask)} Single Gap Events")
    print("\nSample of Single Frame Events [Start, Duration]:")
    print(events[single_mask][:10])
    print("\nSample of Long Frame Events [Start, Duration]:")
    print(events[long_mask][:10])
    print("\nSample of Single Gap Events [Start, Duration]:")
    print(events[gaps_mask])
    return single_mask, long_mask, gaps_mask
