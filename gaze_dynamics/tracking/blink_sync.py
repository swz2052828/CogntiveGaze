"""Detect blinks from eye video via template matching, then refine/clean them.

Ported from ``BlinkSync_swz``. ``clean_blink_data`` is reused from ``events``.
Interactive (IPython) display is replaced by the package's save/show renderer.
"""

import os

import numpy as np
import cv2 as cv
from tqdm import tqdm

from .. import plotting
from ..events import clean_blink_data

plt = plotting.plt


class BlinkSync:
    def __init__(self, loader, directory_path, save_dir="gaze_dynamics_out/blinks"):
        self.loader = loader
        self.directory_path = directory_path
        self.save_dir = save_dir
        self.events = []

    def detect_video_blinks(self, template_img, TM_threshold=0.8, box_size=None):
        """Template-match each task frame; low-score frames are candidate blinks."""
        print(f"Initializing tracking for Subject {self.loader.sub_id}...")
        template = template_img
        frames, scores, frame_indices, blinks_indices = [], [], [], []

        tbs = self.loader.task_begin + self.loader.t_init
        tls = self.loader.task_len
        head0 = head1 = None
        for task_id, (tb, tl) in enumerate(
                tqdm(np.stack([tbs, tls], axis=1), desc="Loading Frames", unit="task")):
            if box_size is not None:
                pts_head = self.loader.load_task_binary(task_id)
                head0, head1 = pts_head[2], pts_head[3]
            for i in range(tl):
                frame_idx = tb + i
                img = cv.imread(os.path.join(self.directory_path, f"{frame_idx:05d}.jpg"),
                                cv.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read frame {frame_idx}")
                    break
                if box_size is not None:
                    minh = head1[i] + box_size[1] * 20
                    minw = head0[i] - box_size[0] * 20
                    img_crop = img[minh:minh + 80, minw:minw + 80]
                else:
                    img_crop = img
                frames.append(img_crop)
                frame_indices.append(frame_idx)
                _, max_val, _, _ = cv.minMaxLoc(cv.matchTemplate(img_crop, template, cv.TM_CCOEFF_NORMED))
                scores.append(max_val)
                if max_val < TM_threshold:
                    blinks_indices.append(frame_idx)

        print('Loading Completed!')
        self.frames = np.array(frames)
        self.frame_indices = np.array(frame_indices)
        self.blinks_indices = np.array(blinks_indices)
        self.scores = np.array(scores)
        return self.frames, self.frame_indices, self.blinks_indices, self.scores

    def _apply_filters(self, input_mask, title, remove_list=None):
        if len(input_mask) <= 0:
            print(f"\n{title}: No events to filter!")
            return
        if title == 'Gap':
            print("\nEvents Before Gap:\tEvents After Gap:")
            for i in range(input_mask.shape[0]):
                print(self.events[input_mask[i]], '\t\t', self.events[input_mask[i] + 1])
        if remove_list is None:
            print(f"\n{title}: No events to remove!")
            return
        if remove_list == 'all':
            mask = input_mask
            if title == 'Fill':
                self.events[input_mask - 1, 1] += self.events[input_mask, 1] + 1
        else:
            mask = input_mask[remove_list]
            if title == 'Fill':
                self.events[input_mask[remove_list] - 1, 1] += self.events[input_mask[remove_list], 1] + 1
        if len(mask) > 0:
            print(f"\nMarking {len(mask)} {title} Events for removal.")
            self.duration_clean[mask] = -1

    def refine_blink_events(self, duration_thresh=20, duration_gap=None, visual_step=2,
                            remove=None, fill=None, new_blinks=None, visualize=True,
                            padding=False, replacing=True):
        """Categorize, optionally visualize, filter, merge and rebuild blink events."""
        singles, longs, gaps = clean_blink_data(self.events, duration_thresh, duration_gap)

        if visualize:
            merged_gap_events = self.events[gaps]
            merged_gap_events[:, 1] = self.events[gaps, 1] + self.events[gaps + 1, 1] + 1
            self.visualize_event_group(self.events[singles], "Single", visual_step)
            self.visualize_event_group(self.events[longs], "Long", visual_step)
            self.visualize_event_group(merged_gap_events, "Gap", visual_step)

        self.duration_clean = self.events[:, 1].copy()
        if remove is None:
            print('No Events to remove!')
            return self.events
        self._apply_filters(singles, 'Single', remove[0])
        self._apply_filters(longs, 'Long', remove[1])
        self._apply_filters(gaps, 'Gap', remove[2])
        self._apply_filters(gaps + 1, 'After Gap', remove[3])
        self._apply_filters(gaps + 1, 'Fill', fill)
        print(f"\nRemoved {np.sum(self.duration_clean < 0)} Events:\n"
              f"{self.events[self.duration_clean < 0]}")

        events_ex = self.events.copy()
        events_ex[:, 1] = self.duration_clean
        print(f"\nRefinement: {events_ex.shape[0]} -> {np.sum(self.duration_clean > 0)} events")
        events_clean = events_ex[self.duration_clean > 0]

        if new_blinks:
            print(f"Appending {len(new_blinks)} manual blinks.")
            to_add = np.array(new_blinks)
            if to_add.ndim == 1:
                to_add = to_add.reshape(1, 2)
            events_clean = np.vstack((events_clean, to_add))
        events_clean = events_clean[events_clean[:, 0].argsort()]

        if padding:
            print('Padding with 1 frame before and after blink events (first loop only).')
            events_clean[:, 0] -= 1
            mask = np.isin(self.frame_indices, events_clean[:, 0])
            if np.any(mask):
                self.visualize_blinks(mask)
            events_clean[:, 1] += 2
            mask = np.isin(self.frame_indices, events_clean[:, 0] + events_clean[:, 1] - 1)
            if np.any(mask):
                self.visualize_blinks(mask)

        if replacing:
            print(f"Events replaced with cleaned version count: {len(events_clean)}")
            self.events = events_clean
        return self.events

    def visualize_blinks(self, mask=True, lower=None, gap=None, tag="blinks"):
        if lower is not None:
            if gap <= 0:
                print('Error: Gap is not positive!')
                return
            mask = (self.scores >= lower) & (self.scores < lower + gap)
        check_frames = self.frames[mask]
        check_indices = self.frame_indices[mask]
        check_scores = self.scores[mask]
        print(f"Total Blinks: {check_frames.shape[0]}")

        cols = 6
        fig = None
        for i in range(check_frames.shape[0]):
            j = i % cols
            if j == 0:
                fig, axes = plt.subplots(1, cols, figsize=(15, 3))
            axes[j].imshow(check_frames[i], cmap="gray")
            axes[j].axis("off")
            axes[j].set_title(f"No.{i} {check_indices[i]:.0f} Score {check_scores[i]:.4f}")
            if i == check_frames.shape[0] - 1 and j < cols - 1:
                for k in range(j + 1, cols):
                    axes[k].axis("off")
            if j == cols - 1 or i == check_frames.shape[0] - 1:
                plt.tight_layout()
                plotting._render(os.path.join(self.save_dir, f"{tag}_{i // cols}.png"), fig)

    def visualize_event_group(self, events, title, visual_step):
        print(f'\n------ Visualizing {title} Events ------')
        if len(events) <= 0:
            print("Error: No events to show.")
            return
        if title == 'Single':
            mask = np.isin(self.frame_indices, events[:, 0])
            if np.any(mask):
                self.visualize_blinks(mask, tag="single")
            return
        for i, event in enumerate(events):
            f, d = event[0], event[1]
            print(f"***** {title} Event {i} (Start: {f}, Dur: {d}) *****")
            mask = np.isin(self.frame_indices, np.arange(f, f + d, visual_step))
            if np.any(mask):
                self.visualize_blinks(mask, tag=f"{title.lower()}_{i}")
