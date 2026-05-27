"""Blink validation + temporal analysis.

Ported from ``BlinkAnalyzer_swz`` onto the shared core: loader from ``io``,
``find_overlap`` from ``preprocess``, figures from ``plotting``, and the cohort
constants (subject range / glasses / offsets) from ``config``.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns

from .. import config
from .. import plotting  # imported first so the matplotlib backend is configured
from ..io import GazeDataLoader
from ..preprocess import find_overlap


def clean_blink_truth(loader, gt_blinks):
    """Project ASCII GT blinks into the task windows and keep the overlapping ones."""
    tb = loader.task_begin + loader.t_init
    tl = loader.task_len
    s2f = loader.s2f
    tb_1000 = (tb + s2f) * 100 / 3
    tl_1000 = tl * 100 / 3
    fs_1000 = np.stack([tb_1000, tb_1000 + tl_1000], axis=1)
    blink_clean, _ = find_overlap(fs_1000, gt_blinks)
    return blink_clean * 0.03 - s2f


class BlinkAnalyzer:
    def __init__(self, data_path='data', blink_path='blinks'):
        self.data_path = data_path
        self.blink_path = blink_path
        self.subject_df = pd.DataFrame()
        self.blink_df = pd.DataFrame()
        self.task_df = pd.DataFrame()
        self.gt_events, self.pred_events, self.task_intervals = [], [], []

        self.subject_range = range(config.SUBJECT_IDS[0], config.SUBJECT_IDS[-1] + 1)
        self.glasses_ids = config.GLASSES_IDS
        self.offsets = config.OFFSETS

        sns.set_context("paper", font_scale=1.2)
        plotting.plt.rcParams.update(
            {'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})

    def load_and_validate(self):
        """Load GT + predicted blinks, build dataframes, score confusion matrix."""
        print(f"{'Sub ID':<8} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8} {'Recall':<8}")
        print("-" * 60)
        subject_results, all_blinks = [], []

        for sub_idx in self.subject_range:
            sub_str = f"{sub_idx:05d}"
            eyewear = 'Glasses' if sub_idx in self.glasses_ids else 'No Glasses'
            npy_file = os.path.join(self.data_path, self.blink_path, sub_str)
            if not os.path.exists(npy_file):
                continue

            with open(npy_file, "rb") as f:
                pred_events = np.load(f)
                pred_metadata = np.load(f)

            if pred_events.size > 0:
                df_sub = pd.DataFrame(pred_events.copy(), columns=['timestamp', 'duration'])
                df_sub['timestamp'] -= self.offsets.get(sub_idx, 0)
                df_sub['subject_id'] = sub_idx
                df_sub['eyewear'] = eyewear
                df_sub['our_count'] = pred_metadata[0]
                df_sub['gt_count'] = pred_metadata[1]
                df_sub['thresh'] = pred_metadata[2]
                df_sub['template_id'] = pred_metadata[3]
                df_sub['offset'] = pred_metadata[4]
                all_blinks.append(df_sub)

            try:
                loader = GazeDataLoader(self.data_path, sub_idx - config.SUBJECT_IDS[0])
                blinks, _, _, _ = loader.load_ascii_data()
                gt_events = clean_blink_truth(loader, gt_blinks=blinks)
                tb, tl = loader.task_begin + loader.t_init, loader.task_len
                task_intervals = np.stack([tb, tb + tl], axis=1)
            except Exception as e:
                print(f"Skipping GT {sub_str}: {e}")
                continue

            pred_copy = pred_events.copy()
            pred_copy[:, 1] += pred_copy[:, 0]
            _, gt_mask = find_overlap(gt_events, task_intervals)
            _, pred_mask = find_overlap(pred_copy, task_intervals)
            gt_events[:, 1] -= gt_events[:, 0]
            task_intervals[:, 1] -= task_intervals[:, 0]

            tp = np.sum((gt_mask == 1) & (pred_mask == 1))
            tn = np.sum((gt_mask == 0) & (pred_mask == 0))
            fp = np.sum((gt_mask == 0) & (pred_mask == 1))
            fn = np.sum((gt_mask == 1) & (pred_mask == 0))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / len(gt_mask) if len(gt_mask) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{sub_str:<8} {tp:<8} {tn:<8} {fp:<8} {fn:<8} {recall:.3f}")

            subject_results.append({
                'subject_id': sub_idx, 'eyewear': eyewear,
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                'Recall': recall, 'Precision': precision, 'Accuracy': accuracy, 'F1': f1,
                'our_count': len(pred_events), 'gt_count': len(gt_events)})
            self.gt_events.append(gt_events)
            self.pred_events.append(pred_events)
            self.task_intervals.append(task_intervals)

        self.subject_df = pd.DataFrame(subject_results)
        self.blink_df = pd.concat(all_blinks, ignore_index=True) if all_blinks else pd.DataFrame()
        return self.subject_df, self.blink_df

    def analyze_tasks(self):
        """Per-task blink counts, rates and mean durations."""
        task_data = []
        for i, row in self.subject_df.iterrows():
            pred_ev, gt_ev, intervals = self.pred_events[i], self.gt_events[i], self.task_intervals[i]
            for t_id, (t_start, t_len) in enumerate(intervals):
                t_len_min = t_len / 1800.0
                if t_len_min <= 0:
                    continue
                p_task = (pred_ev[(pred_ev[:, 0] >= t_start) & (pred_ev[:, 0] <= t_start + t_len)]
                          if pred_ev.size > 0 else np.array([]))
                g_task = (gt_ev[(gt_ev[:, 0] >= t_start) & (gt_ev[:, 0] <= t_start + t_len)]
                          if gt_ev.size > 0 else np.array([]))
                task_data.append({
                    'subject_id': row['subject_id'], 'eyewear': row['eyewear'], 'task_id': t_id,
                    'our_count': len(p_task), 'gt_count': len(g_task),
                    'our_rate': len(p_task) / t_len_min, 'gt_rate': len(g_task) / t_len_min,
                    'our_mean_dur': p_task[:, 1].mean() * 100 / 3 if len(p_task) > 0 else 0,
                    'gt_mean_dur': g_task[:, 1].mean() * 100 / 3 if len(g_task) > 0 else 0})
        self.task_df = pd.DataFrame(task_data)
        print(f"\nExtracted {len(self.task_df)} task blocks.")
        return self.task_df

    def compare_split_half_time(self, save_path='split_half_time.png'):
        """Paired first-50% vs last-50% blink-duration test + figure."""
        if self.subject_df.empty:
            print("Error: subject_df is empty. Run load_and_validate() first.")
            return
        early_data, late_data, valid_count = [], [], 0
        for i, row in self.subject_df.iterrows():
            sub_id = row['subject_id']
            if i >= len(self.task_intervals) or i >= len(self.pred_events):
                continue
            intervals = self.task_intervals[i]
            if len(intervals) == 0:
                continue
            t_max = intervals[-1, 1] + intervals[-1, 0]
            t_mid = t_max * 0.5
            events = self.pred_events[i]
            if events.size == 0:
                continue
            early_blinks = events[events[:, 0] <= t_mid]
            late_blinks = events[events[:, 0] >= t_mid]
            if len(early_blinks) > 0 and len(late_blinks) > 0:
                early_data.append({'subject_id': sub_id, 'Value': early_blinks[:, 1].mean() * 100 / 3})
                late_data.append({'subject_id': sub_id, 'Value': late_blinks[:, 1].mean() * 100 / 3})
                valid_count += 1
        print(f"Found {valid_count} subjects with valid data in both phases.")
        if valid_count < 2:
            print("Not enough data to run a paired T-test (need >= 2 subjects).")
            return

        from scipy.stats import ttest_rel
        paired_df = pd.merge(pd.DataFrame(early_data), pd.DataFrame(late_data),
                             on='subject_id', suffixes=('_early', '_late'))
        t_stat, p_val = ttest_rel(paired_df['Value_early'], paired_df['Value_late'])
        mean_diff = paired_df['Value_late'].mean() - paired_df['Value_early'].mean()
        print(f"Mean First 50%: {paired_df['Value_early'].mean():.2f} ms")
        print(f"Mean Last 50% : {paired_df['Value_late'].mean():.2f} ms")
        print(f"Mean Difference: {mean_diff:.2f} ms")
        print(f"Paired T-test : t={t_stat:.3f}, p={p_val:.4f}")
        plotting.plot_split_half(paired_df, p_val, save_path=save_path)
        return paired_df
