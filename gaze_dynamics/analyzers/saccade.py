"""Saccade / fixation dynamics analyzer.

Ported from ``SaccadeAnalyzer_swz`` onto the shared core: the I-VT detector now
comes from ``events``, screen scaling from ``geometry``, signal metrics from
``metrics``, and the tracking plot from ``plotting``. Target layouts come from
``config.build_saccade_tasks``.
"""

import numpy as np

from .. import config
from ..io import load_gaze_data
from ..geometry import scale_to_screen
from ..events import detect_events
from ..metrics import calculate_signal_metrics
from ..plotting import plot_task_performance


class SaccadeAnalyzer:
    def __init__(self, saccade_vel_thresh=None, sample_rate=None, screen_size=None):
        self.fs = sample_rate or config.SAMPLE_RATE
        self.W, self.H = screen_size or config.SCREEN_RES_PX
        self.saccade_vel_thresh = (saccade_vel_thresh
                                   if saccade_vel_thresh is not None
                                   else config.SACCADE_VEL_THRESH)
        self.target_radius = self.W * 0.11
        self.stim_onsets, self.target_layouts, self.task_names = \
            config.build_saccade_tasks((self.W, self.H))

    def detect_events(self, xy_data):
        return detect_events(xy_data, self.fs, self.saccade_vel_thresh)

    def calculate_directional_error_rate(self, xy_pred, xy_gt, frames, idx):
        """Fraction of time gaze was NOT inside the correct target box."""
        if idx > 4:
            return 1.0, 1.0, None, None
        task_targets = self.target_layouts[idx]
        total_pred = total_gt = total_frames_in_targets = 0
        current_frame = 0
        target_trace_x, target_trace_y = [], []

        for (tx, ty, dur) in task_targets:
            cx, cy = tx, ty
            start_f = current_frame
            end_f = min(current_frame + dur, frames[-1])
            if start_f >= frames[-1]:
                break
            mask = (frames >= start_f) & (frames < end_f)
            gaze_pred = xy_pred[:, mask]
            gaze_gt = xy_gt[:, mask]
            dist_pred = ((np.abs(gaze_pred[0] - cx) <= self.target_radius)
                         & (np.abs(gaze_pred[1] - cy) <= self.target_radius))
            dist_gt = ((np.abs(gaze_gt[0] - cx) <= self.target_radius)
                       & (np.abs(gaze_gt[1] - cy) <= self.target_radius))
            total_pred += np.sum(dist_pred)
            total_gt += np.sum(dist_gt)
            total_frames_in_targets += np.sum(mask)
            target_trace_x.extend([cx] * np.sum(mask))
            target_trace_y.extend([cy] * np.sum(mask))
            current_frame += dur

        if total_frames_in_targets == 0:
            return 1.0, 1.0, None, None
        error_pred = 1.0 - total_pred / total_frames_in_targets
        error_gt = 1.0 - total_gt / total_frames_in_targets
        return error_pred, error_gt, np.array(target_trace_x), np.array(target_trace_y)

    def calculate_saccade_metrics(self, xy_data, sac, vel, frames, idx):
        """Latency of the first valid saccade per stimulus onset + peak velocity."""
        metrics = {'latency': np.nan, 'miss_count': 0,
                   'peak_velocity': np.max(vel) if len(vel) > 0 else 0}
        if idx > 4:
            return metrics
        stim_onsets = self.stim_onsets[idx]
        if sac.shape[0] == 0:
            return metrics
        onsets = frames[sac[:, 0]]
        latencies = []
        for stim_t in stim_onsets:
            responses = onsets[(onsets >= stim_t) & (onsets <= stim_t + 30)]
            if len(responses) > 0:
                latencies.append(responses[0] - stim_t)
            else:
                latencies.append(np.nan)
                metrics['miss_count'] += 1
        metrics['latency'] = np.array(latencies)
        return metrics

    def calculate_fixation_metrics(self, xy_data, fixs):
        """Fixation count, mean duration (ms), and spatial variance (BCEA proxy)."""
        if fixs.shape[0] == 0:
            return {'count': 0, 'mean_duration_ms': 0, 'spatial_variance': 0}
        durations, centroids = [], []
        for start, end in fixs:
            durations.append((end - start) * (1000.0 / self.fs))
            segment = xy_data[:, start:end]
            if segment.shape[1] > 0:
                centroids.append(np.mean(segment, axis=1))
        if len(centroids) > 1:
            centroids = np.array(centroids)
            spatial_var = np.var(centroids[:, 0]) + np.var(centroids[:, 1])
        else:
            spatial_var = 0
        return {'count': len(fixs), 'mean_duration_ms': np.mean(durations),
                'spatial_variance': spatial_var}

    def calculate_all_metrics(self, tasks, sub_id, h_dir, t_start, sm_fix_sac,
                              screen_res=(1920, 1080), visualization=False,
                              out_dir="gaze_dynamics_out"):
        total_miss = 0
        total_fix_p, total_sac_p, total_dyn, total_fix_g, total_sac_g = [], [], [], [], []

        xy_output_list, xy_gaze_list, frame_list = load_gaze_data(h_dir, tasks, sub_id)

        for i, idx in enumerate(tasks):
            xy_pred = scale_to_screen(xy_output_list[i], screen_res, (self.W, self.H))
            xy_gt = scale_to_screen(xy_gaze_list[i], screen_res, (self.W, self.H))
            frames_norm = frame_list[i] - t_start[idx] - 8
            print(f"------- Frame Start: {frames_norm[0]} End: {frames_norm[-1]} -------")

            sac_p, fix_p, vel_p = self.detect_events(xy_pred)
            sac_g, fix_g, vel_g = self.detect_events(xy_gt)
            print(f"(Output) Saccades: {sac_p.shape} Fixations: {fix_p.shape}")
            print(f"(EyeLink) Saccades: {sac_g.shape} Fixations: {fix_g.shape}")

            if idx in sm_fix_sac[0]:
                sig = calculate_signal_metrics(xy_pred, xy_gt)
                print("Signal Metrics:", sig)
                total_dyn.append([sig['RMSE'], sig['DTW']])

            if idx in sm_fix_sac[1]:
                fm_p = self.calculate_fixation_metrics(xy_pred, fix_p)
                fm_g = self.calculate_fixation_metrics(xy_gt, fix_g)
                print("(Output) Fixation:", fm_p)
                print("(EyeLink) Fixation:", fm_g)
                total_fix_p.append([fm_p['count'], fm_p['mean_duration_ms'], fm_p['spatial_variance']])
                total_fix_g.append([fm_g['count'], fm_g['mean_duration_ms'], fm_g['spatial_variance']])

            if idx in sm_fix_sac[2]:
                sm_p = self.calculate_saccade_metrics(xy_pred, sac_p, vel_p, frames_norm, idx)
                sm_g = self.calculate_saccade_metrics(xy_gt, sac_g, vel_g, frames_norm, idx)
                print("(Output) Saccade:", sm_p)
                print("(EyeLink) Saccade:", sm_g)
                err_p, err_g, tx_arr, ty_arr = self.calculate_directional_error_rate(
                    xy_pred, xy_gt, frames_norm, idx)
                print(f"Directional Error Rate: {err_p * 100:.2f}% (Output) {err_g * 100:.2f}% (EyeLink)")
                total_sac_p.append([sm_p['latency'], sm_p['peak_velocity'], err_p * 100])
                total_sac_g.append([sm_g['latency'], sm_g['peak_velocity'], err_g * 100])
                total_miss += sm_p['miss_count'] + sm_g['miss_count']

                if visualization:
                    plot_task_performance(
                        xy_pred, xy_gt, self.fs, self.task_names[idx],
                        self.saccade_vel_thresh, self.target_radius,
                        tx_arr, ty_arr,
                        save_path=f"{out_dir}/saccade_sub{sub_id}_task{idx}.png")

        return total_dyn, total_fix_p, total_fix_g, total_sac_p, total_sac_g, total_miss
