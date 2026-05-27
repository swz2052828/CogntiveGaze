"""Config-driven orchestrator that runs the selected analyzers end-to-end.

This wires the analyzers to one ``GazeConfig`` and one output directory. The
saccade/heatmap entries assume the per-item gaze files that ``io.load_gaze_data``
reads (each holding ``[xy_smooth, xy_compare, frames]``); that is the format the
model side must emit. The blink entry reads the ``[events, metadata]`` ``.npy``
files used by ``BlinkAnalyzer``.
"""

import os

from . import config, plotting
from .analyzers.blink import BlinkAnalyzer
from .analyzers.saccade import SaccadeAnalyzer
from .analyzers.heatmap import HeatmapAnalyzer


class GazeDynamicsPipeline:
    def __init__(self, cfg=None, out_dir="gaze_dynamics_out", show=False):
        self.cfg = cfg or config.GazeConfig()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        plotting.configure(show=show)

    # -- Blink --------------------------------------------------------------
    def run_blink(self, blink_path="blinks"):
        analyzer = BlinkAnalyzer(self.cfg.data_path, blink_path)
        subject_df, blink_df = analyzer.load_and_validate()
        analyzer.analyze_tasks()
        plotting.plot_validation(subject_df, os.path.join(self.out_dir, "blink_validation.png"))
        plotting.plot_temporal_trends(blink_df, save_path=os.path.join(self.out_dir, "blink_temporal.png"))
        plotting.plot_performance_metrics(subject_df, os.path.join(self.out_dir, "blink_performance.png"))
        analyzer.compare_split_half_time(os.path.join(self.out_dir, "blink_split_half.png"))
        return analyzer

    # -- Saccade ------------------------------------------------------------
    def run_saccade(self, gaze_dir, tasks, subjects=None, sm_fix_sac=None, visualization=True):
        subjects = subjects or self.cfg.subject_ids
        task_begin, _, _ = config.build_task_timeline()
        # Which task indices feed signal / fixation / saccade metrics respectively.
        sm_fix_sac = sm_fix_sac or ([3, 4], [0], [1, 2])
        analyzer = SaccadeAnalyzer(saccade_vel_thresh=self.cfg.saccade_vel_thresh,
                                   sample_rate=self.cfg.sample_rate,
                                   screen_size=self.cfg.screen_res_px)
        results = {}
        for sub_id in subjects:
            try:
                results[sub_id] = analyzer.calculate_all_metrics(
                    tasks, sub_id, gaze_dir, task_begin, sm_fix_sac,
                    screen_res=self.cfg.screen_res_px, visualization=visualization,
                    out_dir=self.out_dir)
            except Exception as e:  # one bad subject shouldn't kill the batch
                print(f"Saccade analysis failed for subject {sub_id}: {e}")
        return results

    # -- Heatmap ------------------------------------------------------------
    def run_heatmap(self, gaze_dir, img_index, subjects=None, bg_img_path_base=None,
                    img_ids=None, targets=None):
        subjects = subjects or self.cfg.subject_ids
        analyzer = HeatmapAnalyzer(self.out_dir, self.cfg.width, self.cfg.height)
        xy, hm, ents = analyzer.load_and_process(subjects, gaze_dir, img_index)
        print(f"Computed entropy for {len(ents)} subject(s); mean Output/EyeLink: "
              f"{ents[:, 0].mean():.3f} / {ents[:, 1].mean():.3f}" if len(ents) else "no data")
        if bg_img_path_base and img_ids and xy:
            analyzer.plot_figures(bg_img_path_base, xy, img_ids, targets)
            analyzer.plot_new_heatmaps(bg_img_path_base, img_ids, xy, xy, targets)
        return analyzer, ents

    # -- Dispatch -----------------------------------------------------------
    def run(self, analyses, **kwargs):
        out = {}
        if "blink" in analyses:
            out["blink"] = self.run_blink(kwargs.get("blink_path", "blinks"))
        if "saccade" in analyses:
            out["saccade"] = self.run_saccade(
                kwargs["gaze_dir"], kwargs["tasks"], subjects=kwargs.get("subjects"),
                visualization=kwargs.get("visualization", True))
        if "heatmap" in analyses:
            out["heatmap"] = self.run_heatmap(
                kwargs["gaze_dir"], kwargs["img_index"], subjects=kwargs.get("subjects"),
                bg_img_path_base=kwargs.get("bg_dir"), img_ids=kwargs.get("img_ids"),
                targets=kwargs.get("targets"))
        return out
