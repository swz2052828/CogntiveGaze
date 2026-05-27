"""Command-line entry for the gaze-dynamics pipeline.

Examples
--------
  python -m gaze_dynamics --data-path ./data --analyses blink
  python -m gaze_dynamics --analyses saccade --gaze-dir ./data/saccade --tasks 0,1,2,3,4
  python -m gaze_dynamics --analyses heatmap --gaze-dir ./data/heatmap --img-index 7 \
      --bg-dir ./images --img-ids 1029 --show
"""

import argparse

from . import config
from .pipeline import GazeDynamicsPipeline


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_parser():
    p = argparse.ArgumentParser(description="Unified gaze-dynamics analysis pipeline.")
    p.add_argument("--data-path", default="data", help="Dataset root (ASCII/, blinks/, binaries).")
    p.add_argument("--out-dir", default="gaze_dynamics_out", help="Where figures/results are written.")
    p.add_argument("--analyses", default="blink",
                   help="Comma list of analyses to run: blink,saccade,heatmap.")
    p.add_argument("--show", action="store_true",
                   help="Also display figures interactively (always saved regardless).")
    p.add_argument("--subjects", type=_int_list, default=None,
                   help="Comma list of subject ids; defaults to the configured cohort.")

    # blink
    p.add_argument("--blink-path", default="blinks", help="Subdir under data-path with blink .npy files.")
    # saccade / heatmap shared
    p.add_argument("--gaze-dir", default=None, help="Directory of per-item gaze files (saccade/heatmap).")
    p.add_argument("--tasks", type=_int_list, default=None, help="Saccade task indices, e.g. 0,1,2,3,4.")
    # heatmap
    p.add_argument("--img-index", type=int, default=None, help="Heatmap image index to process.")
    p.add_argument("--bg-dir", default=None, help="Background stimulus image directory.")
    p.add_argument("--img-ids", type=_int_list, default=None, help="Stimulus image ids for figures.")
    p.add_argument("--no-viz", action="store_true", help="Skip per-trial figures (saccade).")
    return p


def main():
    args = build_parser().parse_args()
    analyses = [a.strip() for a in args.analyses.split(",") if a.strip()]
    cfg = config.GazeConfig(data_path=args.data_path)
    if args.subjects:
        cfg.subject_ids = args.subjects

    pipe = GazeDynamicsPipeline(cfg=cfg, out_dir=args.out_dir, show=args.show)

    if "saccade" in analyses and (args.gaze_dir is None or args.tasks is None):
        raise SystemExit("--analyses saccade requires --gaze-dir and --tasks")
    if "heatmap" in analyses and (args.gaze_dir is None or args.img_index is None):
        raise SystemExit("--analyses heatmap requires --gaze-dir and --img-index")

    pipe.run(
        analyses,
        blink_path=args.blink_path,
        gaze_dir=args.gaze_dir,
        tasks=args.tasks,
        img_index=args.img_index,
        bg_dir=args.bg_dir,
        img_ids=args.img_ids,
        subjects=args.subjects,
        visualization=not args.no_viz,
    )
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
