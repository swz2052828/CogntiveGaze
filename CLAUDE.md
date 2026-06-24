# Project guidance for Claude / contributors

## ⛔ NEVER run heavy compute on the login node

The Blythe HPC login nodes (`login01-04.blythe.scrtp.warwick.ac.uk`) are shared
and their Acceptable Use Policy **forbids intensive workloads**. On 2026-06-20 a
usage-policy violation was auto-flagged (temporary limits of 6 cores / 6 GiB
imposed) because CPU-heavy OpenCV jobs (face-detector comparison + Haar
diagnosis) were run directly on the login node at ~680% CPU.

**Rule:** any compute-heavy task MUST go through SLURM on a compute node — never
`Bash run_in_background` on the login node. This includes:
- OpenCV `detectMultiScale` sweeps / video frame-decoding loops
- model training / inference
- large-scale data processing or diagnostics

Light, quick, single-threaded work (a few seconds, small file reads, `squeue`,
`rclone` transfers) on the login node is fine. Sustained multi-core or long
loops are not.

### Partitions
| partition | type | notes |
|---|---|---|
| `compute` | CPU | 168 cores/node, 2-day limit — CPU jobs (face detect, preprocessing, diagnostics) |
| `gpu` / `vis` | GPU | training/inference |
| `hmem` | high-memory CPU | |
| `int` | interactive | |

### How to submit
- Template for CPU diagnostics: `scripts/run_haar_diagnosis.sbatch`
  (`--partition=compute`, `--cpus-per-task=N`, sets `OMP_NUM_THREADS`,
  uses the `envs/facedet` venv).
- Submit with `sbatch --parsable <script>`; watch with a monitor on
  `squeue -h -j <jobid>`.
- Training jobs: `scripts/run_meta_pipeline_springbrook.sbatch`
  (`--partition=gpu,vis`).

### Environments
- `envs/gaze` — training (torch, numpy 2.x, accelerate). Python 3.11.
- `envs/facedet` — face-detection comparison (mediapipe 0.10.14 + solutions API,
  opencv, numpy <2). Python 3.9. Kept separate so mediapipe's numpy<2 pin does
  not disturb the training env.

## Calibration-point clustering & frame selection

The pre-task calibration frames are produced by a curated clustering pipeline,
not raw k-means. Touch these before changing how calibration support is built.

**`cluster_calib_points.py`** clusters per-frame iris data into the 9 grid points
(3x3) per subject. Run with `envs/gaze/bin/python` (numpy 2.x); it's light enough
for the login node. Pipeline order in `process()`:
`segment_30 -> cluster -> apply_overrides -> merge_runs -> cycle_phases -> recompute`.
Outputs `datasets/calib_clusters/subid_<id>_{clusters,centers}.csv` + plots.

- **`merge_runs()`** merges consecutive segments that are the SAME cell +
  temporally continuous (frame gap <= 12) + spatially neighbouring (centre dist
  <= 3px) into one cluster — a fixation split by a brief dropout. Runs AFTER
  overrides.
- **`OVERRIDES` / `apply_overrides()`** — per-subject DSL operating on original
  labels sorted by iris_y (selectors: `all`, `seg_top/bot`, `grp_low`, `rest`).
- **`POST_MERGE_EDITS` / `apply_post_merge_edits()`** — manual edits keyed by the
  MERGED cluster index shown in `plot_calib_timeorder.py`: `start`/`end` (mark
  before/after), `remove`, `relabel`, `only` (restrict a cell to listed
  clusters), `keep` (force-keep, clears before/removed/after + off-grid flag).
- **Principle (the auto default):** each run = MC(center) -> 8 edges one-by-one
  -> MC, repeated TWICE. `cycle_phases`/`cycle_completeness` (min_edges=8,
  min_dur=8) split kept clusters into 2 cycles at MC returns; a complete cycle
  must reach all 8 edges. Target state: all subjects = 2 complete cycles, every
  edge reached. Do NOT remove a cluster unless a rule justifies it.
- **Blinks:** EAR blink detection (`process_calib_windows` -> per-frame `blink`
  in `calib_processed/<id05d>/metadata.mat`) is overlaid via shared
  `load_blink()`; blink frames are tagged (`is_blink`), unlinked, not deleted.

**`select_calib_frames.py`** deterministically picks the K support frames
(replaces leak-prone random-K-from-test). Rules: prefer cycle 2; longer-duration
cluster wins; distinct point+cycle; middle non-blink frame. K levels:
**4** (corners), **9** (points), **18/36/72** (9 points x 2 cycles x m=1/2/4
frames, evenly spaced within a cluster at L*k/(m+1)). Output
`datasets/calib_selection/calib_selection.csv`.

**`generate_calib_support.py`** (SLURM, facedet env) builds the per-K
GazeCapture-format support sets `datasets/calib_support_K{K}/<rec05d>/...`.
Labels are CENTERED cm (see below) — matching the task `labelDotXCam/YCam`.

## Per-subject calibration must be deploy-faithful (no test-set leakage)

Per-subject calibration uses each subject's **pre-task calibration recording**,
NOT frames sampled from the test session. Random-K-from-test support leaks the
evaluation data; the real product enrolls a user on a separate calibration task
*before* the gaze task, so calibration must mirror that.

**Rules:**
- **Support = pre-task calibration frames.** Use the deterministic
  `datasets/calib_support_K{4,9,18,36,72}` sets (built by
  `cluster_calib_points.py` → `select_calib_frames.py` → `generate_calib_support.py`),
  one fixed support set per subject. Never draw the support from the test
  recording.
- **Calibrate each subject on their own frames** (per-recording), not pooled
  across subjects.
- **meta** adapter (`vit_gaze/meta.py`) is meta-trained with
  `--calib-support-root` so the inner-loop support is the subject's calibration
  frames and the query is in-task frames — matching deployment. Without this the
  adapter is a near-no-op (it was trained on a support distribution it never sees
  at deploy).
- **SVR** baselines (`svr` prediction-space, `svr_embed` embedding-space) tune
  `(C, gamma, epsilon)` via the swarm search in `vit_gaze/svr_search.py`
  `--calib-support-root` mode: fit on training subjects' calibration frames,
  score on their in-task frames. Leak-free (held-out subjects never seen by the
  tuner). `fc_ft` keeps fixed Zhu-et-al. defaults (lr 5e-5, 20 steps).
- Gaze labels are camera-centered cm: `((sx-1920/2)/1920*54.4,
  (sy-1080/2)/1080*30.4)` — same convention as the task `labelDotXCam/YCam`.
- Runners: `scripts/run_calib_metatrain.sbatch` (retrain adapters with calib
  support) and `scripts/run_calib_metacompare.sbatch` (`SVR_TUNE=1` for the
  leak-free swarm search) write to `*_calib` checkpoint dirs / CSVs, leaving the
  original random-K results untouched for comparison.
