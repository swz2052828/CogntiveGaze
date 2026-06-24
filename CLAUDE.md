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
