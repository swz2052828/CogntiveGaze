#!/bin/bash
# DeepPrivacy2 (face) generation, two identity regimes.
#
#   per_subject  one fixed latent seed for all of a participant's frames. This
#                reproduces SimSwap's per-participant template, so the two arms
#                differ in EDIT DOMAIN and not in consistency -- the comparison
#                review point M5 asks for.
#
#   per_frame    a fresh identity every frame. This is the fix our falsifiable
#                claim predicts: protection against a release-only attacker is
#                bounded by how far the transformation breaks per-participant
#                consistency. If linkage collapses here, the claim is supported
#                directly; if it does not, the claim is wrong.
#
# Weights came from the HuggingFace space mirrors (the upstream host
# api.loke.aws.unit.no does not resolve from this cluster); md5 of the generator
# matches the value pinned in configs/fdf/stylegan.py.
#SBATCH --job-name=dp2_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_gen_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=4
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
export HF_HUB_OFFLINE=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze

MODES=(per_subject per_frame)
M=${MODES[$SLURM_ARRAY_TASK_ID]}
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_dp2
mkdir -p "$R"

echo "########## DeepPrivacy2 generation, identity=$M"
/springbrook/share/eng/esrpxk/envs/dp2/bin/python scripts/anon/run_deepprivacy2.py \
    --identity "$M" \
    --out-root "$D/ProcessedDP2_$M" \
    --n-per-subject 150 \
    --report "$R/edit_domain_$M.json" || { echo "FAILED $M"; exit 1; }
echo "ALL_DONE $M"
