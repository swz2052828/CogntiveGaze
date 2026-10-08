#!/bin/bash
# Build detectron2 + DensePose into envs/dp2 ONLY (a venv over envs/gaze;
# nothing is installed into gaze). Needed for DeepPrivacy2's full-body path
# (CSE person detector). CUDA 12.1 to match torch 2.5.1+cu121; GCC 12 is the
# newest host compiler CUDA 12.1 accepts.
#SBATCH --job-name=d2_build
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/d2_build_%j.log
set -euo pipefail
module load CUDA/12.1.1 GCC/12.3.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="8.9"   # L40, the only GPU type on the gpu partition
export FORCE_CUDA=1              # compile CUDA ops on a CPU node
export MAX_JOBS=16
PY=/springbrook/share/eng/esrpxk/envs/dp2/bin/python
SRC=/springbrook/share/eng/esrpxk/third_party/detectron2
echo "nvcc: $(nvcc --version | tail -1)  arch: $TORCH_CUDA_ARCH_LIST"
$PY -m pip install --no-build-isolation --no-deps -e "$SRC"
$PY -m pip install --no-build-isolation --no-deps -e "$SRC/projects/DensePose"
# pure-python deps installed from the login node (compute nodes may lack PyPI)
$PY -c "import detectron2, densepose; from detectron2 import _C; print('detectron2', detectron2.__version__, 'OK')"
echo BUILD_OK
