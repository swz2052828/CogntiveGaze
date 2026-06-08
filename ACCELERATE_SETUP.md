# Accelerate Integration Guide

The training pipeline now uses [HuggingFace Accelerate](https://huggingface.co/docs/accelerate) for distributed training and mixed precision. This replaces the custom AMP setup with a unified framework that handles device placement, mixed precision, and gradient accumulation automatically.

## Quick Start

### Single GPU (RTX 5090 or RTX 2070 Super)

No changes needed from the prior workflow. Accelerate auto-detects your hardware:

```bash
# Auto mixed precision: bf16 on RTX 5090, fp16 on RTX 2070 Super
python -m vit_gaze train --epochs 10 --batch-size 8 \
  --data-path /path/to/data --out-path ./output

# Or disable mixed precision (runs in fp32)
python -m vit_gaze train --mixed-precision no ...
```

### Multi-GPU on Same Machine

For 5 L40 + 1 A40 setup, use Accelerate's launcher:

```bash
# Interactive setup (run once)
accelerate config
# → Selects "Multi-GPU", DDP by default
# → Saves config to ~/.cache/huggingface/accelerate/default_config.yaml

# Then launch training
accelerate launch -m vit_gaze train --epochs 10 --batch-size 8 \
  --data-path /path/to/data --out-path ./output
```

Accelerate will automatically:
- Split the dataset across GPUs (DistributedSampler)
- Synchronize model updates (all-reduce)
- Handle device placement
- Apply mixed precision across all GPUs

### Multi-GPU on Slurm Cluster (Bletchley HPC)

Create a config file for your cluster:

```bash
accelerate config
# Select:
# - "Multi-machine"
# - Number of GPUs per machine / total machines
# - Distributed type: "multi_gpu" (DDP) or "deepspeed" (if installed)
# - Mixed precision: "auto" (recommended)
```

Then submit with Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=gaze_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00

# Load modules (cluster-specific)
module load cuda/11.8 pytorch/2.0

# Launch with Accelerate
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  -m vit_gaze train \
  --epochs 30 \
  --batch-size 8 \
  --data-path /scratch/dataset \
  --out-path /scratch/output \
  --fold-index $SLURM_ARRAY_TASK_ID
```

Run with `sbatch train.sh` or use `srun` for immediate execution on allocated nodes.

## Command-Line Arguments

The following new flags control Accelerate behavior:

### `--mixed-precision {auto,no,fp16,bf16}`
**Default:** `auto`

- `auto`: Detect best dtype for your hardware (bf16 on Ampere+/Blackwell, fp16 on Turing)
- `no`: Disable mixed precision (full fp32)
- `fp16`: Force float16 (with fp32 weight/gradient buffers for stability)
- `bf16`: Force bfloat16 (no loss scaling needed; available on Ampere+)

```bash
python -m vit_gaze train --mixed-precision bf16 ...
```

### `--gradient-accumulation-steps N`
**Default:** `1`

Accumulate gradients over N forward/backward passes before `optimizer.step()`. Useful for simulating larger effective batch sizes on memory-constrained GPUs:

```bash
# Simulate batch_size=32 on a 8GB GPU with batch_size=8
python -m vit_gaze train --batch-size 8 --gradient-accumulation-steps 4 ...
```

### `--log-backend {tensorboard,wandb,none}`
**Default:** `tensorboard`

Logging backend for training metrics (optional; you can still log manually):

```bash
python -m vit_gaze train --log-backend wandb ...
```

### `--no-accelerate`
**Default:** (disabled)

Fall back to the old custom AMP implementation (for debugging only):

```bash
python -m vit_gaze train --no-accelerate ...
```

## Under the Hood

### What Changed

**Before (custom AMP):**
```python
amp_enabled, amp_dtype = accel.resolve_amp(device, args.amp)
scaler = accel.make_grad_scaler(amp_enabled, amp_dtype)

with accel.autocast(device, amp_enabled, amp_dtype):
    loss = compute_loss(...)
    
if scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

**After (Accelerate):**
```python
accelerator = init_accelerator(mixed_precision="auto")
model, optimizer = accelerator.prepare(model, optimizer)

with autocast_context(...):
    loss = compute_loss(...)

accelerator.backward(loss)
optimizer.step()
```

### Files Modified

- **`vit_gaze/training.py`**: Main training loop now uses Accelerator
- **`vit_gaze/accelerate_utils.py`**: New wrapper module for initialization
- **`vit_gaze/cli.py`**: New CLI arguments for mixed precision and gradient accumulation
- **`requirements-train.txt`**: New dependency file (includes `accelerate>=0.20.0`)

### Backward Compatibility

The `--amp` flag is **replaced** by `--mixed-precision`:

| Old | New |
|-----|-----|
| `--amp` | `--mixed-precision auto` or `--mixed-precision bf16` (for 5090) |
| (default: off) | (default: auto, detects hardware) |

Existing `--no-tf32` and `--compile` flags still work; TF32 and torch.compile are complementary to Accelerate.

### Device Placement

Accelerate handles device placement automatically:

```python
if use_accelerate:
    device = accelerator.device  # Auto-selected
    model = model.to(device)     # Already done by accelerator.prepare()
    
# Data movement in batches:
gaze = batch["gaze"].to(device)  # Still required for labels
```

## Installation

Install Accelerate via the new requirements file:

```bash
pip install -r requirements-train.txt
```

Or manually:

```bash
pip install accelerate>=0.20.0
pip install tensorboard  # For local logging (optional)
pip install wandb        # For W&B integration (optional)
```

## Advanced: Custom Config Files

Create a custom Accelerate config for your environment:

**`my_accelerate_config.yaml`:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: "no"
gpu_ids: "0,1,2,3,4,5"
machine_rank: 0
main_training_function: train
mixed_precision: bf16
num_machines: 1
num_processes: 6
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Then use it:

```bash
accelerate launch --config_file my_accelerate_config.yaml -m vit_gaze train ...
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size and/or use gradient accumulation:

```bash
python -m vit_gaze train --batch-size 4 --gradient-accumulation-steps 2 ...
```

### Mixed Precision Issues

Some models don't play well with fp16. Try bf16 or disable mixed precision:

```bash
python -m vit_gaze train --mixed-precision bf16 ...
python -m vit_gaze train --mixed-precision no ...
```

### Slow on Multi-GPU

Check if data loading is the bottleneck:

```bash
python -m vit_gaze train --num-workers 8 --prefetch-factor 4 ...
```

Or use the old AMP implementation for comparison:

```bash
python -m vit_gaze train --no-accelerate ...
```

## References

- [HuggingFace Accelerate Docs](https://huggingface.co/docs/accelerate)
- [Distributed Training Tutorial](https://huggingface.co/docs/accelerate/en/basic_tutorials/distributed_training)
- [Slurm Integration](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch)
- [Custom Hardware Setups](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#launching-training-from-a-python-script)
