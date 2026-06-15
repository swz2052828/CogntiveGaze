"""Accelerate library utilities for distributed training and mixed precision.

Wraps HuggingFace Accelerate to provide:
  - Distributed training (DDP, DeepSpeed, FSDP) on single/multi-GPU setups
  - Automatic mixed precision (bf16 on Ampere+, fp16 on Turing)
  - Gradient accumulation
  - Device-agnostic code (works on single GPU, multi-GPU, TPU, etc.)

User can customize behavior via:
  - Command-line arguments: --mixed-precision, --gradient-accumulation-steps, etc.
  - Environment variables (HF_ACCELERATE_MODE, etc.)
  - accelerate CLI config tool: accelerate config
  - ~/.cache/huggingface/accelerate/default_config.yaml

Multi-GPU Setup:
  - Single machine, multi-GPU: Just run with `python -m accelerate.launch`
    $ accelerate config  # interactive setup (DDP, DeepSpeed, etc.)
    $ accelerate launch train.py --epochs 10 --batch-size 8 ...

  - Multi-machine (Slurm): Use environment variables or create a config file
    https://huggingface.co/docs/accelerate/en/basic_tutorials/launch
    https://huggingface.co/docs/accelerate/en/basic_tutorials/distributed_training

  - Multi-GPU on Bletchley HPC: See `accelerate config` for cluster-specific setup
    Typical pattern: sbatch runner that sets LAUNCHER=accelerate and passes config

To disable Accelerate and fall back to the old device/AMP handling:
  $ python -m vit_gaze train --no-accelerate ...
"""

from accelerate import Accelerator


def init_accelerator(
    mixed_precision="auto",
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir=None,
):
    """Initialize the Accelerator for distributed training.

    Args:
        mixed_precision: "auto", "no", "fp16", "bf16". "auto" detects the best
            dtype for the current hardware (bf16 on Ampere+, fp16 on Turing).
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        log_with: Logging backend ("tensorboard", "wandb", "none").
        project_dir: Directory for Accelerator's internal state and checkpoints.

    Returns:
        Accelerator instance ready to wrap models/optimizers/dataloaders.
    """
    # Convert "none" (CLI arg) to None (Accelerator API)
    if log_with == "none":
        log_with = None

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=log_with,
        project_dir=project_dir,
    )
    return accelerator


def should_use_accelerate(args):
    """Quick check: return True if Accelerate should be used based on args.

    Allows a gradual migration: if args.no_accelerate is set, fall back to the
    old custom AMP path (which is still available but deprecated).
    """
    return not getattr(args, "no_accelerate", False)
