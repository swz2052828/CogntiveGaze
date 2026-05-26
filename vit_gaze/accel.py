"""Runtime acceleration helpers shared by training and explanation.

These auto-detect the GPU and turn on speedups that do not change results:

  - cuDNN autotuner (``benchmark``) for fixed-size inputs
  - TF32 matmul/conv on Ampere+ GPUs (silently ignored on Turing such as the
    2070 Super, used on Blackwell such as the 5090)

Mixed precision (AMP) is opt-in. When AMP is off the numerics are identical to
the original fp32 path. When on, bf16 is used where the GPU supports it (e.g.
the 5090) and fp16 with gradient loss scaling otherwise (e.g. the 8 GB 2070
Super, which has no bf16 path but does have fast fp16 tensor cores).

The same source therefore runs well on either machine without code changes:
the capabilities are detected at runtime, not hard-coded.
"""

import contextlib

import torch


def _as_device(device):
    return device if isinstance(device, torch.device) else torch.device(device)


def configure_backends(enable_tf32=True):
    """Enable free, accuracy-preserving CUDA backend speedups.

    Safe to call on CPU (no-op) and on Turing (TF32 flags are simply ignored
    by hardware without that path).
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def amp_dtype_for_device(device):
    """Widest fast autocast dtype the device supports: bf16 if available else fp16."""
    dev = _as_device(device)
    if dev.type != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def resolve_amp(device, use_amp):
    """Return ``(enabled, dtype)`` for autocast given the request flag."""
    dev = _as_device(device)
    if not use_amp or dev.type != "cuda":
        return False, None
    return True, amp_dtype_for_device(dev)


@contextlib.contextmanager
def autocast(device, enabled, dtype):
    """Autocast context that is a no-op unless AMP is enabled on CUDA."""
    dev = _as_device(device)
    if enabled and dev.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def make_grad_scaler(enabled, dtype):
    """GradScaler that is only active for fp16 (bf16 keeps fp32 dynamic range)."""
    use_scaler = bool(enabled) and dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


def describe(device, amp_enabled, amp_dtype):
    """Short human-readable summary for log lines."""
    dev = _as_device(device)
    parts = [f"device={dev}"]
    if dev.type == "cuda":
        try:
            parts.append(f"gpu={torch.cuda.get_device_name(dev)}")
        except Exception:
            pass
        tf32 = torch.backends.cuda.matmul.allow_tf32
        parts.append(f"tf32={'on' if tf32 else 'off'}")
        parts.append(f"cudnn_benchmark={'on' if torch.backends.cudnn.benchmark else 'off'}")
    if amp_enabled:
        dtype_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
        parts.append(f"amp={dtype_name}")
    else:
        parts.append("amp=off")
    return " ".join(parts)
