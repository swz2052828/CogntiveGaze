"""Per-subject adapters for meta-learned calibration.

Both adapters take the fused per-stream feature ``f`` (the output of a
backbone's ``forward_features``) and return a modulated feature ``f'`` that the
shared gaze head then reads. Both initialize to the identity so an un-adapted
model equals the base model, and both expose a *functional* ``func(f, params)``
so the meta-learner can run inner-loop steps on cloned "fast weights" without
mutating the module.

* FiLM (Perez et al. 2018): per-subject scale+shift ``f' = gamma * f + beta``.
  ``2*dim`` params -- small enough to fit from a handful of calibration frames.
* LoRA (Hu et al. 2021): low-rank residual ``f' = f + scaling * (f A^T) B^T``
  with ``B`` init 0. ``2*rank*dim`` params; more expressive, higher overfit risk
  at small K. Realized as a low-rank correction at the head input (uniform with
  FiLM: both map ``f -> f'``), which is the same low-rank family as LoRA on the
  head's first Linear.
"""

import torch
import torch.nn as nn


class FiLMAdapter(nn.Module):
    kind = "film"

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def func(self, f, params):
        gamma, beta = params
        return f * gamma + beta

    def forward(self, f):
        return self.func(f, list(self.parameters()))


class LoRAAdapter(nn.Module):
    kind = "lora"

    def __init__(self, dim, rank=8, alpha=8.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(dim, rank))

    def func(self, f, params):
        A, B = params
        delta = (f @ A.t()) @ B.t()
        return f + self.scaling * delta

    def forward(self, f):
        return self.func(f, list(self.parameters()))


def make_adapter(kind, dim, rank=8, alpha=8.0):
    if kind == "film":
        return FiLMAdapter(dim)
    if kind == "lora":
        return LoRAAdapter(dim, rank=rank, alpha=alpha)
    raise ValueError(f"Unknown adapter kind {kind!r}; choose 'film' or 'lora'.")
