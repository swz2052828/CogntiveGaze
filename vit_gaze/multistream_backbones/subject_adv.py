"""Domain-adversarial subject invariance (DANN) for multistream gaze models.

Idea (Ganin & Lempitsky, JMLR 2016): hang a subject-ID classifier off the
fused per-stream feature vector and insert a Gradient Reversal Layer (GRL)
between them. The discriminator learns to identify the subject; the GRL flips
the sign of its gradient flowing into the encoder, pushing the encoder toward
features from which subject identity (head shape, skin, camera distance
appearance) cannot be recovered while gaze cues are retained. With only ~17
subjects this is mainly a regularizer against subject-specific overfit; it
composes with -- it does not replace -- per-subject calibration.

Training-only: the discriminator is a separate module, never part of the saved
gaze checkpoint, so inference and the gaze_dynamics export bridge are untouched.

Scope notes:
- Multistream only. The feature tap finds the backbone's final regression
  module (``.head`` for the ViT backbone, ``.fc`` for the CNN baselines) and
  captures its input -- exactly the fused vector the gaze head consumes.
- The subject classes are the *training* recordings of the current fold only;
  held-out subjects are never classified, which is what should generalize.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    """Identity forward; gradient is negated and scaled by lambda on backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class SubjectDiscriminator(nn.Module):
    """Small MLP that classifies the fused feature into one of the fold's subjects."""

    def __init__(self, in_dim, num_subjects, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_subjects),
        )

    def forward(self, feat):
        return self.net(feat)


class _FeatureTap:
    """Capture the input to the backbone's final regression module.

    The fused per-stream vector is the input to ``.head`` (ViT) or ``.fc`` (CNN
    baselines), so a forward pre-hook on that module yields exactly the
    representation the gaze head sees -- no backbone code changes needed. Works
    through a torch.compile wrapper by resolving ``_orig_mod`` first.
    """

    HEAD_NAMES = ("head", "fc")

    def __init__(self, model):
        target = self._find_head(model)
        self.captured = None
        self._handle = target.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.captured = args[0]

    @classmethod
    def _find_head(cls, model):
        m = getattr(model, "_orig_mod", model)
        for name in cls.HEAD_NAMES:
            sub = getattr(m, name, None)
            if isinstance(sub, nn.Module):
                return sub
        raise AttributeError(
            "subject-adv: could not find a final regression module "
            f"(looked for {cls.HEAD_NAMES}) on the backbone."
        )

    def remove(self):
        self._handle.remove()


class SubjectAdversary:
    """Owns the GRL, the (lazily built) discriminator and the lambda schedule.

    The discriminator is built on the first batch -- once the fused feature
    width is known -- and its parameters are appended to the supplied optimizer
    so a single backward over ``reg_loss + adv_loss`` updates both: the encoder
    via the reversed gradient (toward invariance) and the discriminator via the
    normal gradient (toward subject classification).
    """

    def __init__(self, model, train_recordings, device, optimizer,
                 max_lambda=0.1, warmup_frac=1.0, total_steps=1):
        self.device = device
        self.optimizer = optimizer
        self.max_lambda = float(max_lambda)
        self.warmup_frac = max(1e-6, float(warmup_frac))
        self.total_steps = max(1, int(total_steps))
        self.tap = _FeatureTap(model)
        recs = sorted(int(r) for r in train_recordings)
        self.rec2idx = {r: i for i, r in enumerate(recs)}
        self.num_subjects = len(recs)
        self.disc = None
        self._step = 0

    def current_lambda(self):
        # Ganin schedule: lambda ramps 0 -> max_lambda so the regressor
        # stabilizes before invariance pressure ramps in.
        p = min(1.0, (self._step / self.total_steps) / self.warmup_frac)
        return self.max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def _ensure_disc(self, feat):
        if self.disc is None:
            self.disc = SubjectDiscriminator(feat.shape[1], self.num_subjects).to(self.device)
            self.optimizer.add_param_group({"params": list(self.disc.parameters())})

    def loss(self, recs):
        """Cross-entropy of the subject discriminator on the last captured feature.

        Lambda is applied inside the GRL (canonical DANN), so the returned loss
        is plain CE: the discriminator gets the full gradient while the encoder
        gets ``-lambda`` times it. Returns ``(adv_loss, lambda)``.
        """
        feat = self.tap.captured
        if feat is None:
            raise RuntimeError("subject-adv: no features captured; run forward first.")
        self._ensure_disc(feat)
        lambd = self.current_lambda()
        logits = self.disc(grad_reverse(feat, lambd))
        idx = torch.tensor([self.rec2idx[int(r)] for r in recs],
                           device=self.device, dtype=torch.long)
        return F.cross_entropy(logits, idx), lambd

    def advance(self):
        self._step += 1

    def remove(self):
        self.tap.remove()
