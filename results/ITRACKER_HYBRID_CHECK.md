# iTracker Hybrid arm — replication check

**Date:** 2026-09-17. **Question:** the CCF draft reports iTracker Full Synthetic
9.36 cm and Hybrid 9.35 cm — a 0.01 cm difference — and concludes that the
"Context-Signal Mismatch is a fundamental architectural incompatibility". The
Hybrid arm differs from Full Synthetic only by restoring the ORIGINAL eye crops,
so a 0.01 cm difference means the real eyes bought nothing.

## Architecture is the original

`vit_gaze/multistream_backbones/itracker.py` reproduces Krafka et al. (2016)
`ITrackerModel.py`: `CrossMapLRN2d(size=5, alpha=1e-4, beta=0.75, k=1.0)`,
AlexNet stack with `groups=2`, 12x12x64 flatten. LRN is present. This is not a
modernised re-implementation, so the comparison below is the same architecture
in a different harness, not a different architecture.

(LRN has no fast cuDNN path, which makes iTracker the slowest of the four
backbones here. That is a speed effect, not a numerical one.)

## Positive control

`face_only_mobile_vit` consumes the face crop only and by construction cannot
see the eye crops, so its two arms must be identical:

| arm | Full Synthetic | Hybrid | difference |
|---|---|---|---|
| template A | 11.426 | 11.426 | **+0.000 ± 0.000** |
| template B | 18.372 | 18.372 | **+0.000 ± 0.001** |

Exactly zero. The harness distinguishes the arms when a model uses the eyes and
correctly reports no difference when it does not.

## Result: the reported failure does not replicate

Paired within fold, Hybrid minus Full Synthetic (negative = the real eye crops
are doing work), `base` metric, K=72 clean enrolment, n=5 folds:

| backbone | template A | template B |
|---|---|---|
| **itracker** | **-2.646 ± 2.163** | **-3.482 ± 1.445** |
| mobile_vit | -3.300 ± 1.160 | -4.707 ± 1.543 |
| convnextv2 | -5.033 ± 0.916 | -5.904 ± 1.256 |
| *face_only_mobile_vit* | *+0.000 ± 0.000* | *+0.000 ± 0.001* |

iTracker per fold, template A:

| fold | Real | Full Synth A | Hybrid A | Hyb - FS |
|---|---|---|---|---|
| 0 | 5.435 | 8.104 | 5.152 | **-2.951** |
| 1 | 5.511 | 10.379 | 6.778 | **-3.601** |
| 2 | 10.236 | 9.967 | 11.057 | +1.090 |
| 3 | 4.138 | 7.425 | 4.129 | **-3.296** |
| 4 | 4.329 | 9.513 | 5.041 | **-4.472** |

Four of five folds recover 3.0-4.5 cm. Fold 2 is the known unseeded-init
predict-the-mean collapse (see `folds34-frozen-loss-rootcause`) and its Real
baseline is already 10.236, so it is not a Hybrid-specific effect.

Arm means (base, n=5): Real 5.930, Full Synthetic A 9.078 (+3.148), Hybrid A
6.432 (**+0.502**), Full Synthetic B 9.520 (+3.590), Hybrid B 6.039 (**+0.109**).
Baseline and Full Synthetic agree closely with the draft (6.10 and +3.26); only
the Hybrid arm diverges.

## What is established, and what is not

**Established:** (1) the same original architecture does not reproduce the
reported Hybrid failure in this harness; (2) the draft's reported signature —
Hybrid minus Full Synthetic = 0.01 cm — matches the signature of the face-only
control, i.e. of a model that cannot see the eye crops.

**Not established:** that the original pipeline contains a defect. Its code is
not available here, so this is a replication, not an audit.

**Not checked:** AFFNet, which the draft reports as failing the same way. There
is no AFFNet implementation in this harness.

## Recommendation

Withdraw the "fundamental architectural incompatibility" claim unless a positive
control in the original pipeline shows the iTracker eye stream is active in the
Hybrid arm (Grad-CAM, an eye-stream ablation, or hashing the two arms' input
tensors). If the arm is found to be misfed, the iTracker row of Table 3 must be
re-run — and the resulting conclusion is *stronger*, not weaker: all four
backbones would then agree that Hybrid recovers most of the synthesis penalty.

Note also that these runs supply the paired per-fold spread the draft's Table 3
lacks. iTracker template A has a 95% CI of [-5.332, +0.040] across folds, which
crosses zero only because of the collapsed fold; template B is [-5.276, -1.687].
Reporting should pair at subject level (n=17) rather than fold level (n=5), or
state explicitly how collapsed folds are handled.
