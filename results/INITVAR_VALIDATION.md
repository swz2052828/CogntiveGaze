# Initialisation-variance replication — the utility axis

**Completed 2026-09-20.** Jobs 1283734–49 (training), 1284531 + 1285486
(evaluation). 80/80 checkpoints, 80/80 evaluations, no failures.

## Design, and why the seed is held fixed

`--seed` in this codebase reaches exactly one place: `recording_kfolds()`, which
permutes which **participants** land in which fold. There is no
`torch.manual_seed` / `np.random.seed` anywhere in `vit_gaze`, so weight
initialisation, DataLoader order and dropout are unseeded.

Re-running training at the **same** seed therefore holds the participant
partition fixed and redraws the initialisation — exactly the variance component
in question, with the split held constant. Changing the seed would confound
initialisation with partition.

Five draws: the original `runs/anon_swap_utility` run plus four repeats, each a
full 4 backbones x 5 folds. Same recipe throughout (`run_base_only_springbrook.
sbatch`, lr 1e-4, 20 epochs, `meanno7_clean`, use-grid), from
`clean_canonical_runs.tsv`.

### Three timeouts, checked and kept

`iv1_itracker` fold 3, `iv3_itracker` folds 2 and 3 hit the 16 h limit (iTracker
is ~8.5 h/fold; `CrossMapLRN2d` has no fast cuDNN path). All three reached 18-19
of 20 epochs and had plateaued — best epoch at 17, 16 and 18 respectively, with
the following epochs worse. Several *completed* runs stopped earlier than that
under early stopping (patience 5, some at 7-11 epochs). The checkpoints are the
best-validation ones and are kept.

## Result: the conclusion is robust

Hybrid minus Full Synthetic, per draw (mean over 5 folds):

| Model | Template | draw 0-4 | mean ± SD | effect / SD |
|---|---|---|---|---|
| iTracker | A | −2.65 −2.57 −2.76 −2.81 −2.68 | **−2.693 ± 0.097** | 27.8 |
| iTracker | B | −3.48 −3.21 −3.69 −3.91 −3.99 | **−3.656 ± 0.320** | 11.4 |
| MobileNet-V3 L | A | −3.20 −3.79 −3.76 −3.37 −3.29 | **−3.481 ± 0.274** | 12.7 |
| MobileNet-V3 L | B | −4.95 −6.46 −6.60 −5.85 −6.14 | **−5.999 ± 0.655** | 9.2 |
| AFFNet | A | −1.97 −2.49 −2.08 −2.12 −2.45 | **−2.221 ± 0.236** | 9.4 |
| AFFNet | B | −3.37 −4.53 −3.99 −3.89 −3.70 | **−3.897 ± 0.428** | 9.1 |
| MGazeNet | A | −2.55 −2.56 −2.41 −2.71 −2.44 | **−2.535 ± 0.119** | 21.3 |
| MGazeNet | B | −4.90 −4.39 −4.09 −4.06 −3.93 | **−4.273 ± 0.389** | 11.0 |

**All 40 draw-level values are negative.** Across-draw SD is an order of
magnitude below the effect in every cell.

## The initialisation noise floor, measured

Real Data baseline across the five draws:

| Model | mean | SD |
|---|---|---|
| iTracker | 5.85 | 0.143 |
| MobileNet-V3 Large | 4.98 | 0.128 |
| AFFNet | 5.67 | 0.203 |
| MGazeNet | 4.96 | 0.040 |

This is the number the original Table 3 needed and did not have. It reported
**±0.03 cm** differences as "Preserved"; the initialisation noise on the baseline
alone is 0.04-0.20 cm, so ±0.03 is an order of magnitude below the floor and
cannot support a claim in either direction.

## A correction to our own wording

With five draws the test has enough power to resolve small effects, and most
Hybrid arms turn out to be **detectably worse** than the real-data baseline —
by very little. Paired within draw:

| Model | Tmpl | Hybrid − Real | 95% CI | Verdict |
|---|---|---|---|---|
| iTracker | A | +0.305 | [+0.104, +0.507] | worse |
| iTracker | B | −0.031 | [−0.136, +0.073] | **indistinguishable** |
| MobileNet-V3 L | A | +0.182 | [+0.007, +0.358] | worse |
| MobileNet-V3 L | B | +1.044 | [+0.691, +1.396] | worse |
| AFFNet | A | +0.185 | [+0.037, +0.334] | worse |
| AFFNet | B | +0.078 | [+0.007, +0.149] | worse |
| MGazeNet | A | +0.243 | [−0.015, +0.501] | **indistinguishable** |
| MGazeNet | B | +0.482 | [+0.182, +0.782] | worse |

Full Synthetic is worse than baseline in 8 of 8 cells, by +2.41 to +7.04 cm.

**So "Hybrid preserves accuracy" is too strong and must not be used.** Six of
eight cells are detectably worse. The defensible claim is a recovery fraction:

| Model | Tmpl | Full Synth cost | Hybrid cost | Penalty recovered |
|---|---|---|---|---|
| iTracker | A | +3.00 | +0.31 | 90% |
| iTracker | B | +3.62 | −0.03 | 101% |
| MobileNet-V3 L | A | +3.66 | +0.18 | 95% |
| MobileNet-V3 L | B | +7.04 | +1.04 | 85% |
| AFFNet | A | +2.41 | +0.19 | 92% |
| AFFNet | B | +3.98 | +0.08 | 98% |
| MGazeNet | A | +2.78 | +0.24 | 91% |
| MGazeNet | B | +4.75 | +0.48 | 90% |

> **Hybrid recovers 85–101% of the synthesis penalty, leaving a small residual
> cost of +0.08 to +1.04 cm that is statistically detectable in six of eight
> cells.**

This does not weaken the paper's argument — which is that the recovery is
achieved by releasing the participant's own ocular pixels — and it keeps our own
wording to the standard we apply to the draft's "+0.03 cm = Preserved".

## Final Table 3

Mean over 5 folds, then mean ± SD over 5 initialisations. Impact relative to that
model's own Real Data baseline.

| Model | Real Data | Full Synthetic (A) | Hybrid (A) | Full Synthetic (B) | Hybrid (B) |
|---|---|---|---|---|---|
| iTracker | 5.85 ± 0.14 | 8.85 ± 0.17 (+3.00) | 6.16 ± 0.20 (+0.31) | 9.48 ± 0.21 (+3.62) | 5.82 ± 0.20 (−0.03) |
| MobileNet-V3 Large | 4.98 ± 0.13 | 8.65 ± 0.18 (+3.66) | 5.17 ± 0.16 (+0.18) | 12.03 ± 0.53 (+7.04) | 6.03 ± 0.25 (+1.04) |
| AFFNet | 5.67 ± 0.20 | 8.07 ± 0.25 (+2.41) | 5.85 ± 0.24 (+0.19) | 9.64 ± 0.53 (+3.98) | 5.74 ± 0.25 (+0.08) |
| MGazeNet | 4.96 ± 0.04 | 7.74 ± 0.20 (+2.78) | 5.21 ± 0.21 (+0.24) | 9.72 ± 0.40 (+4.75) | 5.45 ± 0.25 (+0.48) |

The face-only control (`face_only_mobile_vit`, seed42 draw only) gives Hybrid
minus Full Synthetic = +0.000 ± 0.000 — a model that cannot see the eye crops
reports exactly zero, which is what licenses reading every other row as a real
effect.

## Incidental observation

iTracker collapses to predict-the-mean on **fold 2** in both the seed42 draw
(Real 10.24) and rep3 (10.05). The same participant partition collapses under
different initialisations, so this is a property of that split's training set for
iTracker, not initialisation luck alone. It does not affect the paired
conclusion — both arms use the same checkpoint and the collapse cancels — but it
explains why fold-level intervals are wide for iTracker.

## Files

`runs/initvar/rep{1..4}/<backbone>/base/seed42/` — checkpoints
`runs/initvar_utility/iv_rep{1..4}_<backbone>_<arm>.csv` — evaluations
`results/paper/initvar_summary.json`
`scripts/submit_initvar.sh`, `scripts/anon/run_initvar_eval.sbatch`

**The `meta` column of the initvar CSVs is meaningless** — it pairs each repeat's
base with the seed42 adapter. Only `base` is used.
