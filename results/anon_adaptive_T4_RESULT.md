# T4: can an attacker who knows the method adapt to it?

Answers the second of the four questions the project set out to test. The
zero-effort attacks (T2) apply a pretrained recogniser to the anonymised image
as-is. T4 is the harder threat: the attacker **knows the operator, can run it
themselves, and therefore holds paired (original, anonymised) imagery for people
who are not the target.** They learn what the operator does to an embedding and
undo as much of it as is learnable.

Protocol: gallery = original embeddings, first temporal half, all 18 subjects;
probe = anonymised embeddings, second half, passed through a de-anonymisation map
fitted **leave-one-subject-out**, so the map never sees a frame of the person it
is used against. Metrics follow `id_attack.py` exactly.

Maps in increasing order of attacker effort: `shift` (mean offset, one
512-vector), `procrustes` (best orthogonal map), `ridge` (full linear map, alpha
chosen inside the training subjects).

## Result

| operator | map | d′ | Rank-1 | TAR@1e-3 | | in-sample d′ | in-sample Rank-1 |
|---|---|---|---|---|---|---|---|
| **SimSwap** | none (T2) | 1.97 | 57.7% | 18.9% | | — | — |
| | shift | 1.57 | 56.8% | **22.6%** | | 2.66 | 87.4% |
| | procrustes | −0.27 | 0.0% | 0.0% | | 12.32 | 100% |
| | ridge | −0.01 | 0.0% | 0.0% | | 18.79 | 100% |
| blackbox | none (T2) | 5.02 | 100% | 99.1% | | — | — |
| | shift | 4.90 | 100% | 99.1% | | 5.57 | 100% |
| | ridge | 0.74 | 3.0% | 0.0% | | 19.05 | 100% |
| blur0.50 | none (T2) | 4.04 | 100% | 97.6% | | — | — |
| | shift | 4.00 | 99.8% | 69.4% | | 4.72 | 100% |
| | ridge | 0.60 | 0.0% | 0.0% | | 19.18 | 100% |
| blur0.20 | none (T2) | 4.71 | 100% | 99.6% | | — | — |
| | ridge | 0.65 | 0.0% | 0.0% | | 19.20 | 100% |
| pixel0.25 | none (T2) | 5.57 | 100% | 100% | | — | — |
| | ridge | 0.80 | 2.2% | 0.0% | | 19.26 | 100% |

## Reading

**A linear adaptive attacker gains essentially nothing on a person it could not
train on.** The only adaptation that transfers at all is the mean shift, and even
that is marginal and not consistent across metrics: on SimSwap it moves
TAR@FAR=1e-3 from 18.9% to **22.6%** (+3.7 points) while *lowering* d′ from 1.97
to 1.57. Everywhere else it is neutral or slightly negative.

**The fitted linear maps memorise identities rather than learn the operator.**
Procrustes and ridge reach **Rank-1 100% and d′ 12–19 in-sample** and collapse to
**0–33% and d′ ≈ 0 leave-one-subject-out** — below the `none` baseline, and for
SimSwap below chance (5.6%). The mechanism is visible in the numbers: fitted on
17 subjects, the map projects any input into the span of *those* subjects'
original embeddings, so a new person is mapped onto somebody else and
systematically fails to match themselves.

The in-sample row is why this is reportable. Without it, a below-chance LOO
number is indistinguishable from a bug in the fitting. With it, the fitting is
demonstrably working, and the gap between the two rows **is** the finding.

**The question only has force for SimSwap.** For blur, pixelation and blackbox
the zero-effort attack already reaches 97.6–100% TAR@1e-3, so there is nothing
left for an adaptive attacker to add — those face crops retain the byte-exact eye
boxes, which is the leak already established in §2.1. SimSwap is the only
operator where T2 is far from saturated (18.9%), and it is therefore the only one
where the adaptive question is live. It survives it, with a 3.7-point loss.

## What this does and does not license

**Licensed:** "an attacker who knows the operator and fits a linear
de-anonymisation map on other people's paired imagery gains at most 3.7 points of
TAR@FAR=1e-3 against SimSwap, and nothing against the conventional operators."

**Not licensed:** "SimSwap is robust to adaptive attack." This tests *linear*
adaptation on frozen ArcFace embeddings. The protocol's T4 definition also allows
fine-tuning the recogniser backbone on anonymised imagery, which is a strictly
stronger attacker and is not tested here. The honest statement is that the
cheapest and most obvious adaptive strategies fail, and that a fine-tuning
attacker remains open.

That gap is worth closing before submission, and is cheap relative to what it
buys: fine-tuning one recogniser on anonymised imagery of 17 subjects and testing
on the 18th, repeated leave-one-out.

## Cost

One CPU job, four cores, under a minute, on cached embeddings. No GPU.
`scripts/anon/adaptive_attack.py`, results in `results/anon_adaptive/`.

---

# CORRECTION (2026-09-16): the eye-crop claim needs its attacker named

The section below measures the eye crops against a gallery of ORIGINAL crops,
i.e. an attacker holding the target's enrolment photographs. Measured instead
against the released corpus itself — gallery and probe both drawn from the
anonymised data, which is the attacker this paper's deployment premise actually
implies — the same SimSwap eye crops identify at **42.8% / 62.5%** TAR@FAR=1e-3
(Rank-1 86–89%), not at the floor. Both numbers are correct; they answer
different questions. See `anon_swap2_RESULT.md` §2. Every statement below about
the swap "removing" eye-region identity holds only for the originals-in-hand
attacker.

# T4 on the eye crops — the arm that matters for the method

The face-crop result above leaves a gap. SimSwap's face crop starts with 18.9%
TAR for a zero-effort attacker, so adaptation there had residual signal to work
with. **The eye crops start at the nuisance floor** (0.2–2.0% against a 1.6%
floor), and whether an informed attacker can pull signal out of noise is a
different question that does not follow from the face arm.

It also happens to be the arm the whole method rests on: preserving the original
eyes costs +0.03 MSE of utility and gives up ~63 points of TAR, so if adaptation
could recover identity from *swapped* eyes, the trade-off changes shape.

| eye | map | d′ | Rank-1 | TAR@1e-3 | | in-sample d′ | in-sample Rank-1 | in-sample TAR@1e-3 |
|---|---|---|---|---|---|---|---|---|
| left | none (T2) | 0.40 | 23.1% | 0.0% | | — | — | — |
| | shift | 0.45 | 24.4% | 1.3% | | 0.59 | 29.8% | 1.9% |
| | procrustes | 0.44 | 2.0% | 0.0% | | 2.24 | 87.2% | 59.6% |
| | ridge | 0.38 | 2.0% | 0.0% | | 2.51 | 92.5% | 83.8% |
| right | none (T2) | 0.92 | 29.7% | 2.0% | | — | — | — |
| | shift | 0.91 | 32.3% | 1.5% | | 1.05 | 41.8% | 2.6% |
| | procrustes | 0.66 | 5.6% | 0.1% | | 2.41 | 89.8% | 53.5% |
| | ridge | 0.55 | 3.1% | 0.0% | | 3.17 | 97.4% | 85.4% |

Session-nuisance floor: d′ 0.46, TAR@1e-3 **1.6%**.

The `none` rows reproduce the published periocular figures (left d′ 0.392 /
Rank-1 21.0%, right d′ 0.969 / 31.3%) to within the frame draw, so this is the
same measurement with a map bolted on.

## Reading

**Adaptation does not lift the swapped eye region off the floor.** Every
leave-one-subject-out figure is at or below the 1.6% floor: the best is the mean
shift on the left eye at 1.3%, which is *under* it. Procrustes and ridge again go
to ~0% and ~2% Rank-1 — below the 5.6% chance rate, the same identity-memorising
collapse as the face arm.

**But the in-sample rows say something the face arm did not.** On faces,
in-sample ridge hit d′ 18.8 and 100% TAR — saturated, uninformative about how
much residual there is. On eyes it reaches only d′ 2.5–3.2 and 84–85% TAR. The
swapped eye crops therefore carry **materially less recoverable identity than the
swapped face crops**, even to an attacker with that person's own paired data.
That is a stronger statement than the zero-effort number alone supports, and it
is the right direction for the method.

The residual is not zero, though: 84% TAR in-sample means the information is
present in the embedding and is linearly extractable *given the target's own
paired imagery*. What fails is transfer to a person the attacker could not train
on. The protection is a generalisation barrier, not an information-theoretic one,
and the paper should say so in those words.

## Combined claim, both arms

For an attacker who knows the operator, holds paired imagery of other people, and
fits a linear de-anonymisation map:

| arm | zero-effort TAR@1e-3 | best adapted TAR@1e-3 | gain |
|---|---|---|---|
| SimSwap face crop | 18.9% | 22.6% | **+3.7 pts** |
| SimSwap left eye | 0.0% | 1.3% | +1.3 pts (below the 1.6% floor) |
| SimSwap right eye | 2.0% | 1.5% | **−0.5 pts** |
| blur / pixel / blackbox face crop | 97.6–100% | no gain | — (already saturated) |

Cost: one CPU job for the face arm, one short GPU pass plus a CPU fit for the eye
arm. Both under an hour of wall time in total.
