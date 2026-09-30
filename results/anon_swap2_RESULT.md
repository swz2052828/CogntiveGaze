# SimSwap: a second template, and the threat model the first report conflated

Two questions. Does the SimSwap result depend on which template face was used?
And does "the swap removes identity from the eye region" hold against every
attacker, or only one?

## 1. Template generality — the face-crop results replicate

| measure (attacker holds original enrolment photographs) | template 1 | template 2 |
|---|---|---|
| d′ | 1.98 [1.48, 2.58] | 1.23 [0.71, 1.87] |
| Rank-1 | 57.9% | 36.6% |
| **TAR@FAR=1e-3** | **22.0% [5.2, 44.5]** | **23.3% [7.4, 42.8]** |

The headline number is the same to within a point, on intervals that overlap
almost entirely. Template 2 is *less* identifiable on the rank-based measures and
identical at the strict operating point. **The residual identity in a swapped
face crop is a property of the method, not of one template face.**

Linkage replicates exactly:

| | ARI @ k=18 | k recovered | frames |
|---|---|---|---|
| template 1 | 1.000 | 18 | 1078 |
| template 1, self-aligned | 1.000 | 18 | 1073 |
| template 2 | 1.000 | 18 | 1078 |
| template 2, self-aligned | 1.000 | 18 | **1080 (0 detection failures)** |
| *background floor* | *0.089* | *2* | *1080* |

## 2. The correction: "the swap removes eye identity" is true of ONE attacker

The eye-region claim in `anon_adaptive_T4_RESULT.md` and in the SimSwap reporting
before it was stated without its condition. Two scripts measure the eye crops
with **different galleries**, and the difference is the threat model, not a bug:

- `id_attack.py` — gallery is the **original** crops, probe is the anonymised
  crop. This is T2: the attacker holds enrolment photographs of the target.
- `periocular_attack.py` — gallery and probe are **both** drawn from the released
  anonymised corpus (first/second temporal half). This is the release-only
  attacker: no reference imagery, but any single labelled example names a cluster.

Same data, same recogniser, same metric. Different question, different answer:

| eye crop | attacker holds originals (T2) | attacker holds only the release |
|---|---|---|
| original, left / right | — | 67.3% / 65.6% |
| **SimSwap t1**, left / right | **0.2% / 1.8%** | **42.8% / 62.5%** |
| **SimSwap t2**, left / right | **0.6% / 1.1%** | **56.0% / 57.7%** |
| *session-nuisance floor* | *1.6%* | *—* |

Rank-1 under the release-only attacker is 86–94% for both templates, against 92–97%
for unmodified eye crops. Rank-1 under the originals-in-hand attacker is 15–31%.

**Both templates behave the same in both columns** — cross-domain 0.2%/1.8% (t1)
and 0.6%/1.1% (t2), each at or below the 1.6% floor; release-only 42.8%/62.5% and
56.0%/57.7%. The gap between the columns is therefore a property of
per-subject-consistent face replacement, not of a particular template.

**Mechanism, and why the two columns are consistent.** SimSwap applies a
*per-subject-consistent* transformation: the same person always maps to the same
swapped appearance. That destroys the correspondence with their real face — hence
the floor-level T2 column — while leaving the released corpus perfectly
self-consistent, so frames of one person still match each other. It is the same
mechanism that gives ARI 1.000 in §1, seen through a supervised attack instead of
a clustering one.

**Consequence for the claim.** "SimSwap removes identity from the eye region" is
only defensible as: *it removes the correspondence between the released eye
region and the participant's real eye region.* It does not make the released eye
crops mutually unlinkable, and under the deployment premise of this paper — the
originals never leave the device, so the release is all an attacker has — the
release-only column is the operative one. Every statement about SimSwap must name
its attacker.

## 3. The adaptive attacker replicates across templates

T4, leave-one-subject-out linear de-anonymisation, TAR@FAR=1e-3 (threshold now
aligned to `id_attack.tar_at_far` for every arm):

| arm | zero-effort | best adapted (LOO) | ridge, in-sample |
|---|---|---|---|
| face, template 1 | 19.9% | 23.2% | 100% |
| face, template 2 | 23.6% | 23.6% (no map transfers) | 100% |
| left eye, t1 / t2 | 0.0% / 0.8% | 1.3% / 0.9% | 83.9% / 83.3% |
| right eye, t1 / t2 | 2.1% / 1.0% | 1.5% / 0.1% | 85.7% / 86.7% |

Both templates: a linear adaptive attacker gains at most ~3 points, fitted maps
memorise identities (in-sample 100% / ~85%, leave-one-subject-out ≈ 0), and the
eye-region in-sample ceiling stays well below the face-crop one. The
generalisation-barrier reading in `anon_adaptive_T4_RESULT.md` holds for both.

## 4. Data repair behind these numbers

`ProcessedSwap2`'s eye crops were truncated for two recordings when the group
inode quota hit its hard limit on 2026-09-10 (00020 left eye 0/11718, right
510/11718; 00023 left 0, right 5517 — frames 10614–16880 only). The source zip
had been deleted in the storage cleanup and was re-fetched from the project Drive;
only the four short folders were extracted, verified against `ProcessedData`
(4×11718/11718) and the zip deleted. `ProcessedSwap2_oldeye/00023`, whose symlinks
stopped at the same moment, was rebuilt. Without the repair the periocular arm
would have silently run on 16 subjects for the left eye.
