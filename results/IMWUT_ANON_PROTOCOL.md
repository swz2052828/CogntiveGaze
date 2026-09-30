# Privacy–Utility Characterisation of Face Anonymisation for Gaze Sensing
## Pre-registered experimental protocol (IMWUT submission)

Status: **draft v8, 2026-09-15.** Stages 0-3 complete (§2.1, §2.2, §2.3);
Stage 3b T1/T2/T3 complete incl. the T1 background floor (ARI 0.089, k=2 — the
SimSwap ARI 1.000 is real); P-axis complete and **negative** (no private and
non-inferior point); **condition (iii) complete at K=72 AND K=9** — matched
anonymised enrolment recovers +0.24 to +1.38 cm and makes `mobile_vit`
non-inferior at every operator including `blackbox`, on the strict test (upper CI
bound below the margin) at both enrolment sizes; every cell moves <0.1 cm between
K=72 and K=9 (`results/anon_cond3_RESULT.md`, `results/anon_cond3_K9_RESULT.md`).
**Plan C (§6) complete** — subject-level non-inferiority at Δ=0.40 cm, Holm
corrected, with ICC decomposition, Bland–Altman and paired (iii)-vs-(ii) tests:
matched enrolment takes 7/16 cells to 11/16 non-inferior at both K, and
`mobile_vit`+`blackbox` is non-inferior at p≤0.003 (`results/PLAN_C_RESULT.md`).
**T4 (linear) complete** — an attacker fitting a de-anonymisation map on other
people's paired imagery gains at most +3.7 points TAR@1e-3 against SimSwap and
nothing against the conventional operators; fitted maps memorise identities
(in-sample Rank-1 100%, leave-one-subject-out 0-33%). Extended to the **eye
crops**: adaptation does not lift the swapped eye region off the 1.6% floor
(best 1.3%), and in-sample recovery there tops out at 84% TAR against the face
crop's 100% — the swapped eyes carry materially less recoverable identity. The
protection is a generalisation barrier, not an information-theoretic one. A
fine-tuning attacker remains untested (`results/anon_adaptive_T4_RESULT.md`).
**The low-frequency fusion magnitude is now measured** leave-one-subject-out
(+6.8 to +8.4% in d' on the blur arms, all 18 folds independently choosing the
same weight; zero on black box) — the last number the draft was carrying as
withheld (§ directional bias). Numbers marked *(TBD)* are placeholders.

---

## 0. Deployment scenario

The system this work protects is **smartphone-based, at-home eye tracking for
early medical screening** — ophthalmological and neurological conditions detected
from oculomotor measures captured on a consumer phone, without a clinic visit or
research-grade hardware. That scenario sets every requirement in this protocol:

- **the camera is a phone**, which fixes the ~105 px iris resolution ceiling and
  is why the utility axis cannot be bought back with a better model;
- **capture is at home and unsupervised**, so the imagery leaves a private setting
  and the released corpus is the only artefact anyone downstream sees — which is
  why linkage within the release (T1) is the operative attack, not gallery-based
  re-identification;
- **the measures are clinical**, so utility is not "gaze error in cm" alone but
  the oculomotor biomarkers computed from it (§5.2), and a method that preserves
  cm error while distorting saccade dynamics has failed;
- **the users are patients**, which is why an attribute channel that reveals
  demographics, or an eye-condition covariate that makes one subject identifiable
  (§5.2b), is a harm and not a curiosity.

**Anonymisation exists here to make the imagery shareable at all.** Without it the
raw video cannot leave the device, which forecloses the pooled datasets,
multi-site studies and external validation that a screening tool needs.

---

## 1. Framing

The existing programme (see `COMPREHENSIVE_OPTIMIZATION_ANALYSIS.md`) established a
smartphone-camera gaze pipeline at its input-limited optimum: **2.96 cm calibrated**
(`eyes_only_convnextv2_atto_binocular`), 18 subjects, deploy-faithful 9-point
enrolment. That system is a *privacy hazard by construction*: it continuously
captures face imagery to estimate gaze.

The paper does not ask "can we swap a face?". It asks the deployment question:

> **How much identity can be removed from gaze-sensing imagery before the gaze
> signal — and the oculomotor biomarkers computed from it — stop being usable?**

### The pivot hypothesis (Stage 1, cheap, decisive)

The pipeline's best model is **eyes-only**. If the ~120 px eye crops alone permit
re-identification, then the field's implicit privacy argument ("we only keep the
eyes") is false, and the pixels that *must* be preserved for utility are exactly
the pixels that leak identity. That tension is what makes a Pareto frontier
worth measuring rather than trivial. **Stage 1 tests this before anything else is
built.** If periocular re-identification is at chance, the paper's centre of
gravity moves to full-face operators and the framing is rewritten accordingly.

### Contributions (claimed)

1. **A threat model and attack suite** for anonymised gaze imagery: naive,
   informed/adaptive, attribute, linkage, and periocular attackers (§3).
2. **A privacy–utility Pareto frontier** over 10 anonymisation operators with
   parameter sweeps, on a common protocol (§4, §5).
3. **Two-tier utility**: gaze error *and* downstream oculomotor biomarkers
   (saccade latency, peak velocity, antisaccade error rate, pursuit gain, OKN,
   fixation stability) from the five-task battery in the dataset (§5.2).
4. **Non-inferiority statistics with an empirically-derived margin** — the margin
   is the measured unseeded-init noise floor of this pipeline, not a guess (§6).
5. **A gaze-locked anonymisation objective** (identity suppression + gaze
   consistency + periocular geometry) with a full ablation (§7).

---

## 2. Data and enabling facts (Stage 0 — COMPLETE)

| Fact | Value | How established |
|---|---|---|
| Subjects / recordings | 18 (`00006`–`00023`) | `datasets/ProcessedData` |
| Task frames | 193,578 | `ProcessedData/meanno7/metadata.mat` |
| Original frames | 1080×750, ~11.7k per subject | `datasets/OriginalData` |
| Crop storage | **native resolution, per subject** (face 300–350 px, eyes 110–120 px) | measured |
| Crop geometry | **recoverable exactly** — axis-aligned square, template match corr **0.9993**, MAE ≈ 1.0 (JPEG noise) | `scripts/anon/recover_geometry.py`, job 1266465 |
| Screen | 1920×1080 px = 54.4×30.4 cm, 30 fps | `gaze_dynamics/config.py` |
| Task battery | Fixation, Pro-saccade, Anti-saccade, Smooth pursuit, OKN | `gaze_dynamics/config.py` |
| Glasses subgroup | subjects 7, 10, 12 | `GLASSES_IDS` |
| Attacker models | ArcFace `buffalo_l` (w600k_r50 + genderage), FaceNet vggface2 — installed, weights cached off-home | `envs/idattack` |

**Why exact geometry recovery matters.** Anonymisation is applied to the *full
frame*; the released crops are then cut with the *stored* geometry. The control
arm is therefore byte-identical (up to JPEG) to the existing `ProcessedData`,
so **the entire existing leaderboard remains a valid control** — no control
retraining, and every published number stays comparable.

### 2.1 Stage 1 results — the pivot test (COMPLETE)

18 subjects, gallery = first temporal half, probe = second half, alignment taken
from the original frame. Chance Rank-1 = 5.6%.

**Session-nuisance control (added v3) — run BEFORE reading any identity number.**

Each subject was recorded in a single session, so illumination, seating, camera
distance and background are constant within a subject and differ between them.
The control runs the identical attack on a frame corner containing **no face**:

| Input to the identical attack | d' | Rank-1 (zero-effort) | Rank-1 (probe) | ROC-AUC | TAR@FAR=1e-3 |
|---|---|---|---|---|---|
| Face crop (control arm) | **12.23** | 100% | 100% | 1.000 | 100% |
| Right eye crop | **2.86** | 96.5% | 99.6% | 0.979 | 65.6% |
| Left eye crop | **2.02** | 92.2% | 99.1% | 0.930 | 67.3% |
| **Background patch (no face)** | **0.46** | **50.5%** | **67.0%** | 0.672 | **1.6%** |

**Rank-1 is contaminated and must not carry the argument.** A blank corner of
the room identifies the subject 50-67% of the time (chance 5.6%), so half of any
closed-set Rank-1 number here is session, not face. Verification at strict FAR
is clean: background 1.6% vs. periocular 65-67%, a ~40x separation. d' orders
the three inputs cleanly (12.2 / 2.9-2.0 / 0.46) and, unlike AUC, does not
saturate — it still separates face (12.23) from eye (2.86) where AUC cannot
(1.000 vs 0.979).

**Consequences, binding on every table in the paper:**
- Primary privacy metrics: **TAR@FAR=1e-3 and d'**. Rank-1 moves to the appendix.
- The background control is a **standard control arm run for every operator**,
  not a one-off diagnostic, and its value is printed as the floor in every table.
- Discussion must state that closed-set Rank-1 on single-session data is not
  evidence of de-identification — a methodological warning that applies to
  published work in this area.

**Periocular leakage — the eye crop alone identifies the subject.**

| Attacker (eye crop only, no face detection) | Left Rank-1 | Right Rank-1 | AUC (R) | TAR@FAR=1e-3 (R) |
|---|---|---|---|---|
| Zero-effort: pretrained ArcFace, no training on this data | **92.2%** | **96.5%** | 0.979 | 65.6% |
| Informed probe: logistic regression, temporal-half split | **99.1%** | **99.6%** | 0.99999 | **100%** |

The hypothesis in §1 holds. "We only keep the eyes" is **not** a privacy
argument: an off-the-shelf face recogniser, applied to a 120 px crop it was
never designed for and with no adaptation, still recovers identity at 92-96%.
The pixels the deployed (eyes-only) model requires are the pixels that leak.

**Conventional anonymisation baselines are dominated, not trade-offs.**
Operators applied to the face crop; attacker aligns from the original.

| Operator | Rank-1 | ROC-AUC | TAR@FAR=1e-2 | gender agreement | age MAE (y) |
|---|---|---|---|---|---|
| none (control) | 100% | 1.000 | 100% | 1.000 | 0.0 |
| blur σ=5 | 100% | 1.000 | 100% | 0.874 | 5.5 |
| pixelate b=16 | 100% | 1.000 | 100% | 0.891 | 6.9 |
| blur σ=15 | 100% | 1.000 | 100% | 0.661 | 8.5 |
| blur σ=30 | 81.2% | 0.941 | 53.2% | 0.406 | 10.4 |
| black box | 4.8% (≈chance) | 0.496 | 4.8% | 0.272 | 7.9 |

Operators verified to have modified pixels (MAE vs. original 6.4 / 9.9 / 13.7 /
20.5 / 105.5). Blur and pixelation give **no** identity protection at strengths
that already destroy the image; only total removal works. Note that attribute
inference degrades *faster* than identity (gender agreement falls to 0.66 while
Rank-1 is still 100%) -- identity embeddings are more robust to low-pass
distortion than the attribute head.

**Consequences for the design (changes v1 → v2):**

1. **The operator bank must include periocular-targeted operators** (§4). Every
   v1 operator acts on the *face* region, and most gaze-preserving designs hold
   the eye region byte-exact -- so none of them can move the periocular leakage
   measured above. Added: periocular blur/pixelate sweeps, iris-geometry-
   preserving eyelid/periocular texture replacement.
2. **The identity loss must suppress the periocular embedding too** (§7).
   Otherwise the optimiser has an obvious escape: push identity out of the face
   region and into the eye region, which it is simultaneously required to
   preserve pixel-for-pixel. This is a concrete predicted failure mode and is
   tested explicitly in the ablation.
3. **Closed-set 18-identity Rank-1 flatters the attacker's task.** Add an
   open-set variant: pad the gallery with distractor identities from a public
   face set and report Rank-1 / TAR@FAR against the enlarged gallery.

### 2.2 Stage 2 — geometry recovery (COMPLETE) and a metadata leak

All **210,924 frames / 18 recordings** recovered (`datasets/anon_geometry/*.npz`,
job 1269035, ~25 ms/frame). Only 2 frames fall below 0.99 correlation, both in
00017; their recovered boxes agree with the temporally adjacent frame to within
3 px, so the geometry is correct and the low score reflects motion blur in the
frame itself. **Acceptance criterion corrected**: spatial consistency with the
temporal neighbourhood, not correlation alone. Gate: 18/18 pass.

Cohort inter-ocular distance: mean 153.3 px, sd 12.5, range 130-180 px — a 38%
spread between subjects, which is why operator strength is parameterised by IOD
(§4) rather than in absolute pixels.

**Unexpected finding — the released crop dimensions are an identity channel.**
Recovery showed the crop geometry is *exactly* constant within a subject (IOD
sd = 0.000, eye vertical offset sd = 0.000, face box size sd = 0.000; only the
box position moves). The original preprocessing used a fixed-size, fixed-relative
template per subject and tracked translation only. Consequently the file
dimensions alone identify the subject:

| Metadata visible in the release | Distinct values / 18 subjects | Expected Rank-1 | Identity information |
|---|---|---|---|
| face crop size | 9 | 50.0% | 3.04 / 4.17 bits |
| face + eye crop size | 12 | 66.7% | 3.42 bits |
| face + eye size + IOD | 17 | **94.4%** | **4.06 / 4.17 bits** |

**94.4% of identity is recoverable without reading a single pixel** (chance
5.6%). No pixel-space anonymisation — including the face swap — touches this
channel. The attack experiments in this document are unaffected (all crops are
resized to 112x112 before embedding, discarding size), but a real recipient of
the released files is not so constrained.

**Required mitigation, and a paper section in its own right:** normalise every
released crop to a fixed size, and treat release metadata as part of the attack
surface. This is the class of leak that is invisible to anyone measuring only
pixel-space privacy.

### 2.3 Stage 3 / 3b — the operator sweep, and why blur fails

Stage 3 generated the full operator bank on the recovered geometry (job 1269092,
216/216 tasks, 12 operators x 18 recordings x 11,718 frames x 3 crops ~ 7.6M
images). Stage 3b ran the T2 external-reference attacker over all of it
(job 1271444; 60 gallery + 60 probe per subject, ~1,078/1,076 embeddings).
`none` is the pass-through control: it carries the same JPEG generation as every
anonymised arm, so no difference in the table is attributable to re-encoding.

| Operator | d' | TAR@FAR=1e-3 | Rank-1 | gender agr. | age MAE (y) |
|---|---|---|---|---|---|
| none (control) | 18.40 | 100.0% | 100% | 0.986 | 0.4 |
| blur 0.03·IOD | 16.18 | 100.0% | 100% | 0.855 | 4.1 |
| blur 0.10·IOD | 8.31 | 100.0% | 100% | 0.330 | 11.3 |
| blur 0.20·IOD | 4.67 | 100.0% | 100% | 0.255 | 12.4 |
| blur 0.35·IOD | 4.11 | 97.9% | 99.6% | 0.250 | 12.0 |
| blur 0.50·IOD | 4.02 | 98.3% | 99.7% | 0.250 | 11.7 |
| pixel 0.03·IOD | 18.08 | 100.0% | 100% | 0.929 | 2.2 |
| pixel 0.12·IOD | 12.00 | 100.0% | 100% | 0.900 | 5.0 |
| pixel 0.25·IOD | 5.54 | 100.0% | 100% | 0.670 | 5.9 |
| pixel 0.40·IOD | 4.53 | 95.3% | 99.7% | 0.449 | 7.8 |
| black box (face) | 5.12 | 99.3% | 100% | 0.251 | 7.4 |
| *background floor* | *0.46* | *1.6%* | *50.5%* | — | — |

**(a) Information is removed continuously; the decision never changes.** d' falls
4.6x from 18.40 to 4.02 while TAR@FAR=1e-3 stays between 95.3% and 100%. Every
other metric is pinned — Rank-1 and AUC saturate across most of the sweep — so
without d' the monotone structure is invisible. This is the empirical
justification for the §3 metric choice, and it is the paper's central negative
result: **blur and pixelation do destroy identity information, but never enough
to change an attacker's decision.**

**(b) Mechanism: the operators shrink the similarity scale rather than merging
the distributions.** Genuine mean similarity falls 0.962 → 0.355 across the blur
sweep, but impostor mean similarity falls too, 0.038 → 0.014. Both distributions
translate downward together instead of overlapping, so a threshold set at an
impostor quantile keeps tracking them.

The falling genuine similarity is **template–probe mismatch**, not frame-to-frame
variation of the anonymised image. The gallery is the *original* crops, so a
genuine pair is always sharp-template vs processed-probe; the within-subject
cross-frame variation of the blurred surround is essentially flat across the
sweep (std 1.90 at 0.10·IOD, 1.07 at 1.00·IOD) while d' falls from 8.31 to 4.02,
so it cannot be the driver. *An earlier draft of this section attributed the
effect to blurred-vs-blurred variation; that was wrong and the measurement above
rules it out.* Quantitatively, the FAR=1e-3 threshold
sits at mu_imp + 3.09 sd_imp, and a Gaussian prediction of TAR from (mu_gen,
sd_gen, that threshold) tracks the measured TAR across four orders of operator
strength, including both collapse cases (blur0.20 unmasked: predicted 50.9%,
measured 52.5%; blackbox eye-flat: predicted 0.4%, measured 0.0%).

*Caveat, and it is load-bearing:* the model's residuals are **one-sided** — 7 of
15 cells under-predict the attacker's success, 1 over-predicts (blackbox eye-flat,
0.4 pp, at rounding scale), 7 are saturated ties. The cause is measured, not
assumed: the skew of the genuine score distribution tracks the residual across
the biased cells.

| cell | residual (measured − predicted) | genuine-score skew |
|---|---|---|
| blur 0.50 | +7.7 pp | +0.86 |
| blur 0.35 | +7.4 pp | +0.94 |
| black box | +2.8 pp | +0.70 |
| blur 0.20 | +2.4 pp | +0.90 |
| pixel 0.40 | +2.3 pp | +0.51 |
| blur 0.10 | 0.0 pp | −0.03 (symmetric) |
| pixel 0.12 | 0.0 pp | +0.08 (symmetric) |

Positive skew means more upper-tail mass than the normal fit, so real TAR sits
above the prediction. The saturated cells run the other way (`none` −2.46,
blur 0.03 −2.48, pixel 0.03 −2.41) — that is a **ceiling effect**, genuine mass
pressed against TAR = 100%, not a second mechanism, and the column would look
self-contradictory without saying so. The
model therefore explains the dissociation but **must not be used to extrapolate
where a stronger operator would break the attack**: biased in one direction only,
it would place the break earlier than it occurs — it would make an inadequate
operator look adequate. A break point has to be measured.

**And on the masked arm there is no break point to find.** As sigma grows the face
box tends to a uniform patch carrying no spatial information, so masked TAR is
bounded below by what the preserved eye boxes alone yield; black box *is* that
limiting configuration, at 99.3%. No face-region operator at any strength can
push masked TAR below roughly that value.

**Tested and rejected: the masked privacy curve is not non-monotone.** blur 0.50
(d' 4.02) sits *below* black box (5.12) even though blur preserves strictly more
information. Under the corrected mechanism the gap is template–probe mismatch,
which saturates once the probe stops changing, so d' should flatten; whether it
also *turned up* depended on one thing only — whether a uniform patch carrying the
subject's mean face colour leaks what a black patch does not. Two operators
matched black box in geometry, eye preservation and uniformity and differed from
it only in fill colour:

| Arm | fill colour | d' | Rank-1 | TAR@FAR=1e-3 |
|---|---|---|---|---|
| black box | zero | **5.12** | 100.0% | 99.3% |
| `meanfill` | mean of whole face box | 3.94 | 99.4% | 90.3% |
| `meanfill_skin` | mean **excluding** eye boxes | **3.91** | 99.6% | 92.9% |
| black box, eye-flat | zero, eye texture removed | 0.02 | 5.3% | 0.0% |
| *background floor* | — | *0.46* | *50.5%* | *1.6%* |

`meanfill_skin` lands **below** black box, not above: giving the recogniser the
subject's own face colour makes it *worse* than pure black. The mechanism the
turn required is absent, so **the hypothesis is rejected** and blur 0.75/1.00
masked are expected near 3.9 — flat or slightly declining. (This was our
hypothesis, generated from an earlier and incorrect account of the mechanism;
both the account and the prediction are recorded here as tested and wrong rather
than quietly dropped.)

The eye-region contamination that motivated running the pair was real and
negligible: fill colours differ by L2 5.90 mean / 10.25 max on a 0–255 scale, and
the two arms differ by 0.03 in d'. Quote `meanfill_skin`.

**The colour channel exists; the recogniser does not use it.** Scored against the
session-nuisance floor rather than chance — the right baseline, since a
face-free patch of the room already identifies subjects at Rank-1 50.5% —
identification from the 3-D fill colour alone reaches d' 1.44, Rank-1 88.7%,
AUC 0.987, i.e. 3.1x the floor on d'. So the channel carries identity beyond room
nuisance, and yet a coloured patch is *actively worse* for ArcFace than a black
one. That is a statement about the recogniser, not about the information, and it
is now measured on both sides instead of hedged.

*Labelling:* with one session per subject, face colour is measured under
per-subject-constant illumination, so skin tone and illumination are not
separable in this dataset. The honest term is "the low-frequency colour of the
face region under that subject's session conditions", not "skin tone".

*What the 5.12 vs 3.91 gap is, and is not.* Black box, `meanfill*` and the
strongest blurs are all far out of distribution for a recogniser trained on real
faces. The gap is therefore a fact about **network response to off-distribution
input**, not about identity retention: a black patch is uniform across all
subjects, so the preserved eye regions dominate the embedding, whereas a
subject-varying colour field perturbs the representation without ArcFace being
able to exploit it. That is the whole claim — it carries its own scope limit, and
must not be restated as "black box retains more identity than a coloured patch".
Leakage stays high in every one of these arms, so the headline is untouched.

### Directional bias in the attacker

Two biases are known, both measured, and **both run toward over-stating
protection** — so each one means true leakage is at least what is reported here.

1. *The Gaussian TAR model under-predicts attacker success* (§(b)): one-sided
   residuals, worst +7.7 pp, traced to positive skew in the genuine scores.
2. *The ArcFace-only attacker under-uses a surviving low-frequency channel.*
   Fusing the embedding with an 8×8 thumbnail descriptor raises d' on every blur
   arm tested, and the gain appears on the near-in-distribution arms (blur 0.20,
   0.35) as clearly as on the extreme ones — so it is a property of the sweep,
   not an artefact of the off-distribution cells.

   **Magnitude, re-measured leave-one-subject-out (job 1271465,
   `results/anon_lowfreq_loo/`).** The first pass swept the fusion weight and
   reported the maximum, which selects the weight on the same data it is
   evaluated on; that ≈ +8% was an upper bound, not a measurement. It has been
   re-measured with the weight for each held-out subject chosen by maximising d'
   on the *other 17 only*, so no probe's score depends on a weight fitted using
   its own subject.

   | Arm | d' ArcFace only | d' LOO-fused | gain | gain % | TAR@1e-3 gain | w chosen |
   |---|---|---|---|---|---|---|
   | blur 0.20 | 4.670 | 4.991 | +0.320 | +6.9% | −0.09 pp | 0.25 (18/18) |
   | blur 0.35 | 4.115 | 4.460 | +0.345 | +8.4% | +0.93 pp | 0.25 (18/18) |
   | blur 0.50 | 4.017 | 4.338 | +0.321 | +8.0% | +0.19 pp | 0.25 (18/18) |
   | meanfill-skin | 3.909 | 4.174 | +0.265 | +6.8% | +1.11 pp | 0.25 (18/18) |
   | black box | 5.122 | 5.122 | +0.0000003 | +0.00% | 0.00 pp | **0.0 (18/18)** |

   **The fixed-weight presentation is now earned, not assumed:** all 18 folds
   independently chose the same weight on every arm — which is the evidence the
   protocol required before a single weight could be quoted. The honest LOO gain
   (+6.8 to +8.4% in d') lands essentially on the selected-maximum upper bound,
   so the first pass was optimistic in *procedure* but not, as it turns out, in
   *magnitude*.

   The black-box row is the internal control and it behaves correctly: with the
   face region set to zero there is no surviving low-frequency channel to fuse,
   all 18 folds choose w=0, and the gain is zero to seven decimal places. A
   procedure that manufactured gains would not have found nothing here.

   *Minor caveat, stated for completeness:* the two score matrices are z-scored
   with statistics pooled over all probes, so the held-out subject contributes
   ≈1/18 of the normalisation constants. This is an affine rescaling of the
   whole matrix, not a per-fold fit, and cannot plausibly carry a +0.32 d' gain —
   but the selection itself, which is what the objection was about, is clean.

   The *unselected* descriptor-only measurement stands as taken: d' 2.21 / 2.04 /
   1.92 on blur 0.20/0.35/0.50, against a background floor of 0.46.

Neither changes a TAR-based conclusion, because TAR is saturated — available
headroom is 0.0 / 2.1 / 1.7 pp on those three arms. But absolute percentage
points are the wrong unit near saturation: as a fraction of available headroom
the fusion captures 43% at blur 0.35. This is why d' is the pre-registered
primary metric.

*The metric pair earns its keep in both directions.* The descriptor alone reaches
d' 2.21 / 2.04 / 1.92 on blur 0.20/0.35/0.50 — landing in a range comparable to
the isolated eye crops (2.02 / 2.86), though from a different pipeline and sample
size, so not a matched comparison — while its TAR collapses to 74.5 / 31.3 /
12.1%. That is the mirror image of the main table,
where TAR saturates as d' falls. The dissociation is therefore a property of the
metric pair, not of the operators, and neither metric alone would have been
sufficient.

**(c) Three-arm decomposition: the residual leakage IS the preserved eye region.**
Each arm differs from `exact` in one factor. `flat` keeps the eye boxes at the
same position and size but fills them with constant grey (texture removed, layout
kept); `unmasked` applies the operator to the eye region as well (both removed).
All arms: 18 recordings, 1,200 frames each, 60 gallery + 60 probe per subject.

| Operator | exact (eyes preserved) | eye-flat (layout only) | unmasked (nothing preserved) |
|---|---|---|---|
| | d' / TAR@1e-3 | d' / TAR@1e-3 | d' / TAR@1e-3 |
| blur 0.10·IOD | 8.31 / 100.0% | — | 5.51 / 99.6% |
| blur 0.20·IOD | 4.67 / 100.0% | 1.24 / 5.5% | 2.13 / 52.5% |
| blur 0.35·IOD | 4.11 / 97.9% | — | **0.41 / 1.3%** |
| blur 0.50·IOD | 4.02 / 98.3% | 0.11 / 0.3% | **0.11 / 0.0%** |
| pixel 0.25·IOD | 5.54 / 100.0% | — | 1.33 / 16.6% |
| pixel 0.40·IOD | 4.53 / 95.3% | — | **0.36 / 0.0%** |
| black box | 5.12 / 99.3% | 0.02 / 0.0% | **−0.01 / 5.3%** |
| *background floor* | *0.46 / 1.6%* | | |

This is the paper's central result, and it is sharper than the sweep alone
suggested:

- **The same operator that leaves 97.9% TAR with the eyes preserved drives it to
  1.3% without them.** blur 0.35·IOD unmasked lands *at* the session-nuisance
  floor; blur 0.50 and pixel 0.40 unmasked land below it, at TAR 0.0%. So there
  *is* an operator strength at which face-region processing defeats the attacker
  outright — it just has to be allowed to touch the eyes, which the utility
  constraint forbids. That is the privacy–utility tension in one line, measured
  rather than argued.
- **The surround's contribution vanishes with strength.** Unmasked d' runs
  5.51 → 2.13 → 0.41 → 0.11 across blur 0.10 → 0.50. The processed face region
  carries identity only while the operator is weak; by 0.35·IOD it carries none.
  Yet the masked arm barely moves over the same range (4.67 → 4.02), because what
  it is measuring is no longer the surround.
- **Layout contributes nothing, confirmed at three strengths.** eye-flat tracks
  unmasked rather than exact (blur 0.20: 1.24 vs 2.13 vs 4.67; blur 0.50: 0.11 vs
  0.11 vs 4.02; black box: 0.02 vs −0.01 vs 5.12). Removing the eye *pixels* is
  equivalent to removing the eye *region*, so the subject-constant rectangle
  geometry (§2.2) carries no exploitable identity after the attacker's 112×112
  resize.

**Net: under the eye-preservation constraint the leakage floor is set by the
preserved eye pixels alone, and no face-region operator at any strength in the
sweep gets TAR@FAR=1e-3 below 95.3%.** The same operators, unconstrained, reach
0.0%. Face swapping cannot close that gap, because it preserves exactly the
pixels that carry the residual.

**(d) The attribute head does not degrade — it collapses to a constant, while
identity is untouched.** Measured against ground-truth gender (14 M / 4 W), with
the attribute head run on the original and on the anonymised image:

| Operator | P(male\|orig) | P(male\|anon) | agreement | indep. baseline | balanced acc. | raw acc. |
|---|---|---|---|---|---|---|
| none | 0.749 | 0.755 | 0.985 | 0.627 | 0.729 | 0.800 |
| blur 0.03 | 0.749 | 0.610 | 0.851 | 0.555 | 0.737 | 0.725 |
| **blur 0.10** | 0.749 | **0.072** | 0.322 | 0.286 | **0.541** | 0.292 |
| blur 0.20 | 0.749 | 0.003 | 0.253 | 0.252 | 0.502 | 0.226 |
| blur 0.35 | 0.749 | **0.000** | 0.251 | 0.251 | **0.500** | 0.224 |
| blur 0.50 | 0.749 | **0.000** | 0.251 | 0.251 | **0.500** | 0.224 |
| pixel 0.12 | 0.749 | 0.721 | 0.897 | 0.610 | 0.677 | 0.746 |
| pixel 0.25 | 0.749 | 0.501 | 0.681 | 0.501 | 0.517 | 0.513 |
| pixel 0.40 | 0.749 | 0.214 | 0.452 | 0.358 | 0.504 | 0.345 |
| black box | 0.749 | 0.116 | 0.268 | 0.309 | 0.508 | 0.294 |

Three independent signatures agree that this is **total collapse to one class**,
not a bias toward one: P(male|anon) reaches exactly 0.000 at blur ≥ 0.35; raw
accuracy settles at 0.224 against a cohort female fraction of 0.222, i.e. every
woman right and every man wrong; and balanced accuracy is exactly 0.500 while
agreement exactly equals its independence baseline, which is the signature of one
side being constant.

**The headline dissociation is at blur 0.10·IOD.** There the attribute head has
already collapsed (P(male) 0.072, balanced accuracy 0.541 — chance) while
identity is *completely untouched at the same operator strength*: TAR@FAR=1e-3 =
100.0%, d' 8.31. Not "attributes degrade first" — attributes are destroyed while
identity has not moved.

**Fairness consequence, concrete rather than hypothetical.** Because the collapse
is to one class, the four women's frames are nominally correct and all fourteen
men's are wrong: raw and balanced accuracy diverge by 28 points at exactly the
strength where the operator looks like it is working. Reporting balanced accuracy
here is forced by the data, not by convention.

**The two operator families fail differently, and in opposite directions on the
two axes.** On attributes, blur 0.10 shifts the marginal −0.678 while pixel 0.12
— comparable nominal strength — shifts it −0.028, and pixelation does not
collapse until 0.40. On detection (§2.4) the ordering reverses: pixelation
defeats detection completely (0 of 72) where blur does not (119 of 1080). Neither
family dominates.

**The collapse is in the argmax; the ranking survives, inverted.** A
threshold-free read of the gender logit difference against truth shows blur does
not scramble the representation toward noise — it moves faces *coherently* across
the decision boundary, monotonically with strength:

| | none | blur 0.10 | blur 0.20 | blur 0.35 | blur 0.50 | black box |
|---|---|---|---|---|---|---|
| AUC vs truth | 0.816 | 0.587 | 0.410 | 0.304 | **0.253** | 0.526 |

AUC 0.253 is not absence of information — it is **inverted** information: flip the
sign and it reads 0.747, against 0.816 on clean images. So roughly 78% of the
above-chance discriminative power survives blur 0.50, merely reversed. That
explains the constant argmax rather than merely restating it: if every face
crosses the boundary, the decision is constant while the ranking underneath is
intact and sign-flipped.

*Scope, and it runs in the now-familiar direction:* this is an off-the-shelf
attribute head, imperfect even on clean images (balanced accuracy 0.729 on
`none`). The finding is "this head's decision collapses", **not** "gender is
unrecoverable" — the AUC column shows the information is still there. Like every
other bias catalogued here, that scoping runs toward **over-stating** protection.

**The adapted-attacker control: confirmed where the signal has a determinate
direction, inconclusive elsewhere — and nowhere a protection result.**

| Operator | off-the-shelf | adapted | recovery | AUC | AUC (sign-aware) | direction agreement |
|---|---|---|---|---|---|---|
| none | 0.752 | 0.757 | +0.005 | 0.816 | 0.816 | 18/18 positive |
| blur 0.10 | 0.543 | 0.268 | −0.275 | 0.587 | 0.587 | 17/18 positive |
| blur 0.20 | 0.498 | 0.426 | −0.073 | 0.410 | 0.590 | 17/18 negative |
| **blur 0.35** | 0.501 | **0.646** | **+0.146** | 0.304 | 0.696 | **18/18 negative** |
| **blur 0.50** | 0.501 | **0.674** | **+0.173** | 0.253 | 0.747 | **18/18 negative** |
| pixel 0.40 | 0.509 | 0.379 | −0.131 | 0.544 | 0.544 | 16/18 positive |
| black box | 0.485 | 0.331 | −0.154 | 0.526 | 0.526 | 15/18 positive |

**Direction agreement across the 18 leave-one-subject-out folds predicts recovery
monotonically, with no exceptions** — 18/18 → +0.005, +0.146, +0.173; 17/18 →
−0.275, −0.073; 16/18 → −0.131; 15/18 → −0.154. Where the representation has a
consistent direction across every fold, adaptation recovers; where it does not, a
single fitted threshold fails to generalise.

*Confirmed* at blur 0.35 and 0.50: the off-the-shelf head is at chance (0.501)
while a zero-training adapted attacker reaches 0.646 and 0.674, with unanimous
direction. This is the fourth instance of the pattern below, the same shape as
the adapted-alignment result — and, as there, adaptation recovers **most but not
all**: 0.674 sits below both the 0.816 sign-aware AUC and the 0.752 clean-image
balanced accuracy.

*Not a protection result anywhere.* **The cells where adapted < off-the-shelf must
not be read as protection.** Each held-out subject contributes ~60 frames of a
single class, so a global threshold landing on the wrong side of that subject's
score offset costs all 60 — that is the attacker being inadequate, not the
operator protecting. A stronger adapted attacker (per-subject normalisation, or a
probe on the embedding rather than a threshold on one logit) would likely rescue
those cells, so **0.674 is a lower bound** in the same sense the adapted-alignment
ARI was.

**Past ~0.2·IOD, blurring harder makes gender MORE recoverable.** Direction is
positive at blur 0.10 and negative from blur 0.20 onward, and the sign-aware AUC
then *grows* with strength: 0.587 → 0.590 → 0.696 → 0.747. So blur first weakens
an aligned representation, then inverts it, after which further blurring
strengthens the inverted signal. The 78% survival figure is the endpoint of that
curve, not a single measurement.

This is worth setting against §(b): non-monotone privacy was hypothesised for the
*identity* axis and **rejected** there by the `meanfill_skin` control. It holds on
the *attribute* axis instead, where it is measured. A practitioner increasing σ to
protect attributes is, past 0.2·IOD, moving the wrong way.

*How this control was arrived at, because it is evidence for the claim below.* The
first implementation searched only `score > t`, was structurally unable to exploit
a sign-flipped signal, and returned ≈0.5 — which would have been reported as the
project's one genuine attribute-protection result. It was caught by the
threshold-free AUC column. The direction-agreement diagnostic that now partitions
the table was added in response.

*Age is dropped rather than fixed.* The bands are 20–24 ×1, 25–29 ×8, 30–34 ×5,
35–39 ×2, 50–54 ×1 — a single subject in two of five bands supports no per-band
claim — and the recorded `age_mae_years` is drift between two model predictions,
never touching truth.

### A recurring pattern, and a methodological claim

Three times now an apparent privacy gain has turned out to be a model going
**off-distribution** rather than information being removed:

1. a black patch scores *higher* than a skin-coloured one (§(b)) — ArcFace
   responding to a uniform vs a subject-varying perturbation, not retaining more
   identity;
2. pixelation "protects" by defeating face detection entirely (§2.4), while the
   same operator leaves ARI 1.000 once alignment is supplied;
3. the attribute head "loses" gender by defaulting to a constant class, at a
   strength where identity is untouched.

In all three, an evaluation that measured only the off-the-shelf model's output
would have reported protection that does not exist. **Apparent privacy arising
from off-distribution model behaviour is not privacy**, and distinguishing the
two requires a control that adapts the attacker — the eye-flat and meanfill arms,
the adapted-alignment regime, and the adapted attribute head (§(d)) — which is
now a fourth instance rather than three.

*The failure mode is not a strawman.* The adapted attribute control's own first
implementation fell into it: a directional threshold search, the obvious
implementation, returned a clean ≈0.5 that reads as protection, written by
someone who had named the pattern one step earlier and was actively looking for
it. If it can be committed under those conditions it is the default, not a
hypothetical, and an evaluation that does not include an adapted-attacker control
should not be believed.
This is a methodological claim the paper should make in its own right, because
the field's standard evaluations consist of exactly the measurements that fail
here.

**(e) Alignment, quantified.** The same eye pixels leak far more when the
attacker can place them correctly: isolated eye crops give d' 2.02 (left) and
2.86 (right), whereas the eye regions embedded in an aligned face frame (the
blackbox arm) give 5.12, against a 0.46 floor. *Protocol difference to state
alongside these numbers:* the isolated-crop figures come from the `--no-detect`
path (crop resized straight to 112x112) at a different sample size, so this is
an informative contrast, not a controlled one.

**Consequence for the method.** Face-region processing cannot deliver privacy
under the eye-preservation constraint — that is now measured across 11 operators
and two decades of strength. The periocular preservation axis (§4.1) is not a
refinement of the approach; it is the only remaining route.

**Open cells (not written around):** blur0.50 unmasked (the asymptote of the
surround-only arm) is still generating in job 1271448; blur0.20/0.50 eye-flat are
queued as job 1271451 gated on the generator. *Sample-size note:* the masked arms
use all 11,718 frames per recording, the unmasked and eye-flat arms use 1,200
evenly subsampled; the attack draws 60 gallery + 60 probe per subject in every
case, so the arms are comparable, but the difference is stated here rather than
left for a reader to discover.

### 2.4 Stage 3b / T1 — linkage within the release, the operative attack

Under the deployment premise (§3), originals never leave the lab, so T1 — recover
the identity partition from the released corpus alone, with no reference of any
kind — is the attack that actually applies. Job 1271466/1271488, 18 recordings,
60 probes each.

| Operator | conservative: ARI / purity / k_sel / d' | release-only self-align: ARI / purity / k_sel / frames kept |
|---|---|---|
| none | 1.000 / 1.000 / 18 / 17.23 | 1.000 / 1.000 / 18 / 1078 |
| blur 0.03 | 1.000 / 1.000 / 18 / 16.29 | 1.000 / 1.000 / 18 / 1065 |
| blur 0.10 | 1.000 / 1.000 / 18 / 9.78 | 0.393 / 1.000 / 5 / 183 |
| blur 0.20 | 1.000 / 1.000 / 18 / 7.95 | 0.292 / 1.000 / 4 / 152 |
| blur 0.35 | 1.000 / 1.000 / 18 / 7.84 | 0.301 / 1.000 / 4 / 141 |
| blur 0.50 | 1.000 / 1.000 / 18 / 7.89 | 0.215 / 1.000 / 4 / 119 |
| pixel 0.12 | 1.000 / 1.000 / 18 / 12.03 | — |
| pixel 0.25 | 1.000 / 1.000 / 18 / 9.02 | — |
| pixel 0.40 | 1.000 / 1.000 / 18 / 8.20 | — |
| black box | 1.000 / 1.000 / 18 / 8.89 | 0.658 / 1.000 / 11 / 114 |

**Three attacker regimes, measured.** The conservative column hands the attacker
landmarks from the original frame — information a release-only attacker does not
have. Whether that assumption is doing real work was tested rather than argued,
with a third regime: `adapted`, a release-only attacker who solves alignment from
the release itself by substituting, for frames where detection fails, the median
landmark configuration of frames in the same recording where it succeeded. No
training and no external data — it exploits only the §2.2 finding that crop
geometry is per-subject constant, so it is a **lower** bound on adaptation.

| Operator | off-the-shelf (self-align) | zero-training adapted | alignment solved (conservative) |
|---|---|---|---|
| none | 1.000 | 1.000 | 1.000 |
| blur 0.10 | 0.393 | 0.470 | 1.000 |
| blur 0.20 | 0.292 | 0.463 | 1.000 |
| blur 0.35 | 0.301 | 0.533 | 1.000 |
| blur 0.50 | 0.215 | 0.403 | 1.000 |
| black box | 0.658 | 0.830 | 1.000 |

**The assumption is doing substantial work and the headline must carry it.**
Adaptation helps materially — blur 0.50 goes 0.215 → 0.403, black box 0.658 →
0.830, surviving frames roughly double to triple — but it does **not** reach
1.000. So "every operator leaves the partition exactly recoverable" is true only
for an attacker who has solved alignment, and must be stated with that condition.
The defensible three-line claim:

- off-the-shelf release-only attacker: ARI 0.22–0.30 on blur, 0.66 on black box
- zero-training adapted attacker: ARI 0.40–0.53 on blur, 0.83 on black box
- attacker with alignment solved: ARI 1.000 everywhere, including total face deletion

Even the middle row is far above chance at every operator strength, so linkage is
substantial for a realistic adversary regardless. A trained detector sits
somewhere between the middle and top rows and has not been measured.

*Two caveats that weaken the adapted row, both from its own author:* k_selected
stays at 4–5 on the blur arms under adaptation, unchanged from self-align, so the
ARI gain comes from the forced-k = 18 metric rather than from an unaided
attacker; and black box imputes 546 of 660 frames, making that row mostly
synthetic alignment and the least trustworthy in the table.

*Two details the ARI number alone hides, in the off-the-shelf column:* purity
stays 1.000 throughout — the attacker recovers 4–11 identities *perfectly* and
does not see the rest, which is partial coverage rather than confusion — and the
frames kept collapse from 1078 to 119–183 (11–17%), so the whole effect is
detection failure.

**Pixelation defeats face detection completely, where blur does not.** The two
pixel cells that produced no output were not a bug swallowing an error — they
were a *result* being swallowed. pixel 0.25 self-align yields **0 detections out
of 72 sampled**, not few; the code path for "too few classes to score" returned
without writing JSON, so the cell looked as though it had never run. For
comparison blur 0.50 still leaves 119 of 1080 frames detectable (~11%).

Under the release-only regime that is the strongest apparent protection anywhere
in the sweep — and it is simultaneously the clearest case of protection that
comes from breaking *detection* rather than removing identity, because the
conservative column for the same operator is ARI 1.000. Pixelation is the
operator where the gap between the two regimes is total.

Both silent-failure classes now produce recorded outcomes rather than nothing: an
empty probe root exits non-zero with no JSON, and total detection failure writes
JSON with null ARI, the detection-failure count and a reason string, plus a
warning field if it ever occurs in the conservative regime — where it would
indicate a bug rather than protection. The two are genuinely different: the first
is our infrastructure racing itself, the second is the experiment reporting
something.


---

## 3. Threat model (Plan A)

**Deployment premise (v4).** The original recordings are never released. Only
anonymised imagery is published, for gaze research. So "the attacker holds our
originals" is *not* the operative threat, and the threat model is reorganised
around what an attacker can actually do with a released-only corpus.

Note first that identification with *no* reference is not well posed: attaching a
name to a face requires something to match against. Three attacks are well posed
without any leak of our data, and they are the primary ones:

- **A. Linkage within the release (no reference at all).** Cluster the released
  frames into individuals. Needs nothing external, and is a real harm: it enables
  longitudinal tracking, and combined with any side channel (e.g. the participant
  list) it completes de-anonymisation.
- **B. External reference.** The attacker obtains *any* photograph of the person
  from elsewhere — social media, an institutional page, a badge. This is almost
  always available in practice. We have no external photos of these subjects, so
  **the original frames stand in for that external reference.** This is the
  correct reading of every gallery-based number in this document: it models "the
  attacker has a photo of the target", not "our dataset leaked".
- **C. Attribute inference.** Needs no reference whatsoever.

Gallery and probe are drawn from **disjoint temporal halves** of each recording.

**Alignment rule (conservative).** Face alignment landmarks are computed on the
*original* frame and applied to the anonymised frame. An operator therefore gets
**no credit for merely defeating face detection** — blur that hides the face from
a detector but not from an attacker who knows where the face is scores as leaky.

| ID | Attacker | Reference needed | Priority | Metric |
|---|---|---|---|---|
| **T1** | **Linkage within the release** | none | **primary** | cluster ARI / purity at k=18 and unknown k; same/different d' |
| **T2** | **External reference** (originals stand in) | a photo from elsewhere | **primary** | d', TAR@FAR=1e-3; Rank-1 (appendix) |
| **T3** | **Attribute inference** | none | **primary** | gender / age band / glasses vs. original-image reference |
| T4 | Informed / adaptive | knows the operator; trains on anonymised imagery of **held-out subjects** (leave-subjects-out) | secondary | as T2, + Δ over T2 |
| T5 | Periocular | as T1/T2/T4, restricted to eye crops | secondary | as above |
| T6 | Originals actually leaked | our originals | worst case, one row | as T2 |

**Privacy axis for the Pareto plot: TAR@FAR = 1e-3 and d'** (§2.1 shows Rank-1
carries a ~50% session-nuisance floor and that Rank-1/AUC/TAR@FAR=1e-2 all
saturate at 100%/1.0 across several operators). Every table carries the
background-control row as its floor. Chance Rank-1 = 1/18 = 5.6%.

*Limitation to state in the paper:* one session per subject, so gallery/probe
share session-level nuisance factors (lighting, clothing, camera pose). This
inflates all identity numbers — it is a **conservative** bias for privacy claims
(the attacker is stronger than in the wild) and must be stated as such.

---

## 4. Operator bank (Plan B)

All operators act on the full frame over the FaceMesh-defined face region, then
crops are cut with the recovered geometry. Sweeps produce curves, not points.

**Every operator runs under the same utility constraint as the method** — the
landmark-tight eye protection mask of `gaze_locked_swap`, eye region preserved.
A baseline that blurs the eyes too is a straw man: it is disqualified on the
utility axis before the comparison starts, so beating it proves nothing. The
main Pareto figure therefore uses the *masked* configuration for every operator.
An *unmasked* configuration is run once per operator and reported in a single
table, to show why the naive practice is not viable — not as the comparator.

(Note: the v2/v3 smoke happened to modify only `appleFace` and never the eye
crops, so it was already in the masked configuration. That was an artefact of
operating on stored crops, not a design decision; moving to full-frame operation
makes the mask an explicit, required parameter.)

Sweeps are parameterised **relative to inter-ocular distance (IOD)**, not in
absolute pixels: face crops are 300-350 px and vary per subject, so a fixed σ
means a different strength for each subject and is not comparable across
datasets. The v2 absolute sweep only reached ~0.2 IOD, which is why it merely
found the knee (σ=30 → 81% Rank-1) instead of tracing the full curve.

| Family | Operator | Sweep |
|---|---|---|
| Signal removal | Gaussian blur | σ ∈ {0.03, 0.10, 0.20, 0.35, 0.50} × IOD |
| | Pixelation | block ∈ {0.03, 0.06, 0.12, 0.25, 0.40} × IOD |
| | Black box (face) | — |
| Substitution | Random identity paste (naive) | — |
| Generative | SD inpainting (`generate_fake_faces_diffusion.py`) | strength ∈ {0.62, 0.85} |
| | Gaze-locked swap (IP-Adapter, `gaze_locked_swap`) | strength ∈ {0.85, 0.92} |
| **Ours** | Gaze-locked GAN (§7) | + ablations |
| Control | none | — |

### 4.1 The periocular preservation axis (v4 — the method's main experimental axis)

§2.1 establishes a hard ceiling: with the eye region preserved byte-exact, the
achievable privacy is bounded below by periocular leakage (d' 2.0-2.9,
TAR@FAR=1e-3 65-67%) **no matter how good the face swap is**. Face swapping can
only remove the face-region contribution. The only way past that floor is to
touch the eye region — so how much of it must be preserved becomes the
experimental variable, not a fixed assumption.

The axis is ordered by how much gaze-carrying signal each level keeps. Iris
landmarks come from FaceMesh `refine_landmarks=True` (478 points, iris included),
which the pipeline already runs.

| Level | Preserved | Resynthesised | Gaze risk |
|---|---|---|---|
| **P0** | whole eye crop (byte-exact) | nothing | none (current setting; the floor) |
| **P1** | eye opening: iris, sclera, eyelid margin | periocular skin, brow, orbit | low |
| **P2** | iris + pupil disc, eyelid contour | eyelid texture, sclera boundary, all periocular | moderate (blink/openness cues) |
| **P3** | iris/pupil **geometry only** — a synthetic iris re-rendered at the measured centre and radius | iris texture and everything else | high, but see below |
| P4 | nothing | whole eye region | total (far end of the curve) |

**P3 is the hypothesis worth testing.** Gaze needs iris *geometry* (centre,
radius, position relative to the eye corners); identity may live substantially in
iris and periocular *texture*. If that separation holds, P3 buys a large privacy
gain at small gaze cost and is the paper's methodological contribution.

**Honest prior:** at this capture resolution the eye spans ~105 px and the iris
is only ~30-40 px, so iris *texture* is barely resolved and probably is **not**
the main leak — eyelid shape, eye-corner geometry, brow and periocular skin are
more likely carriers. If so, P1 captures most of the available gain and P3 adds
little over P2. Either outcome is a publishable, mechanistically informative
result; the axis is designed so we learn which regardless.

#### 4.1.1 RESULT (2026-09-10) — the axis is measured, and it is negative

Full numbers and provenance: `results/anon_paxis/PAXIS_RESULT.md`.
Privacy `id_attack.py --no-detect` on both eye folders; utility `mobile_vit`
`fc_ft`, 5 folds paired, 20k paired bootstrap, common `meanno7_clean_paxis`
subset manifest (18,963 rows; the `blackbox` arm reproduces the published
full-manifest number, 3.872 vs 3.874).

| arm | kept | L d' | L Rank-1 | L TAR@1e-3 | fc_ft cm | Δ vs P0 | 95% CI |
|---|---|---|---|---|---|---|---|
| P2 | 2.5% | 0.17 | 10.6% | 0.4% | 11.486 | +7.614 | [+6.90,+8.66] |
| P1 | 5.2% | 0.39 | 14.3% | 0.0% | 7.131 | +3.259 | [+2.61,+3.91] |
| P1d5 | 9.9% | 0.57 | 14.4% | 0.2% | 5.612 | +1.740 | [+1.29,+2.37] |
| P1d10 | 15.7% | 0.81 | 30.9% | 0.3% | 5.387 | +1.515 | [+0.76,+2.43] |
| P1d20 | 30.8% | 1.14 | 51.1% | 0.6% | 4.838 | +0.966 | [+0.39,+1.82] |
| P1d40 | 66.8% | 1.74 | 80.5% | 38.9% | 4.133 | +0.261 | [+0.02,+0.64] |
| P0 | 100% | 2.12 | 91.4% | 66.6% | 3.872 | 0 | — |

The right eye agrees at every level. **The curve does not bend**: privacy and
error trade roughly linearly. P1d40 is the only arm whose point estimate is
inside the Δ=0.40 cm margin, its CI crosses it, and it still leaks (Rank-1
80.5%, TAR@1e-3 38.9%). **No operating point on this axis is both private and
non-inferior.**

The honest prior above was half right. P1 does capture most of the available
*privacy* gain, and iris texture is not the main leak — so P3 is not worth
building. What the prior did not anticipate is the utility side: preserving the
gaze-bearing pixels and deleting the rest does **not** preserve gaze. The model
needs the periocular surround. A facefixed diagnostic (`appleFace` held
byte-identical across arms) decomposes P1's +3.26 cm into **+2.90 cm (89%) eye
stream** and **+0.36 cm (11%) face crop**, so this is a real dependence, not
collateral damage.

This is the input to §7: the GAN needs an explicit **gaze-consistency
objective**, because pixel preservation alone is demonstrably insufficient.

**Two corrections.** (a) The P0 anchor was previously quoted as d' 5.122 /
Rank-1 100% / TAR 99.3%. That is `t2_blackbox.json` — the T2 attack on the
**face** crop, a different crop and attack configuration from the rest of the
axis. Measured through the identical harness, P0 is d' 2.123 / Rank-1 91.4% /
TAR 66.6%; the axis had been exaggerated >2×. The claim that P1 falls *below*
the background nuisance floor is **withdrawn** — it sits at it (0.391 vs 0.361).
The corrected anchor reproduces `peri_ProcessedData_*` to 3 decimals.
(b) **§5.1 needs a fix (peer-owned):** it states `face_only_mobile_vit` "cannot
see the periocular axis at all, so any P-level effect it shows is a confound."
That is not true of these operators. The eye boxes lie inside the face box, so
masking the periocular region necessarily rewrites the released `appleFace`
crop — `face_only_mobile_vit` duly moved **+4.56 cm**, more than `mobile_vit`.
It is a valid falsification arm only on the facefixed roots, where it moves
−0.007 cm.


---

## 5. Utility (Plans C, D)

### 5.1 Tier 1 — gaze error
Three backbones chosen to span *input dependence*, not accuracy:

Eyes-only models are **out of scope** (they belong to a separate paper) and are
also uninformative here: the release preserves the eye region, so at level P0 an
eyes-only model sees no modified pixels and its utility cost is zero by
construction. Utility is measured on **face-using** models:

| Backbone | Input | Control (calibrated) | Role |
|---|---|---|---|
| `face_only_mobile_vit` | face only | 3.03 cm | isolates the face-swap effect; blind to the P-axis |
| `mobile_vit` | face + eyes + grid | 3.33 cm | sees both the swap and the P-axis |
| `convnextv2` | face + eyes + grid | 3.88 cm | cross-architecture generality |
| `itracker` | face + eyes + grid | 4.82 cm | classic architecture, cross-family check |

The face-only / multistream contrast is itself informative: `face_only_mobile_vit`
cannot see the periocular axis at all, so any P-level effect it shows is a
confound, and any P-level effect the multistream models show is genuine.

Protocols (Plan D): **(i) zero-shot** train-real / test-anon; **(ii) retrain** on
anonymised; **(iii) fine-tune** real→anon. Calibrated (K=9 deployable, K=72) and
uncalibrated.

**Calibration support must be anonymised too.** Per-subject calibration works by
fitting each subject's systematic bias; if the support frames are original faces
while deployment sees swapped faces, that is a train/deploy mismatch which would
make the utility numbers both wrong and pessimistic. In deployment the enrolment
recording is anonymised like everything else, so the `calib_support_K*` sets must
be regenerated under each operator. Slices: glasses {7,10,12}, head-pose tertiles, frame-brightness
tertiles, per-subject.

### 5.2 Tier 2 — oculomotor biomarkers (the real utility claim)
Via the existing `gaze_dynamics` bridge (`export.py` → analyzers): saccade
latency, peak velocity, directional (antisaccade) error rate, fixation
stability, heatmap entropy/correlation. A gaze error that does not move these
measures has not damaged the application. This tier is what separates the paper
from a computer-vision benchmark.

*Optional, higher risk:* EyeLink `.EDF` recordings exist under
`datasets/extracted_data/*/Wanzi_*/data/`, and `gaze_dynamics.metrics` already
implements biomarker ICCs and mixed ANOVA against EyeLink. Converting and
aligning them (needs `edf2asc`, plus a session→subject mapping that does not yet
exist) would allow "biomarkers still agree with a research-grade tracker after
anonymisation". Treated as a **stretch goal**, not a dependency.

---

### 5.2b Per-subject covariates, and a falsifiable prediction

Ground-truth attributes come from the participant questionnaires (gender 18/18;
age band 17/18, one "prefer not to say"), not from pseudo-labels.

*Methods note, load-bearing for anyone checking this work:* Google Drive's **web
preview silently omits fields** from these .docx questionnaires — a health
free-text field absent from the browser view is present in the downloaded file.
Extractions here went through a document parser rather than the preview, but
anyone re-reading these sources must download or parse the .docx. A field that
exists in the file but not in the preview is exactly how a covariate gets
silently dropped.

Two per-subject covariates for the subgroup analysis:

- **glasses**: 00007, 00010, 00012
- **eye condition**: 00016 has a **cataract in the right eye**

The cataract is not bookkeeping. It is a genuine ocular pathology inside a cohort
whose study purpose is ophthalmological research, and because it is *monocular*
it yields a falsifiable prediction rather than a footnote: it should appear as a
**left/right asymmetry**. The utility set contains models with explicit per-eye
inputs and one (`face_only_mobile_vit`) with none, so — if 00016 is an outlier in
the eye-stream models but not in `face_only`, the cataract is the cause; if it is
an outlier in both, something else is.

*Attribution caveat:* the subject-code mapping has independent corroboration at
00014, 00009, 00011 and 00021, but not at 00016 specifically, so any claim
resting on that subject's covariate carries it.


### 5.4 Stage 4a results — the utility axis (condition ii)

4 face-using backbones × 12 operators × 5 folds, K = 72, clean-trained
checkpoints evaluated zero-shot on anonymised deployment data with the original
enrolment (condition ii). cm; `base` uncalibrated / `fc_ft` calibrated.

**Pipeline validation first.** The regenerated `none` control reproduces the
published leaderboard exactly:

| backbone | published base / fc_ft | regenerated `none` |
|---|---|---|
| face_only_mobile_vit | 5.43 / 3.54 | 5.43 / 3.55 |
| mobile_vit | 5.12 / 3.33 | 5.12 / 3.33 |
| convnextv2 | 4.33 / 3.88 | 4.33 / 3.88 |
| itracker | 5.93 / 4.82 | 5.93 / 4.82 |

Geometry recovery → full-frame regeneration → re-crop → re-encode → evaluation is
therefore verified end to end, not assumed. Every anonymised number below rests
on this.

| Operator | convnextv2 | mobile_vit | itracker | face_only_mobile_vit |
|---|---|---|---|---|
| none | 4.33 / 3.88 | 5.12 / 3.33 | 5.93 / 4.82 | 5.43 / 3.55 |
| blur 0.10 | 4.37 / 3.84 | 5.16 / 3.72 | 5.66 / 4.61 | 6.05 / 4.82 |
| blur 0.20 | 4.39 / 3.83 | 5.17 / 3.87 | 5.59 / 4.51 | 6.26 / 5.23 |
| blur 0.35 | 4.36 / 3.77 | 5.17 / 3.86 | 5.56 / 4.46 | 6.51 / 5.55 |
| blur 0.50 | 4.34 / 3.75 | 5.17 / 3.86 | 5.54 / 4.45 | 6.58 / 5.64 |
| pixel 0.25 | 4.36 / 3.70 | 5.13 / 3.71 | 6.05 / 4.39 | 6.35 / 5.42 |
| pixel 0.40 | 4.35 / 3.75 | 5.11 / 3.75 | **9.26** / 5.78 | 6.35 / 5.39 |
| black box | **4.24 / 3.65** | 5.17 / 3.87 | **8.68** / 6.09 | 6.62 / 6.05 |

**For three of four face-using models, destroying the face costs almost
nothing.** convnextv2 under total face deletion is 4.24 / 3.65 against a 4.33 /
3.88 control — *better* on both, though −0.23 sits inside the ±0.3–0.4 cm noise
floor and should be read as "unaffected", not "improved". mobile_vit pays +0.54
calibrated. Only `face_only_mobile_vit`, whose entire input is the face, pays a
real price: +2.50 calibrated at black box.

That is the model taxonomy working as designed (§5.1): the face-only control does
see the face-region operators, which confirms the operators act; the multistream
models barely do, which locates their utility in the eye streams.

`itracker` behaves differently again — it *improves* under blur (5.93 → 5.54
base) but breaks at pixel 0.40 and black box (9.26, 8.68). Blur removes something
its face stream was using badly; a hard-edged or empty face crop produces output
its fusion cannot ignore.

**The synthesis with the privacy axis is the paper's Pareto result, and the
frontier is degenerate.** Take black box, the most destructive operator in the
bank. It costs convnextv2 nothing and mobile_vit +0.54 cm — and it buys
TAR@FAR=1e-3 100% → 99.3%, ARI still 1.000. **Near-zero utility cost, near-zero
privacy gain.** The operators are not spread along a trade-off curve; they are
clustered in a corner where they cost little and buy little, and the reason is
the same on both axes: the eye region carries the utility *and* the identity.

### 5.3 Three subjects carry a geometry caveat, for three different reasons

Named separately rather than pooled, because the causes and the consequences
differ.

**00020 — unstable crop geometry.** Eye-box position sd inside the 224
calibration face crop is 5.36 px in y against ≤2.6 px for the other seventeen,
and within-subject IOD sd is 2.42 px against 0.6–1.3 px. Two measurements made by
different means agree, so it is a property of that recording, not of a method.
Any per-subject constant quoted here is weakest for 00020.

**00010 and 00012 — low worst-case template-match quality.** Minimum match
correlation over the K-selection frames is 0.8725 (00010, 10 frames below 0.95)
and 0.9085 (00012, 19 frames), against 0.96–0.97 for the other sixteen. Both are
in `GLASSES_IDS`, so the likely cause is a lens specular reflection defeating the
match on particular frames — single frames, not whole recordings. 00012 also has
the cohort's largest IOD (75.42 px), putting it furthest from any cohort-level
assumption. The enrolment builder screens per frame at a 0.95 correlation floor
and falls back to that subject's median eye box for frames below it, reporting
the fallback count; 6 of 162 frames at K=9.

The median fallback is **off by default** and must be requested. It would
otherwise have been silently applied to exactly these subjects — which is the
worst failure mode for a study whose entire method is per-subject calibration.

---

## 6. Statistics (Plan C)

**Pre-registered non-inferiority margin: Δ = 0.4 cm.** Justification — this
pipeline's measured unseeded-init run-to-run noise floor is ±0.3–0.4 cm
(`flip_right_eye_summary.md`, face-only control expected Δ=0 measured +0.32).
An anonymisation whose utility cost is below the noise of simply retraining the
model is non-inferior in any operationally meaningful sense. The margin is
therefore *derived from the system*, not chosen for convenience. Δ must also be
reported in degrees of visual angle — **requires the viewing distance, which is
not in the repo (open question)**.

- Paired subject-level **bootstrap** (n=18, 10k resamples, BCa CIs).
- **TOST / one-sided non-inferiority** at α=0.025 against Δ.
- **Bland–Altman** (per-subject means and per-frame): bias + limits of agreement.
- **Linear mixed-effects**: `error ~ method * protocol + (1|subject) + (1|gaze_target)`
  (statsmodels MixedLM).
- **Effect sizes** (Cohen's dz) and CIs reported alongside every p; Holm
  correction across operators.
- Pre-declared: with n=18 the study is powered for subject-level differences of
  roughly Δ; per-frame n is not treated as independent evidence.

### 6.1 EXECUTED (2026-09-10) — Stage 4, subject level

`results/PLAN_C_SUBJECT_LEVEL.md` (+ `.txt`, + `anon_stage4_subject.csv`).
Code: `scripts/anon/subject_level.py`, `scripts/anon/plan_c_stats.py`.

Design is **n=17, not n=18** — `meanno7` drops recording 7. The 17 are 17
distinct individuals: three participants recorded twice in the wider study
(subid 3/12, 4/7, 2/6) but only one session of each pair survives here, so no
person contributes two rows. 816 rows, complete and balanced
(4 backbones × 12 operators × 17 recordings), extracted from the per-recording
lines the Stage 4 runs already logged — nothing recomputed.

**Δ in degrees is now available.** Viewing distance is measured at 80–120 cm, so
Δ = 0.40 cm = **0.191–0.286°**. The open question in the preamble is closed.

**The pairing is doing heavy lifting.** Random-intercept fit
`y[s,o] = μ + β_o + u_s + ε` gives ICC **0.94–0.97** for the three eye-using
backbones (between-subject SD 1.07–1.75 cm vs residual 0.19–0.41 cm). Nearly all
variance is *which subject*, and operator effects are tenths of a cm against it.

**Verdicts** (`fc_ft`, Holm-corrected across the 11 operators per backbone):
convnextv2 **11/11 non-inferior** with every point estimate negative;
itracker **10/11** (blackbox inconclusive, +0.161 CI [−0.37,+0.79]);
mobile_vit 3/11 (blackbox inconclusive, +0.550 CI [+0.34,+0.76]);
face_only_mobile_vit **0/11**, INFERIOR on 10/11 up to +2.539 cm — the positive
control behaving as it must, which is what licenses reading the others'
preserved utility as real rather than as models ignoring their input.

**Two limits on how this may be stated.**

*(a) It is calibration-method dependent.* Non-inferior cells out of 44
(11 operators × 4 deploy metrics): convnextv2 40/44, itracker 41/44,
mobile_vit **23/44**, face_only 0/44. mobile_vit is non-inferior under `base`
and `svr_embed` but not `fc_ft` or `meta`. The claim must be reported per
calibration method. `meta` is the weakest column (itracker/blackbox +4.63 cm,
CI [+1.61,+8.17]), consistent with its known instability.

*(b) Mean non-inferiority is not per-subject safety.* Bland–Altman limits of
agreement on none-vs-blackbox span 1.5–5 cm (convnextv2 [−0.97,+0.55];
mobile_vit [−0.35,+1.45]; itracker [−2.31,+2.63]) with no proportional bias
(|corr| ≤ 0.40). Individual subjects can be materially worse off while the
cohort mean is unchanged. **The deployment claim is about the cohort mean and
must say so.**

*Note on the earlier fold-level numbers.* mobile_vit/blackbox read INFERIOR at
fold level (CI [+0.46,+0.62]) and inconclusive at subject level
(CI [+0.34,+0.76]) — a wider interval from more points, which is correct: a fold
mean averages 3–4 subjects and understates how much a *new subject* varies.
Subject level answers the deployment question and supersedes the fold-level
table.

---

## 7. Method (Plan E)

Extend `gaze_preserving_swap/losses.py` (already has LSGAN, TV, and
`FrozenGazeCriterion`):

```
L = L_adv
  + λ_gaze · ‖ f(x_anon) − f(x_real) ‖        # prediction CONSISTENCY, not label error
  + λ_eye  · L1(x_anon, x_real | periocular mask) + λ_geom · iris/landmark geometry
  − λ_id   · [ cos( e_arc(x_anon), e_arc(x_real) )        # face identity
             + cos( e_peri(x_anon), e_peri(x_real) ) ]   # periocular identity (v2)
  + λ_tv   · TV
```

Note `L_gaze` is changed from the existing label-error form to a **consistency**
term: the requirement is that anonymisation does not move the deployed model's
prediction, which is not the same as being accurate. Ablation: drop each term in
turn; report both axes for every ablation (an ablation table that reports only
utility would beg the question). **The periocular identity term is not optional:**
without it the objective can satisfy both constraints by relocating identity into
the eye region it is required to preserve. The ablation must show this happening.

---

## 8. Compute plan and staging

Each stage is smoke-tested before its full run; full runs need explicit approval.

| Stage | Work | Cost | State |
|---|---|---|---|
| 0 | Feasibility: geometry recovery, attacker stack | ~1 CPU-h | **done** |
| 1 | Pivot test + session-nuisance control + d'/IOD calibration of the metrics | ~2 CPU-h | **done** (§2.1) |
| 2 | Geometry table, all 193,578 frames (measured 20 ms/frame) | ~1 CPU-h, 18-way array | **done** (§2.2), job 1269035 |
| 3 | Cheap operator bank, IOD-parameterised, masked + unmasked, full frames | ~40 CPU-h, ~190 GB | **done** (§2.3), job 1269092, 12 ops x 18 recs, 648/648 |
| 3b | Attack suite T1–T3 primary (+T4–T6) on Stage 3 output | ~10 GPU-h | **done** — T1/T2/T3 incl. the T1 background floor (ARI 0.089, k=2); T5 periocular done (`peri_*`); T4 **linear** done (`anon_adaptive_T4_RESULT.md`). T6 = T2 as run (originals are the stand-in gallery). **Open: T4 fine-tuning attacker** |
| 4 | Zero-shot utility, 4 face-using backbones, no retraining | ~25 GPU-h | **done** — incl. condition (iii) matched anonymised enrolment at K=72 and K=9 (`anon_cond3_RESULT.md`, `anon_cond3_K9_RESULT.md`) |
| **4b** | **P-axis, swap-based**: P0–P4 periocular levels by aligned donor-region replacement (fast, no diffusion) → full dataset, both axes | ~20 CPU-h + 15 GPU-h | **done and NEGATIVE** — periocular preservation trades linearly; no point is both private and non-inferior (P1 costs +3.26 cm, 8x the margin). `anon_paxis/PAXIS_RESULT.md`. **P3 not built** |
| 5 | Generative operators (SD inpaint, IP-Adapter swap, and diffusion P-levels) — **stratified subsample** ~1.5k frames/subject, since inpainting is ~1–2 s/frame (full set = 50–100 GPU-h *per variant*) | ~40 GPU-h | **partial** — SimSwap done (`ProcessedSwap{,2}{,_oldeye}`) and attacked. SD inpaint and IP-Adapter **not run** |
| 5b | Regenerate `calib_support_K{9,72}` under each retained operator | ~5 GPU-h | **done** — `datasets/anon_calib_geometry_k{9,72}`; this is what condition (iii) consumes |
| 6 | Gaze-locked GAN (§7) training + ablations, incl. the periocular identity term | ~40 GPU-h | **not started** — and see §9 before starting: the P-axis being negative removes this method's assumed headroom |
| 7 | Retrain / fine-tune protocols: 4 backbones × 5 folds × ~5 retained conditions | ~200 GPU-h | **deferred** (§9.3), and condition (iii) already delivers non-inferiority without it |
| 8 | Statistics, biomarkers, figures, writing | CPU | **partial** — Plan C statistics **done** (`PLAN_C_RESULT.md`: subject-level TOST at Δ=0.40 cm, Holm, ICC, Bland–Altman). **Tier-2 oculomotor biomarkers (§5.2) not started** — the largest remaining gap, and §0 makes them the clinical utility claim. Figures and writing pending |

**Cost control.** The P-axis (§4.1) is implemented twice: a cheap donor-region
replacement that runs on the full dataset (4b), and a diffusion version on the
subsample (5). If the two agree on the privacy–utility ordering, the cheap one
carries the main figures and diffusion is reported as a quality check — which is
what keeps Stage 5 from dominating the budget.

**Subsampling must be validated**, not assumed: before Stage 5, confirm the
stratified subsample reproduces full-set gaze error within 0.05 cm on the
control arm. Cheap check, prevents an unfixable confound.

**Stage 7 is the schedule risk**, and it is only worth paying once Stages 3b/4b
have shown which operators are on the frontier. Do not pre-commit it.

---

## 9. Scope decisions and their resolutions

**Resolved 2026-09-08:** the periocular preservation axis is **in scope** and is
the method's main experimental axis (§4.1); the eyes-only zero-cost control is
**dropped** (conflicts with a separate paper), and eyes-only models are out of
the utility set (§5.1).

1. **Ethics — amendment pack drafted 2026-09-10**, `results/ethics_amendment/`
   (summary of changes, application form v3 with changes highlighted and the
   §4.D/§4.X/§8.2 ticks flipped, PIL v2, data-protection annex, covering note).
   Route is the **Legacy Amendment** path (study approved 2022, i.e. before
   3 Aug 2026); classification is **substantial**, on the grounds of a new
   work package, a new derived data set, and §8.2 changing to Yes because
   WP2 processes biometric data to identify a natural person. Two items block
   submission and are not resolvable from the file: whether Prof Pecchia is
   still PI, and whether §8.5/8.7/8.9 (no external platform, no external
   sharing, nothing outside the UK) still match where the data actually sits.
   **The approved completion date 31/03/2026 has passed** — an extension rides
   along as a minor item and the overrun is disclosed rather than hidden.
   The consent form carries a future-research clause
   ("I am happy for my data to be used in future research"), and the PIL
   conditions such use on independent REC review, so a **new-use submission to
   BSREC 08/22-23** is in progress rather than re-consent. Two points are
   disclosed there: the leaflet described eye-region-only video whereas full-face
   video was captured (full-face data is retained locally and never distributed);
   and §2.1 shows the leaflet's privacy rationale — "since the video will be
   limited to the region of the eyes there will be little possibility of
   identification" — is **empirically false**, which is this paper's strongest
   real-world motivation. Figures use the researchers' own faces, so the consent
   form's separate publication clause is not engaged. The participant
   questionnaire (age, sex, health) supplies **ground-truth attribute labels**,
   replacing the pseudo-label scheme for T3.
2. **Viewing distance — resolved by measurement**: 80-120 cm; active display
   53.3 x 29.6 cm (16:9, consistent). Report cm as primary, degrees as a range
   (2.96 cm = 1.4-2.1 deg; the margin Δ = 0.4 cm = 0.19-0.29 deg). **Do not use the
   EyeLink `ppd` field**: at 107.08 px/deg it implies ~170 cm, contradicting the
   measurement, so the EyeLink host geometry was misconfigured — every
   degree-valued EyeLink output is miscalibrated by that factor, and its
   velocity fields are the -32768 missing sentinel anyway. Gaze is stored in
   PsychoPy `height` units and is reliable; any EyeLink biomarker must be
   recomputed from the raw samples using the measured geometry.
3. **Target cycle — November**, confirmed. **Stage 7 is deferred**, contingent on
   the earlier stages finishing in time.
4. **EyeLink** — the `.hdf5` files hold everything needed (no EDF, no SR licence,
   no `edf2asc`); the only missing artefact is the participant-code to `subid`
   mapping, which is being supplied. Coverage is the open risk: the 13 populated
   sessions all fall in 2023-08-21..31 while 12 of the 18 subjects were recorded
   from 2023-09-01 onward, so the usable overlap may be as low as 4-5 subjects.
   Decide after the mapping arrives; supplementary material at best.
