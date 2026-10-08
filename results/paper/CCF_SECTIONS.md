# CCF draft — rationale, alternatives and reviewer risks

**Rewritten 2026-10-08.** This file used to carry a second copy of the paper
text. That copy drifted (it still said "preserves accuracy" and quoted the
pre-replication single-draw Table 3), so it has been removed.

- **Paste text and every number:** `CCF_PASTE_ORDER.md` — the only source.
- **This file:** why each section is written the way it is, what was considered
  and rejected, what a reviewer will attack, and which claims were withdrawn.

Section letters below follow the old A–H layout; the corresponding
`CCF_PASTE_ORDER.md` section is given in brackets.

---

## The argument in one paragraph

Gaze utility and identity live in the same pixels. Restoring the genuine eyes
recovers the utility of real data and releases the biometric with it; replacing
the eyes removes the link to the real face but costs most of the gaze signal and
none of the linkability of the release. Breaking release-only linkage needs two
things at once — a fresh synthetic identity (per frame here; per session would
be the deployable form) and nothing in the released crop that the transformation
does not edit — and the only configuration we found that does both repaints the
eyes. Across a replacement family (SimSwap) and a removal family (DeepPrivacy2,
face and full-body), and under three training/testing conditions, no point is
both private and useful (`figs_core/fig_core3_tradeoff`).

---

## A — Abstract [§1]

- **No "anonymised" for the output.** Everything measured is pseudonymised; the
  terminology rule in the paste copy's header applies throughout.
- **"Recent work reports that it hides identity effectively"** is a deliberate
  forward reference to Muştu & Ekenel (2025). We contradict a published result;
  saying so in the first paragraph is better than being found out in review.
- **The EyeLink stays** as an instrument claim, not a "we also validated" claim:
  physiological ground truth is what makes sub-centimetre differences
  interpretable at all. (Table 1, the EyeLink ICC biomarker table, does not go in
  the paper; the claim does.)
- **Why the abstract now names two utility conditions.** It used to say
  "retraining on de-identified data" while the numbers were deploy-only. It now
  says both, because both are measured and they answer different questions
  (see H).
- **Length.** ~285 words. If capped at 250, the paste copy lists what to cut;
  never cut the removal-by-inpainting sentence before the tiered-release clause
  unless forced — it is the strongest new utility result.

## B — Introduction and contributions [§2]

- **Dataset third, deliberately.** Leading with it frames a dataset paper with
  de-identification attached and invites "why is the privacy section here?".
  Led by the evaluation protocol and the structural claim, the dataset reads as
  the instrument that makes those claims measurable.
- **The falsifiable claim was corrected, not abandoned.** The first draft said
  protection "is bounded by how far the transformation breaks per-participant
  consistency". DeepPrivacy2 showed that half right; the contribution is now the
  two-part condition, tested as a 2×2. Do not revert to the one-part wording.
- **Title.** The current title stresses the half that turned out easy (utility
  recovery). Alternatives in §0. If "High-Precision Benchmark" stays, it must say
  it refers to the EyeLink ground truth, not the 4.9–5.9 cm smartphone baselines.

## C — Related Work [§3]

This field is more crowded than the original draft implies and contains a
directly competing published result. Thin Related Work here reads as not knowing
the field, and with a competing claim it reads as avoidance.

- **C1 Smartphone gaze.** iTracker/GazeCapture, Valliappan et al. 2020, AFFNet,
  MGazeNet. Compress to one point: all consume face and eye ROIs, so all inherit
  the release problem.
- **C2 Replacement vs removal.** The expected split (removal destroys gaze,
  replacement preserves it) is now *measured*: the first holds outright
  (retrained on DeepPrivacy2 eyes, models are no better than the mean), the
  second only partly (SimSwap synthetic eyes cost 2–5 cm after retraining). On
  privacy the objective does not decide the outcome. DeepPrivacy2's own paper is
  about full-body anonymisation and motivates it by exactly our concern — face
  anonymisation leaves identifiers outside the face. Cite that; it is support,
  not competition.
- **C3 The competing claim — cannot be omitted.**
  - *Muştu & Ekenel 2025* (arXiv 2505.20985): face swapping "effectively hides
    identities". Their anonymity strength is an A1 quantity, on which we agree
    (face ROI 22% vs 100%). They do not test release-only linkage, which is the
    question a public release poses.
  - *Yang et al., Practical Digital Disguises* (arXiv 2204.03559): nearest
    neighbour in the medical domain; measures gaze/expression survival vs
    blurring. Our differentiation: the A2 attacker, the periocular control, and
    the retained ocular region being the biometric. **Open item:** confirm from
    the paper whether it runs any linkage attack; if it does, reword.
  - *Nature Medicine "digital mask" (2022)* and its published critique: one
    sentence each — whether generative de-identification suffices for clinical
    release is contested.
  - *Gaze-constrained face swapping* (arXiv 2305.16138) belongs here.
  - *DeepPrivacy2's own re-identification test* (Market1501, enrolment images
    against the anonymised release) is an A1 test; it falls to the mask-out
    level, consistent with our A1. It does not test release-only linkage.
- **C4 Eye movement as a biometric.** Rigas & Komogortsev and successors,
  BioEye/EMVIC, HCI work recovering identity/gender/age from gaze; the gaze-privacy
  literature (differential privacy for gaze streams, RL obfuscation,
  privacy-preserving scanpath comparison, a decade review), and specifically
  **Alsakar, Alotaibi, Khamis & Stumpf, ACM TOPS 2025, on handheld mobile
  devices** — our exact device class.
  - **Rejected: the resolution argument for the open tier.** We first meant to
    argue that 30 Hz at ~5 cm is too coarse to identify anyone. The literature
    does not support it (0.1° / 30 Hz reported sufficient for oculomotor-plant
    biometrics; gentle degradation with sampling rate; above chance at 0.5°
    added noise). We ran the attack instead (ARI 0.033 vs a ~0 permutation null).
    Do not reintroduce the argument.

## D — Threat model [§7]

- **Two attackers because they disagree on identical pixels** (A1 0.2% vs A2
  43% on the same synthetic eye crops). A privacy number without its attacker is
  uninterpretable; this is the protocol contribution.
- **A2 is the operative attacker for a public release**: the recipient of a
  public dataset is by definition A2.
- **Each recogniser against its own floor.** FaceNet's floor (3.52%) is more than
  twice ArcFace's (1.57%); mixing them inflates or deflates by 2×.
- **Alignment from the original frame**, so no operator is credited for defeating
  the detector. Self-aligned attacks are labelled and report surviving frames
  (full-body DeepPrivacy2 lost 32–66 of ~1080 to detection; excluded, not imputed).
- **Rank-1 is not evidence** on single-session data (floor Rank-1 ~50%).

## E — Identity leakage [§8]

- **Why the eye-ROI linkage attack exists.** The face-crop floor is a patch of
  the room; hair, spectacles, outline are attributes of the person. The 120 px
  eye ROI removes them. If linkage survived only on the face crop, "it's the
  hair" would be the review.
- **Why the illumination probe.** Single session confounds person with imaging
  conditions. Normalisation and a within-session lighting change leave linkage
  intact; photometric statistics alone nevertheless link at ARI 0.87–0.90. Both
  halves go in: the recogniser is not using photometry, *and* photometry is a
  second channel. Neither replaces the second recording (Limitations).
- **Why the DeepPrivacy2 2×2.** It turns the two-part condition from an inference
  into a factorial result, and answers "you didn't use DP2's strongest mode".
  The resolution confound (full body generates at 288×160) is controlled by the
  design: the fixed-identity full-body cell has the same degradation and still
  links at 0.982.
- **Wording for the 0.297 cell:** "reduces", never "prevents" — it is above the
  0.089 floor and pair AUC is 0.78.
- **Bootstrap intervals are subject-level.** The first version of the bootstrap
  gave duplicated participants distinct labels and produced intervals that
  excluded their own point estimates (8 of 10 arms); fixed, guarded, and the
  biased output is quarantined. Report only `results/anon_ari_ci/` and the
  full-body bootstrap files.
- **0.795 → 0.805.** The DeepPrivacy2 fixed-identity left-eye ARI was first
  reported from a single KMeans fit (0.795); standardised on the bootstrap
  pipeline's point estimate, 0.805 [0.63, 0.94]. Every other cell agreed.

## F — Edits to the existing text [§6, §9d, §9e, §4]

Deletions are **not** softenings; the consistency fixes are ones a reviewer will
otherwise catch.

| # | Location | What | Why |
|---|---|---|---|
| 1 | Privacy-Utility Experiments, final para | iTracker/AFFNet "failed the Hybrid test", "architectural incompatibility" | Does not replicate in this pipeline; AFFNet recovers best. Delete. |
| 2 | Contributions (V4), GAN section | "encrypted" | Category error; nothing is encrypted. Delete. |
| 3 | Privacy-Utility Experiments, para 1 | "viable pathway for sharing clinical datasets publicly" | Contradicted by §8. Replace. |
| 4 | GAN section opening + Hybrid figure caption | "eliminating the patient's biometric identity" | Not supported for either arm. Replace. |
| 5 | Table 6 | "(Failure)" on an improvement, "(Stable Failure)" | Conflates performance with stability. Restructure. |
| 6 | Dataset protocol | "initially comprised 23 volunteers" vs N=18 | Resolved by restoring the V4 demographics and exclusions; confirm when pasting. |

## G — Limitations [§10]

- **Single session is the first limitation, not the last.** It bounds the
  photometric result, and the deployable fix (per-session identity) is exactly
  what it cannot test. A second recording is the decisive experiment.
- **Condition (i) is one initialisation, three architectures.** Say so; it
  supports direction and rough size, not differences under ~0.4 cm.
- **Full-body DeepPrivacy2 utility is a bound, not a measurement.** Its eyes are
  repainted at lower resolution than the face mode whose eyes already carry no
  gaze signal; the figure draws it at the no-information level with an arrow.
- **Missing `styleganL_nocse`** (410 Gone upstream): no person in our frames
  needed it; on other footage that path would leave people unedited.
- **Conclusion: controlled access, not open deposit.** It follows from the
  measurements; do not soften it into "future work may enable release".

## H — Utility [§9]

- **Why three conditions.** Deploy-only (ii) mixes information loss with domain
  shift. Retraining (i) removes the shift, so comparing (ii) and (i) separates
  them. Testing (i) on real faces answers the release use case (a third party
  trains on the corpus and deploys on its users); testing on de-identified faces
  answers benchmarking on the release.
- **What the comparison showed, and why it strengthens the paper.**
  - SimSwap: the penalty survives retraining → it is lost information. A Full
    Synthetic release is of little use for training a model for real users (two
    of three architectures at or beyond the no-information reference).
  - DeepPrivacy2: two thirds of the (ii) penalty was shift, but what remains is
    the no-information level in 60 of 60 fold results. The (ii) figure of
    +11–15 cm must not be quoted as DeepPrivacy2's utility cost on its own; it
    is "confidently wrong", worse than predicting the mean.
  - Hybrid: at baseline in every condition, both families → the central utility
    claim does not depend on how the model was trained.
- **Why the no-information reference.** Without it "+4.5 cm" sounds like a
  moderate cost; it is in fact "learned nothing". 9.37 cm is the error of
  predicting the training participants' mean gaze point, per fold.
- **Why condition (ii) keeps five draws and (i) has one.** (ii) is Table 3 and
  carries the precision claims (init noise floor 0.04–0.20 cm). (i) answers a
  directional question whose effects are 2–5 cm.
- **"Recovers", not "preserves".** Six of eight Hybrid arms in Table 3 are
  detectably worse than baseline at the measured noise floor.
- **Clinical anchoring (M2).** Compare calibrated error (1.8–3.7°) with the
  literature's 1.42°, which is itself post-calibration; quoting 1.42° beside an
  uncalibrated number overstates the gap. Requirement column needs a clinician.

## Figures [§11]

- **The privacy–utility plane is the one main-text privacy figure.** It carries
  the conclusion in a single panel; the threat-model figure carries a
  methodological point §7 already makes in prose.
- **Every figure is generated from result files** by
  `scripts/anon/make_core_figs.py`; regenerate rather than edit by hand.
- No figure shows participant imagery. Keep it that way: the smoke-test montages
  used for checking DeepPrivacy2 output stay local.

## Page budget

Compress the four per-architecture descriptions (eqs. 2–3, iTracker three-stream
prose) into one comparison table plus two sentences — they are published
architectures — and collapse the seven task descriptions to the two that later
sections use plus one sentence.

## Claims withdrawn during this revision — do not reintroduce

| Claim | Why withdrawn |
|---|---|
| iTracker/AFFNet "architectural incompatibility" with Hybrid | Did not replicate; 0.01 cm arm gap was an inactive-eye-stream signature |
| Hybrid "preserves accuracy" | Six of eight arms detectably worse than baseline; use "recovers 85–101%" |
| Open tier safe because gaze is too coarse to identify | Unsupported by the literature; measured instead (ARI 0.033) |
| Our error "vs 1.42°" with uncalibrated numbers | 1.42° is post-calibration; compare against calibrated 1.8–3.7° |
| Breaking per-participant consistency suffices | Half right; two-part condition |
| DeepPrivacy2 costs +11–15 cm | That is deploy-only and includes domain shift; retrained, it is the no-information level |
| "Methods with a wider edit domain remain to be tested" | Tested: DeepPrivacy2 face and full body |
| Eye-ROI ARI 0.795 (DP2 face, fixed) | Single fit; bootstrap point estimate 0.805 |

## Open items

Tracked in `CCF_PASTE_ORDER.md` → *Open items*.
