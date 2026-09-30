# Three controls the privacy axis was missing

**Date:** 2026-09-17, job 1283732 plus one CPU run. All three were raised by our
own review of the draft; all three are now measured rather than argued.

---

## 1. An eye-crop-native floor

Every floor quoted so far (TAR 1.6%, Rank-1 50.5%, ARI 0.089) came from
`t1bg_p120`, which runs the **face-crop** pipeline on a face-free patch. That
controls for room and session but is not strictly an eye-crop quantity, and the
draft had to label it "borrowed".

Running the **periocular** pipeline itself on a 120 px face-free patch of the same
frames — same frame selection, same recogniser, same downstream steps, only the
pixels differ:

| | TAR@FAR=1e-3 | Rank-1 | d′ |
|---|---|---|---|
| ArcFace, eye-native floor | **1.57%** | 50.6% | 0.458 |
| *(previously borrowed from the face pipeline)* | *1.6%* | *50.5%* | *0.46* |

Agreement to three significant figures. The limitation is closed, and figure 5b's
"floor borrowed from panel a" annotation can be replaced with the measured value.
Left and right are identical because the background patch does not depend on the
eye folder — a passing consistency check on the new `--background` path.

---

## 2. A second recogniser family

Every identity number in the study came from ArcFace (ResNet-50, WebFace600K,
margin softmax). FaceNet (Inception-ResNet-v1, VGGFace2, triplet) is
architecturally unrelated and trained on a different corpus, so agreement is
evidence about the imagery rather than about one model's inductive bias.
Detection/alignment is not involved: the periocular attack embeds the resized
crop directly.

Release-only attacker (gallery and probe both from the released corpus):

| Eye crop | ArcFace TAR@1e-3 | FaceNet TAR@1e-3 |
|---|---|---|
| Original, left / right | 66.6% / 62.2% | **67.1% / 80.5%** |
| SimSwap t1, left / right | 42.8% / 62.5% | **42.8% / 55.8%** |
| SimSwap t2, left / right | 56% / 58% | **34.4% / 51.0%** |
| *Floor* | *1.57%* | *3.52%* |

**Both recognisers agree on both load-bearing claims.** Unmodified ocular ROIs are
strongly identifying — 19–23x FaceNet's own floor, 40x ArcFace's — and the swap
does not bring the release-only attack anywhere near the floor.

**FaceNet's floor is 3.52%, more than twice ArcFace's.** It is more susceptible to
session nuisance, so FaceNet figures must always be quoted against the FaceNet
floor. Mixing the two floors would overstate the leakage by a factor of two.

### 2b. The A1 cell, also cross-checked

`id_attack` gained the same `--recogniser` switch (job 1284550). FaceNet is
embeddings-only, so the flag refuses to run without `--no-detect` and refuses
`--attributes` rather than failing halfway through a gallery. Gallery is
`ProcessedData` (original enrolment imagery), probe is the released root:

| Eye crop | ArcFace A1 TAR@1e-3 | FaceNet A1 TAR@1e-3 |
|---|---|---|
| Original, left / right | 66.6% / 62.2% | **69.1% / 81.2%** |
| SimSwap t1, left / right | 0.2% / 1.8% | **0.00% / 1.48%** |
| SimSwap t2, left / right | 0.6% / 1.1% | **0.93% / 0.19%** |

The two recognisers now agree across **both threat models and both templates**.
Under A1 the swap takes the ocular ROIs to 0–1.5%, below any plausible floor
(ArcFace's A1 floor is 1.6%; FaceNet's release-only floor is 3.52%). FaceNet's
Rank-1 on swapped crops is 8–23% against 5.6% chance — the same "faint residue,
unusable for verification" picture ArcFace gives.

**Gap, stated:** no FaceNet-specific *A1* floor was measured — `id_attack` has no
`--background` path, only `periocular_attack` does. Values of 0.00% and 0.19%
cannot exceed any floor, so the conclusion is unaffected, but the floor quoted
alongside these figures is ArcFace's.

---

## 3. Does the open tier leak?

The release plan puts per-frame gaze estimates and oculomotor measures in an open
tier because it holds no imagery and no face embeddings. Eye movement is itself a
biometric, so that had to be tested.

**The resolution argument we intended to use does not hold and must not be used.**
We were going to note that our instrument (30 Hz, ~5 cm ≈ 2.4–3.6° at 80–120 cm)
sits far below the operating points usually quoted for oculomotor biometrics. But
one line of work reports **0.1° / 30 Hz** as sufficient; degradation with sampling
rate is gentle (EER 0.073 / 0.082 / 0.090 at 1000 / 250 / 125 Hz); and
identification stays above chance with 0.5° of added spatial noise.

So we ran the attack. Attacker A2 holds only the released gaze series; each
recording is cut into 5 s fragments (150 frames at 30 fps) and described by
fifteen dispersion, velocity and saccade/fixation statistics; features are
z-scored and clustered exactly as the pixel linkage is.

| | value |
|---|---|
| ARI at known k = 17 | **0.033** |
| Purity | **0.157** (chance 0.059) |
| k selected unaided | 4 (ARI 0.033) |
| Label-permutation null (20 reps) | **−0.0001**, max 0.0015 |
| *Pixel tiers, for scale* | *0.68 – 1.000* |

**The open tier leaks, and the leak is small.** It is clearly above the
permutation null and about 2.7x chance purity, but two orders of magnitude below
the pixel tiers. The honest description is low — not zero — residual risk.

**Validity check.** `labelDotXCam/YCam` is the participant's EyeLink gaze, not the
stimulus target: 18,338 distinct x values, and two participants at the same frame
index differ by 15.1 cm on average with **no** identical frames. Had these been
target positions the experiment would have been vacuous.

---

## Files

- `peri_floor_{arcface,facenet}_apple{Left,Right}Eye.json` — floors
- `peri_facenet_{ProcessedData,ProcessedSwap,ProcessedSwap2}_*.json` — second recogniser
- `gaze_signal_linkage.json` — open tier
- `scripts/anon/recognisers.py`, `scripts/anon/gaze_signal_linkage.py`,
  `scripts/anon/run_peri_controls.sbatch`
- `periocular_attack.py` gained `--recogniser` and `--background`; defaults are
  unchanged, so every earlier result reproduces.
