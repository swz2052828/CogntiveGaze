# Is the periocular leak an illumination artefact?

**Date:** 2026-09-21, job 1286157. 16 attacks, no failures.

The objection: every recording is a single session, so exposure, white balance,
focus distance and the light source's angle on the cornea are constant within a
participant and differ between participants. The face-free floor controls for the
ROOM, not for those, because they are properties of the imaging of the PERSON.
Our §Results paragraph attributed the residual linkage to "illumination, skin
tone, periocular geometry and crop scale" — and illumination is a session
attribute, so that attribution was not safe.

A second recording settles this and we cannot run one. Three probes attack the
named channel directly with the data we have.

## Results

Release-only linkage, ARI at known k = 18, pooled halves, 120 px eye ROI:

| Condition | Original L / R | SimSwap L / R |
|---|---|---|
| Baseline (temporal half split) | 0.844 / 0.853 | 0.684 / 0.789 |
| **z-score** (per-crop brightness and contrast removed) | 0.835 / 0.858 | **0.742 / 0.772** |
| **CLAHE** (local histogram equalisation) | 0.858 / 0.833 | **0.637 / 0.653** |
| **Cross-task-block** (large within-session illumination change) | **0.918 / 0.931** | **0.717 / 0.762** |
| *Photometric features only* | *0.870 / 0.902* | *0.898 / 0.901* |

Verification under the same conditions, TAR@FAR=1e-3:

| Condition | Original L / R | SimSwap L / R |
|---|---|---|
| Baseline | 66.6% / 62.2% | 42.8% / 62.5% |
| z-score | 68.0% / 73.0% | 34.4% / 59.1% |
| CLAHE | 60.7% / 66.3% | 32.2% / 46.9% |
| Cross-task-block | 63.2% / 62.2% | 36.2% / 59.6% |

## Two independent carriers, not one

**The recogniser's linkage does not depend on illumination.** Removing per-crop
brightness and contrast entirely leaves it intact — the swapped arm *rises* from
0.684 to 0.742 under z-score — and CLAHE costs 0.05–0.14 without collapsing it.
A large within-session illumination change (dark-screen stimulus blocks as
gallery, natural-image blocks as probe) leaves it intact, and on the original
crops raises it to 0.918 / 0.931. Verification behaves the same way: 62.2% → 73.0%
on the right eye under z-score.

**But photometric statistics alone also link, at 0.87–0.90.** Twenty-one summary
numbers per crop — per-channel mean, SD, percentiles, white-balance ratios,
dynamic range, no structure at all — recover the participants about as well as
the face recogniser does. This is a separate and simpler attack path that needs
no face model, and the swap does not touch it: SimSwap scores 0.898 / 0.901
against the originals' 0.870 / 0.902.

So the two channels are **redundant carriers**, not the same carrier. Suppressing
the photometric one does not degrade the recogniser, which means the recogniser
was not relying on it.

## What this does and does not settle

**Settles:** the attribution in §Results survives. The residual linkage the
recogniser exploits is not an illumination artefact.

**Adds:** a purely photometric attack links the corpus at 0.87–0.90. Worth
reporting on its own — releasing crops with unnormalised photometry is itself a
linkage channel, and normalising it would be cheap.

**Does not settle:** cross-task-block changes the *screen*, which is the dominant
light source at 80–120 cm in a dim room, but not the ambient lamp, camera
position, exposure regime or seating. It is a partial dissociation. And in a
single session "this person's skin tone under this lamp" and "this recording's
exposure" remain confounded, which is exactly why the photometric control scores
as high as it does. **A second recording remains the decisive experiment.**

What has changed is the status of the objection: "this might be an illumination
artefact" was an open alternative explanation and is now a tested and largely
rejected one, with the residual confound named and bounded.

## Files

`illum_{ProcessedData,ProcessedSwap}_apple{Left,Right}Eye.json` — photometric-only
`peri_{normzscore,normclahe,block}_*.json` — attacks
`emb_*.npz` — embeddings, for the linkage clustering
`scripts/anon/session_confound.py`; `periocular_attack.py --normalise / --split`
