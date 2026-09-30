# Periocular linkage (T1) — the control the face-crop result was missing

> See also `ITRACKER_HYBRID_CHECK.md` and, for the rebuilt utility axis,
> `paper/CCF_SECTIONS.md` §H.


**Date:** 2026-09-16. **Prompted by:** the objection that ARI 1.000 on the
`appleFace` crop proves nothing, because hair, ears, neck, clothing and build are
in frame and SimSwap edits only the inner face.

The objection is correct about the control that existed. The session-nuisance
floor (`t1bg_p120`, ARI 0.089) is a **face-free patch of the room**. It rules out
room and session cues. It does *not* rule out person-but-not-face cues, because
those are inside the face crop and absent from the room patch.

## What the face crop actually contains

Inspected directly (8 subjects, `ProcessedSwap/*/appleFace`, 320x320). The crop
runs eyebrows to chin. **Ears, neck, clothing and build are out of frame**; hair
appears only as edge wisps on long-haired subjects. So four of the five channels
named in the objection are not available to this attacker in the first place.

What *is* in frame and is *not* swapped: spectacles (retained verbatim), skin
tone and illumination (SimSwap blends the template onto the source skin), and
head pose / face outline / camera distance (preserved by design — this is what
keeps gaze geometry intact).

## The control: linkage on the 120 px eye crop

The eye crop excludes hair, ears, neck, clothing and the face outline. In the
fully swapped arm the eye pixels are themselves synthetic. Clustering follows
`linkage_attack.py` exactly: KMeans at known k, silhouette-selected
agglomerative (cosine, average) at unknown k. Script: `scripts/anon/peri_linkage.py`.
Inputs are the embeddings already saved by `periocular_attack.py --emb-out`;
no new inference was run.

ARI @ known k=18, pooled halves (per-half values in the JSONs):

| Eye crop | left | right | T2 TAR @ FAR 1e-3 (left / right) |
|---|---|---|---|
| Original | 0.844 | 0.853 | 66.6% / 62.2% |
| SimSwap template 1 | 0.684 | 0.789 | 0.2% / 1.8% |
| SimSwap template 2 | 0.737 | 0.761 | at floor |
| *Floor (face-free patch)* | *0.089* | *0.089* | *1.6%* |

## What this establishes

A dissociation, measured on the same pixels with the same model:

- **Identity-to-name** (needs external reference imagery): the swap destroys it,
  62–67% -> 0.2–1.8%, a ~40x drop to the floor.
- **Identity-to-itself** (within the released corpus): the swap barely touches
  it, 0.85 -> 0.68–0.79 against a 0.089 floor.

Crucially this holds in a crop containing none of the channels the objection
named. The leak is therefore **not** a matter of cropping too widely. The
residual channels are skin tone, illumination, periocular geometry and crop
scale — which are exactly the signals gaze estimation needs. Removing them costs
the utility (full-synthetic arm, base 8.19 cm vs 5.07 cm clean).

This is a constraint of the task, not a defect of SimSwap: **linkability and
gaze utility share a signal**. The claim generalises to any method that edits
only the inner face.

## Limitations, to be reported

- **Unknown-k clustering mostly fails on eye crops.** Silhouette usually selects
  k=2 and ARI collapses to ~0. Every number above assumes an attacker who knows
  n=18. That premise is realistic for a public dataset (the README states the
  cohort size) but it is a premise, and the face crop does not need it — there
  silhouette recovers k=18 unaided.
- **The one place the swap does protect linkage** is unsupervised discoverability
  on the right eye: original k_auto=26 at ARI 0.674, swapped k_auto=2 at ARI
  0.011. The swap breaks discovery of the partition, not its separability.
- The floor is borrowed from the face-crop pipeline (a 120 px room patch through
  the same model and clustering). Scale and model match; the crop content does
  not. A periocular-specific floor has not been run.

## Correction to earlier material

`results/ethics_amendment/07_data_release_plan.md` §1b says "Realism is what
makes it good for utility and what makes it linkable." That attributes linkage
to the swap's realism, which is **not supported** — the swapped faces all look
like one template person and linkage is unchanged. Replace with the mechanism
above. The release-tier decisions are unaffected: they were set by the measured
residual, and the residual is confirmed, not revised.
