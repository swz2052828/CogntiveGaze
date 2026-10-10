# privaudit

Release-time privacy audit for appearance-based gaze datasets (smartphone or
webcam: face crop, eye crops, face grid). It runs the attacks that decide whether
a candidate release can be shared, against the floors that make their numbers
interpretable, and maps each result onto a release checklist
([CHECKLIST.md](CHECKLIST.md)).

It was built for, and validated on, the CognitiveGaze study, and reproduces that
study's eye-stream numbers to floating-point precision
(`validate_cognitivegaze.py`).

## Input

GazeCapture layout, frame names sorting in temporal order:

```
release/
  <participant>/
    appleFace/      000123.jpg ...
    appleLeftEye/   000123.jpg ...
    appleRightEye/  000123.jpg ...
```

Optional, same layout: `--original` (unprocessed crops; null control),
`--enrolment` (photographs of the same people; attacker A1). Optional
`--floor-frames <root>/<participant>/<frame>.jpg`: the full frames, from which a
face-free patch gives the session-nuisance floor. **Without a floor, verdicts
fall back to a statistical null and overstate leakage on single-session data.**

## Run

```bash
pip install insightface onnxruntime-gpu scikit-learn opencv-python pillow numpy
python -m privaudit --release data/release \
    --streams appleFace appleLeftEye appleRightEye \
    --floor-frames data/frames --original data/unprocessed \
    --gpu 0 --out audit/
```

Writes `audit/audit.md` (the table below), `audit/audit.json` (every number) and
the embeddings. An 18-participant release takes minutes on one GPU, mostly the
bootstrap (`--n-boot`).

Excerpt, CognitiveGaze with SimSwap identity A (the face-free floor, measured on
the same recordings, is TAR 1.57% and ARI 0.074):

```
| Item                           | Stream       | Status | Evidence                                          |
| R2 release-only linkage        | appleLeftEye | FAIL   | ARI 0.683 [0.591, 0.872]                          |
| R3 release-only verification   | appleLeftEye | FAIL   | TAR@FAR=1e-3 42.78% [5.92%, 62.47%]               |
| R4 enrolment re-identification | appleLeftEye | PASS   | TAR@FAR=1e-3 0.00% [0.00%, 0.00%]                 |
| R5 photometric channel         | appleLeftEye | FAIL   | 21 summary statistics, no recogniser: ARI 0.898   |
| R6 crop-geometry metadata      | all streams  | FAIL   | expected Rank-1 66.7% (chance 5.6%), no pixels    |
```

The same synthetic eye crops pass against an attacker with photographs (R4) and
fail against one holding only the release (R2, R3): the two attackers must both
be run.

## What it measures

| Item | Attack | Threat |
|---|---|---|
| R0 | the attacks below on a face-free patch of the same frames | floor |
| R2 | KMeans partition of the release at the true cohort size; ARI with participant-bootstrap CI | A2 |
| R3 | first-half templates vs second-half probes; TAR at FAR 1e-3 with CI | A2 |
| R4 | enrolment templates vs released probes | A1 |
| R5 | 21 photometric statistics per crop, no recogniser | A2 |
| R6 | crop dimensions as an identifier, expected Rank-1 and bits | A2 |
| R7 | partition with the cohort size unknown (silhouette) | A2 |
| R8 | release vs original, mean absolute difference | integrity |

Eye crops are embedded directly at 112×112 — a face detector finds nothing in a
120 px eye crop, and detector failure must not be scored as privacy. Face crops
can be embedded the same way or, with `--align-detect appleFace`, after detecting
a face *on the released crop* (failures counted, not imputed).

**Decision rule.** A channel leaks when the lower bound of its 95%
participant-level bootstrap interval exceeds the floor. Intervals resample
participants, not frames; a duplicated participant keeps its own label, and an
interval that excludes its own point estimate is refused rather than reported.

## Limits

- Not a certificate. It measures the attacks it implements, on the sessions you
  recorded. Cross-session linkage — the longitudinal threat — needs a second
  session to measure.
- Pretrained recognisers carry demographic bias; small or homogeneous cohorts
  limit how far a verdict generalises.
- The non-pixel tier (gaze traces, oculomotor features) is not covered; see
  CHECKLIST.md C13.
