# Condition (iii): the calib_processed source bug, 2026-09-10

## What happened

The first condition (iii) run (job 1272012) was invalid and is quarantined under
`runs/anon_stage4_cond3/contaminated/`. `build_anon_calib_support.py` iterated
frame names from `calib_support_K<K>` but read PIXELS from
`datasets/calib_processed` — a different root holding a different face box for
the same frames.

| source | IOD / face-crop width |
|---|---|
| task data (`ProcessedData`, what the models were trained on) | 0.400 – 0.485 |
| `calib_support_K72` (conditions (i)/(ii), P-axis) | 0.383 – 0.449 |
| **`calib_processed`** (what the builder actually read) | **0.286 – 0.306** |

`calib_support_K72`'s face crop is a **1.40x zoom** of `calib_processed`'s
(multi-scale template match, corr 0.992). Measured MAE between them on
`appleFace`: **39.4** levels, against a JPEG re-encode floor of **0.11**. Eye
crops were unaffected (MAE 1–9).

## How it surfaced

On the `none` arm the operator is a no-op, so condition (iii) must equal
condition (ii). It did not: `face_only_mobile_vit` fc_ft read **5.727 cm against
3.548 cm**. Two features of the discrepancy located the fault:

- `base` matched condition (ii) to three decimals on every arm. `base` never
  touches the calibration support, so the fault was in the enrolment path.
- The damage was worst on the **face-only** backbone and mild on the multistream
  ones (convnextv2 +0.06, mobile_vit +0.27), which is the signature of a
  face-crop error rather than an operator effect.

Nothing else would have caught it. The crops were the right size, the right
frames, and visually the right face.

## Fixes

1. **`build_anon_calib_support.py` now reads pixels from the root whose filenames
   it iterates** — it anonymises the released calibration crops in place. There is
   no second source root any more, so this mistake is now unrepresentable.
2. **IOD comes from the geometry file**, not a hard-coded table, so it cannot
   drift out of step with the crops it describes. `iod_for()` refuses to build if
   IOD/face-width falls outside 0.36–0.52.
3. **`calib_crop_geometry.py` grew `--src-root`** (default `calib_support_K72`)
   and an `--expect-ratio` guard that refuses to write geometry recovered against
   a face-box convention that does not match the task data.

## A second bug found while fixing the first: truncated eye templates

`cv2.matchTemplate` only evaluates positions where the whole template fits, so an
eye box that extends past the face-crop edge is unrepresentable — the best the
search can do is pin the template at the edge, which misaligns it and depresses
the correlation. Read as a match failure, it dropped frames:

| recording | left-eye corr before | after | frames below 0.95 before → after |
|---|---|---|---|
| 00012 | 0.795 | **0.963** | 68/72 → 17/72 |

00012's left eye sits at x ≤ 0 in **99%** of frames — it has among the tightest
face crops (IOD/width 0.449). Rejecting those frames would have left it 5 usable
enrolment frames, tripped the enrolment guard, and **silently removed the subject
the result is most sensitive to**.

`match_in_crop` now searches on a border-replicated pad and returns coordinates
that may be negative; callers preserve the intersection with the crop, which is
the correct answer — the rest of the eye genuinely is not in the released image.
Control: for 16 of the other 17 recordings the padded search reproduces the
unpadded boxes **bit-identically**. The one change is a fix of the same
pathology — 00009 frame 03227, right eye:

| | x, y, w, h | corr |
|---|---|---|
| unpadded | 98, **0**, 65, 65 | 0.765 |
| padded | 93, **−12**, 78, 78 | **0.929** |

78 px is that recording's normal box size; the unpadded search had shrunk the
template to 65 px to make it fit against the edge.

## Residual, accepted

**19 of 1296 enrolment frames (1.5%)** remain below corr 0.95 — 17 in 00012, one
each in 00007 and 00009 — all structurally truncated, so part of the template
necessarily lands on replicated border and the correlation is capped below 1 even
at the correct position. These take the per-subject **median box**
(`--allow-median-geometry`), which is sound: the box is stable to 0.7–3.4 px
within a subject. Fallback counts are reported by the builder and are 19 for
every operator, as expected.

**Enrolment crops are ~7% wider relative to the face than task crops**
(median difference in IOD/width: −0.035, range −0.007 to −0.073). This is a
property of `calib_support_K72` itself — `generate_calib_support.py` crops from
the calibration video with its own FaceMesh box — and is therefore shared by
conditions (i), (ii) and the P-axis. It cancels in every operator-vs-control
comparison. Not introduced by this fix; recorded so it is not rediscovered.

## Verification of the rebuild

| op | appleFace MAE vs source | eye crop MAE | eye region inside face crop |
|---|---|---|---|
| none | **0.08** | 0.04 | 0.09 |
| blur0.10 | 8.65 | 0.04 | 0.32 |
| blur0.20 | 12.13 | 0.04 | 0.34 |
| blur0.50 | 16.23 | 0.04 | 0.33 |
| blackbox | 73.13 | 0.04 | 0.73 |

`none` sits at the re-encode floor (0.11), so the control is a control again.
Eye crops are untouched; the eye region inside the face crop is preserved to
within JPEG ringing even for blackbox. All five roots: 18 recordings x 72 frames,
0 missing.

## Coverage

Condition (iii) covers `none`, `blur0.10`, `blur0.20`, `blur0.50`, `blackbox`.
The `blur0.03`, `blur0.35` and five `pixel*` deployment roots were removed in the
storage cleanup, so those arms cannot run without regenerating ~2.4M frames. The
surviving ladder (control, three blur strengths, total removal) supports the
condition (iii) claim; pixelation tracked blur closely on every privacy axis
measured in Stage 1, so the gap is a completeness gap, not an evidential one.

Re-run: job 1272086.

---

# Blast radius of the source bug: 38 calibration supports quarantined

The bug affected every `calib_support_K<K>_<op>` root, not just the K72 arms in
use. Moved to `datasets/_badgeom_calib_supports/` with a manifest and a README:

| set | fate |
|---|---|
| `calib_support_K18_<op>` (12), `calib_support_K36_<op>` (12) | quarantined — never wired into any run |
| `calib_support_K72_<op>` and `calib_support_K9_<op>` for `blur0.03`, `blur0.35`, `pixel0.03/0.06/0.12/0.25/0.40` (14) | quarantined — their deployment roots were deleted in the storage cleanup, so no arm can use them |
| `calib_support_K{9,72}_{none,blur0.10,blur0.20,blur0.50,blackbox}` (10) | **rebuilt correctly** and in use |

None of the quarantined roots backs a published number — only K72 was ever wired
into `run_stage4_cond3.sbatch`. They are quarantined rather than left in place
because they inspect as perfectly healthy: right frames, right size, a real
upright face. Restoring any of them means recovering geometry for that K first
(`run_calib_geom_k.sbatch` with `K=<k>`; the K levels are **not** nested, so each
needs its own) and then re-running `run_rebuild_calib_support.sbatch` with the
same K.

# K=9: does the recovery survive a deployment-realistic enrolment?

K=72 is 72 calibration frames per subject. A deployed at-home screening app
realistically gets a **nine-point** calibration lasting a few seconds. The
condition (iii) recovery was measured at K=72, so it is worth nothing if it does
not survive an 8x smaller enrolment.

Built for K=9: geometry (`anon_calib_geometry_k9`, 18/18 recordings, 9/9 frames
each, IOD/face-width 0.382-0.449 — matching the K72 values per subject, e.g.
00006 0.422 vs 0.424, 00012 0.449 vs 0.449) and the five operator supports
(18x9 = 162 frames each, 0 missing, 5 median fallbacks, `none` at MAE 0.08 vs
the source — the re-encode floor).

`run_stage4_cond3.sbatch` and `run_stage4_full.sbatch` now take `K` and `KTAG`,
so conditions (ii) and (iii) run at any K without a forked copy.
