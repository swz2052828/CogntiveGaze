# Raw-frame single-input ViT vs facemesh+multistream (task #6) — VERDICT: multistream wins decisively

RawFrameViTGaze = timm ViT-S/16 @384², trained on the raw above-shoulder 1080x750
frames (no face detection, no eye crops, no grid; attention must localize eyes).
5 folds, meanno7_clean, seed 42. Calibration = fc_ft (Zhu defaults) on MATCHED
random-K draws (same support indices both pipelines; leaky-K, so absolute values
are optimistic vs deploy-faithful — the comparison itself is fair).
Data: `runs/raw_vs_ms/*.csv`; facemesh latency job 1170134.

## Accuracy (cm, fold-mean)
| | ms atto_binocular (3.7M) | raw ViT-S/384 (22.1M) | ms vit (87.3M) |
|---|---|---|---|
| base            | **4.36** | 5.88 | 4.37 |
| fc_ft @K9       | **2.40** | 4.83 | 2.44 |
| fc_ft @K72      | **2.03** | 4.53 | 2.04 |

**The gap WIDENS after calibration** (+1.5 base → +2.5 @K72): calibration
transforms multistream (-2.3) but barely helps raw (-1.3). The raw model's
features lack the iris detail calibration exploits — same law as
"svr_embed rescues a bad readout but not a weak encoder". Root cause matches
[eye-crop-resolution-ceiling]: eyes span ~105px native; in a 384² full frame
they shrink to ~37px. Global attention cannot recover downsampled iris detail.

## Compute (NVIDIA L40, batch 1)
| | end-to-end/frame | GPU fwd | facemesh (CPU, 4thr) | params | peak GPU MB |
|---|---|---|---|---|---|
| ms atto_binocular | ~6.4 ms | 2.78 ms | 3.33 + 0.26 crop | **3.7M** | 135 |
| raw ViT-S/384     | **~2.7 ms** | 2.74 ms | 0 | 22.1M | 129 (452 at fwd) |
| ms vit            | ~11.4 ms | 7.8 ms | 3.33 + 0.26 | 87.3M | 451 |

## Conclusion
Dropping facemesh saves only ~3.7 ms/frame CPU (both pipelines are comfortably
real-time) but costs **+2.5 cm calibrated accuracy** and **6x the parameters**
vs the multistream leader. The face/eye-crop pipeline is not an overhead to be
optimized away — it is a hard-coded attention prior that preserves native iris
resolution, and it is worth far more than its milliseconds. Raw-frame end-to-end
gaze from this camera is a dead end at this input resolution; it could only
become competitive with a much higher-resolution source or a cropping-free
foveation mechanism (e.g. learned zoom), both out of scope.

---

# UPDATE (2026-07-04): deploy-faithful calibration + MobileViT/Foveal variants + 10k-frame timing

## Deploy-faithful raw calibration (support = OriginalCalib, SAME frame IDs as multistream; cm, 5-fold mean)
| model | params | base | fcft@K9 | fcft@K72 | svrE@K72 |
|---|---|---|---|---|---|
| raw_mobile_vit (MobileViT-S) | 5.3M | 5.15 | 4.04 | **3.90** | 4.69 |
| raw_vit (ViT-S/384) | 22.1M | 5.88 | 5.60 | 5.54 | 6.27 |
| raw_foveal_vit (2-view ViT-S) | 22.3M | 6.36 | 5.85 | 5.69 | 6.13 |
| *multistream reference (deploy-faithful, clean metacompare)* | 3.7M | 4.36 | ~3.4 | ~3.3 | 2.97 |

- **raw_mobile_vit is the best raw model by far** (3.90 vs 5.54): conv inductive
  bias beats global attention on the raw frame too — same law as the crop study.
- **The fovea design failed** (6.36 base, worse than plain raw_vit): the fixed
  top-center crop misses faces off-center; a learned localizer would be needed —
  which is what facemesh already is.
- Even the best raw model stays **~0.9-1.0 cm behind** the multistream leaders
  under identical deploy-faithful calibration.

## 10,000-frame inference time (user protocol; facemesh 70.0 s incl. decode; GPU b1 forwards)
| pipeline | total s/10k | vs best raw |
|---|---|---|
| raw_vit | **26.4** | — |
| raw_mobile_vit | 37.2 | — |
| raw_foveal_vit | 52.7 | — |
| facemesh + atto_binocular | 97.0 | +70.6 |
| facemesh + binocular | 97.0 | +70.6 |
| facemesh + face_only_mobile_vit | 107.4 | +81.0 |
| facemesh + foveal_vit | 115.9 | +89.5 |
| facemesh + mobile_vit | 181.9 | +155.5 |

Raw pipelines save 70-155 s per 10k frames (7-16 ms/frame) — i.e., all pipelines
run comfortably real-time (≥55 fps end-to-end even for the slowest). **The time
saved does not buy back the 0.9-2.6 cm accuracy cost.** Verdict unchanged:
facemesh+multistream wins; raw_mobile_vit is the only raw variant worth noting
(a viable fallback where face detection is impossible).
