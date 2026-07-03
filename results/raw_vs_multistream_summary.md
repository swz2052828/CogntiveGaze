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
