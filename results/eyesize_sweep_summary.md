# Eye-size sweep (opt #4 as requested) — eyes_only_convnextv2_atto_binocular, 5 folds, clean data

| --eye-size | base val error (cm) | vs 224 |
|---|---|---|
| 112 | 7.740 | +3.39 |
| 168 | 5.394 | +1.05 |
| **224 (baseline)** | **4.348** | — |
| 336 | 4.767 | +0.42 |

**224 is the optimum.** Downsizing is severely costly (112 nearly doubles the
error — the model needs every pixel of the ~105-120px native eye detail,
presented at the encoder's pretrained resolution). Upscaling to 336 is also
WORSE (+0.42): it adds zero true detail (source is 120px) while moving the
encoder off its pretrained 224 resolution and diluting attention over
interpolated pixels. Consistent with the native-resolution ceiling measured in
native_eye_span.md: the information stops at the source; the loader's job is
just to present it at the encoder's native input size.
