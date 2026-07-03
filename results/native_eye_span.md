# Native eye-region span in raw OriginalData (1080x750), px

ProcessedData eye crops are 120x120 at source (upscaled to 224 in the loader).
Native span = square eye box (GazeCapture-style 1.7 margin) from FaceMesh.

| rec | eye median px | eye p90 px | face median px | n |
|---|---|---|---|---|
| 00006 | 96 | 100 | 336 | 60 |
| 00007 | 87 | 94 | 341 | 60 |
| 00008 | 117 | 132 | 459 | 60 |
| 00009 | 103 | 112 | 403 | 60 |
| 00010 | 109 | 119 | 432 | 60 |
| 00011 | 107 | 115 | 390 | 60 |
| 00012 | 108 | 119 | 330 | 60 |
| 00013 | 112 | 122 | 402 | 60 |
| 00014 | 124 | 131 | 469 | 60 |
| 00015 | 96 | 106 | 347 | 60 |
| 00016 | 111 | 123 | 417 | 60 |
| 00017 | 109 | 116 | 424 | 60 |
| 00018 | 104 | 107 | 374 | 60 |
| 00019 | 101 | 111 | 382 | 60 |
| 00020 | 96 | 103 | 333 | 60 |
| 00021 | 102 | 105 | 384 | 60 |
| 00022 | 116 | 119 | 428 | 60 |
| 00023 | 99 | 104 | 374 | 60 |

**Cohort: eye median 105px, min 87, max 124.**

Verdict: existing 120px crops are near-native; recrop adds little
