# 30-frame drift-shift correction (cm)

| method | mean | median |
|---|---|---|
| svr_4d_mm_drift30_roll | 4.116 | 4.127 |
| svr_4d_drift30_roll | 4.309 | 4.394 |
| rbf_2d_drift30_roll | 4.328 | 4.424 |
| projective_2d_drift30_roll | 6.174 | 4.545 |
| svr_4d_mm_drift30_causal | 8.064 | 8.089 |
| projective_2d_drift30_causal | 8.278 | 5.680 |
| projective_2d_drift30_settle | 8.306 | 6.396 |
| svr_4d_mm_raw | 8.397 | 8.359 |
| projective_2d_drift30_start | 8.451 | 6.720 |
| svr_4d_mm_drift30_start | 8.964 | 8.659 |
| svr_4d_drift30_causal | 9.025 | 9.088 |
| rbf_2d_drift30_causal | 9.033 | 9.095 |
| rbf_2d_raw | 9.123 | 9.154 |
| svr_4d_raw | 9.130 | 9.157 |
| svr_4d_mm_drift30_settle | 9.557 | 9.463 |
| rbf_2d_drift30_start | 10.077 | 10.021 |
| svr_4d_drift30_start | 10.080 | 10.046 |
| svr_4d_drift30_settle | 10.766 | 10.766 |
| rbf_2d_drift30_settle | 10.769 | 10.756 |
| projective_2d_raw | 21.677 | 22.199 |

## Conclusion
Rolling (per-second GT anchor) reaches 4.12cm but is not deployable. Causal
(previous-second anchor) recovers almost nothing (8.06 vs raw 8.40) and
settle-skipped start anchors HURT (9.5-10.8): the per-block shift decorrelates
within ~1 second — the residual is fast wobble (detector/micro-movement), not
slow head drift, so any anchor is stale by the time it is applied. Start-of-task
anchoring therefore cannot work on this signal. Best deployable remains
svr_4d + moment matching with per-dot median aggregation (5.23cm, at the
signal ceiling); rolling-GT correction only adds value where the target is
continuously known (e.g. smooth-pursuit UIs), where it is worth ~4.3cm.
