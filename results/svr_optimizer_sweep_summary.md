# SVR calibration optimizer sweep: PSO vs JAYA vs MVO (33 backbones x 5 folds, cm)

Swarm-tuned (C, gamma, epsilon) for the embedding-space SVR calibrator
(Zhu et al. bounds), leak-free deploy-faithful fitness (train subjects only:
support = pre-task calib K72, query = task frames), evaluated on held-out
subjects at K = 9/18/32/36/72 (evenly-spaced subsample of the 72 calib frames).
Fixed baseline = C=1, gamma=scale, eps=0.1 (what the earlier metacompare used
when SVR_TUNE was off). runs/svr_opt_sweep/, jobs 2026-07-19, 165/165 clean.

## Cohort mean (33 backbones)
| method | K9 | K18 | K32 | K36 | K72 | tune s/fold |
|---|---|---|---|---|---|---|
| fixed | 8.70 | 8.12 | 7.20 | 6.99 | 5.41 | 0 |
| pso | 7.93 | 7.05 | 5.95 | 5.84 | 5.02 | 72 |
| **jaya** | 7.89 | 6.97 | **5.72** | **5.59** | **4.69** | 70 |
| mvo | **7.70** | **6.86** | 5.77 | 5.64 | 4.80 | 71 |

- **Swarm tuning decisively beats fixed HPs**: at K<=36 tuned wins on 27-33 of
  33 backbones (every optimizer); at K72 still 23-27/33. The old "svr_embed is
  weak" verdict was substantially a FIXED-HP artifact.
- **JAYA is the best single optimizer** (best cohort mean at K32/36/72; most
  per-backbone wins at K72: 17/33), and the parameter-less one. MVO best in the
  low-K regime (K9/18). PSO trails overall and diverges on some backbones
  (mgazenet 7.10, eva02 7.12, mobilenet_v4 7.51 at K72 -- swarm collapse to a
  sharp fitness minimum that does not transfer).
- **New best svr_embed numbers on the leaders**: atto_binocular 2.96 (mvo) /
  2.98 (pso) vs 4.36 fixed; face_only_mobile_vit 3.02; binocular 3.24. These
  match or beat the previous best-of-any-method podium (~2.9-3.0) using ONLY
  the SVR calibrator.
- Failure cases (tuning worse than fixed at K72): mobilenet_v4, itracker,
  eyes_only_mobile_vit, eyes_only_mobilenet_v4 -- weaker embeddings where the
  train-subject fitness landscape does not transfer to held-out subjects.
- Optimizer agreement is imperfect (per-backbone winners split); a cheap
  ensemble ("run all three, pick by LOO on enrollment") would capture the
  per-backbone best -- same pattern as the LOO method-selection result.
