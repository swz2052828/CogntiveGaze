# CLEAN metacompare — cross-backbone (deploy-faithful calib support), cm

base/base_adv = uncalibrated clean base vs clean adv. svr/svr_embed/fc_ft/meta = calibrated (from clean base). meta_adv = calibrated adv (adapter on adv features).
adv_raw = base - base_adv (>0 adv better UNcalib); adv_meta = meta - meta_adv (>0 adv better after meta calib).

## At K=72 (best-K regime)
```
backbone                          base base_adv svr_emb   fc_ft    meta meta_adv adv_raw adv_meta         best@K  cf
--------------------------------------------------------------------------------------------------------------------
face_only_mobile_vit             5.430    4.907   3.029   3.544   3.788    3.799  +0.523   -0.010   svr_embed@18  25
eyes_only_convnextv2_atto_binocular   4.374    5.586   2.971   3.257   4.221    5.136  -1.211   -0.915   svr_embed@72  25
eyes_only_convnextv2_binocular   4.588    5.332   3.245   3.251   4.327    4.767  -0.744   -0.440   svr_embed@72  25
foveal_vit                       4.439    4.428   3.291   3.345   4.011    3.999  +0.012   +0.011   svr_embed@18  25
mobile_vit                       5.115    4.738   3.656   3.326   3.704    3.760  +0.378   -0.056       fc_ft@72  25
eyes_only_mobilenet_v3           5.004    4.927   3.828   4.120   3.837    3.493  +0.077   +0.344    meta_adv@18  25
cnn_transformer                  4.711    4.837   5.599   3.680   4.130    3.843  -0.126   +0.287    svr_embed@9  25
convnextv2_atto                  4.337    4.567   4.118   3.508   4.249    4.339  -0.230   -0.090       fc_ft@36  25
mobilenet_v3                     5.038    4.910   3.526   4.031   3.704    3.625  +0.128   +0.079   svr_embed@72  25
vit                              4.379    4.324   3.940   3.681   4.063    4.165  +0.055   -0.101       fc_ft@36  25
eyes_only_vit                    4.400    4.478   4.671   3.796   4.058    4.431  -0.078   -0.372       fc_ft@72  25
convnext                         4.397    4.603   3.983   3.809   4.341    4.607  -0.206   -0.266       fc_ft@36  25
mobilevitv2                      4.779    4.918   5.801   4.053   3.819    7.683  -0.140   -3.864         meta@9  25
eyes_only_mobile_vit             4.213    4.885   7.328   4.003   3.857    4.803  -0.672   -0.946        meta@36  25
convnextv2                       4.331    4.453   4.570   3.882   4.032    4.164  -0.122   -0.132       fc_ft@72  25
eyes_only_eva02_tiny             4.931    4.815   3.900   4.011   4.947    4.657  +0.116   +0.291   svr_embed@72  25
repvit                           4.667    5.271   4.016   4.314   3.914    4.128  -0.604   -0.215        meta@72  25
convnextv2_dualenc               4.418    4.466   4.765   3.966   4.154    4.082  -0.048   +0.072       fc_ft@18  25
eyes_only_mgazenet               5.114    5.208   5.063   4.429   4.016    4.273  -0.094   -0.257        meta@72  25
eyes_only_mobilenet_v4           4.333    4.401   7.894   4.446   4.227    4.074  -0.068   +0.153    meta_adv@72  25
mobilenet_v4                     4.508    4.367   6.544   4.109   4.574    4.223  +0.141   +0.351       fc_ft@72  25
normface_convnext                4.600    4.965   4.595   4.143   4.598    5.008  -0.365   -0.410       fc_ft@36  25
eyes_only_fastvit                4.464    4.510   6.206   4.659   4.311    4.247  -0.045   +0.063     meta_adv@9  25
eyes_only_convnextv2_atto        4.610    4.643   5.251   4.280   4.502    4.479  -0.033   +0.023       fc_ft@36  25
eyes_only_convnextv2             4.470    4.671   5.874   4.420   4.284    4.408  -0.200   -0.123        meta@72  25
eyes_only_mobilevitv2            5.459    5.596   5.486   4.989   4.938    6.061  -0.137   -1.122        meta@36  25
mgazenet                         4.947    5.232   5.148   4.660   4.716    5.909  -0.285   -1.193       fc_ft@72  25
dinov2                           5.363    5.425   5.873   4.786   5.378    5.539  -0.062   -0.161       fc_ft@36  25
eva02                            5.089    5.182   5.546   4.779   5.149    5.152  -0.093   -0.003       fc_ft@36  25
itracker                         5.930    6.670   6.642   4.824   5.132    6.162  -0.740   -1.030       fc_ft@72  25
convnextv2_film                  5.559    6.205   5.304   5.048   5.482    5.467  -0.647   +0.015   svr_embed@18  25
affnet                           5.791    5.259   5.905   5.308   5.115   69.619  +0.532  -64.505        meta@72  25
convnextv2_nano                  7.358    9.357   7.474   7.109   7.348    9.394  -1.999   -2.046       fc_ft@72  25
--------------------------------------------------------------------------------------------------------------------
COHORT MEAN                    adv_raw=-0.212 (adv better raw 9/33)  adv_meta=-2.320 (adv better meta 11/33)  best_err_mean=4.062
```

## At K=9 (real-world low-K regime)
```
backbone                          base svr_emb   fc_ft    meta meta_adv
----------------------------------------------------------------------
face_only_mobile_vit             5.430   3.273   3.684   4.039    3.850
eyes_only_convnextv2_atto_binocular   4.374   3.288   3.416   4.235    5.150
eyes_only_convnextv2_binocular   4.588   3.476   3.315   4.335    4.866
foveal_vit                       4.439   3.356   3.424   4.014    4.019
mobile_vit                       5.115   4.132   3.477   3.677    3.817
eyes_only_mobilenet_v3           5.004   4.209   4.191   3.751    3.657
cnn_transformer                  4.711   3.491   3.761   4.178    3.907
convnextv2_atto                  4.337   4.429   3.613   4.251    4.362
mobilenet_v3                     5.038   3.826   4.070   3.705    3.700
vit                              4.379   4.169   3.767   4.080    4.170
eyes_only_vit                    4.400   4.877   3.904   4.081    4.433
convnext                         4.397   3.956   3.920   4.341    4.617
mobilevitv2                      4.779   5.852   4.060   3.801    7.695
eyes_only_mobile_vit             4.213   5.358   4.015   3.984    5.008
convnextv2                       4.331   4.567   3.981   4.043    4.164
eyes_only_eva02_tiny             4.931   4.307   4.086   4.952    4.653
repvit                           4.667   4.446   4.650   3.966    4.156
convnextv2_dualenc               4.418   4.482   4.011   4.159    4.086
eyes_only_mgazenet               5.114   5.865   4.420   4.041    4.199
eyes_only_mobilenet_v4           4.333   6.568   4.450   4.232    4.123
mobilenet_v4                     4.508   5.543   4.147   4.600    4.261
normface_convnext                4.600   5.226   4.149   4.595    5.000
eyes_only_fastvit                4.464   5.073   4.675   4.332    4.232
eyes_only_convnextv2_atto        4.610   4.929   4.312   4.505    4.486
eyes_only_convnextv2             4.470   4.975   4.441   4.298    4.423
eyes_only_mobilevitv2            5.459   5.135   5.025   5.233    5.379
mgazenet                         4.947   4.981   4.759   4.761    5.752
dinov2                           5.363   6.210   4.833   5.384    5.557
eva02                            5.089   5.154   4.876   5.143    5.150
itracker                         5.930   7.110   4.886   5.126    7.589
convnextv2_film                  5.559   5.133   5.118   5.492    5.512
affnet                           5.791   5.469   5.296   5.122  106.311
convnextv2_nano                  7.358   7.742   7.149   7.349    9.394
```

# clean metacompare: 33 backbones aggregated
