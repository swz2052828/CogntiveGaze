# ADV clean-retrain (Step-2 for subject-adversarial models), cm

adv_dirty/cl = existing adv on clean val; adv_clean/cl = adv retrained on clean.
adv_gain = adv_dirty/cl - adv_clean/cl (>0 = cleaning train helped adv).
adv_vs_base = base_clean/cl - adv_clean/cl (>0 = adv beats base, both clean-trained).

```
backbone                         adv_dirty/cl adv_clean/cl  adv_gain base_clean/cl adv_vs_base  cf
--------------------------------------------------------------------------------------------------
vit                                    4.4620       4.3104   +0.1516        4.3695     +0.0590   5
mobilenet_v4                           4.1861       4.3561   -0.1700        4.5033     +0.1472   5
eyes_only_mobilenet_v4                 4.2423       4.4093   -0.1670        4.3258     -0.0835   5
foveal_vit                             4.4731       4.4113   +0.0618        4.4236     +0.0123   5
convnextv2                             4.4728       4.4654   +0.0074        4.3211     -0.1443   5
eyes_only_vit                          4.4307       4.4674   -0.0367        4.3852     -0.0822   5
convnextv2_dualenc                     3.8343       4.4758   -0.6415        4.4025     -0.0733   5
eyes_only_fastvit                      4.7319       4.5376   +0.1943        4.4723     -0.0653   5
convnextv2_atto                        4.5283       4.5661   -0.0378        4.3202     -0.2459   5
convnext                               4.2244       4.5888   -0.3644        4.3858     -0.2030   5
eyes_only_convnextv2_atto              4.5031       4.6579   -0.1548        4.6008     -0.0571   5
eyes_only_convnextv2                   4.5443       4.6643   -0.1200        4.4809     -0.1834   5
mobile_vit                             4.6309       4.7316   -0.1007        5.1161     +0.3845   5
eyes_only_eva02_tiny                   4.7921       4.8164   -0.0243        4.9198     +0.1034   5
cnn_transformer                        5.0150       4.8342   +0.1808        4.7042     -0.1300   5
eyes_only_mobile_vit                   4.7025       4.8880   -0.1855        4.2046     -0.6835   5
face_only_mobile_vit                   5.2520       4.9055   +0.3465        5.4216     +0.5161   5
mobilenet_v3                           4.8389       4.9321   -0.0932        5.0432     +0.1111   5
mobilevitv2                            4.7464       4.9323   -0.1859        4.7909     -0.1414   5
eyes_only_mobilenet_v3                 4.8927       4.9345   -0.0417        4.9917     +0.0572   5
normface_convnext                      4.7992       4.9533   -0.1541        4.5884     -0.3650   5
eva02                                  5.6316       5.1702   +0.4614        5.0855     -0.0847   5
mgazenet                               5.1106       5.1821   -0.0715        4.9629     -0.2193   5
eyes_only_mgazenet                     5.0001       5.2284   -0.2283        5.1068     -0.1217   5
affnet                                 5.5501       5.2474   +0.3026        5.7923     +0.5448   5
repvit                                 5.0288       5.2672   -0.2384        4.6543     -0.6129   5
eyes_only_convnextv2_binocular         4.5001       5.3453   -0.8451        4.5868     -0.7584   5
dinov2                                 5.9280       5.4249   +0.5031        5.3577     -0.0672   5
eyes_only_convnextv2_atto_binocular       4.5190       5.5732   -1.0542        4.3481     -1.2251   5
eyes_only_mobilevitv2                  5.5270       5.5812   -0.0542        5.4710     -0.1102   5
convnextv2_film                        5.8799       6.2014   -0.3215        5.5411     -0.6602   5
itracker                               6.4720       6.6776   -0.2056        5.9217     -0.7559   5
convnextv2_nano                        9.3472       9.3474   -0.0002        7.3611     -1.9863   5
--------------------------------------------------------------------------------------------------
COHORT MEAN (done)                     4.9939       5.0935   -0.0996
# adv clean-retrain done: 33 backbones; adv beats base (clean-trained) on 9/33
```
