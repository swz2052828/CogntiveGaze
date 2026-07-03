# Temporal smoothing at inference (opt #5), cm

Per-recording mean error, averaged over recordings+folds. 'all' includes blink
frames; 'clean' = blink-free frames only. med=centered median (offline),
ema=causal EMA (real-time), blinkI=linear interp over blink frames.

## convnext
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 4.4846      4.4029      3.8742      3.7808      4.4446      4.3598
med3                4.4900      4.4022      3.8740      3.7772      4.4469      4.3579
med5                4.5013      4.4053      3.8824      3.7788      4.4564      4.3600
med9                4.5197      4.4189      3.8974      3.7896      4.4735      4.3731
med15               4.5957      4.4934      3.9728      3.8637      4.5470      4.4451
ema0.3              4.9079      4.8126      4.3063      4.2006      4.8543      4.7572
ema0.5              4.6342      4.5462      4.0218      3.9220      4.5862      4.4959
ema0.7              4.5299      4.4472      3.9151      3.8204      4.4854      4.4002
blinkI              4.5198      4.4029      3.9031      3.7808      4.4757      4.3598
blinkI+med5         4.5221      4.4053      3.9011      3.7787      4.4757      4.3599
# base: best=raw gain=+0.0000 (vs raw, all frames)
# fcft: best=med3 gain=+0.0002 (vs raw, all frames)
# meta: best=raw gain=+0.0000 (vs raw, all frames)
```

## convnextv2
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 4.4308      4.3369      3.9857      3.8814      4.1355      4.0460
med3                4.4351      4.3355      3.9838      3.8767      4.1376      4.0435
med5                4.4440      4.3382      3.9896      3.8772      4.1476      4.0457
med9                4.4596      4.3501      4.0009      3.8854      4.1641      4.0577
med15               4.5330      4.4219      4.0724      3.9554      4.2419      4.1332
ema0.3              4.8375      4.7310      4.3962      4.2819      4.5581      4.4558
ema0.5              4.5746      4.4742      4.1246      4.0154      4.2838      4.1881
ema0.7              4.4741      4.3784      4.0228      3.9179      4.1790      4.0887
blinkI              4.4604      4.3369      4.0104      3.8814      4.1679      4.0460
blinkI+med5         4.4613      4.3378      4.0056      3.8765      4.1675      4.0455
# base: best=raw gain=+0.0000 (vs raw, all frames)
# fcft: best=med3 gain=+0.0019 (vs raw, all frames)
# meta: best=raw gain=+0.0000 (vs raw, all frames)
```

## eyes_only_convnextv2_atto_binocular
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 4.4541      4.3646      3.5938      3.4947      4.2862      4.1953
med3                4.4569      4.3630      3.5914      3.4894      4.2892      4.1930
med5                4.4658      4.3656      3.5976      3.4902      4.2982      4.1951
med9                4.4839      4.3789      3.6131      3.5011      4.3147      4.2072
med15               4.5616      4.4546      3.6980      3.5839      4.3936      4.2838
ema0.3              4.8793      4.7782      4.0719      3.9615      4.7209      4.6181
ema0.5              4.6066      4.5106      3.7623      3.6569      4.4404      4.3434
ema0.7              4.5017      4.4100      3.6440      3.5430      4.3329      4.2406
blinkI              4.4846      4.3646      3.6223      3.4947      4.3181      4.1953
blinkI+med5         4.4853      4.3654      3.6176      3.4899      4.3178      4.1950
# base: best=raw gain=+0.0000 (vs raw, all frames)
# fcft: best=med3 gain=+0.0024 (vs raw, all frames)
# meta: best=raw gain=+0.0000 (vs raw, all frames)
```

## eyes_only_convnextv2_binocular
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 4.6395      4.5515      3.6964      3.5948      4.3747      4.2851
med3                4.6392      4.5495      3.6913      3.5893      4.3756      4.2827
med5                4.6462      4.5522      3.6955      3.5899      4.3840      4.2850
med9                4.6636      4.5652      3.7094      3.6008      4.4019      4.2982
med15               4.7410      4.6401      3.7938      3.6832      4.4809      4.3749
ema0.3              5.0522      4.9518      4.1605      4.0485      4.7988      4.6963
ema0.5              4.7876      4.6916      3.8603      3.7516      4.5253      4.4280
ema0.7              4.6857      4.5945      3.7451      3.6408      4.4204      4.3282
blinkI              4.6640      4.5515      3.7177      3.5948      4.4029      4.2851
blinkI+med5         4.6644      4.5518      3.7121      3.5891      4.4027      4.2847
# base: best=med3 gain=+0.0003 (vs raw, all frames)
# fcft: best=med3 gain=+0.0051 (vs raw, all frames)
# meta: best=raw gain=+0.0000 (vs raw, all frames)
```

## face_only_mobile_vit
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 5.4925      5.4015      4.4607      4.3590      3.9519      3.8373
med3                5.4857      5.3954      4.4513      4.3500      3.9300      3.8186
med5                5.4897      5.3957      4.4547      4.3497      3.9285      3.8140
med9                5.5038      5.4061      4.4710      4.3622      3.9374      3.8200
med15               5.5692      5.4695      4.5495      4.4387      4.0150      3.8967
ema0.3              5.8417      5.7420      4.8774      4.7671      4.3527      4.2345
ema0.5              5.6094      5.5142      4.6025      4.4962      4.0788      3.9624
ema0.7              5.5242      5.4323      4.5002      4.3973      3.9800      3.8660
blinkI              5.5121      5.4015      4.4814      4.3590      3.9651      3.8373
blinkI+med5         5.5058      5.3951      4.4719      4.3492      3.9417      3.8136
# base: best=med3 gain=+0.0067 (vs raw, all frames)
# fcft: best=med3 gain=+0.0094 (vs raw, all frames)
# meta: best=med5 gain=+0.0233 (vs raw, all frames)
```

## mobile_vit
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 5.2324      5.1604      3.4752      3.3625      3.8407      3.7161
med3                5.2407      5.1627      3.4649      3.3519      3.8209      3.7018
med5                5.2536      5.1689      3.4677      3.3504      3.8158      3.6974
med9                5.2780      5.1893      3.4800      3.3604      3.8229      3.7052
med15               5.3598      5.2696      3.5649      3.4445      3.8987      3.7815
ema0.3              5.6129      5.5266      3.9353      3.8133      4.2544      4.1302
ema0.5              5.3797      5.2980      3.6311      3.5122      3.9731      3.8496
ema0.7              5.2861      5.2094      3.5180      3.4030      3.8714      3.7489
blinkI              5.2611      5.1604      3.4947      3.3625      3.8463      3.7161
blinkI+med5         5.2699      5.1694      3.4826      3.3501      3.8268      3.6962
# base: best=raw gain=+0.0000 (vs raw, all frames)
# fcft: best=med3 gain=+0.0103 (vs raw, all frames)
# meta: best=med5 gain=+0.0249 (vs raw, all frames)
```

## mobilenet_v3
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 5.1406      5.0510      4.6216      4.5288      3.7758      3.6769
med3                5.1383      5.0449      4.6179      4.5214      3.7692      3.6665
med5                5.1471      5.0465      4.6263      4.5226      3.7760      3.6663
med9                5.1620      5.0583      4.6419      4.5351      3.7909      3.6779
med15               5.2315      5.1267      4.7175      4.6096      3.8733      3.7589
ema0.3              5.5081      5.4088      5.0223      4.9198      4.2166      4.1076
ema0.5              5.2658      5.1715      4.7587      4.6612      3.9255      3.8219
ema0.7              5.1763      5.0857      4.6605      4.5669      3.8168      3.7174
blinkI              5.1670      5.0510      4.6483      4.5288      3.8046      3.6769
blinkI+med5         5.1621      5.0460      4.6418      4.5221      3.7940      3.6661
# base: best=med3 gain=+0.0023 (vs raw, all frames)
# fcft: best=med3 gain=+0.0037 (vs raw, all frames)
# meta: best=med3 gain=+0.0067 (vs raw, all frames)
```

## vit
```
filter           base(all)    base(cl)   fcft(all)    fcft(cl)   meta(all)    meta(cl)
raw                 4.5017      4.4145      3.8107      3.7081      4.1706      4.0793
med3                4.5069      4.4139      3.8093      3.7039      4.1750      4.0779
med5                4.5157      4.4171      3.8142      3.7055      4.1838      4.0808
med9                4.5331      4.4307      3.8269      3.7172      4.2000      4.0939
med15               4.6106      4.5064      3.9096      3.7991      4.2810      4.1734
ema0.3              4.9198      4.8192      4.2712      4.1590      4.6125      4.5074
ema0.5              4.6517      4.5564      3.9736      3.8647      4.3281      4.2286
ema0.7              4.5487      4.4583      3.8593      3.7543      4.2188      4.1246
blinkI              4.5327      4.4145      3.8320      3.7081      4.2012      4.0793
blinkI+med5         4.5350      4.4168      3.8285      3.7045      4.2023      4.0804
# base: best=raw gain=+0.0000 (vs raw, all frames)
# fcft: best=med3 gain=+0.0013 (vs raw, all frames)
# meta: best=raw gain=+0.0000 (vs raw, all frames)
```

