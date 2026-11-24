# Original code, entry points and configuration provenance

## Exact documented environment

- MATLAB **R2024a Update 4**, version **24.1.0.2628055**.
- Statistics and Machine Learning Toolbox **24.1**; Deep Learning Toolbox **24.1**; Fuzzy Logic Toolbox **24.1**; Parallel Computing Toolbox **24.1** (parallel/GPU execution where available).
- Reported primary workstation: 13th-generation Intel Core i5, 16 GB RAM, NVIDIA GeForce RTX 3070. Identical hardware is not essential to logic, but can affect runtime and stochastic results.
- See the public repository's `tools/capture_environment.m` to record the user's installation. Compatibility beyond R2024a has not been established.

## Main ten-fold source entry points in the existing GitHub repository

- `rffp_10fold/run_RFFP_10fold.m` — fixed LSBoost/KNN/AHC/FCM pipeline.
- `rffp_dnn/run_nested_tuned_RFFP_DNN_10fold.m` — tuned DNN evaluation.
- `run_RFFP_Hierarchical_FastKAN_V2_10fold.m` — FastKAN ten-fold driver.
- `rffp_parser/materializeAlignedRFFP.m`, `rffp_10fold/prepareFoldRepresentation.m` — feature materialization.
- `rffp_10fold/makeRouteGroupedFolds.m` — fold builder; archived fold assignment must be used for exact comparison.
- `rffp_10fold/positioningMetrics.m` — summary metrics.

**Important:** The existing v0.1.2 GitHub snapshot does not include the newer compact MLP/ELM and all cross-campaign implementations. Do not claim complete eight-method source availability until those exact experimental files have been reviewed, added and tested in a new versioned source release.

## Settings reported by the manuscript and source

- 10 outer held-out-LogID folds, 118,930 observations, 38 complete logs; saved fold assignment must not be regenerated when seeking numerical identity.
- Representation C: 5,415 radio identities (906 LTE, 4,509 Wi-Fi).
- Shared baseline/DNN materialization: at least 50 training observations per radio; up to 128 LTE and 384 Wi-Fi identities, with detection indicators; missing LTE/WLAN floors −140/−105 dBm.
- LSBoost: 350 regression trees per coordinate, max 256 splits, learning rate 0.05; no systematic outer-fold tuning.
- KNN: K=5 fixed; AHC and FCM: 600 prototypes, 20 clusters, and five-neighbor prototype interpolation; FCM fuzzifier 2.
- Tuned DNN: 10 predefined candidates, grouped inner validation, final fit on outer development set; inspect `makeRFFPDNNCandidates.m` for complete per-candidate settings.
- Compact MLP: two width candidates; ELM: three hidden widths × three ridge values; grouped inner validation. Exact experimental scripts must accompany a complete code release.
- Hierarchical FastKAN: distinct power/count encoding and inner-validation/expert fusion; inspect the FastKAN source and manuscript for layer settings and calibration. It does not share the DNN inner split.
- Cross-campaign: September 2014 Samsung GT-I9305 (32,798 rows, 16 logs); May 2015 Samsung SM-G900F (86,132 rows, 22 logs). Two directions; device, year, routes and environment are confounded.

## Important execution distinction

These entry points refer to the **existing repository**, not executable copies bundled in this aggregate-only supplement. Authorized-data users should clone a pinned, reviewed source release, read each entry point's function signature, supply the permitted MAT representations and saved fold checkpoint in a private directory, and verify dependencies before running. A clean end-to-end reproduction of the new MLP/ELM and cross-campaign pipelines has not been independently demonstrated by this supplement.
