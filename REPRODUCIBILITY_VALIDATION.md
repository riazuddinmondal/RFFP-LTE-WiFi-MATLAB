# Reproducibility verification — October 2026

**Scope:** Representation C, route-grouped outer Fold 1; 12,651 test fingerprints. Independent execution was reported for LSBoost, KNN, AHC, FCM, tuned DNN and hierarchical FastKAN. The archived ten-fold aggregate metrics were also reconstructed from private saved outputs. This is not a public-data reproduction and not a fresh ten-fold retraining.

| Method | Fold 1 result | Comparison granularity |
|---|---|---|
| LSBoost | Passed | Six metrics and 12,651 sample-level errors |
| KNN | Passed | Six summary metrics |
| AHC | Passed | Six summary metrics |
| FCM | Passed | Six metrics and 12,651 sample-level errors |
| Tuned DNN | Passed | Six metrics and 12,651 sample-level errors |
| Hierarchical FastKAN V2 | Passed | Six metrics and 12,651 sample-level errors |

## Independently executed DNN Fold 1

Ten candidate configurations were evaluated; `StrongReg` selected (selection score 29.99), followed by 67 epochs of final GPU training. Original/reproduced errors (m): mean 21.910, median 17.348, RMSE 30.208, P68 23.969, P95 52.463, maximum 464.060. Reported maximum and mean absolute differences between 12,651 individual errors: 0.000000000 m.

## Independently executed hierarchical FastKAN V2 Fold 1

GPU: NVIDIA GeForce RTX 3070; seed 17; 256 global radio features; 15/15 local experts accepted. Runtime 858.82 s. Original/reproduced errors (m): mean 22.225, median 18.007, RMSE 28.511, P68 24.604, P95 52.359, maximum 273.770. Reported maximum and mean absolute differences between 12,651 individual errors: 0.000000000 m.

## Evidence limitations

The private MATLAB result MAT files and detailed validation reports were examined as archive entries but are deliberately **not distributed**. These metrics were reported by the validation scripts and user; MATLAB was not executed during public packaging. To substantiate independence, maintain archived original-file provenance, creation dates and checksums in a restricted audit folder. Reproduction by an external researcher requires separately authorized data access.
