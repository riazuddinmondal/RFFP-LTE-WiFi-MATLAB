# Release notes

## v0.1.2 — Complete MATLAB Source-Code Archive

### Changes
- Restored the MATLAB source-code folders omitted from the earlier GitHub release.
- Included the publicly releasable source files for route-grouped 10-fold evaluation, DNN positioning, fingerprint parsing, supporting utilities, and validation.
- Updated release documentation and citation metadata.
- Prepared the complete software snapshot for permanent archival through Zenodo.

### Reproducibility
This release provides the public MATLAB evaluation framework for the outdoor LTE/Wi-Fi RF fingerprint positioning study.

The release does not include restricted RF fingerprint datasets, reference coordinates, trained models, or private research outputs. Reproducing the published numerical results requires authorized access to the relevant datasets.

### Previous Release
Version v0.1.1 was archived on Zenodo but did not contain all intended MATLAB source-code folders. Version v0.1.2 supersedes that incomplete software snapshot for reproducibility purposes.

## v0.1.0 — Initial public software release

- Published MATLAB source for outdoor LTE/Wi-Fi RF fingerprint positioning experiments: LSBoost, KNN, AHC, FCM, tuned DNN, and hierarchical FastKAN.
- Included route-grouped evaluation utilities, environment capture, and selected Fold 1 verification scripts.
- Documented the MATLAB R2024a Update 4 experimental environment and the limits of independent reproduction checks.
- Distributed the original software under the MIT License (copyright 2026 Riaz Uddin Mondal).
- Excluded restricted fingerprint datasets, route-fold assignments, per-sample predictions, and model checkpoints.

The public release is **source code only** and does not establish a public dataset DOI or a Zenodo DOI. See `DATA_ACCESS_POLICY.md` and `REPRODUCIBILITY_VALIDATION.md`.
