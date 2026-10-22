# Outdoor LTE/Wi-Fi RF Fingerprint Positioning — MATLAB Evaluation Software

MATLAB implementations and evaluation utilities for outdoor LTE/Wi-Fi radio-frequency fingerprint positioning (RFFP). The framework compares LSBoost, KNN, agglomerative hierarchical clustering (AHC), fuzzy c-means (FCM), a tuned deep neural network (DNN), and hierarchical FastKAN using route-grouped evaluation.

**Repository:** https://github.com/riazuddinmondal/RFFP-LTE-WiFi-MATLAB  
**Software version:** 0.1.0  
**License:** MIT (software only)

## Contents

- `rffp_10fold/` — baseline evaluation, fold preparation, and positioning metrics.
- `rffp_dnn/` — DNN architectures, candidate selection, training, and evaluation.
- `rffp_parser/` — aligned fingerprint representation materialization from authorized inputs.
- `FastKANLayer.m` and `run_RFFP_Hierarchical_FastKAN_V2*.m` — FastKAN implementation and single-/ten-fold runners.
- `validation/` — selected independent Fold 1 verification scripts and instructions.
- `tools/` — environment capture, release checks, and saved-output analysis utilities.

## Dependencies

The RFFP evaluation framework was developed and evaluated using **MATLAB R2024a Update 4 (Version 24.1.0.2628055)**. The installed MathWorks toolboxes used in the experiments were from **Release R2024a (Version 24.1)**.

The framework uses the following toolboxes:

- **Statistics and Machine Learning Toolbox:** LSBoost regression, nearest-neighbor positioning, clustering-related operations, and other statistical learning functions.
- **Deep Learning Toolbox:** Implementation, training, and evaluation of the DNN and hierarchical FastKAN models.
- **Fuzzy Logic Toolbox:** Fuzzy C-Means (FCM) clustering.
- **Parallel Computing Toolbox:** Parallel execution of LSBoost training and GPU-accelerated DNN and FastKAN experiments.

The computational experiments were performed using MATLAB on a workstation equipped with a 13th-generation Intel Core i5 processor, 16 GB RAM, and an NVIDIA GeForce RTX 3070 GPU.

**Software compatibility:** MATLAB R2024a Update 4 is the verified experimental environment. Compatibility with earlier or later MATLAB releases has not been systematically evaluated. Researchers seeking to reproduce the reported results are encouraged to use the same MATLAB and toolbox versions.

The utility script `tools/capture_environment.m` can be used to document the MATLAB installation, toolbox versions, and computing hardware on another system.

## Data access

**The original and processed LTE/Wi-Fi fingerprint datasets are not included.** The public software does not contain individual coordinates, sample-level predictions, restricted fold assignments, or trained model checkpoints. Access to the underlying research data requires separate authorization and is not guaranteed. See [DATA_ACCESS_POLICY.md](DATA_ACCESS_POLICY.md).

The code cannot independently reproduce the published numerical results without the required restricted inputs. Users with authorized access should keep those inputs **outside the public Git working tree**, add the source folders to the MATLAB path, and consult the relevant runner documentation and configuration.

## Reproducibility status

All six methods were independently executed for route-grouped outer Fold 1 (Representation C; 12,651 test samples), with agreement against archived summary metrics. Sample-level positioning-error comparisons were also completed for LSBoost, FCM, tuned DNN, and hierarchical FastKAN. Archived aggregate results covering all ten folds were reconstructed from private saved outputs. **The remaining nine folds have not been independently retrained as part of this release validation.** See [REPRODUCIBILITY_VALIDATION.md](REPRODUCIBILITY_VALIDATION.md).

## License and citation

Copyright (c) 2026 Riaz Uddin Mondal. The MATLAB software is distributed under the [MIT License](LICENSE). The license does **not** apply to third-party dependencies or restricted research data. Citation metadata are in [CITATION.cff](CITATION.cff). A persistent DOI may be added after archival publication.

## Responsible use

Do not commit private MAT files, raw measurement logs, sample-level outputs, reference coordinates, or restricted research archives to this repository. The repository's `.gitignore` offers basic safeguards but is not a substitute for reviewing files before publication.
