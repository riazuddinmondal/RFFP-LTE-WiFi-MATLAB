# Outdoor LTE/Wi-Fi RF Fingerprint Positioning — MATLAB Evaluation Software

MATLAB implementations and evaluation utilities for outdoor LTE/Wi-Fi radio-frequency fingerprint positioning (RFFP). The framework compares LSBoost, KNN, agglomerative hierarchical clustering (AHC), fuzzy c-means (FCM), a tuned deep neural network (DNN), hierarchical FastKAN, compact MLP and ELM using route-grouped evaluation. Six pipelines are also evaluated across campaigns.

**Repository:** https://github.com/riazuddinmondal/RFFP-LTE-WiFi-MATLAB  
**Software version:** 0.1.3  
**License:** MIT (software only)

## Contents

- `rffp_10fold/` — baseline evaluation, fold preparation, and positioning metrics.
- `rffp_dnn/` — DNN architectures, candidate selection, training, and evaluation.
- `rffp_parser/` — aligned fingerprint representation materialization from authorized inputs.
- `FastKANLayer.m` and `run_RFFP_Hierarchical_FastKAN_V2*.m` — FastKAN implementation and single-/ten-fold runners.
- `validation/` — selected independent Fold 1 verification scripts and instructions.
- `rffp_lightweight/` — compact MLP and ELM.
- `rffp_crosscampaign/` — campaign transfer runners.
- `public_verification_v0.1.3/` — aggregate tables and no-data verification scripts.
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

## Responsible Use and Data Protection

This repository contains MATLAB source code and documentation intended for research and reproducibility purposes. The original and processed RF fingerprint datasets, raw measurement logs, reference coordinates, route-fold assignments, sample-level positioning results, trained models, and other restricted research materials must not be committed or publicly distributed without appropriate authorization from the relevant rights holders.

The repository's `.gitignore` provides basic safeguards against accidental inclusion of restricted files; however, users are responsible for reviewing all files before committing or publishing them.

## Software Release and Archiving

The MATLAB evaluation software is maintained in this GitHub repository and archived through Zenodo for long-term preservation and citation.

Version **v0.1.2** restores the complete publicly releasable MATLAB source-code directory structure, including the 10-fold evaluation, DNN, fingerprint parsing, supporting utilities, and validation folders.

The repository provides implementations and evaluation utilities for LSBoost, KNN, AHC, FCM, DNN, and hierarchical FastKAN approaches to outdoor LTE/Wi-Fi radio-frequency fingerprint positioning.

The original and processed RF fingerprint datasets and other restricted research materials are not included. Reproduction of numerical results requires authorized access to the corresponding research data.

**Earlier version DOI (v0.1.2, author-reported):** https://doi.org/10.5281/zenodo.23567257. This DOI does not cover v0.1.3. New version DOI pending deposit.


## Supplementary source and aggregate verification (v0.1.3)

Additional original MATLAB entry points for compact MLP/ELM ten-fold and bidirectional cross-campaign experiments are included in `rffp_lightweight/` and `rffp_crosscampaign/`. Aggregate-only public verification is in `public_verification_v0.1.3/`. See `SOURCE_REVIEW_v0.1.3.md` for audit scope and outstanding checks. **The original v0.1.2 archival DOI does not cover these additions. The v0.1.3 DOI will be recorded after deposit.** No restricted input data or saved prediction arrays are distributed.

## Verification on 24 November 2026

See [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) for author-side MATLAB tests, including the ELM match and MLP numerical variation. Static dependency analysis additionally detected System Identification Toolbox for several neural-network runners. All detected products were installed; this is not a minimal-toolbox compatibility test.

For no-data checks, enter `public_verification_v0.1.3/` and run `verify_public_aggregates` in MATLAB or `python verify_public_aggregates.py` in Python. Keep authorized inputs and all generated model outputs outside the repository. The misleadingly named `tools/prepare_public_data.m` was excluded because it retains restricted data. `tools/check_release.m` requires private inputs and is not a public smoke test.
