# RFFP public benchmark and cross-campaign verification supplement

**Version:** v0.1.3   
**Main source repository:** https://github.com/riazuddinmondal/RFFP-LTE-WiFi-MATLAB  
**Existing archival record (earlier release):** https://doi.org/10.5281/zenodo.23567257 (verify version and contents before citing this supplement).

## Scope

The principal benchmark comprises **eight methods** (LSBoost, KNN, AHC, FCM, tuned DNN, hierarchical FastKAN, compact MLP, ELM), 118,930 observations, 38 measurement logs and ten held-out-log folds. The separate **cross-campaign** evaluation includes six methods (LSBoost, tuned DNN, compact MLP, ELM, global FastKAN and hierarchical FastKAN), each in both 2014→2015 and 2015→2014 directions. The six cross-campaign methods must not be confused with the eight-method ten-fold benchmark.

## Public tables

- `tenfold_pooled_metrics.csv`: archived pooled eight-method metrics, including median/P68/P95 (not independently recoverable from public summaries).
- `tenfold_fold_sufficient_statistics.csv`: 80 fold-method records; sample counts, means, RMSEs and sufficient error sums. For six original methods sums were freshly computed from the restricted archived sample errors; MLP/ELM sums were derived from archived fold mean/RMSE with rounding.
- `tenfold_threshold_counts_by_fold.csv`: counts at 10, 25, 50 and 100 m for the original six methods, reconstructed from restricted saved error arrays (Table 11).
- `tenfold_coarse_error_histogram.csv`: broad error bins for the original six methods; bins containing fewer than 10 observations are suppressed. These bins do **not** reproduce exact pooled quantiles.
- `fold_campaign_device_composition.csv`: fold-level aggregate campaign/device counts (Table 12), including total row.
- `fold_integrity_aggregate.csv`: aggregate assertions calculated by the authors from restricted fold assignments. The public table verifies its arithmetic, **not the underlying mapping**.
- `crosscampaign_aggregate_metrics.csv`: 12 author-audited method-direction rows from the earlier cross-campaign supplement.

## Run the public checks

Python 3.8+ standard library only: `python verify_public_aggregates.py`  
MATLAB R2024a: `verify_public_aggregates(pwd)` when the current directory is this supplement folder.

These scripts **actually recalculate** pooled mean error and pooled RMSE from fold-level sufficient statistics and verify the threshold-count ordering and fold-composition arithmetic. This is stronger than merely checking positive CSV fields, but is **not** independent confirmation that the source measurements or predictions are correct. The inputs are author-generated aggregates.

**Exact pooled median, P68, and P95 cannot be reconstructed** from fold means/RMSEs, fold percentiles, or the coarse histogram. They remain archived author-reported results. A histogram only bounds a quantile to its bin, and small-bin suppression further limits that exercise. Public aggregate consistency is not full independent experimental reproduction.

## Data restriction

No sample-level fingerprints, transmitter identities, geographic coordinates, log-to-fold maps, individual prediction coordinates, individual errors, or trained models are included. Private inputs and all original MAT archives must remain outside public GitHub/Zenodo releases. The author confirmed public release of the supplementary software and aggregate tables on 24 November 2026.

## Reproducing with authorized data

See `AUTHORIZED_DATA_REPRODUCTION.md` and `SOURCE_ENTRYPOINTS_AND_CONFIG.md`. Data access is subject to permission from the relevant owners; approval is not guaranteed. This package alone cannot rerun model fitting.

## Status

This package does not itself establish that a tagged GitHub release or new Zenodo version exists. Python checks passed in the packaging environment and MATLAB checks passed on the author’s R2024a installation. See ../VERIFICATION_REPORT.md. Fold and pooled extrema columns were omitted to minimize disclosure of individual error values.
