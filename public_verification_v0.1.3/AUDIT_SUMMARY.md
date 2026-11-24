# Author-side audit and public-check scope

**Original six methods:** The author-side numerical extraction used the restricted `out_of_fold_results.csv.gz` from the supplied results archive. Each method has 118,930 finite positioning errors. Summing error and squared error by saved fold reproduces the archive's pooled mean and RMSE to floating-point precision. Threshold counts reconstruct the percentages of Table 11 (10/25/50/100 m). The private log-to-fold mapping has 38 unique logs, none assigned to more than one fold.

**Compact MLP and ELM:** The supplied `Lightweight_10Fold_v01` archive includes ten fold summaries and pooled summaries. The public fold sufficient sums for these two methods are computed from the archived fold means and RMSEs; they are not a fresh re-extraction of individual errors. This permits pooled mean/RMSE reconciliation, not independent percentile reconstruction.

**Cross-campaign:** The 12-row CSV carries six methods in both transfer directions and is taken from the previously prepared author-side audit. Its public presence does not constitute an independent rerun.

**Independent public verification:** `verify_public_aggregates.py` and `.m` recompute pooled mean and RMSE from public sufficient statistics and validate threshold-count ordering and fold-composition arithmetic. The public scripts do not verify that the author-generated sufficient statistics correspond to restricted raw observations. They do not recompute exact median/P68/P95, or demonstrate independent model training.

**Scientific positioning:** This is a documented comparative study with a privacy-preserving audit supplement and **limited public reproducibility**, not a fully open-data independently rerunnable benchmark. The author-side validation and public arithmetic checks must be described separately.
