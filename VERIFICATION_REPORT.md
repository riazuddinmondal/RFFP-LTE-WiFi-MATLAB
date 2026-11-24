# Release verification — 24 November 2026

Status: v0.1.3; author confirmed public release of supplementary software and aggregate tables. Evidence below is from author-provided MATLAB output, not an independent third-party replication.

## Environment and scope

MATLAB R2024a 24.1.0.2537033, Windows 11 Pro; installed toolboxes 24.1. Original experimental environment reported as R2024a Update 4 (24.1.0.2628055). No license number, local paths, private predictions or model files are included.

- MATLAB and Python public aggregate checks: eight methods, 118,930 observations each; pooled mean and RMSE match to displayed six decimals.
- Source availability: 12 selected files resolved within the isolated candidate directory, including one FastKANLayer copy.
- Code Analyzer: 13 messages (NASGU 1, AGROW 5, MSNU 1, MEXCEP 1, ISCL 3, DEFNU 2); no explicit syntax errors reported. Original algorithms retained.
- Static dependency audit: seven runners, zero detected external non-MathWorks files; detected products MATLAB, Deep Learning, System Identification, Statistics and Machine Learning, Parallel Computing. Static analysis cannot establish dynamic dependency completeness.

## Authorized-data execution: 2014 training / 2015 testing

32,798 training and 86,132 test observations. Both runs completed with 100% coverage in fresh private output directories.

| Method | Archived mean (m) | Rerun mean (m) | Archived P95 (m) | Rerun P95 (m) | Wall time (s) |
|---|---:|---:|---:|---:|---:|
| ELM | 107.887515 | 107.888 | 226.226476 | 226.226 | 4.19 |
| Compact MLP | 54.141229 | 55.0473 | 142.009646 | 144.9760 | 85.95 |

ELM matched the archived values at reported precision; selected 1024 hidden neurons and ridge lambda 1 from nine inner-validation candidates. Inner fit/validation counts 26,280/6,518; three validation logs. Rerun median 97.705, RMSE 125.163, P68 128.408 m.

MLP completed both candidates (validation means 35.164 and 32.091 m); rerun median 40.0012, RMSE 80.8083, P68 55.0721 m. Mean increased approximately 1.67% and P95 2.09%. Cause not established; stochastic training and environment differences are plausible. Original archived benchmark values are retained, not replaced with rerun values.

## Limitations

No fresh full ten-fold retraining or reverse-direction rerun was performed for this release. Public aggregate checks cannot verify original predictions, exact percentiles or private log assignments. The two model reruns used authorized data and are not publicly reproducible without it. The final packaging changes are documentation, metadata, exclusion of a sensitive export utility and removal of extrema columns; the Python verifier was rerun on the final tables. MATLAB aggregate checks subsequently passed for all eight methods on the final package and the downloaded GitHub snapshot. All 67 entries in the pre-finalization GitHub checksum manifest matched. No training code was changed.

The author confirmed public release of supplementary software and aggregate tables on 24 November 2026. Restricted inputs remain subject to separate authorization. Final manuscript reconciliation and archival DOI confirmation remain outstanding.
