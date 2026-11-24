# Supplementary source inclusion and static review

**Status: release candidate; not approved for public release.** Original v0.1.2 repository files retained, with supplementary MATLAB sources added without changing training logic. `CITATION.cff` schema version corrected to `1.2.0` (software version updated to 0.1.3). Public aggregate verification files are under `public_verification_v0.1.3/`.

## Added MATLAB sources

- Ten-fold MLP/ELM: `rffp_lightweight/run_RFFP_lightweight_MLP_ELM_10fold_v01.m`
- Fold 1 MLP/ELM development: `rffp_lightweight/run_RFFP_lightweight_MLP_ELM_fold1_v03.m`
- Campaign transfer: `rffp_crosscampaign/run_RFFP_compact_MLP_crosscampaign.m`, `run_RFFP_ELM_crosscampaign.m`, `run_RFFP_LSBoost_crosscampaign_fixed.m`, `run_RFFP_tuned_DNN_review_validation.m`, `run_RFFP_Hierarchical_FastKAN_crosscampaign.m`
- Historical partial orchestration: `rffp_crosscampaign/run_RFFP_crosscampaign_all.m`

## Checks and remaining risks

- Verified the selected MATLAB files have named entry points and use expected repository helper functions; no scientific algorithms or hyperparameters were changed.
- Verified the new source-only package contains no raw MAT datasets, original per-sample results, or coordinate CSVs. **Static inspection cannot guarantee all source comments or paths are disclosure-approved.**
- The source scripts generate restricted outputs when run, including `predictions.mat` and checkpoint files. Their output must not be committed.
- The original `tools/prepare_public_data.m` was excluded from this snapshot because it retains fingerprints, identifiers and coordinates. No experimental runner was modified.
- The original `tools/check_release.m` requires private data; it is not a public no-data smoke test.
- MATLAB execution was reported by the author on 24 November 2026: aggregate checks and two direction-1 transfer runners. See VERIFICATION_REPORT.md. No MATLAB runtime is available in the packaging environment. Coauthor/data-owner review remains required.
- Verify all numerical and source correspondence to the submitted manuscript, check license/rights for new code, and create an actual GitHub release/Zenodo deposit before describing it as published.
