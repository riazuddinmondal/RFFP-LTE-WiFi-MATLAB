# Cross-campaign source code — review candidate

The six evaluated transfer pipelines are **LSBoost, tuned DNN, compact MLP, ELM, global FastKAN and hierarchical FastKAN**. Global and hierarchical FastKAN are both calculated by `run_RFFP_Hierarchical_FastKAN_crosscampaign.m`. Two directions: direction 1 trains on 2014 and tests on 2015; direction 2 reverses the campaigns.

## MATLAB setup (authorized data only)

Tested research environment reported by authors: MATLAB R2024a Update 4 (24.1.0.2628055). Required toolboxes depend on method: Statistics and Machine Learning, Deep Learning, and optional Parallel Computing (GPU). Fuzzy Logic is used by other baseline algorithms. Exact cross-version compatibility has **not** been established.

From the repository root, in MATLAB:

```matlab
addpath(genpath(pwd));
dataFile = fullfile('PRIVATE_DATA','RFFP_corrected_representations.mat');
foldFile = fullfile('PRIVATE_DATA','corrected_route10fold_checkpoint.mat');
% Run one direction at a time, with a private output directory:
run_RFFP_compact_MLP_crosscampaign(dataFile, 'PRIVATE_RESULTS/MLP', 1);
run_RFFP_ELM_crosscampaign(dataFile, 'PRIVATE_RESULTS/ELM', 1);
run_RFFP_LSBoost_crosscampaign_fixed(dataFile, 'PRIVATE_RESULTS/LSBoost', 1, false);
run_RFFP_tuned_DNN_review_validation(dataFile,foldFile,'crosscampaign',1,'PRIVATE_RESULTS/DNN');
run_RFFP_Hierarchical_FastKAN_crosscampaign(dataFile,1,'OutputDir','PRIVATE_RESULTS/KAN');
```

Inspect each function signature before execution. Some methods generate `.mat` files with predictions, test-row indices, identifiers, and/or trained models; **all output directories must remain private**. Do not publish any `PRIVATE_DATA`, `PRIVATE_RESULTS`, `predictions.mat`, `completed_result.mat`, checkpoint, or model outputs without a separate rights/privacy review. The `run_RFFP_crosscampaign_all.m` file is an original partial orchestration example, **not** a validated all-six-method driver.

The FastKAN cross-campaign implementation includes both global and hierarchical predictions; no separate global-only entry point is required for its archived metrics.

**Review status:** author-side direction-1 ELM and compact MLP execution completed on 24 November 2026; see ../VERIFICATION_REPORT.md. Other transfer runners were statically checked, not freshly retrained.
