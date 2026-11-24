# Lightweight MLP and ELM ten-fold source

`run_RFFP_lightweight_MLP_ELM_10fold_v01.m` is the saved ten-fold entry point for compact MLP and ELM. `run_RFFP_lightweight_MLP_ELM_fold1_v03.m` is a separate Fold 1 development/verification script, **not** the ten-fold driver.

Requires authorized `RFFP_corrected_representations.mat` and `corrected_route10fold_checkpoint.mat`, and repository helpers `prepareFoldRepresentation.m`, `materializeAlignedRFFP.m`, `makeRFFPInnerValidation.m`. From repository root in MATLAB R2024a:

```matlab
addpath(genpath(pwd));
report = run_RFFP_lightweight_MLP_ELM_10fold_v01( ...
 'PRIVATE_DATA/RFFP_corrected_representations.mat', ...
 'PRIVATE_DATA/corrected_route10fold_checkpoint.mat', ...
 'PRIVATE_RESULTS/Lightweight_10Fold_v01');
```

All saved fold MAT files, combined MAT files, and any sample-level outputs must remain private. Do not regenerate folds when trying to reproduce the published partition. Source included without modifications to experiment settings. Ten-fold MLP/ELM execution has not been repeated for this snapshot; see the root verification report for limited cross-campaign execution.
