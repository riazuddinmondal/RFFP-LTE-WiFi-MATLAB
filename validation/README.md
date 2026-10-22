# Independent Fold 1 validation (restricted-data workflow)

These six MATLAB scripts are the versions supplied after the successful October 2026 Fold 1 checks. They are **not self-contained**: they require separately authorized, restricted data and reference result files. The scripts use the author's Windows staging path `C:\RFFP_Release_Validation`; edit the `root` variable to match your environment.

Run `prepare_*` before the corresponding `validate_*` script. The LSBoost validation was compared with the archived checkpoint; DNN and FastKAN compared archived and newly trained sample-level errors. KNN and AHC summary comparisons and FCM validation were reported during the evaluation but their standalone comparison scripts were not supplied in this upload; do not claim they are included here.

Never place private MAT files, checkpoint files, per-sample predictions or raw fingerprints in a public repository. The original experiment's independent Fold 1 comparisons do not substitute for full ten-fold retraining.
