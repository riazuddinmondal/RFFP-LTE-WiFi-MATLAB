# Authorized-data access and reproduction procedure

1. Contact the corresponding author using the address in the manuscript or repository. Describe your research institution, intended purpose, exact dataset/derivative requested, planned retention, and whether any redistribution is intended.
2. The authors must determine the applicable original data-owner permissions and institutional obligations. They cannot guarantee access, and this package grants no license to restricted third-party measurements.
3. If access is explicitly approved, obtain the permitted source data and processed Representation C / fold checkpoint **through the authorized private channel**, not GitHub or a public issue attachment. Comply with any agreement and retain data in access-controlled storage.
4. Clone the approved, version-pinned MATLAB source repository. In MATLAB R2024a Update 4, add its source directories to the path, capture `ver` and hardware details, and use the private dataset paths when invoking the documented runners.
5. Verify `N=118930`, the 38 complete LogIDs, the 10 archived outer folds, no shared LogID between train and test, and campaign counts 32,798/86,132. Never use test labels to tune a model.
6. Execute original baseline, tuned DNN, FastKAN, and compact MLP/ELM programs with their archived configuration. The cross-campaign runners use two whole-campaign splits rather than the ten-fold partition.
7. Recompute per-fold errors and pooled metrics on private storage. Compare to public aggregate tables. Expect stochastic/runtime variation where training is not bitwise deterministic.
8. Do not publish individual fingerprints, transmitter IDs, reference/predicted coordinates, sample-level errors, model checkpoints, log-to-fold maps or routes. Publish only vetted aggregates after appropriate approval.

The public verification scripts can run without steps 1–7, but they check only author-provided aggregate identities. Independent model-training reproducibility remains conditional on authorized data and complete reviewed source availability.
