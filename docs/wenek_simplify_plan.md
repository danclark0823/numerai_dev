# Wenek simplification test — frozen before execution

Compare exactly three interaction models: saved rank-8/penalty-0.01 baseline,
rank 2 with penalty unchanged, and penalty 0.1 with rank unchanged. Keep all
other settings, feature order, training row IDs, ridge out-of-fold residuals,
seeds, and full test eras identical to dc_wenek_1024_v1. Do not tune epochs,
learning rate, feature count, row count, or blending weights. Check the existing
10% and 25% blends for all three variants as secondary comparisons.

Primary question: does either simplification contribute positively beyond ridge
in at least two of the three historical windows? If neither does, shelve this
interaction formulation. A passing screen only supports further research, not
deployment: these periods have been examined repeatedly and actual MMC exists
for just one 30-era window. Report MMC, CORR, ridge contribution, and paired
differences to the original interaction model within each contiguous window.
Use the same 13-era block bootstrap, acknowledging its limited effective sample.

For each model report training residual Pearson correlation/MSE and later-test
residual Pearson correlation/MSE, plus the zero-residual MSE baseline and raw
prediction standard deviation. Training residuals come from out-of-fold ridge
predictions, but the interaction model has fitted these rows, so these are
**in-sample interaction diagnostics**, not out-of-fold interaction performance.
Later test residuals use y minus the strictly historical outer ridge prediction.
A large training/test gap is consistent with overfitting, though it does not
separate overfitting conclusively from changing regimes or residual distributions.

Verify baseline source hashes, feature/row alignment, raw residual labels,
embargo boundaries, saved baseline prediction parity, zero linear coefficients,
serialization and batch parity. Hash production files before/after. Save new
models and all results in a separate directory; do not alter baseline artifacts
or any of the four running models. No automatic bigger run, refit or upload.
