# dc_alscot — frozen first experiment

Named for Alscot, the mercenary warmage listed at
https://spellmongershall.com/characters. Research only; no live model changes.

Hypothesis: jointly penalizing variation in feature/target covariance across
historical blocks can retain more stable signal than ordinary ridge. This is
a covariance-stability proxy, NOT direct minimization of worst-period loss or
optimization of MMC. It can suppress genuine changing signals or fit sampling noise.

For each training era compute centered feature/target covariance, average these
within fixed 52-calendar-era blocks, and form their population covariance R.
Normalize R to trace / feature count = 1 (unless exactly zero). Fit centered
least squares with penalty alpha*I + lambda*R, alpha=10. Lambda=0 must reproduce
ordinary ridge. Features remain in their original five-bin scale. Each era has
equal weight in R; the squared-error part retains the matched ridge row weights.

Freeze before evaluation: lambda grid [0, 1, 10, 100]; choose separately in each
outer fold by highest worst inner-window mean official CORR, then highest mean
CORR, then smaller lambda. Use three 52-calendar-era inner validation windows
immediately before the outer embargo, separated by 13-era gaps; each fit excludes
13 eras before its validation window. Inner validation uses only the existing
1,024-row/every-fourth-era sample (roughly 13 sampled eras/window), not full eras.
Record all inner scores; outer labels cannot affect selection. Refit the selected
setting on eligible outer training rows. Zero is an explicit ridge fallback.

Reuse Wenek's exact 128 features, target_ender_60, seed 42 and training sample.
Evaluate all rows of eras 850–879, 1000–1029, 1133–1162 with 13-era embargo.
Check reference row IDs, labels, eras and matched ridge predictions. Historical
52-era holdout remains excluded. These development windows have been inspected
repeatedly; actual MMC exists only for the last 30 eras. No fresh confirmation.

Screen only the inner-selected pipeline, not individual grid members: require
positive contribution beyond ridge in at least two windows, CORR gains in at
least two windows and pooled CORR, nonnegative MMC no worse than ridge wherever
available, plus improved worst-window CORR. Passing permits a separately planned
replication, not deployment. Failure stops this formulation without retuning.

Tests: lambda-zero ridge parity; positive-semidefinite penalty; covariance
reconstruction; shrinkage of a synthetic reversing signal; batch/serialization
parity; invalid-input rejection; future-label isolation in nested selection.
Real-run checks include chronological boundaries, selected grid membership,
full-era submission ranks/range, nonconstant finite predictions, and protected
production-file hashes before/after. Save source snapshots, config, training IDs,
inner-selection audits, models, predictions, official scores and paired block
bootstrap intervals. No hosted packaging, upload, full-data refit or live edits.
