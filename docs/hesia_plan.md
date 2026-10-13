# dc_hesia — frozen feature-specialist experiment

Named after Hesia, the supporting warmage listed at
https://spellmongershall.com/characters. Research only; preserve the four live models.

Hypothesis: separately fitting fixed subsets of the full feature universe and
averaging normalized predictions preserves useful signals that a joint ridge fit
may suppress. This tests random feature specialization, not semantic group expertise.
The all-feature ridge control separates greater information coverage from the
effect of the ensemble. A linear subspace ensemble is not a new nonlinear learner;
only per-era rank normalization makes its blend differ from a single linear score.

Frozen design, before evaluation:

- Use all 3,555 v5.3 features in metadata order. Generate two seeded random
  partitions (seed 42), each with eight near-equal disjoint subsets (444–445
  features). Thus 16 overlapping specialists, every feature in exactly two.
  No feature selection, semantic grouping, target lookup, or outcome-based weights.
- Train every specialist and both controls on exactly the same rows and raw
  target_ender_60 labels. Ridge alpha=10 throughout, penalty n*alpha. Controls:
  the exact historical 128-feature ridge and one ridge using all 3,555 features.
- Normalize each specialist over each complete era with tie-preserving Gaussian
  ranks, then average equally. Do not select individual specialists on results.
- Seed 42, 1,024 rows per sampled training era, every fourth era; outer test
  windows 850–879, 1000–1029, 1133–1162, 13-era embargo. Full-era evaluation.
  Reuse prior sample rules/IDs and references. Keep the historical 52-era holdout
  excluded. All three windows were inspected before; this is exploratory.
- One configuration, no grid search. Screen Hesia against ALL-feature ridge:
  positive contribution in at least two windows, CORR gains in at least two
  windows plus pooled gain, and actual MMC nonnegative and no worse than the
  all-feature control wherever available. MMC covers only the latest 30 eras.
  Report worst-window CORR and comparisons against the small control too.
- A pass warrants a separate frozen replication plan, not deployment. A failure
  stops this formulation without changing partitions, alpha, or blend weights.

Implementation: accumulate raw sufficient statistics incrementally over eligible
training rows with bounded float batches. Fit specialists from exact marginal
Gram matrices; fit controls from the same statistics. Predictions use one batched
matrix product for all member coefficients, then full-era normalization. Limit
BLAS to two threads; avoid multiple full-size float training matrices.

Tests: seeded exact feature coverage and no duplicate columns per specialist;
marginal-stat fits vs independently fitted ridge; incremental-stat parity;
label-free partitions; full-era ensemble formula; reordered named feature input;
batch and serialization parity; invalid-input rejection; future-label isolation.
Real checks: unique/aligned IDs, eras/labels/reference windows, exact historical
128-feature ridge parity, full-era nonconstant finite valid submissions, explicit
chronology, and ten protected production-file hashes unchanged. Save source and
metadata hashes, feature memberships, training IDs, models, predictions, metrics,
per-era scores, and separate-window paired block-bootstrap intervals. No upload,
hosted packaging, or final full-data refit.
