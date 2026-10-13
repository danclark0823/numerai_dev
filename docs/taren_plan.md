# dc_taren — frozen recency-weighted residual ridge experiment

Named for Taren, a warmage listed at https://spellmongershall.com/characters.
Research only; preserve the four live models.

Hypothesis: Numerai signals drift. A ridge model that weights recent eras more
heavily than the equal-weight history used by dc_rondal (and, plausibly, by most
of the meta model) will make different errors, adding contribution beyond
Rondal and nonnegative MMC. Evidence motivating this: retraining on recent eras
was the largest gain in this project (results round 4); holdout CORR was ~4x
below dev; Wenek's interactions reversed on later data; Alscot, which suppressed
changing signals, lost everywhere. This tests diversification through *time*,
not architecture. It is not direct MMC optimization.

What changed from the shelved experiments, deliberately:

- All 3,555 v5.3 features and every row of every eligible training era, not
  128 features and 1,024 sampled rows. Hesia showed the small setup understates
  ridge badly (pooled CORR 0.006 vs 0.025 on the same rows).
- The control is a walk-forward recreation of dc_rondal, not raw-target ridge:
  the fifth model must add to the live portfolio, not beat a weak baseline.

Frozen design, before evaluation:

- Label: target_ender_60 residualized per era against gaussianized
  v53_lgbm_ender60 and v53_lgbm_ender20 (`targets.residualize`, exactly as
  `run_experiment.label_frame`). Rows lacking either benchmark or the target are
  excluded. Training eras: train plus validation, era stride 1.
- Model: centered weighted ridge. Era e in a fit ending at era T gets weight
  w = 0.5^((T - e)/h), constant within the era; calendar era numbers, so gaps
  are respected. Penalty alpha * sum(w) with alpha=10, so h = inf reproduces
  `RidgeModel` (Rondal's recipe) exactly. Weighted feature means and label mean
  center the fit; intercept unpenalized.
- Half-life grid h in eras: [52, 104, 208, 416, inf] (1, 2, 4, 8 years, none).
- Nested selection per outer fold, same geometry as Alscot's `select_strength`:
  three 52-era inner validation windows ending at outer_start - 13, separated by
  13-era gaps; each inner fit uses eras < inner_start - 13, decay measured from
  that fit's last era. Inner validation uses every row of each era, scored with
  official CORR on the raw target_ender_60. Choose the h with highest mean of the
  three inner-window CORRs; ties (|diff| < 1e-6) go to the larger h. inf is the
  explicit Rondal fallback. Record all inner scores; outer labels cannot affect
  selection. Refit the selected h on all eligible outer training eras.
- Outer windows: all rows of eras 850–879, 1000–1029, 1133–1162; outer fits use
  eras < start - 13. Historical 52-era holdout and its embargo remain excluded.
- One pass over the data in era order: compute each era's Gram matrix X'X and
  X'y once (float64, bounded row batches), add into one accumulator per grid h
  after decaying it by 0.5^(delta_era/h). At each of the 12 required fit
  endpoints (3 outer + 9 inner), solve all five h. Memory stays at a few p x p
  matrices; no full feature matrix. BLAS limited to two threads.

Screen, applied to the inner-selected pipeline only (grid members are reported
as diagnostics, never selected on outer results). Versus the Rondal control:

1. Positive contribution beyond Rondal (`correlation_contribution` with the
   control as reference) in at least two of three windows.
2. Actual MMC on 1133–1162 nonnegative and no worse than the control's.
3. Positive CORR in every window, and pooled 90-era CORR no more than 0.002
   below the control's. CORR gains are not required: this is a diversifier.

Report per-window CORR, MMC, BMC, contribution beyond Rondal, payout proxy,
mean per-era rank correlation with Rondal and with the meta model, selected h per
fold, and separate-window paired 13-era block-bootstrap intervals. If every fold
selects inf, Taren equals Rondal and fails criterion 1; record that as no
evidence that recency weighting helps, not as a reason to widen the grid.

A pass warrants a separate frozen replication plan (fresh resolved live eras
or a new window set), then a full-data fit and hosted packaging through the
existing ridge path in `build_hosted_models.py`. A failure shelves this
formulation without changing the grid, alpha, selection rule or label.

Caveats: these three windows have been inspected repeatedly; actual MMC exists
for only 30 eras (~2.3 bootstrap blocks). The benchmark models may have been
trained on validation eras, so residual labels on training eras could carry
mild information about later periods; Rondal shares this, and both arms use the
same labels, but absolute scores may be optimistic.

Tests: h=inf parity with `RidgeModel.fit_stream`; recursive decayed accumulation
equals direct weighted least squares on synthetic data; synthetic drifting
signal favors finite h and stationary signal favors inf under the selection
rule; future-label isolation (perturbing labels after an endpoint leaves its
coefficients unchanged); inner/outer embargo geometry; serialization and batch
parity; invalid-input rejection. Real-run checks: chronology, residual-label
parity with `label_frame`, selected h in grid, finite nonconstant predictions,
full-era submission validity, protected production-file hashes unchanged.
Save source snapshots and hashes, config, per-endpoint training era ranges and
row counts, inner audits, models, full-era predictions, metrics, per-era scores,
paired intervals and the decision. No upload, hosted packaging, full-data
refit or live edits. Existing output directories must not be overwritten.
