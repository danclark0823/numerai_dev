# dc_carmella — frozen portfolio-residual experiment

Named for Carmella, a warmage listed at https://spellmongershall.com/characters.
Research only. dc_carmella would be a separate fifth model: live dc_rondal,
dc_tyndal and the other live models are never refit, reweighted or replaced.
Rondal and Tyndal appear only as walk-forward research recreations.

Hypothesis: our live diversifiers already capture the linear (Rondal) and
nonlinear (Tyndal) signal left after removing both benchmarks. A nonlinear
learner trained on what remains after also removing a leakage-safe Rondal
prediction will add contribution beyond both Rondal and Tyndal. The diversifier
slate tested this idea with weak 128-feature teachers on sampled rows; here the
teacher is the full-strength Rondal recipe and the control is a matched Tyndal.
Taren and Azar showed that changing Rondal's training data or target does not
add to Rondal; this changes the learner and removes Rondal's signal explicitly.

Frozen design, before evaluation:

- Forward-chained Rondal predictions (pass 1). Use Rondal's exact recipe: all
  features, target_ender_60 residualized against both benchmarks, ridge
  alpha=10, unweighted, every row. Era blocks are 26 eras long, starting at the
  first labeled era plus 117 (104 eras of warm-up plus the 13-era embargo). Each
  block's rows are predicted by a fit on eras < block_start − 13 only. That uses
  no future data, so one set of predictions serves all outer folds. Rows before
  the first block have no prediction and train neither LightGBM. Outer-window
  Rondal recreations come from the same pass, fitted at outer_start − 14.
- Labels, per era on rows with a forward-chained prediction:
  - Tyndal control: target_ender_60 residualized against gaussianized
    v53_lgbm_ender60 and v53_lgbm_ender20 (`targets.residualize`).
  - Carmella: the same, plus the gaussianized forward-chained Rondal prediction
    as a third reference.
- Learner (pass 2): LightGBM with Tyndal's exact parameters (2000 trees, learning
  rate 0.01, depth 5, 31 leaves, colsample 0.1, seed 42). All features, streamed
  through `ParquetRows`. Both models train on identical rows: every fourth eligible
  era (offset 0) up to outer_start − 14, including validation eras. Striding keeps
  peak memory near the measured 16 GB for about 1.4M rows. No hyperparameter
  search and no inner selection: one fixed configuration.
- Outer windows: all rows of eras 850–879, 1000–1029, 1133–1162; 13-era embargo.
  Historical 52-era holdout and its embargo remain excluded.

Screen. Rondal is the outer-endpoint ridge recreation and Tyndal the matched
LightGBM control. "Portfolio" is their equal-weight per-era Gaussianized blend.

1. Contribution beyond the portfolio greater than 1e-12 in at least two of
   three windows.
2. Contribution beyond the Tyndal recreation greater than 1e-12 in at least two
   of three windows.
3. Actual MMC on 1133–1162 nonnegative.
4. Positive CORR on target_ender_60 in every window.

Report per-window CORR, MMC, BMC, payout proxy, all contributions, mean per-era
rank correlations with Rondal, Tyndal and the meta model, and paired
Carmella − Tyndal 13-era block-bootstrap intervals per window. Also report the
forward-chained Rondal predictions' CORR on training eras, to show how strong
the removed signal was.

A pass warrants a separate frozen replication plan on fresh live eras before
any full-data fit, hosted packaging or new live model entry. A failure shelves
this formulation without changing parameters, block size, warm-up or stride.

Caveats: these windows have been inspected repeatedly; actual MMC covers 30
eras (~2.3 bootstrap blocks). Early forward-chained Rondal fits use less history
than later ones, so the residual removes less Rondal signal in early eras.
Benchmark predictions may have been fit on validation eras, shared by every arm.
The Tyndal control trains on stride-4 rows, not live Tyndal's full rows.

Tests: forward-chained predictions use only eras < block_start − 13; block and
warm-up geometry; three-reference residuals match `targets.residualize`;
future-label isolation; screen. Real-run checks: Rondal-recreation parity with
`dc_taren_full_v1` on outer windows, label parity on sampled eras, identical
training rows for both LightGBMs, chronology, finite nonconstant predictions,
full-era submission validity, protected production-file hashes unchanged. No
upload, hosted packaging, full-data refit or live edits.
