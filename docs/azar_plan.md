# dc_azar — frozen alternate-target residual ridge experiment

Named for Azar, a warmage listed at https://spellmongershall.com/characters.
Research only. dc_azar would be a separate fifth model: live dc_rondal and the
other three live models are never refit, reweighted or replaced. Rondal appears
only as a walk-forward research recreation used as the comparison yardstick.

Hypothesis: Numerai's alternative 60-day targets encode return definitions that
differ from Ender. Ridge trained on one of them, after residualizing against both
benchmarks, will make errors different from Rondal (trained on residualized Ender).
That would give positive contribution beyond Rondal and nonnegative MMC when scored
on target_ender_60. Round 2 found that LightGBM on teager2b_60 or jasper_60
diversified, but those were raw-target models with train-only, every-fourth-era
data and medium-proxy neutralization, so this is a different test.

Frozen design, before evaluation:

- Candidate training targets: all 60-day v5.3 targets defined on every row where
  target_ender_60 and both benchmarks are defined. That gives 19 targets: agnes,
  alpha, bravo, caroline, charlie, claudia, cyrusd, delta, echo, ender (the
  fallback), jasper, ralph, rowan, sam, teager2b, tyler, victor, waldo and xerxes.
  target_jeremy_60 is excluded because it is missing within 400+ eras; this was
  decided from label availability only, not outcomes.
- Labels: each candidate residualized per era against gaussianized
  v53_lgbm_ender60 and v53_lgbm_ender20, identical to `targets.residualize`. They
  are computed for all candidates at once, which is exact because the candidates
  share their row set within every era.
- Model: unweighted ridge, alpha=10, penalty alpha*n, all 3,555 features, every row
  of every eligible era (train plus validation). All candidates share X'X; each has
  its own X'y. The ender candidate is exactly Rondal's recipe and is the control.
- Nested selection per outer fold. Use the same inner geometry as Taren and Alscot:
  three 52-era inner windows ending at outer_start − 13, separated by 13-era gaps,
  each fit using eras < inner_start − 13. Score every row of each inner era on raw
  target_ender_60. The criterion is mean per-era `correlation_contribution` of the
  candidate with the same-endpoint Rondal recreation as reference, averaged per
  window and then across the three windows. Pick the highest mean. If no alternate
  target has a positive mean, select ender, which equals Rondal and fails the
  screen. Record all inner scores; outer labels cannot affect selection.
- Outer windows: all rows of eras 850–879, 1000–1029, 1133–1162; fits use eras
  < start − 13. Historical 52-era holdout and its embargo remain excluded.
- One pass over the data in era order: compute exact per-era X'X once, X'Y for all
  candidates, and solve every candidate at the 12 fit endpoints with one Cholesky
  factorization each. BLAS limited to two threads.

Screen, for the inner-selected pipeline only. All 19 targets' outer scores are
reported as diagnostics and are never selected on outer results. Because Azar
would be a separate model, it need not beat Rondal; it must add to it:

1. Contribution beyond the Rondal recreation greater than 1e-12 in at least two of
   three windows.
2. Actual MMC on 1133–1162 nonnegative.
3. Positive CORR on target_ender_60 in every window.

Report per-window CORR, MMC, BMC, contribution beyond Rondal, payout proxy, the
selected target per fold, mean per-era rank correlation with Rondal and the meta
model, and paired Azar − Rondal 13-era block-bootstrap intervals per window.

A pass warrants a separate frozen replication plan, preferably on newly resolved
live eras. Only after that would a full-data fit, hosted packaging and a new live
model entry follow. A failure shelves this formulation without changing the target
list, alpha, selection criterion or label.

Caveats: these windows have been inspected repeatedly, including by Taren
diagnostics; actual MMC covers only 30 eras (~2.3 bootstrap blocks). Benchmark
predictions may have been fit on validation eras; Rondal shares this. Selecting
among 18 alternatives adds selection noise, which the nested design is meant to
contain, not eliminate.

Tests: multi-target fit equals independent `RidgeModel` fits; vectorized residual
labels equal `targets.residualize`; exact per-era statistics; future-label
isolation; selection rule including the ender fallback; screen; serialization and
batch parity; invalid inputs. Real-run checks: residual-label parity with
`targets.residualize` on sampled eras, Rondal-recreation parity with
`dc_taren_full_v1`, chronology, complete eras, finite nonconstant predictions,
full-era submission validity, protected production-file hashes unchanged. No
upload, hosted packaging, full-data refit or live edits.
