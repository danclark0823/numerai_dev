# dc_taren — recency-weighted residual ridge results

Status: tested and shelved, 2026-10-13. Not deployed; the four live models are
unchanged. Named for Taren, a warmage listed at
https://spellmongershall.com/characters. The [frozen plan](taren_plan.md)
specifies the hypothesis, grid, selection rule and screen.

## What we tried

Rondal's recipe (all 3,555 features, target_ender_60 residualized per era against
both benchmarks, ridge alpha=10) with era weights 0.5^((last era - era)/h). Each
outer window chose h from [52, 104, 208, 416, inf] by mean CORR over three earlier
52-era inner windows. h = inf is exactly Rondal, which served as the walk-forward
control. Every row of every eligible era was used (5.5M training rows by the
last window; eras 1–157 lack benchmark predictions, so their residual labels
are undefined, as in Rondal's own fit).

## Results

Run `artifacts/dc_taren_full_v1` streamed train and validation once in era order,
fitting all 60 models (12 endpoints x 5 half-lives) from exact per-era sufficient
statistics. It took 618 seconds with 5.6 GB peak RSS.

Inner selection chose h = inf in all three folds, so Taren equals Rondal and fails
the screen: zero contribution beyond Rondal in every window. Inner CORR increased
monotonically with half-life in every fold:

| Outer window | h=52 | h=104 | h=208 | h=416 | inf |
|---|---:|---:|---:|---:|---:|
| 850 | 0.0233 | 0.0291 | 0.0333 | 0.0357 | 0.0377 |
| 1000 | 0.0146 | 0.0191 | 0.0225 | 0.0242 | 0.0253 |
| 1133 | 0.0064 | 0.0095 | 0.0139 | 0.0175 | 0.0216 |

Within this design, more history beat recency-weighted history. That differs
from the round-4 finding: adding the most recent eras still helps, while
downweighting older eras discards useful signal.

## Walk-forward Rondal (the control)

This is the first leakage-safe historical recreation of Rondal: full rows, all
features and embargoed walk-forward fits. Pooled CORR was 0.0242 over 90 eras and
MMC 0.0114 on 1133–1162. Its per-era rank correlation with the meta model was
0.094 there. Window CORRs were 0.0306, 0.0266 and 0.0156. This supports Rondal's
deployment, although these windows were previously inspected.

## Diagnostics, not candidates

Outer-window grid scores are recorded but cannot be selected under the plan.
Shorter half-lives lost pooled CORR. h=52 had higher latest-window MMC (0.0166
vs 0.0114), but its contribution beyond Rondal was +0.0057, −0.0144 and +0.0105.
That sign change between windows is unstable, like Wenek's interaction component.
Treating MMC as a reason to rescue h=52 would select on outer results.

## Decision

Shelve this formulation without changing the grid, alpha, selection rule or
label. It rejects recency-weighted single ridge on these windows, not all uses
of recent data. Live dc_rondal stays as it is; any follow-up would be a separate
model. Examples are a standalone short-memory model, or a half-life chosen by
contribution beyond Rondal instead of CORR. Either is a new hypothesis that needs
its own frozen plan, preferably evaluated on windows that this diagnostic has not
influenced.

## Verification and reproduction

All 67 repository tests and lint passed. New tests cover exact float32 Gram
matrices, h=inf parity with `RidgeModel`, recursive decay versus direct weighted
least squares with irregular era gaps, future-label isolation, inner-window
geometry, drift versus stationary selection, tie-breaking, serialization and
batch parity, invalid inputs, and the screen. That includes a numerical-zero
threshold so a run identical to Rondal cannot count as positive contribution.
A pre-run smoke test on 42 features found that problem; its artifacts were
deleted and not logged.

Real-run checks passed for monotonic era streaming, feature-bin validity, endpoint
chronology, complete inner and outer eras, finite nonconstant predictions,
full-era submission validity, joblib reload/batch parity, and unchanged protected
production files. Artifacts retain source snapshots and hashes, configuration,
per-endpoint training ranges and effective row counts, inner scores, all fitted
models, full-era predictions, metrics, per-era scores, paired intervals and the
decision. Entry points: `src/numerai_dev/taren.py`, `scripts/run_taren.py`,
`configs/taren_full.yaml`, `tests/test_taren.py`.
