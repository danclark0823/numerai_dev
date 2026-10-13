# dc_alscot — results

Status: shelved after its first frozen experiment, 2026-10-13. Not deployed.
Named for Alscot, the supporting mercenary warmage in Spellmonger; see the
[character listing](https://spellmongershall.com/characters).

## What we tried

Jointly learned ridge coefficients with an additional penalty on variation in
feature/target covariance across historical 52-era blocks. This is a tractable
stability proxy, not a direct worst-loss or MMC objective. Unlike the earlier
stable feature portfolio, it fits coefficients jointly rather than selecting
features by individual directional consistency. See the [frozen plan](alscot_plan.md).

The matched experiment used Wenek's 128 features, original Ender60 labels,
1,024 rows per sampled training era, every fourth era, and ridge alpha=10.
Each outer fold independently selected lambda from [0, 1, 10, 100] using three
earlier embargoed inner splits. Selection maximized worst inner-window CORR,
then mean CORR, then favored smaller penalties. Zero provided a ridge fallback.
Inner validation used sampled rows/eras; outer validation used complete eras.

All three folds selected lambda=100. It improved the worst inner-window scores,
but those gains did not carry to any outer window. No grid expansion or
post-result parameter tuning was performed.

## Results

Run `artifacts/dc_alscot_1024_v1` completed in 28.8 seconds, evaluating 563,466
rows across 90 full eras. MMC covers only the last 30, not the whole CORR period.

| Measure | Matched ridge | Alscot |
|---|---:|---:|
| CORR, 90 eras | 0.006146 | 0.003586 |
| MMC, latest 30 eras | -0.004378 | -0.005447 |
| Worst-window CORR | -0.003338 | -0.004781 |
| Positive contribution beyond ridge | — | 0/3 windows |

| Window | Ridge CORR | Alscot CORR | Contribution beyond ridge |
|---|---:|---:|---:|
| 850–879 | 0.007763 | 0.004149 | -0.001935 |
| 1000–1029 | 0.014012 | 0.011390 | -0.001493 |
| 1133–1162 | -0.003338 | -0.004781 | -0.001391 |

## Why we are not using it

It failed every performance requirement: no outer-window CORR wins, no positive
contribution windows, lower pooled CORR, worse worst-window CORR, and negative
MMC worse than ridge. The historical covariance penalty fell substantially, so
the intended mechanism operated, but reducing that training proxy did not
produce useful out-of-sample stability. Suppression of useful changing signal
and noisy block estimates are possible explanations, not established causes.

The selected lambda reached the grid boundary in every fold; that is recorded,
not permission to keep searching after seeing outer results. This rejects the
tested covariance-penalized formulation, not every era-robust learning method.
No scale-up, full-data refit, hosted packaging, or upload followed.

All evaluation windows have been inspected repeatedly. Their results are
exploratory, and the 30-era MMC period contains only about 2.3 thirteen-era
blocks. The existing 52-era historical holdout remains excluded; it was not
opened for confirmation. A passing future candidate still needs a separate
frozen confirmation before deployment.

## Verification and artifacts

All 50 repository tests passed and lint passed. New coverage includes zero-penalty
ridge equivalence, PSD/covariance reconstruction, synthetic reversing-signal
shrinkage, future-label isolation, input validation, and serialization/batch parity.
Real-data checks passed for exact reference IDs/eras/labels, historical ridge
prediction parity, reload/batch parity, nonconstant predictions, and complete-era
submission ranks. All ten protected production-file hashes were unchanged.

The artifact directory retains source snapshots, configuration, training IDs,
fold-specific inner scores and selected penalties, serialized models, full-era
predictions, metrics, paired bootstrap comparisons, and a failed screen decision.
The runner refuses to overwrite an existing artifact directory.

Entry points: `scripts/run_alscot.py`, `configs/alscot_1024.yaml`,
`src/numerai_dev/alscot.py`, and `tests/test_alscot.py`. No changes to any of the
four live models were needed.
