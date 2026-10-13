# dc_azar — alternate-target residual ridge results

Status: tested and shelved, 2026-10-13. Not deployed. dc_azar was planned as a
separate fifth model; live dc_rondal and the other three live models are
unchanged. Named for Azar, a warmage listed at
https://spellmongershall.com/characters. The [frozen plan](azar_plan.md)
specifies the hypothesis, target list, selection rule and screen.

## What we tried

Ridge (all 3,555 features, alpha=10, every row of every eligible era) trained on
one of 18 alternative 60-day targets, each residualized per era against both
benchmarks. Each outer window chose the target whose inner-window predictions
had the highest mean contribution beyond a same-endpoint Rondal recreation,
scored on target_ender_60. Rondal's own target was the fallback. All 19 targets
shared one exact X'X per endpoint.

## Results

Run `artifacts/dc_azar_full_v1` took 593 seconds with 6.1 GB peak RSS. The
inner-selected pipeline failed the screen: its contribution beyond Rondal was
negative in all three windows and its MMC was negative.

| Window | Selected target | Azar CORR | Rondal CORR | Contribution beyond Rondal | Azar MMC | Rondal MMC |
|---|---|---:|---:|---:|---:|---:|
| 850 | tyler | 0.0183 | 0.0306 | −0.0019 | — | — |
| 1000 | rowan | 0.0168 | 0.0266 | −0.0025 | — | — |
| 1133 | delta | 0.0079 | 0.0156 | −0.0003 | −0.0013 | 0.0114 |

Each fold selected a different target, and none of the selected targets kept
its inner-window advantage in the next window. Inner contributions were small:
the winners led at 0.0047, 0.0021 and 0.0013. Spread across 18 alternatives,
gaps that size appear to be selection noise rather than a stable target
property. Paired Azar − Rondal CORR intervals excluded zero in every window,
for example [−0.0097, −0.0057] in 1133–1162; MMC was −0.0127 [−0.0166, −0.0119].

## Diagnostics, not candidates

All 18 targets' outer scores are saved but cannot be selected under the plan.
Jasper had positive contribution beyond Rondal in all three windows: +0.0016,
+0.0019 and +0.0009. Its pooled CORR matched Rondal's (0.0243) and its MMC was
0.0115. Teager2b, the target behind dc_resid_ensemble's second member, swung
from +0.0031 to −0.0025. Choosing jasper now would select on outer results
across 18 options. Any jasper model would need a new frozen plan with jasper
fixed in advance and windows or live eras these diagnostics have not influenced.

## Decision

Shelve this formulation without changing the target list, alpha, selection
criterion or label. Live models are unaffected. The result rejects selecting a
target from short inner windows, not every alternative-target model.

## Verification and reproduction

All 75 repository tests and lint passed. New tests cover exact per-era
statistics, multi-target fits matching independent `RidgeModel` fits, vectorized
residual labels matching `targets.residualize`, future-label isolation, the
selection fallback, serialization and batch parity, invalid inputs and the
screen. A 42-feature smoke run found a shape quirk in the label-parity check and
float32-versus-float64 rounding of about 1e-8. The parity tolerance is 1e-6 for
that reason. Smoke artifacts were deleted and not logged.

Real-run checks passed for residual-label parity on three sampled eras per
split, exact Rondal-recreation parity with `dc_taren_full_v1`, monotonic
streaming, chronology, complete inner and outer eras, finite nonconstant
predictions, full-era submission validity, joblib reload/batch parity, and
unchanged protected production files. Entry points: `src/numerai_dev/azar.py`,
`scripts/run_azar.py`, `configs/azar_full.yaml`, `tests/test_azar.py`.
