# dc_mavone Phase 0-1 audit

Dataset v5.3, 3555 features (order sha256 d8f2180f57a256f2), 40 named targets plus the `target` alias ({"train": {"same_missingness": true, "equal_values": true}, "validation": {"same_missingness": true, "equal_values": true}}).

## Files

| File | Bytes | Modified | sha256 |
|---|---:|---|---|
| features.json | 387,149 | 2026-10-10T15:12:23 | 27de6b598b0d4794 |
| live_1374.parquet | 12,146,279 | 2026-10-11T09:24:57 | 6717c02d6a6c681b |
| meta_model.parquet | 14,886,916 | 2026-10-10T15:30:54 | 9c186a2f4171d612 |
| train.parquet | 3,296,841,026 | 2026-10-10T15:23:02 | 05d99c729369abb2 |
| train_benchmark_models.parquet | 65,045,676 | 2026-10-10T15:31:21 | 1655bca3291a1d7c |
| validation.parquet | 5,687,751,662 | 2026-10-10T15:30:51 | 252a8cf329998b18 |
| validation_benchmark_models.parquet | 127,863,257 | 2026-10-10T15:31:14 | 2002a51844c12c69 |
| validation_example_preds.parquet | 98,342,838 | 2026-10-10T15:31:02 | 4ca8fa086b5d8903 |

## Availability

| Source | Eras | First | Last | Notes |
|---|---:|---:|---:|---|
| train_benchmark (v53_lgbm_ender60) | 417 | 158 | 574 | row coverage 0.776 |
| validation_benchmark (v53_lgbm_ender60) | 666 | 575 | 1240 | row coverage 1.000 |
| meta model (local) | 95 | 1133 | 1227 | contiguous 95; min era row coverage 1.000 |
| meta model (remote, 2026-10-14) | 96 | 1133 | 1228 | era column only; newer eras are reserved and their values were not read |
| resolved target_ender_60 (validation) | | 575 | 1227 | |

Data types: `{"train": {"train": {"first": 1, "last": 574, "eras": 574, "rows": 2746268}}, "validation": {"test": {"first": 1228, "last": 1240, "eras": 13, "rows": 90852}, "validation": {"first": 575, "last": 1227, "eras": 653, "rows": 4078596}}}`

## Target maturity and missingness

| target | split | rows_nonnull | rows | first_era | last_era | full_eras | partial_eras | empty_eras |
|---|---|---|---|---|---|---|---|---|
| target_ender_60 | train | 2746268 | 2746268 | 1 | 574 | 574 | 0 | 0 |
| target_jasper_60 | train | 2746268 | 2746268 | 1 | 574 | 574 | 0 | 0 |
| target_rowan_60 | train | 2746268 | 2746268 | 1 | 574 | 574 | 0 | 0 |
| target_teager2b_60 | train | 2746268 | 2746268 | 1 | 574 | 574 | 0 | 0 |
| target_ender_60 | validation | 4078596 | 4169448 | 575 | 1227 | 653 | 0 | 13 |
| target_jasper_60 | validation | 4078596 | 4169448 | 575 | 1227 | 653 | 0 | 13 |
| target_rowan_60 | validation | 4078596 | 4169448 | 575 | 1227 | 653 | 0 | 13 |
| target_teager2b_60 | validation | 4078596 | 4169448 | 575 | 1227 | 653 | 0 | 13 |

All 41 target columns are in `target_inventory.csv`.

## Periods already used (all development evidence)

| Eras | Use | Status |
|---|---|---|
| 1-574 | training for every earlier model | training |
| 587-1162 | validation scoring for round 1-3 LightGBM, ridge and target sweeps (including jasper, rowan, teager2b) and ensemble selection | development |
| 850-879, 1000-1029, 1133-1162 | walk-forward windows: diversifiers, Wenek, Bendonal, Alscot, Hesia, Taren, Azar, Carmella | development |
| 1176-1227 | 52-era holdout, inspected 3 times; chose recent_* live members | development |
| 1-1162 | training of recent_* live members | training |
| 1-1227 | training of final_ender60 and final_ridge_resid_a10 | training |
| 1133-1227 | every era with local meta-model coverage; all previously inspected | development |
| >=1228 | unresolved in the 2026-10-10 snapshot; reserved for fresh confirmation, outcomes not inspected | reserved |

No resolved era in the local snapshot is unused. Eras from 1228 on are reserved for fresh confirmation; their outcomes are not inspected here.

Label availability: No point-in-time publication dates exist locally; a label for era e is assumed available 13 weekly eras later (60D2L resolves in about 12 weeks). Results are a retrospective forward-label simulation, not an exact historical live replay.

## Prediction-artifact provenance

47 logged runs (`artifact_provenance.csv`). 10 trained on validation eras (recent_*/final_*), so they are not valid forward controls. Earlier walk-forward artifacts used different targets, features or labels. No cached prediction is reused; every Mavone prediction is generated fresh with recorded provenance.

## Phase 1: target audit on eras <= 1119 (historical training only)

| member | target | rows | nonnull | first_era | eras_partial | eras_empty | distinct_values | mean_era_mean | mean_era_std | min_era_std | max_era_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ender | target_ender_60 | 6110704 | 6110704 | 1 | 0 | 0 | 5 | 0.5000 | 0.2237 | 0.2236 | 0.2240 |
| teager2b | target_teager2b_60 | 6110704 | 6110704 | 1 | 0 | 0 | 5 | 0.5000 | 0.2237 | 0.2236 | 0.2240 |
| jasper | target_jasper_60 | 6110704 | 6110704 | 1 | 0 | 0 | 5 | 0.5000 | 0.2237 | 0.2236 | 0.2240 |
| rowan | target_rowan_60 | 6110704 | 6110704 | 1 | 0 | 0 | 5 | 0.5000 | 0.2237 | 0.2236 | 0.2240 |

Value distributions: 
- ender: 0:0.050, 0.25:0.200, 0.5:0.500, 0.75:0.200, 1:0.050
- teager2b: 0:0.050, 0.25:0.200, 0.5:0.500, 0.75:0.200, 1:0.050
- jasper: 0:0.050, 0.25:0.200, 0.5:0.500, 0.75:0.200, 1:0.050
- rowan: 0:0.050, 0.25:0.200, 0.5:0.500, 0.75:0.200, 1:0.050

Mean per-era Spearman correlation between targets (std over eras in brackets):

| target | ender | teager2b | jasper | rowan |
|---|---|---|---|---|
| ender | 1.000 (0.000) | 0.791 (0.017) | 0.793 (0.015) | 0.754 (0.022) |
| teager2b | 0.791 (0.017) | 1.000 (0.000) | 0.704 (0.022) | 0.760 (0.021) |
| jasper | 0.793 (0.015) | 0.704 (0.022) | 1.000 (0.000) | 0.680 (0.025) |
| rowan | 0.754 (0.022) | 0.760 (0.021) | 0.680 (0.025) | 1.000 (0.000) |

Joint-intersection exclusions on stride eras <= 1119: 0 of 1,527,361 rows; by target {'missing_target_ender_60': 0, 'missing_target_teager2b_60': 0, 'missing_target_jasper_60': 0, 'missing_target_rowan_60': 0} (`history_exclusions_by_era.csv`).

## Split calendar

| Block | Eras | Fit on eras <= | Labels available by (approx.) | In file | Resolved | Meta | Prospective | Stride training eras |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 1133 | 1133-1158 | 1119 | era 1132 (2024-09-14) | 26 | 26 | 26 | 0 | 280 |
| 1159 | 1159-1184 | 1145 | era 1158 (2025-03-15) | 26 | 26 | 26 | 0 | 287 |
| 1185 | 1185-1210 | 1171 | era 1184 (2025-09-13) | 26 | 26 | 26 | 0 | 293 |
| 1211 | 1211-1236 | 1197 | era 1210 (2026-03-14) | 26 | 17 | 17 | 9 | 300 |

Phase 4 MMC-history gate: needs 195 contiguous forward eras with MMC, has 95 (1133-1227). Gate passed: False; learned MMC weights deferred.

Scoring rules: {"source": "https://docs.numer.ai/numerai-tournament/scoring/ (fetched 2026-10-14)", "summary": "From round 1343 CORR, MMC and BMC use 60-day Ender; v5.3 diagnostics use target_ender_60 and v53_lgbm_ender60. Payout 3*CORR + 15*MMC (repo constant)."}

Environment: `{"python": "3.12.15", "lightgbm": "4.7.0", "numpy": "2.5.3", "pandas": "3.0.6", "pyarrow": "25.0.0", "scipy": "1.18.1", "numerai_tools": "0.7.2", "cpu_count": 10, "machine": "arm64", "OMP_NUM_THREADS": null}`
