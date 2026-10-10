# Numerai Project Plan (revised 2026-10-10)

**Goal:** Numerai Classic performance, for fun. An AI research agent ("Jev") may come
later as a tool for that goal, not as a goal in itself. "Generalizable infrastructure"
is dropped.

## Verified facts (checked 2026-10-10, current round 1374)

| Item | Current value | Implication |
|---|---|---|
| Dataset | `v5.3` | Pin the version in config; never use `target` alias |
| Main target | `target_ender_60` (60 market days); `target_ender_20` also exists | Optimize 60D first, use other targets for ensembles |
| Payout | `3 × CORR60 + 15 × MMC60` (atomic staking, from round 1343) | MMC is weighted heavily → being *different* from the meta model matters |
| Eras | Weekly in train/validation, daily in live | Embargo must be measured in weekly eras |
| Live resolution | ~12 weeks for 60D rounds | Start live submissions early; feedback is slow |
| Scoring code | `numerai-tools` (`numerai_corr`, `correlation_contribution`, …) | Use it; don't reimplement |
| Meta model | `meta_model.parquet` covers validation eras | Lets us compute MMC locally |

Files (≈9.3 GB total): `train.parquet` 3.3 GB, `validation.parquet` 5.7 GB,
`meta_model.parquet`, `*_benchmark_models.parquet`, `validation_example_preds.parquet`,
`live.parquet`, `features.json`.

## Methodology

- **Splits.** Train on `train.parquet`. Split `validation.parquet` chronologically into
  *dev* (model selection) and a *locked holdout* (most recent ~2 years with resolved
  targets). Every holdout evaluation is logged; it's looked at rarely.
- **Embargo.** 60 market days ≈ 12 weeks + lag → **gap of 13 eras** between any
  training and evaluation window for 60D targets (5 for 20D).
- **Primary metric:** mean per-era payout proxy `3·CORR + 15·MMC` on dev, plus
  CORR and MMC separately, Sharpe (mean/std), max drawdown, % positive eras,
  correlation with benchmark models.
- **Trial accounting.** Every experiment is appended to `experiments/log.jsonl`
  with config, git hash, and metrics. Promotion decisions consider the number of
  trials run.
- **Final model** for live is retrained on train + validation using frozen settings.

## Compute plan (Apple M4, 24 GB RAM)

- Load features as `int8`, only needed columns.
- Start with the `medium` feature set; train on every 4th era (standard trick,
  also reduces target overlap).
- Full feature set / all eras only after measuring peak memory.

## Phases

| # | Goal | Done when |
|---|---|---|
| 0 | Download data, reproduce the official example model on validation, submit unstaked | Our CORR matches `validation_example_preds` closely; one live submission |
| 1 | Evaluation harness: splits + embargo, numerai-tools metrics incl. MMC, holdout lock, results log, leakage tests | `make baseline` reproduces a logged result |
| 2 | Manual research: targets ensemble, feature neutralization, LGBM vs XGB vs linear, era-boosting, MMC-oriented diversification | A model beats the example model on dev payout proxy and holdout |
| 3 | Automated live pipeline (Numerai model upload) | Weekly-daily submissions without manual steps |
| 4 | (optional) Jev: agent proposes experiments; compared against Optuna at equal budget | Agent beats Optuna baseline, or we learn it doesn't |

## Out of scope (for now)

MLflow, databases, distributed compute, staking automation, multi-problem generality.
