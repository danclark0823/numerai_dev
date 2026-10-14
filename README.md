# numerai_dev

A hobby research pipeline for the [Numerai Classic](https://numer.ai) tournament.
The goal is tournament performance, with experiments that are reproducible and
compared with sound statistics.

Current best on dev eras: **CORR60 0.044** (4 era-bagged LightGBMs on all 3,555
features). See [docs/results.md](docs/results.md).

## Setup

Requires Python 3.12. On Apple Silicon, use conda-forge so LightGBM gets OpenMP:

```bash
conda create -n numerai -c conda-forge python=3.12 lightgbm pandas pyarrow polars numpy scipy scikit-learn pytest ruff pyyaml
conda activate numerai
pip install numerapi numerai-tools
pip install -e .
```

Download the data (about 9.3 GB into `data/v5.3/`, which git ignores):

```bash
python scripts/download_data.py
```

## Usage

Each experiment is a YAML config in `configs/`:

```bash
python scripts/run_experiment.py configs/lgbm_all_bag4.yaml
```

This trains the model, scores it on the dev eras, saves the model and predictions
to `artifacts/<name>/`, and appends a record (config, git hash, metrics, runtime,
peak memory) to `experiments/log.jsonl`. To also score the locked holdout, add
`--holdout`, but only for final candidates.

Compare two experiments era by era:

```bash
python scripts/compare.py lgbm_all_stride4 lgbm_all_bag4
```

Run the tests:

```bash
pytest
```

## Live submissions

Numerai models and what they submit are defined in [configs/live.yaml](configs/live.yaml):

| Numerai model | Submits |
|---|---|
| `lgbm_bag4` | 4 era-bagged LightGBMs on `ender_60` (CORR-focused) |
| `dc_resid_ensemble` | benchmark-residual + teager2b_60 models, 50% neutralized (diversification) |
| `dc_tyndal` | LightGBM on target residualized against both benchmarks (experimental diversifier) |
| `dc_rondal` | Ridge regression on target residualized against both benchmarks (most independent) |

New models are named after war mages from the *Spellmonger* series (`dc_tyndal`,
`dc_rondal`, `dc_minalan`, ...). `configs/live.yaml` holds each one's technical description.

Credentials go in `.env` (`NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`; git-ignored).
Use an API key with only upload + read scopes, never staking.

```bash
python scripts/submit_live.py --check-validation  # live code path == validated preds
python scripts/submit_live.py                     # dry run: predict + validate
python scripts/submit_live.py --upload            # upload this round
```

Prediction files go to `predictions/<round>/`, and every run is logged to
`experiments/submissions.jsonl`. Live models use the saved `artifacts/`, which
exist only locally, so the experiments must be run once on this machine first.

## Methodology

- **Data:** dataset v5.3, target `target_ender_60` (60-day horizon).
- **Splits:** train on `train.parquet`. Validation is split chronologically into
  *dev* (for model selection) and a locked *holdout* (the last 52 resolved eras).
  A 13-era embargo separates each window, because 60-day targets overlap
  neighboring weekly eras.
- **Metrics:** scored with Numerai's official `numerai-tools`. We report CORR,
  MMC (meta model contribution; covers only eras 1133+), BMC (contribution
  against Numerai's benchmark model, used as the dev-set stand-in for MMC), and
  the payout proxy `3·CORR + 15·MMC`.
- **Comparisons:** paired per-era differences with a 13-era block bootstrap CI.
- **Memory:** features are streamed as int8 into numpy and freed after LightGBM
  binning, so the full feature set trains in about 12 GB.

## Layout

```text
configs/        experiment configs (YAML)
scripts/        download_data.py, run_experiment.py, compare.py
src/numerai_dev data loading, splits, evaluation, experiment log
tests/          pytest suite
docs/           plan.md (roadmap), results.md (results so far)
experiments/    log.jsonl, the append-only record of every run
```

## Roadmap

The experimental three-model diversifier slate and its nested chronological
evaluation runner are documented in [docs/diversifiers.md](docs/diversifiers.md).
The fifth-model ridge-plus-interaction experiment, **dc_wenek**, is documented in
[docs/wenek.md](docs/wenek.md), with its frozen design in [docs/wenek_plan.md](docs/wenek_plan.md).
Wenek is now shelved after its simplification test. The next candidate,
**dc_bendonal**, is now [implemented and evaluated](docs/bendonal.md). It improved
on ridge in all three development windows but failed the nonnegative-MMC screen;
the [original plan](docs/bendonal_plan.md) and results are retained for audit.
The subsequent covariance-stability experiment, **dc_alscot**, is
[implemented and evaluated](docs/alscot.md). Its inner-selected penalty worsened
CORR in every outer window and worsened MMC, so this formulation is also shelved.
The full-feature specialist ensemble, **dc_hesia**, is
[implemented and evaluated](docs/hesia.md). It achieved positive MMC but lost to
matched all-feature ridge, so it is shelved. Broader feature coverage substantially
improved the ridge control compared with the earlier 128-feature experiments.
The recency-weighted residual ridge, **dc_taren**, is
[implemented and evaluated](docs/taren.md) on full data against a walk-forward
recreation of dc_rondal ([plan](docs/taren_plan.md)). Nested selection chose no
decay in every fold, so it is shelved; the Rondal recreation scored CORR 0.024 and
MMC 0.011 on the development windows. The alternate-target residual ridge,
**dc_azar** (a separate model; [plan](docs/azar_plan.md)), was also
[shelved](docs/azar.md): its inner-selected target changed every fold and
contributed negatively beyond Rondal in all three windows. The portfolio-residual
LightGBM, **dc_carmella** (a separate model; [plan](docs/carmella_plan.md)),
[passed its development screen](docs/carmella.md). It added to matched Rondal
and Tyndal recreations in all three windows with MMC 0.014, but stays close to
Tyndal. Next is a frozen replication on fresh live eras, not deployment.
The multi-target ensemble **dc_mavone** ([plan](docs/multi_target_ensemble_plan.md))
completed Phases 0–3 ([results](docs/mavone.md)). An equal-weight blend of
LightGBM models on four raw 60-day targets did not beat matched Ender alone on
forward eras 1133–1227 (MMC 0.0105 vs 0.0116), so hypothesis 1 failed. The raw-target
members correlate 0.80–0.89 with each other and about 0.8 with the benchmark.
A benchmark-residual member (MMC 0.0165) and the dc_resid_ensemble recreation
(MMC 0.0152) did much better. The MMC-history gate for weight optimization fails
(95 of 195 eras), so learned weights are deferred.

See [docs/plan.md](docs/plan.md). Next up: target ensembles and feature
neutralization to raise MMC, then automated live submissions (unstaked).
