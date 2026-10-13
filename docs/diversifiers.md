# Diversifier research slate

This is an experimental walk-forward runner, separate from live deployment. It
tests three mechanisms for predicting information missed by a historical teacher
ensemble. The teachers are ridge, ExtraTrees, and shallow LightGBM; their
per-era Gaussianized predictions are equally weighted.

- **Stable portfolio:** rank features within each era, measure residual-target
  correlation across eras, and select features whose direction persists across
  chronological blocks. Reject strong mean absolute exposure to the teacher
  consensus. Average signed feature ranks; do not optimize portfolio weights.
- **Factorization machine:** standardized linear terms plus low-rank pairwise
  interactions, trained in mini-batches using Adam and L2 regularization.
- **Anti-consensus LightGBM:** a shallow nonlinear learner trained on the same
  historical consensus residuals. This isolates the value of the residual target
  from the value of a new architecture.

An equal-weight blend of their Gaussianized predictions is evaluated alongside
each teacher and the teacher consensus. No ensemble weights are selected on the
evaluation windows. A portfolio with no qualifying features fails explicitly;
the runner does not silently relax its criteria.

## Running

From the repository root, using the existing `numerai` conda environment:

```bash
conda run --no-capture-output -n numerai python scripts/run_diversifiers.py configs/diversifiers_smoke.yaml
conda run --no-capture-output -n numerai python scripts/run_diversifiers.py configs/diversifiers_research.yaml
```

The smoke config tests two small windows using 42 features and 64 rows per era.
The research config uses three 30-era windows, a seeded subset of 128 medium
features, and 256 rows per era. These are affordable exploratory settings, not
full-universe diagnostics. Features are selected without looking at labels.
Features are read in parquet batches, but the selected int8 sample is retained
in RAM. Increasing feature/row limits increases memory and training cost.

An existing output directory causes an error rather than overwriting a run.
For another run, choose a new `name` in a copied config. A failed run keeps its
partial artifacts and a manifest marked `running`; only successful completion
marks it `complete` and appends an experiment record.

## Chronology and interpretation

Each outer window uses only labels from eras strictly more than 13 eras before
its start (5 for a 20-day target). Training-era subsampling does not change this
calendar-era embargo. Inside that historical training set, expanding teacher
fits generate out-of-fold predictions with the same embargo. Warm-up rows have
no out-of-fold prediction and cannot train a residual learner. Targets are
centered and projected off the Gaussianized out-of-fold consensus within each
era. All feature selection and learner fitting happen before the outer window.

The last 52 resolved validation eras and their preceding embargo are reserved:
the runner rejects test windows touching either. Later outer windows may train
on labels from earlier test windows once the embargo has elapsed, as would a
scheduled historical retraining process.

Reports include CORR, actual MMC only where meta-model predictions exist, BMC,
contribution against the historical teacher consensus, per-era scores, and
prediction correlations. The consensus contribution is a research proxy; it is
not actual MMC. Contribution against the consensus itself is omitted.

The chosen historical windows have been used by previous research. A correct
walk-forward computation does not restore their status as unseen data. Small
sample scores, especially the smoke run, are integration checks and cannot
support deployment or claims of expected live performance. There is no claim
that low correlation alone is useful. The existing production models are not
used as historical baselines because some have trained on these eras.

## Artifacts

`artifacts/<name>/` contains:

- `manifest.json`: exact feature order, config, git version, embargo boundaries,
  inner-fold audit, selected portfolio features, runtime, and completion state.
- `window_<era>.joblib`: fitted candidate models with feature order and audit.
  Load only trusted locally generated joblib files. The portfolio needs complete
  per-era groups to calculate cross-sectional ranks at prediction time.
- `oof_<era>.parquet`: row IDs, era, teacher consensus, and residual training label.
- `validation_preds.parquet`: sampled test row IDs, targets, references, and every
  model's predictions, with window labels.
- `metrics.json`, `per_era.csv`, `prediction_correlations.csv`: aggregate and
  window-level performance, individual era scores, and diversity diagnostics.

Successful runs append to `experiments/log.jsonl`. This runner does not package
hosted models or change `configs/live.yaml`; promotion requires a separate
full-universe evaluation and serving integration.

The initial `diversifiers_smoke_v1` integration run accidentally excluded the
original training split because of its `data_type` filter. Its record is retained
for audit, but must not be used for comparisons. `diversifiers_smoke_v2` includes
both splits and is the corrected integration run. A regression test covers this.

## First research run — 2026-10-13

`diversifiers_research_v1` completed in 117 seconds on 297,472 sampled rows and
128 features. Evaluation covers eras 850–879, 1000–1029, and 1133–1162, with 256
sampled rows per era. Only the final window has meta-model predictions. Thus CORR
below covers 90 eras, while MMC covers 30; they describe different periods.

| Candidate | CORR (90 eras) | MMC (30 eras) | Teacher-consensus contribution (90 eras) |
|---|---:|---:|---:|
| Stable portfolio | 0.00401 | -0.00640 | -0.00148 |
| Factorization machine | 0.00801 | 0.00021 | 0.00358 |
| Anti-consensus LightGBM | 0.01272 | -0.00215 | 0.00441 |
| Equal-weight ensemble | 0.00991 | -0.00461 | 0.00277 |

These sampled results do not establish useful MMC. The factorization machine and
anti-consensus learner have positive contribution to the teacher consensus in
each window, but this does not translate into convincing actual MMC. In the
latest window their CORRs are -0.00652 and 0.00664 respectively. The stable
portfolio contributes negatively against the teachers in all three windows.

The current teacher ensemble is also weak: its sampled MMC is -0.01101. Beating
it does not establish that a candidate improves a strong meta model. The next
research step is a larger cross-sectional sample with stronger historical
teachers and fixed settings, before considering more architecture tuning. The
historical holdout remains excluded from this experiment.
