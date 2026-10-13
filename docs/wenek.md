# dc_wenek

**Status: shelved after the fixed simplification follow-up, 2026-10-13. Never
deployed.** Baseline and both simplifications added value beyond ridge in only
one of three windows. Earlier proposed next steps below are historical context;
the final decision is to stop this formulation. The next additive-model proposal
is [dc_bendonal](bendonal_plan.md).

The outcome comments subsequently added to `src/numerai_dev/wenek.py` change its
file hash without changing executable code. Historical manifests retain their
original hashes. The simplification runner deliberately rejects source-hash
mismatches; do not bypass that check or rewrite old manifests to rerun it. Any
future replication should explicitly capture a new baseline and source version.

Wenek is a separate research model: ridge predictions plus a low-rank pairwise
interaction model trained on out-of-fold ridge errors. The original four live
models and their configuration are not inputs to, or outputs of, its fitting
process. Their files are hashed before and after each run to verify preservation.

The [frozen experiment plan](wenek_plan.md) specifies the hypothesis, training
settings, controls, time boundaries, and conditional scaling rule. The ordinary
FM and interaction-only FM share an optimizer, but the latter never updates
linear coefficients. Its intercept is allowed; per-era ranking removes constants.

## Execute

```bash
conda run --no-capture-output -n numerai python scripts/run_wenek.py configs/wenek_1024.yaml
# Only if artifacts/dc_wenek_1024_v1/decision.json passes the scaling screen:
conda run --no-capture-output -n numerai python scripts/run_wenek.py configs/wenek_2048.yaml
conda run -n numerai pytest -q
conda run -n numerai ruff check .
```

Existing output directories cause an error instead of overwriting results. Choose
a new run name in a copied config for any rerun. Sampling is reproducible and
nested: the 2,048-row sample contains the 1,024-row sample for each eligible era.
The training matrix includes every fourth era; test windows include every
resolved validation row in every era. Normalization uses the complete test era,
even when raw model predictions are calculated in smaller numerical batches.

## Saved outputs

Each run writes under `artifacts/dc_wenek_<sample>_v1/`:

- `manifest.json`: configuration, exact feature order, source hashes, timestamps,
  training boundaries, inner-fold audits, runtime, peak RSS, production hashes,
  and completion/failure state.
- `window_<era>.joblib`: three FM controls and ridge, plus features and fold audit.
  These are trusted local research bundles, not Numerai-hosted deployment files.
- `oof_<era>.parquet`: sampled training row IDs, targets, out-of-fold ridge
  predictions, and the exact residual labels used by the interaction model.
- `validation_preds.parquet`: full-era candidate predictions and scoring references.
- `metrics.json`, `per_era.csv`, and `report.md`: pooled and window-level results.
- `paired_vs_ridge.json`: paired differences and separate 13-era block-bootstrap
  intervals for each contiguous window. No interval spans a gap between windows.
- `prediction_correlations.csv` and `correlations_<era>.csv`: diversity diagnostics.
- `decision.json`: result of the predefined sample-size scaling screen. A passing
  screen authorizes a larger research sample, not production deployment.

Every fold is reloaded immediately after saving. Predictions are recomputed with
a different batch size and compared numerically; the zero interaction blend must
equal ridge exactly. The research submission path is checked for every complete
test era. Successful runs are appended to the existing experiment log.

## Interpretation

CORR covers 90 eras; actual MMC is available for only 30 of them. Compare paired
MMC and payout on identical eras, and inspect each window separately. A 30-era
window supplies only about 2.3 thirteen-era blocks, so the confidence intervals
are exploratory. The teacher ensemble is a proxy reference, not the Numerai
meta model. The current ridge control uses original targets and 128 features;
it is not the deployed residual-target, all-feature dc_rondal model.

The existing 52-era holdout has already influenced previous research and remains
excluded. These development periods are also previously examined. Fresh resolved
eras and a separate frozen evaluation are needed before proposing a production
candidate. Full-data refitting, hosted packaging, and live uploading are
conditional later steps, not automatic effects of this runner.

## First result — 2026-10-13

`dc_wenek_1024_v1` completed in 183.5 seconds, using 286,720 sampled training
rows across the outer windows and evaluating 563,466 full-era rows. Peak RSS
was 3.30 GB (not a measurement of compressed macOS physical footprint).
All three saved folds passed reload, numerical-batch, and submission checks.
Ten protected production files, covering configuration, model members, and
hosted bundles, had identical before/after SHA-256 hashes.

| Model | CORR, 90 eras | MMC, 30 eras |
|---|---:|---:|
| Ridge | 0.00615 | -0.00438 |
| Normal FM, original target | -0.00505 | -0.00929 |
| Normal FM, matched training rows | -0.00033 | -0.00761 |
| Interaction-only residual model | -0.00410 | 0.00371 |
| Ridge + 10% interactions | 0.00553 | -0.00391 |
| Ridge + 25% interactions | 0.00416 | -0.00296 |

The interaction model contributed negatively beyond ridge in the first two
windows (-0.00380 and -0.00955), but positively in the last (+0.00455). Both
blends improved CORR and MMC in the final window and worsened CORR in both
earlier windows. Therefore the predefined scaling screen failed: only one
window supported incremental interactions, rather than at least two.

The final-window MMC difference between the interaction model and ridge was
+0.00809, with exploratory paired 95% block-bootstrap interval
[-0.00073, +0.01071]. The interval includes zero; the two blend intervals also
include zero. Positive MMC in that single period does not establish robustness.

No 2,048-row run, full-data production refit, hosted package, or upload was
performed. The 2,048 config remains available for a future explicitly revised
experiment. The current version is a functioning research implementation, not a
validated fifth live model. Further work should investigate the unstable
interaction component through a separately specified regularization/capacity
experiment, rather than selecting the one favorable historical window.

## Fixed simplification follow-up

The next experiment is specified in [wenek_simplify_plan.md](wenek_simplify_plan.md).
It compares the saved original interaction model with rank 2 (instead of 8) and
penalty 0.1 (instead of 0.01), changing only one setting per variant. Run with:

```bash
conda run --no-capture-output -n numerai python scripts/simplify_wenek.py configs/wenek_simplify.yaml
```

The runner reuses exact saved residual labels and checks training IDs, source
hashes, baseline prediction parity, embargoes, and model reload parity. Outputs
are saved separately in `artifacts/dc_wenek_simplify_v1/`, including training versus
later residual diagnostics, per-window scores, paired confidence intervals, and
the predefined continue-or-shelve decision. Training diagnostics are in-sample
for the interaction learner despite using out-of-fold ridge residual targets.

### Simplification outcome — 2026-10-13

The fixed follow-up completed in 27.1 seconds. All baseline/residual alignment,
embargo, reload and batch-parity checks passed; production hashes are unchanged.

| Interaction variant | CORR, 90 eras | MMC, 30 eras | Windows contributing positively beyond ridge |
|---|---:|---:|---:|
| Original: rank 8, penalty 0.01 | -0.00410 | 0.00371 | 1/3 |
| Fewer factors: rank 2 | -0.00594 | 0.00186 | 1/3 |
| Stronger regularization: penalty 0.1 | 0.00118 | 0.00401 | 1/3 |

Stronger regularization reduced the harmful predictions in earlier windows and
improved pooled CORR. However, its contribution beyond ridge remained negative
in both earlier windows (-0.00035 and -0.00160), versus positive in the latest
(0.00473). Reducing rank did not help. All three variants fail the predefined
requirement of positive incremental contribution in at least two windows.

The original model's mean training residual correlations were 0.01465 and
0.01645 in the first two windows, falling to -0.00305 and -0.01152 on later data.
With stronger regularization these were 0.00230/0.00510 in training and
0.00131/0.00280 later. This is consistent with excessive fitting of unstable
interactions, but does not prove that overfitting rather than regime change is
the sole cause. Better raw residual Pearson correlations do not necessarily
mean positive contribution under Numerai's ranked scoring transformation.

**Decision: shelve this ridge-plus-factorization-interaction formulation.** Keep
the artifacts and the Wenek name for a different, separately specified approach;
do not scale, deploy, or continue tuning these variants against the same windows.
Detailed results remain in `artifacts/dc_wenek_simplify_v1/report.md`, with
per-era scores, training/test MSE diagnostics, and paired confidence intervals.
