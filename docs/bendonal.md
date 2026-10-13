# dc_bendonal — implementation and first result

Status: shelved; not selected as the fifth live model. The tested configuration
improved on ridge but failed the predefined nonnegative-MMC screen.
No larger-sample run, final refit, hosted package or
deployment was performed. The four existing live models are unchanged.

## Implementation

`src/numerai_dev/bendonal.py` implements five-bin additive response curves as
linear terms plus three fixed orthonormal nonlinear contrasts per feature. A
streamed, centered Gram matrix supports separate slope/curvature penalties and
an unpenalized intercept. Fitted curves, basis, training means and coefficients
are serialized together; prediction ranks are computed across complete eras.

The parquet loader validates selected feature values before converting to int8.
Missing values, fractional bins and out-of-range bins are rejected rather than
silently recoded. In this run all 834,826 loaded rows had valid finite bins, and
all five bins were present for every selected feature in every training window.

The [original plan](bendonal_plan.md) was followed with 128 fixed medium features,
1,024 rows per sampled training era, a training stride of four, and a strict
13-era embargo. Evaluations use every resolved row in the three test windows.
The historical holdout and its embargo were excluded.

## First experiment: dc_bendonal_1024_v1

Completed 2026-10-13 in 51.1 seconds, evaluating 563,466 rows across 90 eras.
Peak RSS was 2.90 GB; this can understate macOS compressed physical footprint.

| Candidate | CORR, 90 eras | MMC, 30 eras | Positive contribution beyond ridge |
|---|---:|---:|---:|
| Ridge control | 0.006146 | -0.004378 | — |
| Bendonal nonlinear penalty 10 | 0.007442 | -0.002517 | 3/3 windows |
| Bendonal nonlinear penalty 100 | 0.006296 | -0.004171 | 3/3 windows |

Both candidates improved CORR over ridge in all three windows. For penalty 10,
the window CORR improvements were +0.000467, +0.000696, and +0.002727. Unlike
Wenek's interaction component, additive curvature contributed positively beyond
ridge in every window.

However, the predefined screen also required nonnegative actual MMC. Neither
candidate met that requirement, so neither was selected for the 2,048-row
replication. The latest window's CORR remained negative as well (-0.000611 for
penalty 10). This is evidence of an incremental improvement to a weak baseline,
not a validated fifth live model.

Penalty 10 improved latest-window MMC over ridge by +0.001861, with an
exploratory paired 95% block-bootstrap interval of [+0.001497, +0.002964]. The
earlier-window CORR improvement intervals include zero. Each window supplies
only about 2.3 thirteen-era blocks and has been examined repeatedly, so these
intervals do not establish fresh statistical confirmation or future live MMC.

Training CORR for penalty 10 ranged from 0.0275 to 0.0313, above later-window
performance. Nonlinear coefficient norms and all per-feature curves are saved.
Curvature-only scores are diagnostic outputs, not additional candidates chosen
after looking at results.

## Verification

All 41 repository tests passed, along with lint. Coverage includes basis/lookup
reconstruction, exact linear-only ridge equivalence, synthetic linear and U-shaped
signal recovery, inability to fit a pure centered interaction, invalid/missing
bin rejection, absent training bins, streamed/reference fit parity, shrinkage,
future-label isolation, row/feature alignment, serialization and submission checks.

For each real-data fold, newly fitted ridge reproduced Wenek's saved baseline;
the additive model's linear-only mode reproduced that ridge. Serialized models
reproduced predictions with a different batch size, lookup curves reproduced
the same predictions, and complete-era submission validity checks passed. The
before/after hashes of the ten protected production files were identical.

## Artifacts and reproduction

The runner is `scripts/run_bendonal.py`, configured by `configs/bendonal_1024.yaml`:

```bash
conda run --no-capture-output -n numerai python scripts/run_bendonal.py configs/bendonal_1024.yaml
```

The completed output directory already exists, so this exact command refuses to
overwrite it. Any explicit replication needs a new run name. Saved outputs under
`artifacts/dc_bendonal_1024_v1/` include:

- Source snapshots and hashes, exact configuration/features, fold audits,
  protected-file hashes, runtime/memory measurements, and completion state.
- Training row IDs, labels and era boundaries for each fold, plus bin counts.
- Serialized candidate models and ridge controls, and per-feature lookup curves.
- Full-era predictions, metrics, paired comparisons, prediction correlations,
  training diagnostics, and the continue/shelve decision.

The tested configuration is shelved under the original rule. A different
target or formulation would require a separately recorded hypothesis; no
additional tuning, residualization, or feature expansion was done in this run.
