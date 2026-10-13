# dc_hesia — feature-specialist ensemble results

Status: tested and shelved, 2026-10-13. Positive MMC, but no advantage over the
matched all-feature ridge control. No deployment or changes to the four live models.
Named for [Hesia](https://spellmongershall.com/characters), a supporting warmage.

## What we tried

Sixteen ridge specialists, each seeing 444–445 of the 3,555 available features.
Two seeded random eight-way partitions give every feature exactly two memberships.
Fit each on the same sampled rows and original Ender60 target, normalize member
predictions over complete eras, and average equally. Alpha=10 for every member
and both controls; partitions and weights were fixed without looking at results.
The [frozen plan](hesia_plan.md) specifies the hypothesis and stopping rules.

Controls were the exact prior 128-feature ridge and an all-feature ridge. All
three approaches used 1,024 rows per sampled training era, every fourth era,
with a 13-era embargo and complete outer-era evaluation. We did not residualize
targets or use any already-deployed model as a historical prediction baseline.

## Results

Run `artifacts/dc_hesia_1024_v1` completed in 109.3 seconds and evaluated 563,466
rows. Peak RSS was 7.22 GB, which can understate macOS compressed physical footprint.

| Model | CORR, 90 eras | MMC, latest 30 eras | Worst-window CORR |
|---|---:|---:|---:|
| 128-feature ridge | 0.006146 | -0.004378 | -0.003338 |
| All-feature ridge | 0.025220 | 0.007350 | 0.021066 |
| Hesia specialists | 0.022130 | 0.005688 | 0.019024 |

| Window | 128-feature CORR | All-feature CORR | Hesia CORR | Hesia contribution beyond all-feature ridge |
|---|---:|---:|---:|---:|
| 850–879 | 0.007763 | 0.025797 | 0.020844 | -0.002886 |
| 1000–1029 | 0.014012 | 0.028799 | 0.026522 | 0.001292 |
| 1133–1162 | -0.003338 | 0.021066 | 0.019024 | -0.001073 |

Hesia's mean per-era prediction rank correlation with all-feature ridge was
0.9025. It contributed positively beyond that control in only one window and
lost CORR in all three. Although its MMC was positive, it was 0.001662 below the
all-feature control. The exploratory paired 13-era block-bootstrap interval for
this MMC difference was [-0.004131, -0.000178]. These repeatedly inspected windows
and roughly 2.3 blocks per window are not fresh statistical confirmation.

## Decision and interpretation

Shelve this fixed specialist formulation: it failed the predefined requirement
to outperform all-feature ridge. Beating the small control alone was explicitly
insufficient. No subset replacement, blend tuning, alpha search, larger fit,
hosted package, or upload followed.

The useful finding is that all-feature ridge substantially beat the 128-feature
control on the same rows, targets and periods. Within this experiment, broader
feature coverage supplied useful signal; dividing it among specialists did not
add enough value. This supports treating input coverage as a limitation of the
earlier small-feature experiments. It does not prove those discarded architectures
would succeed with more features or imply that all feature-group ensembles fail.

The all-feature control is not a newly validated fifth model: we already run
an all-feature residual-target ridge (Rondal), and this control uses original
targets. There was no matched comparison to a leakage-safe historical recreation
of Rondal. Its live fitted artifact must not be scored retrospectively on training
eras. All-feature versus specialist results here are development evidence only.

Actual MMC covers just eras 1133–1162. The existing 52-era historical holdout
remains excluded. No new holdout confirmation was attempted.

## Verification and reproduction

All 55 repository tests passed, as did lint. Tests cover exact seeded coverage,
incremental and marginal sufficient-statistic parity with independent ridge,
equal-weight full-era blend reconstruction, future-label exclusion, input checks,
serialization, batch-size parity, and named-column reordering.

All three real-data folds passed exact reference ID/era/label alignment, prior
128-feature ridge prediction parity, reload/batch parity, full-era submission
validity, feature-order checks and chronological embargo checks. All ten protected
production files retained identical hashes. The model computes normalized ranks
over complete eras, not independently over numerical prediction batches.

Artifacts retain source snapshots, metadata hash, configuration, exact feature
memberships, training IDs, serialized bundles, full-era predictions, per-era and
aggregate scores, paired intervals, and the failed decision. The entry points are
`src/numerai_dev/hesia.py`, `scripts/run_hesia.py`, and `configs/hesia_1024.yaml`.
Reusing the completed run name deliberately fails rather than overwriting it.
