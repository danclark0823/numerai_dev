# dc_mavone — multi-target ensemble forward-validation plan

Status: proposed, not executed. Revised 2026-10-14 after auditing the original
proposal and the repository's existing research. This document authorizes no
training, data refresh, deployment, or changes to the four live models by itself.

## Name and implementation handoff

Research model name: **dc_mavone**, after Mavone, a supporting warmage in the
Spellmonger series ([character reference](https://spellmongershall.com/characters)).
This is a separate research ensemble, not a rename or replacement of a live model.

Cursor handoff: use this document as the implementation specification. Start
with Phases 0–3 and stop for review at that boundary; preserve the conditional
MMC-history gate and all deployment approval requirements. Suggested new entry
points are `src/numerai_dev/mavone.py`, `scripts/run_mavone.py`,
`configs/mavone_research.yaml`, and `tests/test_mavone.py`. Use unique artifact
names prefixed `dc_mavone_`; never overwrite completed runs. No implementation
or training has been performed as part of preparing this plan.

## Objective

Test two separate hypotheses:

1. A fixed, small ensemble of different-target LightGBM models provides useful
   complementary signal beyond a matched primary-target model.
2. Conservative weight selection improves forward MMC beyond equal weighting.

Neither hypothesis is assumed true. More models, lower prediction correlation,
or better selection-period performance are not sufficient evidence of improvement.
If learned weighting fails, retain equal weighting only if hypothesis 1 passes.
If both fail, retain the existing models and document the result.

## Scope and prior evidence

Reuse the existing data loaders, training code, official scoring wrappers,
prediction artifacts where their provenance permits, and experiment tracking.
Do not redesign the project or change the four running models.

This is not our first target-ensemble experiment:

- Earlier LightGBM research tested eight auxiliary targets. The deployed
  `dc_resid_ensemble` combines benchmark-residual Ender and Teager2b predictions.
  Equal weighting over the entire earlier candidate collection diluted results.
- Azar's short-window selection among alternate targets did not generalize.
  Its outer diagnostics cannot now be treated as an untouched selection set.
- Hesia showed a substantial gain from broader feature coverage, but its
  specialist ensemble lost to matched all-feature ridge.
- Taren and Carmella provide historical recreations of existing models that may
  be useful controls after their exact training/evaluation provenance is checked.

References: [earlier results](results.md), [Azar](azar.md), [Hesia](hesia.md),
[Taren](taren.md), and [Carmella](carmella.md).

## Phase 0 — Data sufficiency and provenance gate

Before training, create an inventory of:

- Dataset version, file hashes, acquisition dates where known, feature order,
  target columns, target maturity and missingness, and available benchmark/meta
  predictions. Treat the target alias as an alias, not an extra target.
- Prediction artifacts: row IDs, era coverage, training endpoints, targets,
  feature sets, preprocessing, hyperparameters, code versions, and whether any
  selection or early stopping used their evaluation labels.
- All periods already used for training, model selection, diagnostics, or
  human inspection. Mark them as development evidence, not fresh confirmation.
- Availability evidence for historical labels and meta-model predictions.
  A current file containing old values does not establish when they became
  available. Where point-in-time provenance is absent, label results as a
  retrospective forward-label simulation, not an exact historical live replay.

The audit found 40 named targets and 95 resolved eras with local meta-model
coverage, 1133–1227. Only 30 of those were in our recent development windows;
the former 52-era holdout has already influenced previous decisions. These
counts describe the audited local snapshot, not a guarantee about current remote
availability. Inspect whether a newer snapshot is available when execution is
approved; preserve old data and hashes rather than silently overwriting them.

Deliverable: an availability table and explicit split calendar. If no genuinely
unused evaluation period exists, record that limitation and reserve future
observations before inspecting their outcomes. Do not rename an old holdout.

## Phase 1 — Freeze four targets and a matched recipe

Use explicit `target_ender_60` as the scoring target. Confirm the tournament's
current target and scoring rules at execution time; do not silently mix scoring
regimes or compare Ender-20 leaderboard history with Ender-60 backtests.

The proposed initial collection is:

| Training target | Role and rationale |
|---|---|
| `target_ender_60` | Primary-target anchor and single-model control |
| `target_teager2b_60` | Established auxiliary in our existing target ensemble |
| `target_jasper_60` | Auxiliary with prior evidence of complementary signal; not a fresh discovery |
| `target_rowan_60` | Additional target formulation already supported by our pipeline; usefulness remains unproven |

This shortlist is informed by prior research, including inspected results. Do
not claim it was chosen independently of that history. Before generating new
evaluation predictions, audit coverage, per-era distributions and pairwise
target rank correlations using only the designated historical training period.
These statistics explain the shortlist; they do not justify searching the
outer periods for replacements. If a target is absent or incompatible, pause
and revise the plan explicitly before training. Do not quietly substitute a
target because its observed evaluation performance looks better.

Use all 3,555 features from the pinned v5.3 metadata. If the dataset version or
feature universe changes, document and freeze a revised experiment first.
Use the existing all-feature LightGBM recipe uniformly:

- 2,000 trees, learning rate 0.01, max depth 5, 31 leaves.
- Feature fraction 0.1, seed 42; pin library versions and thread settings.
- Original target labels, no benchmark residualization in these four members.
- Every fourth eligible training era, with all eligible rows in those eras.
  Fix the stride anchor globally; do not shift it separately for each target.
- Identical preprocessing, training endpoints, feature order, and training IDs.
  Use the joint nonmissing-target intersection for the initial comparison and
  report exclusions by target and era. Do not silently fill missing labels.
- No outer-window early stopping or hyperparameter tuning.

Run a small integration test first, then one representative full-feature fit
to estimate elapsed time and memory. Limit concurrency to the machine's actual
capacity. Do not extrapolate tiny 128-feature timings to full-feature training.

## Phase 2 — Generate reusable forward predictions

Use chronological forward fits with explicit label-availability cutoffs.
At each prediction block, train only on historical rows whose selected labels
would have been available before that block began. All four targets are 60-day
targets to simplify the initial schedule.

The current code uses a conservative 13-weekly-era gap: train eras must satisfy
`era < prediction_start - 13`. Confirm the calendar/maturity mapping before
execution; use actual availability dates where available, and document any
additional exclusions required by publication delays.

Choose a fixed 26-era prediction-block/retraining schedule in the Phase 0
calendar. Use the same schedule for all four models. A model fitted at the
start of a block remains fixed throughout that block.

For every prediction, retain its row ID, era, model/target, model training
endpoint, maximum training-label availability date or documented approximation,
and artifact hash. Predict complete evaluation eras on an identical row universe.
Use explicit joins and completeness assertions; never silently inner-join away
missing component predictions. Save raw predictions as well as derived blends.

Cached predictions can be reused only if this provenance is established.
Current production artifacts trained on evaluation eras are not valid controls.

## Phase 3 — Diversity and simple ensemble evaluation

Normalize each member with the same tie-preserving per-era Gaussian-rank
transformation, over the complete era. Form the arithmetic weighted blend, then
apply the same final submission ranking used at inference. Numerical batching
must not change the normalization universe.

Primary comparisons:

1. Matched Ender-60 LightGBM alone.
2. Equal-weight four-target ensemble: 0.25 per member.
3. Leakage-safe recreation of the existing two-member target ensemble, where
   feasible. Label it a contextual recipe comparison because residual labels
   and feature neutralization differ from the controlled four-target experiment.

For each member and blend, report per-era CORR, available MMC and diagnostic
benchmark contribution; prediction rank correlations; per-era score correlations;
and fixed incremental-addition diagnostics. Use paired common-era comparisons.
Do not select members or tune weights from outer diagnostic results.

For the controlled equal-weight hypothesis, require positive mean CORR,
nonnegative available MMC, positive mean contribution beyond the matched primary
model, and positive paired mean MMC improvement over that model. Report the
uncertainty and consistency across periods, not just the aggregate pass/fail.
A historical pass advances research; it does not justify deployment.

Deliverable: target rationale/coverage table, individual and ensemble metrics,
correlation matrix/heatmap, per-era predictions, and compute-cost report.
Stop for review after Phases 0–3. Later phases are conditional, not automatic.

## Phase 4 — Gate MMC weight selection on usable history

Exact retrospective MMC can be computed only on aligned, mature target rows
with valid meta-model predictions. Use the official `numerai-tools` scoring
implementation and pin its version. Benchmark contribution is not MMC, and
diagnostic benchmark contribution is not necessarily the live benchmark metric.

Initial operational minimum for this experiment:

- 104 mature weekly eras of forward component predictions and MMC references
  available for the first weight-selection window.
- A 13-era maturity gap before evaluation, extended if availability requires it.
- At least three subsequent, nonoverlapping 26-era forward evaluation blocks.

Under the weekly-gap approximation, that needs at least 195 contiguous eras of
usable coverage for the first complete development exercise, before reserving
fresh confirmation. This is an operational floor, not a statistical guarantee
or a universal rule about sample size. Dependence and selection still matter.

The audited 95-era snapshot fails this gate. If coverage remains insufficient,
do not shorten the window after seeing scores, optimize on the old 30 eras, or
substitute BMC while claiming MMC optimization. Continue equal-weight research
and preserve prospective predictions; defer learned weights. Any separately
authorized proxy-weight experiment must be labeled as such and cannot satisfy
the MMC hypothesis.

## Phase 5 — One conservative weighting procedure

Avoid unconstrained continuous optimization in the first experiment. Use exactly
five predefined weight vectors:

- Equal weighting: `[0.25, 0.25, 0.25, 0.25]`.
- Four modest tilts, one toward each target: its weight is 0.40 and each other
  weight is 0.20. No negative weights, no dropped members, no weight above 0.40.

These tilts are a restricted, strongly regularized search around equal weights.
They are not a claim to find the global optimum. Do not add best-three searches,
different risk objectives, concentration grids, or window-length grids to this run.

For each outer evaluation block:

1. Take the latest 104 fully available historical eras of forward predictions,
   ending before the outer block's maturity gap.
2. Construct and officially score the actual blended predictions for each of
   the five vectors, per complete era. The blend's MMC is not generally the
   weighted mean of its members' MMC because scoring normalizes the blend.
3. Reject tilted candidates whose mean CORR is below 90% of equal weighting's
   positive mean CORR. If equal weighting has nonpositive CORR, retain it as the
   fallback and flag the window as lacking a viable anchor.
4. Among eligible candidates maximize `mean(MMC) - 0.25 * std(MMC)` using
   sample standard deviation. The multiplier is fixed, not tuned on evaluation.
5. A tilt must strictly beat equal weighting's objective; otherwise use equal
   weights. Break equal objective ties in favor of equal weighting, then the
   fixed target order in Phase 1.
6. Freeze the weights before evaluating the next 26-era block. Repeat without
   changing any selection rule. Evaluate equal weighting on exactly the same eras.

Risk adjustment expresses a preference for steadier scores; it is not protection
against selection bias. The untouched forward comparisons provide that check.
Later folds may use earlier evaluation labels only once those labels mature,
as specified by the frozen rolling procedure.

## Phase 6 — Robustness and research decisions

Report for the concatenated forward predictions and each contiguous period:

- CORR and MMC means, standard deviations, positive-era fractions, worst eras,
  Sharpe-like ratios, cumulative scores and drawdowns.
- Diagnostic benchmark contribution separately, with the reference identified.
- Paired differences against equal weights and the primary-target model.
- Prediction correlations, weight histories, selection frequencies and turnover.
- Sensitivity to removing any one evaluation block, strictly as a diagnostic,
  not permission to redesign the strategy using the most favorable subset.

Use paired moving-block bootstrap comparisons with at least 13-era blocks;
report a predeclared 26-era-block sensitivity check where history permits.
Do not bridge gaps between disjoint windows or bootstrap stocks as independent
observations. Report actual era coverage and block counts. Do not annualize
Sharpe-like ratios assuming overlapping weekly outcomes are independent.

The learned-weight hypothesis advances only if forward mean MMC exceeds equal
weighting, CORR remains at least 90% of a positive equal-weight CORR, and the MMC
difference is positive in at least two of the three initial evaluation blocks.
If improvement is marginal, uncertain, or unstable, prefer equal weights.
These are development screens, not proof or authorization for deployment.

Do not report cumulative MMC as an investment return. Comparisons with other
Numerai models must match metric, dates, target horizon, and live-vs-backtest
status; leaderboard reputation is not directly comparable with our short windows.

## Phase 7 — Fresh confirmation and approval

Before observing confirmation outcomes, freeze the entire selected procedure:
targets, feature set, sampling, base-model recipe, retraining schedule,
normalization, weight-update rule, and decision criteria. Any prescribed weight
updates during confirmation may use only labels already available at that time.

Reserve at least 52 genuinely unused resolved weekly eras for confirmation,
with appropriate maturity separation from selection. This is a minimum planning
duration, not a guarantee of enough evidence; if uncertainty remains, do not claim
success or extend the window selectively until it looks favorable.

To recommend a learned-weight ensemble, require positive confirmation MMC,
positive paired MMC gain over equal weighting with a 95% paired block interval
above zero, and the fixed CORR floor. For equal weighting, require the analogous
gain over the primary model and positive CORR. Report contextual existing-ensemble
comparisons as well; unresolved performance conflicts require review.

If confirmation is unavailable, label the result exploratory and stop before
deployment. The already-inspected historical holdout cannot satisfy this step.

Only after confirmation and explicit user approval: refit the selected components
on eligible data, package the ensemble, verify inference parity, and generate a
submission file. Uploading, live configuration changes, and replacing any of the
four running models require separate approval. Prospective live submissions also
require approval; saving research predictions locally does not imply deployment.

## Testing and retained artifacts

Required tests/checks:

- Future-label perturbations cannot change earlier fits, target decisions,
  selected weights, or predictions.
- Both base-model and weight-selection maturity boundaries are enforced.
- All components use the same intended training IDs and complete evaluation IDs;
  duplicate, missing, reordered, and misaligned rows are handled explicitly.
- Zero/missing meta-model coverage disables MMC optimization rather than silently
  scoring another reference. Null targets and partial-era coverage are reported.
- Scoring matches pinned official functions, including ties and final blending.
- Equal-weight and tilted-vector formulas, simplex constraints, CORR floor,
  fallback behavior, and deterministic tie-breaking are covered by tests.
- Serialization, reordered feature columns, different numerical batch sizes,
  and research/inference paths reproduce predictions and valid submissions.
- All protected production files retain their before/after hashes.

Retain configs, code/library/data hashes, exact target and feature lists,
training/split/provenance manifests, raw and normalized forward predictions,
candidate selection-period scores, frozen weights, per-era outcomes, confidence
intervals, decisions, runtime/memory observations, and a human-readable report.
Never overwrite a completed run. Record failed and null results as well as wins.

## Execution boundary

When implementation is approved, execute Phases 0–3 first. Present the audit,
frozen experiment design, simple-ensemble findings, and the MMC-history gate
result before requesting approval to proceed. Do not claim improvement until
the relevant hypothesis survives genuinely forward evaluation, and do not claim
production readiness without fresh confirmation.

Official references, to recheck at execution:

- [Numerai dataset and meta-model coverage](https://numer.ai/data)
- [Scoring targets and timelines](https://docs.numer.ai/numerai-tournament/scoring/)
- [MMC calculation and diagnostic BMC distinction](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc)
