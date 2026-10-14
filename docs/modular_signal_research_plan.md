# Modular signal research: build useful pieces, then combine them

Status: proposed follow-on plan, 2026-10-14. Documentation only; no experiments
are authorized or started by this document.

## Objective

Find a small number of complementary predictive signals. A component does not
need to be the best standalone model; it must improve a fixed reference when
added at a predeclared weight. Low prediction correlation alone is not success.

Separate three questions:
1. Does a component predict anything useful?
2. Does it improve our reference ensemble?
3. Does that improvement survive genuinely uninspected data?

## 0. Do not interrupt Cursor's current work

The residual-capacity experiment described in
`docs/residual_capacity_ensemble_plan.md` remains its own frozen experiment.
Cursor is working on it; that file's status may lag execution. Do not infer that
it is safe to start another run from the document's status alone.

- Do not change its recipes, 50/50 blends, endpoints, purge, scoring, tests,
  manifests, checkpoints, or output directories.
- Do not launch competing training, data refreshes, or full-dataset reads while
  it is running. Do not modify shared code or dependency versions under its run.
- Preserve all existing uncommitted work. In particular, do not append to the
  shared experiment log from a second process.
- Leave the four production models, `configs/live.yaml`, and hosted artifacts
  unchanged. No submission, deployment, staking, commit, or push is implied.

For now, review this plan only. Implementation starts after Cursor reports its
current stopping point and the user approves the next bounded experiment.

## 1. Close the current experiment and establish the reference

First audit the completed capacity results and provenance. Answer separately:
did the larger tree recipe help, and did either 50/50 Rondal blend help its tree
and Rondal comparators? Preserve negative and inconclusive results.

Use the existing advancement rules, not new thresholds chosen after seeing
results. If the run is incomplete, stop here rather than selecting from partial
blocks. If no replacement qualifies, retain the current-recipe residual tree as
the research reference. If multiple outputs qualify, select and document one
reference in a short decision record before evaluating any new additions.

Freeze that reference for the entire next research batch. Selection on these
historical results is development selection, not independent validation.
Do not repeatedly replace the reference as new candidates arrive.

Record:
- Exact member recipes, weights, preprocessing, target, and fit schedule.
- Dataset, feature-order, training-ID, code, configuration, and artifact hashes.
- The eligible evaluation IDs and eras, and where true MMC is available.
- The full list of historical experiments and periods already inspected.

## 2. Build a small reusable prediction library

After current work finishes, reuse verified prediction artifacts where possible.
Do not retrain just to standardize filenames. Introduce a separate adapter and
new research namespace if needed; never rewrite completed run artifacts.

Each component needs an immutable manifest and predictions keyed by row ID,
era, outer block, and training cutoff. Save raw predictions and the exact
normalization recipe. Score full eligible cross-sections, not tiny row samples.

Require identical evaluation IDs, no duplicates, finite values, complete eras,
and explicit joins. Reject mismatches; do not silently inner-join away rows.
Only compare matched evaluation periods. Report MMC coverage separately from
CORR coverage, and never label benchmark/teacher contribution as actual MMC.

Use forward predictions with label-availability-safe cutoffs. Never evaluate
the deployed models retrospectively on eras included in their training.
Any learned feature selection, preprocessing, residual teacher, or blend weight
must be fit using only information available before its evaluation block.

Retain the current experiment's calendar for direct historical comparisons.
Before fresh confirmation, audit the target horizon, data delay, and purge;
document the conservative schedule chosen. If endpoints change, recreate both
reference and candidates under the new schedule in a new run. Do not mix
incompatible cached predictions. Existing benchmark residual labels carry the
documented non-point-in-time provenance limitation; do not erase that caveat.

## 3. Test one hypothesis at a time

The capacity run already covers stronger trees and a linear complement. Do not
repeat it or reinterpret a failed 50/50 blend by immediately sweeping weights.

The first follow-on batch contains at most TWO new components, run sequentially:

### Component A: alternate-target signal

Question: does one previously studied alternate target add information to the
frozen reference, rather than merely predict similarly to Ender?

- Default candidate: Mavone's existing Jasper recipe and verified forward
  predictions. This is a cheap reuse test, not a claim that Jasper is novel.
- Audit target identity, preprocessing, endpoints, and hashes before reuse.
  If incompatible, propose a matched refit and cost estimate before running.
- Evaluate standalone and at the fixed addition weight below. Do not sweep
  targets, horizons, target mixtures, or residual definitions in this batch.
- Review previous Mavone results first: this test asks about marginal value
  against the frozen reference, not whether the failed four-target blend wins.

### Component B: restricted-feature specialist

Question: does removing access to most features cause a tree to learn useful
relationships that the all-feature reference underuses?

- Use one feature group defined by the pinned dataset metadata, selected and
  recorded before scoring. Choose by a documented structural hypothesis, not
  historical target correlations or whichever group's backtest looks best.
- If metadata provides no defensible group, propose one fixed, seeded,
  label-independent subset instead; freeze the exact feature list first.
- Hold the tree recipe, label, training schedule, and scoring pipeline fixed
  to the tree component of the reference wherever applicable. Change only
  feature access. No feature-group sweep or performance-based feature selection.
- Low standalone accuracy is acceptable only if the fixed blend adds value.

Before each component: write a one-page experiment card with hypothesis,
recipe, comparator, hashes, compute estimate, maximum runtime, and stop rule.
Obtain approval for the bounded run. Tiny smoke runs verify plumbing only.

Defer additional architectures, adaptive regimes, recent-history models,
learned residual correctors, and weight optimization to a later decision.
Previous failed experiments are evidence to consult, not a queue to rerun.

## 4. Freeze the marginal-value test

For this new batch only, evaluate each candidate C as:

`blend = rank(0.80 * normalize(reference) + 0.20 * normalize(C))`

Normalize each complete era using average-rank Gaussianization and unit
standard deviation; reject constant predictions. Apply identical final ranking
to reference, component, and blend. Freeze tie handling and implementation.
The 20% allocation is an experimental design choice, not an estimated optimum.
Do not also try 10%, 30%, 50%, sign flips, or neutralization after seeing scores.
The current capacity experiment's 50/50 tests remain unchanged.

Primary comparison: blend minus frozen reference, paired by era, on actual MMC.
Also report standalone component, CORR, BMC, per-block results, worst era,
worst block, score variability, and prediction/score correlations. Report
13-era moving-block bootstrap intervals and a 26-era sensitivity check on the
existing 95-era development sample. These intervals are descriptive after
repeated research, not a correction for all historical model selection.

Development advancement screen:
- Blend mean MMC is positive and exceeds the reference.
- Paired MMC improvement is positive in at least three of four blocks.
- Blend mean CORR is at least 90% of a positive reference mean CORR.
- If the reference is nonpositive on mean CORR or MMC, stop and reassess the
  anchor instead of automatically promoting an addition to a weak baseline.
- Review worst-block and variability trade-offs explicitly; no automatic
  deployment even if the numerical screen passes.

Classify each result as rejected for this use, promising but uncertain, or
eligible for confirmation. An interval spanning zero means uncertainty, not
proof of benefit or proof of no effect. A failure applies to this recipe and
weight, not all possible uses of the model family. Log every attempted test.

## 5. Assemble only after component testing

If neither candidate passes, stop this batch. Do not force an ensemble.
If one passes, its existing 80/20 blend is the confirmation candidate.
If both pass, evaluate exactly one predeclared combination:

`combined = rank(0.60 * normalize(reference)
                 + 0.20 * normalize(A) + 0.20 * normalize(B))`

Compare against the reference and both tested 80/20 blends on matched eras.
Use the same advancement screen; if it does not improve all comparators, do
not add complexity merely because both components passed independently.
If one candidate is redundant with a member already in the reference, flag
that before testing rather than unknowingly double-counting it.

This batch permits two component comparisons and at most one combined
candidate, not a search over all subsets. Choose at most one finalist and
freeze its exact recipe before accessing confirmation outcomes.

## 6. Fresh confirmation, then prospective observation

Eras 1133–1227 have already been inspected. Forward fitting does not make them
fresh again. Do not score reserved eras 1228+ under this development plan.

Write a separate confirmation protocol before revealing any reserved labels:
- Audit which eras actually remain uninspected and which predictions were
  saved before outcomes became available. Do not assume a reserved label means
  a clean holdout if other work has already inspected it.
- Freeze one finalist, reference, scoring, availability/purge rules, sample
  size, decision thresholds, and one review date. Use development paired-score
  variance and serial dependence to justify the required number of eras and
  the smallest practically useful MMC gain before opening outcomes.
- Nine reserved eras alone are not persuasive confirmation; if insufficient
  fresh data exists, record the limitation and accumulate prospective data.
- Compare only the locked finalist and reference. No repeated peeking,
  finalist switching, or tuning on confirmation outcomes. A redesign consumes
  development data and requires new confirmation data.

Only after separate approval consider prospective shadow predictions, or a
separate Numerai model slot. Keep existing live models unchanged. Live partial
scores are monitoring observations, not final CORR60/MMC60 evidence. Deployment
and staking require separate decisions.

## 7. Required tests and deliverables

Tests before any substantive new run:
- Future-label mutation does not change earlier fitted predictions.
- Training cutoffs, purges, full-era normalization, and reserved-era guards.
- Duplicate/missing ID, nonfinite/constant prediction, and feature-order errors.
- Official scoring parity; deterministic blend formula and tie handling.
- Manifest mismatch/corruption rejection; immutable completed artifacts.
- Identical comparator coverage and no modification to production artifacts.

Deliverables per component: frozen experiment card, immutable predictions,
manifest, per-era and per-block scores, paired deltas and uncertainty, runtime,
and a short decision explaining what worked or failed. Negative findings stay
in the research record. Final batch report identifies the single next decision,
not a promise of leaderboard performance.

## Immediate handoff to Cursor

Finish the current residual-capacity task without changing its scope. Report
its results or approval checkpoint first. Then review this document and propose
the reference decision plus Component A's provenance check and compute budget.
Do not begin a new training run or access reserved outcomes until approved.
