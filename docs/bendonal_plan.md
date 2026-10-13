# dc_bendonal — additive nonlinear regression

Draft research plan, 2026-10-13. Named after Bendonal the Outlaw, a supporting
warmage in Spellmonger ([character listing](https://spellmongershall.com/characters)).
This is a new candidate for the fifth model. Wenek stays shelved, and the four
running models, their configurations, artifacts and hosted packages stay intact.
This document authorizes no automatic deployment and records no completed runs.

## Hypothesis

Each discretized feature may have a useful nonlinear response: for example,
extreme bins may behave differently from the middle. Ordinary ridge assumes a
linear response; Wenek tried pairwise interactions and failed the stability
screen. Bendonal will learn a separate small curve for each feature and sum the
curves, with no feature-pair interactions. It may improve generalization while
retaining signal beyond linear ridge. Distinct architecture alone does not imply
positive MMC, and this approach is not claimed to be novel within Numerai.

Model: prediction = intercept + sum of individual feature response curves.
For a feature with values 0–4, a five-bin lookup can represent any such curve.
Implement that lookup as one linear term plus three nonlinear contrast terms;
the constant is absorbed by the global intercept. This gives four effective
parameters per feature and makes the linear-only control explicit.

## Phase 1: implementation and controls

1. Verify the selected data's values and missingness. Explicitly reject unknown
   bins at inference. Define any missing-bin policy from training data before
   evaluation; never cast missing values silently to a valid integer bin.
2. Construct a fixed orthonormal nonlinear contrast basis over bins 0–4,
   orthogonal to a constant and the linear bin vector under equal bin weights.
   The basis does not depend on targets or evaluation data. Center the resulting
   columns using training means only. Do not claim empirical independence between
   slope and curvature: actual bin frequencies need not be uniform.
3. Fit penalized least squares with an unpenalized intercept. Use the existing
   ridge convention (penalty multiplied by training-row count), with independent
   penalties for slopes and nonlinear coefficients. Stream sufficient statistics
   or use a bounded-memory solver; do not form a dense all-row one-hot matrix.
4. Predict raw scores in batches, then rank across complete eras. Save exact
   feature order, basis, centering, coefficients, solver settings and training
   boundaries in a separate dc_bendonal artifact bundle.

Start with 128 features: 512 coefficient columns make a streamed Gram matrix
practical. Do not automatically extend dense normal equations to all 3,555
features (14,220 columns); benchmark memory and choose a sparse/iterative solver
before a full-feature experiment.

## Phase 2: frozen initial experiment

Reuse Wenek's label-free selection of 128 medium features, seed 42, nested
1,024-row training samples, every-fourth-era training cadence, and full-era
evaluation windows 850–879, 1000–1029, 1133–1162. Use a 13-calendar-era embargo.
Require exact row/feature parity against the saved baseline where available.

Fit the original target_ender_60 directly. Do not introduce residual targets,
feature selection by performance, or fitted ensemble weights in this first test.

| Candidate | Linear penalty | Nonlinear penalty | Purpose |
|---|---:|---:|---|
| Existing ridge control | 10 | No nonlinear terms | Exact baseline |
| Bendonal moderate shrinkage | 10 | 10 | Test additive curvature |
| Bendonal strong shrinkage | 10 | 100 | Test conservative curvature |

These are three predefined candidates, not the start of a regularization search.
Penalties act on the documented basis; equal numeric penalties across different
feature representations do not imply equal model complexity. Include a
linear-only implementation parity check to isolate solver/preprocessing errors.
Any later change to this matrix requires a new, recorded hypothesis.

## Phase 3: correctness tests

- Enumerate the five bins to verify basis dimensions, orthogonality, and exact
  lookup reconstruction. Check handling of absent training bins and invalid input.
- Recover synthetic linear and U-shaped signals on independent test samples;
  ensure the additive model cannot represent a pure interaction of independent,
  centered inputs without marginal signal.
- Disable nonlinear terms and reproduce the existing RidgeModel's fitted values
  and predictions using the same centering and penalty convention.
- Verify stronger penalties shrink coefficient norms on a controlled fixture.
- Verify streamed fitting against an in-memory small reference fit, including
  intercept handling and multiple input batches.
- Perturb future labels and confirm earlier fits and predictions do not change.
  Assert embargoes and exclude the historical holdout and its preceding gap.
- Check row-ID/feature-order alignment, serialization, raw batch parity, full-era
  rank parity, and submission range, finiteness, IDs and ties.
- Run the repository tests and lint; verify hashes of protected production files
  before and after the experiment. Changes remain confined to research code.

## Phase 4: evaluation and stopping rules

Report per-era and per-window CORR, actual MMC where available, contribution
beyond the independently fitted ridge baseline, teacher-consensus contribution
as a secondary proxy, and prediction correlations. Inspect the nonlinear
component alone as a diagnostic, but do not select it as another submission
candidate after seeing its scores. Also report training/test gaps, coefficient
norms, feature curves, runtime and peak-memory limitations.

Report paired differences versus ridge and 13-era block-bootstrap intervals
separately within each contiguous window. Ninety eras have CORR, only thirty
have MMC, and thirty eras provide about 2.3 bootstrap blocks. Never bridge gaps
or present the three development periods as fresh independent confirmation.

A configuration passes the initial research screen only if all of these hold:

1. Positive mean contribution beyond ridge in at least two of three windows.
2. Better mean CORR than ridge in at least two windows and better pooled CORR.
3. Nonnegative MMC, with MMC no worse than ridge, in the available MMC window.

If neither passes, shelve the approach without adjusting settings repeatedly on
the same windows. If a candidate passes, select using the count of positive
contribution windows, then the median per-window contribution (tie-break: stronger
regularization). This is an exploratory screening rule, not statistical proof.

Repeat the selected candidate and ridge at 2,048 training rows per era with all
other settings fixed and complete test eras. Require the same screen to pass.
Do not expand the feature set, introduce partial residualization, or fit blend
weights during this replication. Those would be separate later experiments.

## Phase 5: confirmation and conditional deployment preparation

The 52-era historical holdout was used in earlier research and remains excluded.
The three development windows have also been examined repeatedly. Before a final
confirmation run, identify genuinely unexamined resolved eras, record their
exact boundaries and availability, freeze the selected settings, and fit only
eligible earlier data. Do not invent a fresh holdout by relabeling old eras.

When enough unexamined eras are available for meaningful evaluation, compare
against a contemporaneously trained ridge baseline using the same metrics and
embargo. Do not tune on the confirmation result. If no fresh data are available,
report that limitation and retain a research-only candidate.

Only after satisfactory confirmation: propose a final fit and separate hosted
bundle, test local/hosted parity on a complete cached live round, and prepare an
unstaked live-trial handoff. Do not register, upload, stake, or modify any running
model as an automatic consequence of historical research.

## Deliverables

- A dedicated additive model implementation, unit tests and frozen configs.
- Separate artifacts containing source/config hashes, feature and training-row
  provenance, coefficients, full-era predictions, per-era scores, confidence
  intervals, and an explicit continue/shelve decision.
- A concise research report explaining what worked or failed, with a clear
  distinction between observed performance and claims about future live MMC.

First milestone: passing correctness tests and the three-candidate 1,024-row
comparison. No Bendonal training has been run as part of drafting this plan.
