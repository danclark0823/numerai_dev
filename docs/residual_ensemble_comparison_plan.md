# Residual ensemble comparison plan

Status: **proposed—not approved for execution.** This document authorizes no training,
data refresh, scoring of reserved eras, packaging, upload, or change to the four live
models (lgbm_bag4, dc_resid_ensemble, dc_tyndal, dc_rondal). Any resulting model would
be a separate research model. Thresholds marked **[needs approval]** must be confirmed
before execution.

## Research question

Does a fixed blend of benchmark-residual Ender and a leakage-safe Rondal recreation
improve on benchmark-residual Ender alone?

This question comes from [dc_mavone](mavone.md)'s contextual results. Those were seen on
inspected eras, so the question itself is shaped by earlier outcomes. No result on those
eras can count as confirmation.

## Comparisons

The controlled hypothesis compares only A and B. C is reported separately as context.

**A. Benchmark-residual Ender alone (reference).** LightGBM on `target_ender_60`,
residualized per era against gaussianized `v53_lgbm_ender60`. The recipe is the same as
Mavone's `ender_resid` member: all 3,555 features; 2,000 trees, learning rate 0.01,
depth 5, 31 leaves, colsample 0.1, seed 42; every row on eras with era % 4 == 1; zero
weight where the benchmark is missing (train eras < 158). Final form: per-era rank of the
raw prediction (equivalent to ranking its Gaussian rank).

**B. Fixed 50/50 blend of A and Rondal (candidate).** The Rondal recreation is ridge
(alpha = 10 × rows) on all 3,555 features and every eligible row. Its label is
`target_ender_60` residualized per era against both `v53_lgbm_ender60` and
`v53_lgbm_ender20` (eras ≥ 158). This is the dc_rondal recipe, fitted only at the
endpoints below. Blend procedure:

1. Over each complete era (every validation-file row in that era), transform A and
   Rondal by a tie-preserving Gaussian rank: average-rank ties, then
   `norm.ppf((rank − 0.5) / n)`.
2. Take 0.5 × A + 0.5 × Rondal.
3. Apply the final submission ranking: per-era `rank(pct=True)`, average ties, into (0, 1].

There is no neutralization, weight search, or other member. Batching must not change
the normalization universe. This is `mavone.blend(preds, {"a": 0.5, "rondal": 0.5}, eras)`.

**C. Existing residual-Ender + Teager2b recipe (contextual control).** Preserved exactly
as in `configs/live.yaml` and Mavone's `resid_ens`:

1. Gaussian-rank each member per complete era, as above. The members are A's model and a
   raw-label `target_teager2b_60` LightGBM with the same recipe and training rows.
2. Take the equal-weight mean.
3. Neutralize 50% against the `medium` feature set, per era (`live.neutralize`): center
   the features and predictions in float32, least-squares fit, subtract 0.5 × the
   projection.
4. Apply the final per-era `rank(pct=True)`.

C differs from live dc_resid_ensemble in stride (4 vs 1) and fit endpoints. It is labeled
contextual, because its residual labels, extra member and neutralization differ from the
A/B contrast.

## Evaluation calendar

Use the same fixed 26-era blocks and 13-era maturity gap as Mavone. Each block is
predicted by a fit on eras ≤ start − 14, and that fit stays fixed throughout the block.

| Block | Fit on eras ≤ | Stage | Eras |
|---|---:|---|---|
| 1133–1158 | 1119 | exploratory (inspected) | 26 resolved |
| 1159–1184 | 1145 | exploratory (inspected) | 26 resolved |
| 1185–1210 | 1171 | exploratory (inspected) | 26 resolved |
| 1211–1236 | 1197 | 1211–1227 exploratory; 1228–1236 reserved | 17 + 9 |
| 1237 onward | 1223, 1249, … | fresh confirmation (future) | not yet available |

Exploratory: eras 1133–1227, 95 eras, all with meta-model MMC. Every one was inspected
by earlier research (dev scoring, walk-forward windows, the 52-era holdout and Mavone),
so results there are **exploratory only**.

Reserved: eras ≥ 1228. Do not score eras 1228–1236, and do not read newer target or
meta-model values for them, before the candidate and the confirmation procedure are
frozen.

## Step 1 — Prediction and provenance audit (before any fitting)

An audit of existing artifacts on 2026-10-14 found:

| Source | Eras | Fit endpoints | Compatible with this calendar? |
|---|---|---|---|
| Mavone `ender_resid` (A) | 1133–1236, all rows | 1119, 1145, 1171, 1197 | Yes, if hashes verify (see below) |
| Mavone `teager2b` (C member) | 1133–1236 | same | Yes, same conditions |
| Taren/Carmella Rondal recreation | 850–879, 1000–1029, 1133–1162 | 836, 986, 1119 | Only eras 1133–1158 (endpoint 1119). Eras 1159–1162 used endpoint 1119, but this calendar needs 1145. |
| Saved Rondal models `window_1133.joblib`, `rondal_1133.joblib` | — | 1119 | Endpoint 1119 only |
| Carmella forward-chained Rondal | training eras ≤ 1119 only | per block | Not saved; does not cover evaluation eras |
| Live `final_ridge_resid_a10` (dc_rondal) | — | trained through 1227 | **No**: trained on the evaluation eras |
| Live `recent_*` members | — | trained through 1162 | **No**: trained on evaluation eras 1133–1162, and not refit per block |

**Conclusion: existing predictions cannot support the full B comparison.** Rondal
predictions with the required endpoints exist only for eras 1133–1158. Joining the Taren
window predictions (endpoint 1119) onto later Mavone blocks (endpoints 1145+) would mix
schedules, and calling that a matched experiment is not allowed.

The audit must also confirm, with explicit exceptions on failure:

- **A and C reuse:** Mavone's block prediction and model files match the hashes in
  `artifacts/dc_mavone_research_v1/manifest.json`. That manifest predates provenance
  version 2: it lacks dataset-file hashes. Reuse is permitted only if (a) the current
  data files match the hashes in `artifacts/dc_mavone_audit_v2/audit.json`, which was
  recorded the same day, before the forward run, and (b) the limitation is stated in the
  report. Otherwise regenerate A and C (cost below). **[needs approval: reuse or
  regenerate]**
- **Row universe:** identical validation IDs per era for A, Rondal and C, joined
  explicitly with completeness checks (`mavone.assert_complete`), never inner-joined.
- **Training endpoints and maturity:** every member's training eras are ≤ start − 14.
- **Feature coverage:** all members use the pinned 3,555 v5.3 features, and the
  feature-order hash matches.
- **Label provenance:** benchmark-residual labels use today's benchmark files. The
  v53 benchmark models and their historical predictions did not exist at those historical
  dates, and their own training provenance is not point-in-time. Residualizing old labels
  with them uses information that was unavailable then. So every result is a
  **retrospective forward-label simulation**, not a point-in-time historical replay. The
  same caveat applies to A, B and C.
- **Parity:** the new Rondal fit at endpoint 1119 must reproduce the Taren `rondal`
  column on eras 1133–1158 (rtol 1e-5), as Carmella's did.

## Step 2 — Retraining required if approved (not performed)

| Item | Work | Estimated cost |
|---|---|---|
| Rondal recreation | One era-ordered streaming pass with `taren.EndpointFitter` (h = inf), endpoints 1119, 1145, 1171, 1197; predict all validation rows in 1133–1236 | About 10–15 min, ≤ 6 GB (Taren fitted 60 models in one pass in 618 s at 5.6 GB) |
| A and C, if not reused | Per block, one shared binned Dataset with two boosters (`ender_resid`, `teager2b`) | About 4 × (2 × 7.5 + 2) min ≈ 70 min, peak about 11 GB (Mavone measured) |
| Blends and evaluation | Normalize, blend, neutralize C, score | About 2–3 min |

The total is about 15 minutes if A and C are reused, or about 1.5 hours if regenerated.
Run with a new artifact name prefixed `dc_resid_cmp_`, using provenance version 2 (the
config, implementation, feature-order and dataset hashes recorded at the start). This
needs a small runner, or a Mavone-runner extension, reviewed before execution.

## Step 3 — Exploratory analysis on eras 1133–1227

All differences are paired by era, on the common resolved eras with meta-model coverage.

- Per-era CORR (`numerai_corr`), MMC (`correlation_contribution` against
  `numerai_meta_model`) and diagnostic BMC (against `v53_lgbm_ender60`), via the pinned
  numerai-tools 0.7.2.
- Paired differences B − A for MMC, CORR and payout (3·CORR + 15·MMC), reported overall
  and by block, with the share of positive eras.
- Uncertainty: a 13-era moving-block bootstrap (`wenek.paired_block_ci`, 3,000
  resamples) on the single contiguous window. It is exploratory (about 7 effective
  blocks) and never bridges gaps.
- Diagnostics, not criteria: prediction rank correlations among A, Rondal, B, C, the
  meta model and the benchmark; per-era score correlations; Rondal's standalone scores.
- C versus A and C versus B are reported in a separate section. They do not enter the
  decision.

## Predeclared success criteria (against A)

The primary metric is paired MMC improvement, with a CORR safeguard.
**All thresholds below need approval before execution.**

Exploratory screen (eras 1133–1227). B advances to confirmation only if all hold:

1. Paired mean MMC(B − A) ≥ **+0.0005**. [needs approval]
2. MMC(B − A) is positive in at least **3 of 4** blocks. [needs approval]
3. CORR safeguard: paired mean CORR(B − A) ≥ **−0.0020**, and B's mean CORR > 0.
   [needs approval]
4. B's mean MMC ≥ 0.

Rationale: under payout 3·CORR + 15·MMC, a +0.0005 MMC gain is worth +0.0075. The
largest CORR loss the safeguard allows (−0.0020) costs −0.0060, so a passing B cannot
lower expected payout in the observed data. The bootstrap interval is reported, but is
not required to exclude zero at this stage, because the eras are inspected and the
sample is small.

Fresh confirmation (stage below). The candidate is confirmed only if:

1. Paired mean MMC(B − A) > **0** over the whole confirmation window. [needs approval]
2. Paired mean CORR(B − A) ≥ **−0.0020**, and B's mean CORR > 0. [needs approval]
3. B's mean MMC ≥ 0, with the bootstrap interval reported.

A failed screen or confirmation is recorded as a null result. Thresholds, the members,
the 50/50 weight and the normalization are not adjusted afterwards.

## Fresh-confirmation stage

Freeze before inspecting any reserved outcome: the B procedure, the Rondal and A recipes,
the block schedule, the thresholds above, the confirmation window, and the analysis code
(hashes recorded).

- Window: the first **52** resolved eras from 1228 on (1228–1279). [needs approval;
  minimum 26]. Each future block is fit at its scheduled endpoint (1223, 1249, …) with the
  frozen recipe, once its labels have matured. Eras 1228–1236 use the existing
  endpoint-1197 predictions.
- Freeze the reserved-era predictions for A, B and C (eras 1228–1236 now, later blocks
  as their features appear), with hashes, before their targets or meta-model values are
  read.
- One analysis at the end of the window, with no interim looks that could change the
  decision.
- Data refreshes for new eras need separate approval. Each refresh is stored as a new
  dated snapshot with its own hashes, never overwriting the 2026-10-10 snapshot.
- Timing: era 1279 resolves around era 1292, roughly 13 months after the current
  snapshot. A 26-era window would resolve in about 7 months.

## Not allowed

- No weight optimization, no adding candidates based on inspected scores, no
  neutralization in B.
- No retrospective use of live production artifacts on their own training eras.
- No joins of incompatible cached predictions presented as a matched experiment.
- No scoring of eras 1228–1236, or reading their newer target or meta-model values, before
  the confirmation procedure is frozen.
- No changes to the four live models, `configs/live.yaml` or hosted packages, and no
  uploads.

## Decisions needed before execution

1. Approve or adjust the thresholds: +0.0005 MMC, 3 of 4 blocks, −0.0020 CORR safeguard,
   confirmation MMC > 0.
2. Reuse Mavone's A and C predictions (hash-verified against the same-day audit), or
   regenerate them under provenance version 2 (about 70 extra minutes).
3. Confirmation window length (52 or 26 eras) and approval for future dataset refreshes.
4. A research name for the candidate (a Spellmonger war mage) if it advances.
