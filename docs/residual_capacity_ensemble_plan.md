# dc_rescap — residual tree capacity and fixed Rondal blends

Status: **frozen before scoring; approved and run 2026-10-15 (13 h budget, 13-era gap
kept). Nothing advanced; see Results.** This is a controlled
research experiment, not an attempt to reproduce any leaderboard strategy. It is a
separate experiment: it does not refit, reweight or replace lgbm_bag4, dc_resid_ensemble,
dc_tyndal or dc_rondal, and it changes nothing in `configs/live.yaml` or `artifacts/hosted`.
There is no upload, deployment, staking, data refresh, or scoring of reserved eras (1228+).

Code: `src/numerai_dev/rescap.py`, `scripts/run_rescap.py`, `configs/rescap_research.yaml`,
`tests/test_rescap.py`; shared provenance guards in `src/numerai_dev/provenance.py`.
Artifacts: `artifacts/dc_rescap_research_v1` (main run), `dc_rescap_timing_v1` (bounded
timing), and `dc_rescap_smoke_v1` (64-feature integration test).

## Questions

1. Is the current LightGBM recipe limiting the benchmark-residual Ender model? This
   compares two complete recipe packages; it cannot attribute any difference to one
   parameter.
2. Does a fixed 50/50 blend with a leakage-safe Rondal recreation add value beyond both
   of its members?

## Data, label and calendar (identical for both tree recipes)

- Pinned local v5.3 snapshot (file hashes in `docs/mavone_audit.md`), all 3,555 features
  in metadata order (the feature-order hash is recorded).
- Evaluation target: explicit `target_ender_60`. Payout reference 3·CORR + 15·MMC.
- Tree label: Mavone's `ender_resid`. That is `target_ender_60` minus its per-era
  least-squares fit on the gaussianized `v53_lgbm_ender60`, computed over every eligible
  row with that benchmark. Rows without the benchmark (train eras 1–157) get zero
  weight.
- Tree training rows: every row on eras with era % 4 == 1 (a global anchor), in train and
  in validation rows typed `validation`, with eras ≤ the block's fit endpoint. Both recipes
  train on one shared binned LightGBM Dataset, so their training IDs, bin edges, weights
  and feature order are identical by construction. The training-ID hash is recorded per
  block.
- Fixed 26-era blocks with the existing 13-era gap; each block is fit on eras
  ≤ start − 14:

| Block | Fit on eras ≤ | Evaluation eras |
|---|---:|---|
| 1133 | 1119 | 1133–1158 |
| 1159 | 1145 | 1159–1184 |
| 1185 | 1171 | 1185–1210 |
| 1211 | 1197 | 1211–1227 (17 eras; 1228+ reserved and not predicted) |

That is 95 evaluation eras, all with meta-model MMC. **All were inspected by earlier
research** (dev scoring, walk-forward windows, the 52-era holdout and Mavone). Every
result is development evidence from a retrospective forward-label simulation. The
benchmark predictions used to build residual labels are not point-in-time.
- No early stopping, and no selection or tuning on any evaluation label.

Gap note: Numerai's own benchmark walk-forward purges 16 eras for 60-day targets
([models page](https://docs.numer.ai/numerai-tournament/models)). This plan keeps the
repository's 13-era gap, as instructed, for comparability with Mavone, Taren and
Carmella. A 16-era gap would be more conservative, and it is listed under decisions.

## Tree recipes (exactly two)

Source for B: [Numerai docs, Models → "Standard Large LGBM params"](https://docs.numer.ai/numerai-tournament/models),
accessed 2026-10-14. The published dict is `n_estimators=20000, learning_rate=0.001,
max_depth=6, num_leaves=2**6, colsample_bytree=0.1`.

| Parameter | A: current | B: standard large |
|---|---:|---:|
| n_estimators | 2,000 | 20,000 |
| learning_rate | 0.01 | 0.001 |
| max_depth | 5 | 6 |
| num_leaves | 31 | 64 |
| colsample_bytree | 0.1 | 0.1 |
| objective | regression | regression (not published; LightGBM default) |
| random_state | 42 | 42 (not published; held equal to A) |
| num_threads | 10 | 10 |
| everything else | LightGBM 4.7.0 defaults | LightGBM 4.7.0 defaults |

Deviations from the published recipe: an explicit seed and objective, both held equal to
A, and our residual label with zero-weighted rows instead of a raw target. Library
versions: LightGBM 4.7.0, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, numerai-tools 0.7.2,
Python 3.12.15, on a 10-core arm64 machine with 24 GB. The deep 30,000-tree recipe, seed
sweeps, other targets, neutralization and other hyperparameters are excluded.

## Rondal recreation

dc_rondal's established recipe: ridge with alpha = 10 (scaled by row count) on all
3,555 features. Its label is `target_ender_60` residualized per era against both
`v53_lgbm_ender60` and `v53_lgbm_ender20`. It trains on every row of every eligible era,
which means eras 158 onward, where both benchmarks exist. One era-ordered streaming pass
(exact float32 sufficient statistics, `taren.EndpointFitter`) solves the fits at
endpoints 1119, 1145, 1171 and 1197, each before any later era is added. Each evaluation
row is predicted by its own block's endpoint, on exactly the tree models' evaluation IDs.

Differences from the tree members, documented rather than equalized:
- Label: residualized against two benchmarks, versus one for the trees.
- Coverage: every row, versus every fourth era.
- Model class: linear, versus trees.

These differences are part of what makes it a candidate diversifier.

Parity check: the endpoint-1119 fit must reproduce `dc_taren_full_v1`'s walk-forward
Rondal on eras 1133–1158 (rtol 1e-5), or the run stops.

## Predeclared outputs

1. `tree_current`: recipe A.
2. `tree_large`: recipe B.
3. `rondal`: Rondal recreation.
4. `blend_current`: fixed 50/50 of A and Rondal.
5. `blend_large`: fixed 50/50 of B and Rondal.

Blends: Gaussian-rank each member over each complete era (average-rank ties,
`norm.ppf((r − 0.5)/n)`), take 0.5 × tree + 0.5 × Rondal, then apply the final per-era
`rank(pct=True)` used for submissions. Single members get the same final ranking. There
is no weight optimization and no neutralization. Both blends are evaluated regardless of
which tree looks better. Numerai's published ensembles also standardize to unit standard
deviation after gaussianizing. That changes nothing here, because both members' per-era
Gaussian ranks have the same standard deviation.

## Scoring and comparisons

Official numerai-tools 0.7.2: `numerai_corr`, and `correlation_contribution` against
`numerai_meta_model` (MMC) and `v53_lgbm_ender60` (diagnostic BMC).

Primary paired comparisons, on identical eras:
- `tree_large` − `tree_current`
- `blend_current` − `tree_current`
- `blend_current` − `rondal`
- `blend_large` − `tree_large`
- `blend_large` − `rondal`

The report covers:
- Per-era and per-block CORR, MMC and BMC.
- Mean, standard deviation, share of positive eras, worst era and worst block.
- Prediction rank correlations, including with the meta model and benchmark, plus a heatmap.
- Per-era score correlations.
- Paired differences with 13-era moving-block bootstrap intervals, plus a 26-era
  sensitivity check (95 contiguous eras allows both).
- Runtime and memory by recipe.

### Development advancement rule (per candidate vs comparator)

1. Mean MMC > 0.
2. Paired mean MMC gain over the comparator > 0.
3. MMC difference > 0 in a majority of blocks (at least 3 of 4).
4. Mean CORR ≥ 90% of the comparator's mean CORR, when the comparator's CORR is positive.
   If the comparator's CORR is ≤ 0, it is flagged as an inadequate anchor and the
   candidate does not advance automatically.

A blend **adds value beyond both members** only if it advances against its tree member
and against Rondal. Higher CORR with lower MMC is reported as a trade-off. Intervals that
span zero mean the improvement is uncertain. Advancing earns a separately frozen
confirmation plan on reserved eras; it never establishes production readiness.

## Provenance audit: cached controls

| Candidate cache | Label | Rows and endpoints | Provenance | Use |
|---|---|---|---|---|
| Mavone `ender_resid` (blocks 1133–1211) | Same | Same stride, anchor and endpoints; evaluation IDs a superset | Legacy manifest without dataset hashes; trained in a different process | **Not reused.** Recipe A is refit beside B on the shared Dataset. Mavone's predictions serve only as a recorded determinism diagnostic. |
| `dc_taren_full_v1` / `dc_carmella_full_v1` Rondal | Same Rondal label | Endpoint 1119 for 1133–1162 only; 836 and 986 elsewhere | Complete | **Not reused**: wrong endpoints after 1158. Used as the parity reference for the 1119 refit. |
| Carmella forward-chained Rondal | Same | Training eras ≤ 1119 only | Not saved | Not usable. |
| Live `final_ridge_resid_a10`, `final_ender60` | — | Trained through 1227 | Production | **Excluded**: trained on the evaluation eras. |
| Live `recent_*` members | — | Trained through 1162 | Production | **Excluded**: overlaps the evaluation eras, not refit per block. |

The main run uses provenance version 2. Resume and evaluation reject any change in
config, implementation-file hashes, feature order or dataset-file hashes. Completed
blocks and checkpointed models (each 20,000-tree booster is checkpointed) are re-hashed
before reuse. Completed runs and evaluations are never reopened. Failures raise
`ProvenanceError`, pointing to a new run name.

## Compute budget and concurrency

See the cost check below. One process at a time, LightGBM `num_threads=10`, no
concurrent blocks (one block's Dataset peaks at about 11 GB of 24 GB). The forward run
requires `--budget-hours`, recorded per run. It stops between models once the budget is
spent and reports; it never shrinks the recipe or one candidate's data to fit.

### Cost check (2026-10-14; scores from these runs were not used)

- Tests: 120 passed (whole suite); ruff clean; `git diff --check` clean.
- Smoke integration (`dc_rescap_smoke_v1`): 64 features, 20 and 40 trees, blocks 1133 and
  1211. Forward and evaluation ran end to end in 30 s and 15 s, peaking at 3.4 GB.
  A second evaluation was refused. The Taren and Mavone parity checks are skipped in
  smoke mode and run first in the main run.
- Bounded timing (`dc_rescap_timing_v1`): block 1211, all features, 1,656,713 training
  rows, 10 threads.

| Stage | Measured | Full-run extrapolation |
|---|---|---|
| Shared Dataset build | 257 s, 9.2 GB | about 4 min per block |
| Recipe A | 200 trees in 82 s (0.41 s per tree) | 2,000 trees: about 14 min per block, about 0.9 h total |
| Recipe B | 400 trees in 159 s (0.40 s per tree) | 20,000 trees: about 2.2 h per block, about 8.8 h total |
| Prediction | about 5×10⁻⁵ s per tree per 1,000 rows | about 2 min per block |
| Rondal stream | Taren reference: 618 s, 5.6 GB | about 0.25 h |
| Evaluation | smoke reference | about 0.1 h |
| **Total** | | **about 10.1 h; peak about 10 GB of 24 GB** |

Per block: 1133 about 2.34 h, 1159 about 2.41 h, 1185 about 2.47 h, 1211 about 2.53 h,
scaled by training rows. The per-tree cost is extrapolated linearly from early trees. The
machine must stay awake and unthrottled for about 10 h.

**Proposed budget: 13 wall-clock hours** (`--budget-hours 13`, about 30% headroom).
Concurrency: one process, `num_threads=10`, nothing else heavy running alongside. If
the budget runs out, the run stops at the next model boundary with status
`stopped_budget` and keeps every completed checkpoint. A later `--resume` under
the same identity continues it only with your approval. Recipes and data are never cut
to fit.

## Leaderboard context (read-only)

`scripts/leaderboard_snapshot.py` → `artifacts/leaderboard_snapshot_v2`, retrieved
2026-10-14 19:43 UTC with public queries only (our model IDs came from the read-only
`get_models` call).

- **Model level:** lgbm_bag4, dc_resid_ensemble, dc_tyndal and dc_rondal have **no
  resolved rounds** yet, for canon CORR, MMC or BMC. So there is no overlapping resolved
  history, and no model-level comparison with BPA (29 models), MISTREATED (84) or
  KOWALOT (100) is possible. Those accounts' model histories were deliberately not
  pulled.
- **Account level** (public leaderboard, aggregates over every model in an account;
  horizon and weighting as defined by Numerai; not comparable with any single model or
  backtest):

| Account | Rank | corJ60 | MMC | v2Corr20 |
|---|---:|---:|---:|---:|
| bpa | 1 | 0.0256 | 0.0048 | 0.0169 |
| mistreated | 2 | 0.0193 | 0.0053 | 0.0152 |
| kowalot | 3 | 0.0195 | 0.0054 | 0.0148 |

These are context only. They are not targets for this backtest, and nothing about
those accounts' architectures is inferred. The current stake and the return fields in
the snapshot are not historical MMC and are not used. Rerun the snapshot once our models
have resolved rounds.

## Verification

Tests: `tests/test_rescap.py`, `tests/test_mavone_provenance.py`, `tests/test_mavone.py`.
They cover:
- Cutoff and reserved-era enforcement.
- Future-label isolation for the trees and the ridge.
- Matched training rows and zero-weight isolation for both recipes.
- Blend formula and official-scoring parity, including ties.
- Serialization, feature-order and batch-size parity.
- Identity enforcement on resume and evaluation, checkpoint corruption, and refusal to
  reopen completed runs before anything is written.

At run time the runner also checks:
- Residual-label parity, and Rondal parity with Taren.
- Reload and batch parity for each block.
- Complete, explicitly joined evaluation IDs.
- Unchanged hashes of the protected production files.

## Stopping point

Deliver the frozen plan, audit, tests and compute estimate, then stop for approval
before the long run. After an approved run, report whether the larger recipe improved the
baseline, whether either blend added value beyond both members, and whether anything
warrants a separately frozen confirmation plan. No deployment, model replacement,
upload, staking, commit or push without explicit approval.

## Decisions taken

1. Budget: 13 h approved. The run took 5.5 h wall-clock (about 70 min per block for
   recipe B and 7 min for recipe A), peaking at 11.2 GB.
2. Gap: the 13-era gap was kept.

## Results (`artifacts/dc_rescap_research_v1`, 2026-10-15)

95 eras (1133–1227), all previously inspected development data. Reserved eras 1228+
were not predicted or scored. Run status `forward_complete`; production hashes were
identical before and after.

Parity:
- `tree_current` matches Mavone's `ender_resid` exactly in all four blocks.
- Rondal matches Taren's endpoint-1119 Rondal to within 9e-10.

| Output | CORR | MMC | MMC std | MMC % positive eras | Worst-era MMC | BMC |
|---|---:|---:|---:|---:|---:|---:|
| tree_current | 0.0248 | 0.0165 | 0.0078 | 97% | −0.0072 | 0.0113 |
| tree_large | 0.0276 | 0.0173 | 0.0097 | 94% | −0.0088 | 0.0112 |
| rondal | 0.0181 | 0.0138 | 0.0072 | 97% | −0.0048 | 0.0116 |
| blend_current | 0.0234 | 0.0164 | 0.0064 | 100% | +0.0015 | 0.0125 |
| blend_large | 0.0249 | 0.0170 | 0.0070 | 99% | −0.0025 | 0.0125 |

| Comparison | Paired MMC (13-era 95% CI) | Positive MMC blocks | Advances |
|---|---|---:|---|
| tree_large − tree_current | +0.0009 [−0.0015, +0.0038] | 2/4 | No |
| blend_current − tree_current | −0.0000 [−0.0025, +0.0017] | 1/4 | No |
| blend_current − rondal | +0.0027 [+0.0007, +0.0051] | 3/4 | Yes |
| blend_large − tree_large | −0.0003 [−0.0037, +0.0020] | 2/4 | No (CORR is also below the 90% floor) |
| blend_large − rondal | +0.0033 [+0.0006, +0.0068] | 3/4 | Yes |

Interpretation:
- **Capacity:** the standard-large recipe's CORR and MMC gains are uncertain (both
  intervals span zero). They are concentrated in block 1133, and its MMC was lower than
  the current recipe's in blocks 1185 and 1211. There is no evidence that the current
  recipe is the binding constraint.
- **Blends:** both improve clearly on Rondal alone, but neither adds value beyond both
  members, because neither improves on its own tree's mean MMC. They do reduce MMC
  variability and raise the worst era, a stability benefit the predeclared rule does not
  score.
- **Latest block (1211):** Rondal was the strongest output (MMC 0.0166, against 0.0084
  for tree_current and 0.0066 for tree_large), and both blends beat both trees there.
  This is a single 17-era block, noted as a pattern to watch rather than evidence.

Under the plan, no output earns a confirmation run on reserved eras, and nothing is
deployed.
