# dc_mavone — multi-target ensemble, Phases 0–3

Research ensemble named after Mavone (Spellmonger war mage). A separate model:
nothing here refits, reweights or replaces lgbm_bag4, dc_resid_ensemble, dc_tyndal or
dc_rondal, and nothing was uploaded, packaged or added to `configs/live.yaml`.
Specification: [multi_target_ensemble_plan.md](multi_target_ensemble_plan.md).

Code: `src/numerai_dev/mavone.py`, `scripts/run_mavone.py`,
`configs/mavone_research.yaml`, `tests/test_mavone.py`.
Artifacts: `artifacts/dc_mavone_audit_v2` (v1 had wrong approximate era dates),
`artifacts/dc_mavone_research_v1`, integration test `artifacts/dc_mavone_smoke_v1`.

## Phase 0 — data sufficiency and provenance

Full audit: [mavone_audit.md](mavone_audit.md).

- Local snapshot v5.3, downloaded 2026-10-10/11, file hashes recorded. v5.3 is still
  the newest remote version. The remote meta-model file now reaches era 1228 (local:
  1227); only its era column was read. The local snapshot is kept unchanged.
- 40 named targets plus the `target` alias (identical to `target_ender_60`).
- Resolved `target_ender_60`: eras 1–1227. Eras 1228–1240 are `test` rows without targets.
- Meta model: 95 contiguous eras, 1133–1227, full row coverage. Benchmark
  `v53_lgbm_ender60`: train eras 158–574 and validation eras 575–1240.
- Scoring rules rechecked 2026-10-14: since round 1343, CORR, MMC and BMC use 60-day Ender.
- **No resolved era is unused.** Eras 587–1162 were dev-scored, three walk-forward
  windows and the 1176–1227 holdout were inspected repeatedly, and live members
  trained through 1162 or 1227. Every result below is development evidence.
  Eras ≥ 1228 are reserved for fresh confirmation.
- No point-in-time label publication dates exist. A label for era e is assumed
  available at e + 13, so this is a **retrospective forward-label simulation**.
- No cached predictions are reused. All 10 recent_*/final_* runs trained on
  evaluation eras, and earlier walk-forward artifacts used other targets or labels.

## Phase 1 — frozen targets and recipe

Targets (pre-registered order, also the incremental-addition order): `target_ender_60`
(anchor), `target_teager2b_60`, `target_jasper_60`, `target_rowan_60`. This shortlist
came from earlier inspected research; it was not chosen independently.

On the historical period (eras ≤ 1119), all four targets are complete, with identical
5-bin distributions. Their mean per-era rank correlations with Ender are 0.79 (teager2b),
0.79 (jasper) and 0.75 (rowan); jasper–rowan is the least correlated pair at 0.68. The
joint nonmissing intersection excludes no rows.

Recipe: all 3,555 features, raw labels, LightGBM with 2,000 trees, learning rate 0.01,
depth 5, 31 leaves, colsample 0.1, seed 42, num_threads 10 (10-core M4, 24 GB), no early
stopping or tuning. Training uses every eligible row on eras with era % 4 == 1 in both
splits (a global anchor). All members share one binned Dataset, so they have identical
training IDs (hash recorded per block).

Integration test: 64 features, 20 trees, blocks 1133 and 1211, all checks passed.
Representative full fit: block 1133 (see compute cost below).

## Phase 2 — forward predictions

Four fixed 26-era blocks starting at 1133, 1159, 1185 and 1211. Each is fit on eras
≤ start − 14, so labels have matured before the block begins:

| Block | Fit on eras ≤ | Resolved eras | Prospective eras |
|---|---:|---:|---:|
| 1133–1158 | 1119 | 26 | 0 |
| 1159–1184 | 1145 | 26 | 0 |
| 1185–1210 | 1171 | 26 | 0 |
| 1211–1236 | 1197 | 17 | 9 (unscored, reserved) |

Each block saves raw predictions for every validation-file row in its eras: id, era,
data type, block, fit endpoint, label-availability era and date, and model and
prediction-file hashes. Every block also checks that reloaded models reproduce its
predictions and that splitting rows into different batch sizes doesn't change them.

The contextual control is a fifth booster per block: Ender-60 residualized against
`v53_lgbm_ender60`, on the same rows, with zero weight where the benchmark is missing
(train eras < 158). It is blended equally with the raw Teager2b member, then 50%
neutralized against medium features. This recreates the dc_resid_ensemble recipe at
stride 4; live used stride 1.

## Phase 3 — results

Full report: [mavone_phase3.md](mavone_phase3.md). Heatmap:
[mavone_pred_corr.svg](mavone_pred_corr.svg). These are 95 forward eras (1133–1227),
all with meta-model MMC and all previously inspected, so this is development
evidence only.

| Model | CORR | MMC | BMC | Beyond Ender | Corr with meta | Corr with benchmark |
|---|---:|---:|---:|---:|---:|---:|
| Ender alone | 0.0260 | 0.0116 | 0.0004 | — | 0.50 | 0.81 |
| Equal-weight 4 targets | 0.0249 | 0.0105 | −0.0007 | 0.0002 | 0.51 | 0.82 |
| Ender + Teager2b | 0.0254 | 0.0108 | −0.0004 | 0.0001 | | |
| Ender + Teager2b + Jasper | 0.0252 | 0.0110 | −0.0001 | 0.0005 | | |
| Ender residual vs benchmark (member) | 0.0248 | 0.0165 | 0.0113 | 0.0107 | 0.30 | 0.43 |
| dc_resid_ensemble recreation (contextual) | 0.0282 | 0.0152 | 0.0060 | 0.0071 | 0.45 | 0.69 |

**Hypothesis 1 fails.** The equal-weight blend has positive CORR, nonnegative MMC and
slightly positive contribution beyond Ender (0.0002). But its paired MMC difference from
Ender alone is −0.0011 (95% CI −0.0023 to 0.0005; positive in 36% of eras). It loses
MMC to Ender in three of four blocks (block 1159: +0.0003). Each incremental addition is
also slightly negative.

Why: the raw-target members are near copies. Their predictions correlate 0.80–0.89 with
each other, their per-era CORR scores 0.88–0.94, and each correlates about 0.8 with the
benchmark. Averaging them mostly averages away noise Ender already handles, and adds
nothing the meta model lacks.

What does help, as context: removing the benchmark from the label. The residual member
alone has the best MMC (0.0165, sharpe 2.1) and lowest meta correlation (0.30). The
contextual dc_resid_ensemble recreation beats Ender alone by +0.0036 MMC
(CI 0.0020–0.0069) and beats the equal-weight blend in three of four blocks. This was not
the controlled test, and its member recipe came from earlier inspected research.

**Phase 4 gate fails:** MMC weight selection needs 195 contiguous forward eras with MMC;
there are 95. Learned weights are deferred, and BMC is not substituted.

Compute: 2.7 h for four blocks of five boosters, 6–8.5 min per booster on 1.53–1.66M
rows, peak RSS 11.0 GB. Raw and normalized forward predictions are in
`artifacts/dc_mavone_research_v1/evaluation/forward_preds.parquet`. The 9 prospective
eras (1228–1236) are in `prospective_preds.parquet`, unscored.

Stopped for review before Phase 4. No upload, packaging or live change; the protected
production file hashes were unchanged on every run.
