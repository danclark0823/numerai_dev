# Results log

Dev eras ≈ 587–1162 (575–576 eras), target `target_ender_60`. Holdout (last 52
resolved eras) not yet scored. Comparisons use `scripts/compare.py` (paired,
13-era block bootstrap). Full metrics in `experiments/log.jsonl`.

| Experiment | CORR | Sharpe | BMC | Train time | Peak RAM | Verdict |
|---|---|---|---|---|---|---|
| `baseline_lgbm_medium` — example recipe, 780 features | 0.0314 | 1.99 | 0.0011 | 1 min | 7.3 GB | reference |
| `lgbm_all_stride4` — all 3555 features | 0.0422 | 2.34 | 0.0016 | 4 min | 11.2 GB | **+0.011 CORR, CI [+.008, +.014]** |
| `lgbm_all_deep` — depth 10, 1023 leaves | 0.0432 | 2.56 | 0.0009 | 8 min | 13.2 GB | not significant (CI [−.0005, +.0024]); BMC worse |
| `lgbm_all_bag4` — 4 era-offset models averaged | 0.0441 | 2.33 | 0.0013 | 16 min | 12.3 GB | **+0.002 CORR, CI [+.001, +.003]** vs single |
| Numerai `v53_lgbm_ender60` (reference preds) | 0.056 | 3.16 | — | — | — | possibly trained on validation eras |

## Notes

- Example predictions == `v53_lgbm_ender60` benchmark. Its validation-era CORR
  exceeds its own out-of-sample train-era CORR, so it may be in-sample; treat
  as an upper reference, not a target.
- `meta_model.parquet` covers only eras 1133–1227, so dev MMC has 30 eras; BMC
  is the dev proxy.
- BMC ≈ 0 for every model so far: we mostly reproduce what the benchmark knows.
  With payout = 3·CORR + 15·MMC, diversification is the next lever.

## Round 2: diversification (2026-10-10)

Single all-feature, every-4th-era models trained on other targets (all scored
on `target_ender_60`, full dev):

| Train target | CORR | BMC | MMC (30 eras) |
|---|---|---|---|
| ender_60 (`lgbm_all_stride4`) | 0.0422 | 0.0016 | 0.0133 |
| **ender_60 residualized vs benchmark** | 0.0361 | **0.0048** | 0.0115 |
| teager2b_60 | 0.0410 | 0.0024 | 0.0063 |
| jasper_60 | 0.0398 | 0.0011 | 0.0112 |
| rowan_60 | 0.0387 | 0.0023 | 0.0013 |
| ender_20 | 0.0358 | 0.0016 | 0.0037 |
| echo_60 | 0.0328 | 0.0003 | −0.0016 |
| tyler_60 | 0.0323 | −0.0004 | −0.0005 |
| tyler_20 | 0.0278 | −0.0011 | −0.0030 |
| agnes_60 | 0.0204 | 0.0005 | −0.0006 |

Ensemble (`scripts/ensemble.py`): members and neutralization chosen on dev
half A (eras 588–874), scored on half B (875–1162, 288 eras).

- Greedy selection picked **residual + teager2b_60**; equal-weight over all 10
  was worse (dilution by weak targets).
- Neutralization against `medium` features: best proportion 0.5 on A.

| On B | CORR | BMC | MMC (30 eras) | 3·CORR+15·BMC |
|---|---|---|---|---|
| `lgbm_all_bag4` | 0.0347 | 0.0018 | 0.0111 | 0.131 |
| `ens_targets_resid_neutral` | 0.0323 | 0.0037 | 0.0115 | 0.152 |
| paired diff, 95% CI | −0.0024 [−.004, −.001] | +0.0019 [+.0003, +.0035] | — | +0.021 [−.007, +.050] |

**Verdict:** trades a little CORR for twice the BMC; the payout-proxy gain is
suggestive (P≈0.08), not conclusive. Caveat: the benchmark correlates only
0.63 with the meta model, so BMC is a loose proxy for MMC, and real MMC is
flat on the 30 eras we can measure. Live MMC is the real test.
