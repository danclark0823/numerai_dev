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
