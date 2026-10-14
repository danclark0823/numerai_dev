# dc_mavone Phase 3: forward multi-target ensemble

Forward predictions for eras 1133-1227 (95 resolved eras, 95 with meta-model MMC) from 4 fixed 26-era blocks, each fit on stride-4 eras before a 13-era gap. Every evaluated era was inspected by earlier research, so this is development evidence from a retrospective forward-label simulation, not fresh confirmation.

Prospective predictions for 9 unresolved eras are saved unscored for later confirmation.

## Members and blends (all forward eras)

| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | Payout (3C+15M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ender | 0.0260 | 1.55 | 0.0116 | 1.10 | 0.0004 | 0.0000 | 0.2514 |
| teager2b | 0.0235 | 1.54 | 0.0094 | 0.91 | -0.0012 | 0.0003 | 0.2118 |
| jasper | 0.0234 | 1.70 | 0.0108 | 1.15 | 0.0003 | 0.0011 | 0.2323 |
| rowan | 0.0218 | 1.52 | 0.0080 | 0.74 | -0.0021 | -0.0008 | 0.1849 |
| ender_resid | 0.0248 | 2.41 | 0.0165 | 2.10 | 0.0113 | 0.0107 | 0.3211 |
| ender_only | 0.0260 | 1.55 | 0.0116 | 1.10 | 0.0004 | 0.0000 | 0.2514 |
| inc_ender_teager2b | 0.0254 | 1.56 | 0.0108 | 1.03 | -0.0004 | 0.0001 | 0.2381 |
| inc_ender_teager2b_jasper | 0.0252 | 1.64 | 0.0110 | 1.10 | -0.0001 | 0.0005 | 0.2415 |
| eq4 | 0.0249 | 1.63 | 0.0105 | 1.03 | -0.0007 | 0.0002 | 0.2322 |
| resid_ens | 0.0282 | 2.02 | 0.0152 | 1.75 | 0.0060 | 0.0071 | 0.3125 |

`resid_ens` is the contextual dc_resid_ensemble recreation (residual labels and neutralization differ from the controlled experiment; stride 4, unlike live).

## By block

### Block 1133

| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | Payout (3C+15M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ender_only | 0.0402 | 4.54 | 0.0173 | 2.45 | -0.0001 | 0.0000 | 0.3806 |
| eq4 | 0.0374 | 4.43 | 0.0153 | 2.28 | -0.0024 | -0.0002 | 0.3415 |
| resid_ens | 0.0392 | 4.36 | 0.0192 | 2.94 | 0.0035 | 0.0065 | 0.4058 |

### Block 1159

| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | Payout (3C+15M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ender_only | 0.0188 | 1.39 | 0.0112 | 1.20 | -0.0005 | 0.0000 | 0.2239 |
| eq4 | 0.0188 | 1.52 | 0.0115 | 1.27 | -0.0003 | 0.0011 | 0.2282 |
| resid_ens | 0.0230 | 1.63 | 0.0165 | 1.82 | 0.0062 | 0.0078 | 0.3169 |

### Block 1185

| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | Payout (3C+15M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ender_only | 0.0154 | 1.15 | 0.0036 | 0.37 | -0.0010 | 0.0000 | 0.1002 |
| eq4 | 0.0141 | 1.12 | 0.0018 | 0.18 | -0.0026 | -0.0009 | 0.0691 |
| resid_ens | 0.0214 | 2.54 | 0.0106 | 1.52 | 0.0073 | 0.0089 | 0.2229 |

### Block 1211

| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | Payout (3C+15M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ender_only | 0.0317 | 1.75 | 0.0155 | 1.45 | 0.0049 | 0.0000 | 0.3273 |
| eq4 | 0.0318 | 2.11 | 0.0150 | 1.75 | 0.0044 | 0.0010 | 0.3203 |
| resid_ens | 0.0300 | 1.81 | 0.0140 | 1.35 | 0.0074 | 0.0044 | 0.3000 |

## Paired common-era differences (13-era block bootstrap)

| Comparison | Score | Mean | 95% CI | Eras | % positive | By block |
|---|---|---:|---|---:|---:|---|
| eq4_minus_ender_only | corr | -0.0011 | [-0.0025, 0.0007] | 95 | 0.42 | 1133: -0.0028, 1159: 0.0000, 1185: -0.0013, 1211: 0.0000 |
| eq4_minus_ender_only | mmc | -0.0011 | [-0.0023, 0.0005] | 95 | 0.36 | 1133: -0.0021, 1159: 0.0003, 1185: -0.0018, 1211: -0.0005 |
| eq4_minus_ender_only | bmc | -0.0011 | [-0.0023, 0.0004] | 95 | 0.35 | 1133: -0.0023, 1159: 0.0002, 1185: -0.0016, 1211: -0.0005 |
| resid_ens_minus_ender_only | corr | 0.0022 | [0.0004, 0.0063] | 95 | 0.63 | 1133: -0.0010, 1159: 0.0042, 1185: 0.0060, 1211: -0.0017 |
| resid_ens_minus_ender_only | mmc | 0.0036 | [0.0020, 0.0069] | 95 | 0.72 | 1133: 0.0019, 1159: 0.0054, 1185: 0.0070, 1211: -0.0015 |
| resid_ens_minus_ender_only | bmc | 0.0055 | [0.0041, 0.0080] | 95 | 0.82 | 1133: 0.0036, 1159: 0.0067, 1185: 0.0083, 1211: 0.0025 |
| eq4_minus_resid_ens | corr | -0.0033 | [-0.0071, -0.0015] | 95 | 0.31 | 1133: -0.0018, 1159: -0.0042, 1185: -0.0073, 1211: 0.0018 |
| eq4_minus_resid_ens | mmc | -0.0047 | [-0.0080, -0.0029] | 95 | 0.23 | 1133: -0.0039, 1159: -0.0051, 1185: -0.0088, 1211: 0.0010 |
| eq4_minus_resid_ens | bmc | -0.0066 | [-0.0093, -0.0050] | 95 | 0.15 | 1133: -0.0059, 1159: -0.0065, 1185: -0.0099, 1211: -0.0030 |
| inc_ender_teager2b_minus_ender_only | corr | -0.0006 | [-0.0015, 0.0004] | 95 | 0.41 | 1133: -0.0018, 1159: -0.0007, 1185: -0.0009, 1211: 0.0015 |
| inc_ender_teager2b_minus_ender_only | mmc | -0.0008 | [-0.0015, 0.0003] | 95 | 0.38 | 1133: -0.0016, 1159: -0.0007, 1185: -0.0013, 1211: 0.0011 |
| inc_ender_teager2b_minus_ender_only | bmc | -0.0008 | [-0.0016, 0.0002] | 95 | 0.39 | 1133: -0.0017, 1159: -0.0008, 1185: -0.0012, 1211: 0.0011 |
| inc_ender_teager2b_jasper_minus_ender_only | corr | -0.0008 | [-0.0019, 0.0007] | 95 | 0.42 | 1133: -0.0023, 1159: -0.0003, 1185: -0.0008, 1211: 0.0006 |
| inc_ender_teager2b_jasper_minus_ender_only | mmc | -0.0005 | [-0.0015, 0.0008] | 95 | 0.42 | 1133: -0.0014, 1159: 0.0001, 1185: -0.0009, 1211: 0.0005 |
| inc_ender_teager2b_jasper_minus_ender_only | bmc | -0.0006 | [-0.0016, 0.0007] | 95 | 0.43 | 1133: -0.0016, 1159: -0.0000, 1185: -0.0008, 1211: 0.0004 |

## Prediction rank correlations (eras with meta model)

| model | ender | teager2b | jasper | rowan | ender_resid | eq4 | resid_ens | meta | bench |
|---|---|---|---|---|---|---|---|---|---|
| ender | 1.000 | 0.875 | 0.889 | 0.821 | 0.536 | 0.952 | 0.782 | 0.502 | 0.811 |
| teager2b | 0.875 | 1.000 | 0.839 | 0.861 | 0.469 | 0.949 | 0.820 | 0.490 | 0.778 |
| jasper | 0.889 | 0.839 | 1.000 | 0.799 | 0.473 | 0.936 | 0.716 | 0.484 | 0.771 |
| rowan | 0.821 | 0.861 | 0.799 | 1.000 | 0.468 | 0.924 | 0.743 | 0.462 | 0.730 |
| ender_resid | 0.536 | 0.469 | 0.473 | 0.468 | 1.000 | 0.514 | 0.805 | 0.302 | 0.429 |
| eq4 | 0.952 | 0.949 | 0.936 | 0.924 | 0.514 | 1.000 | 0.812 | 0.513 | 0.820 |
| resid_ens | 0.782 | 0.820 | 0.716 | 0.743 | 0.805 | 0.812 | 1.000 | 0.446 | 0.693 |
| meta | 0.502 | 0.490 | 0.484 | 0.462 | 0.302 | 0.513 | 0.446 | 1.000 | 0.634 |
| bench | 0.811 | 0.778 | 0.771 | 0.730 | 0.429 | 0.820 | 0.693 | 0.634 | 1.000 |

Heatmap: `prediction_rank_corr.svg`. All-era matrix: `prediction_rank_corr.csv`.

## Per-era score correlations (CORR)

| model | ender | teager2b | jasper | rowan | ender_resid |
|---|---|---|---|---|---|
| ender | 1.000 | 0.944 | 0.932 | 0.901 | 0.591 |
| teager2b | 0.944 | 1.000 | 0.897 | 0.902 | 0.544 |
| jasper | 0.932 | 0.897 | 1.000 | 0.877 | 0.499 |
| rowan | 0.901 | 0.902 | 0.877 | 1.000 | 0.564 |
| ender_resid | 0.591 | 0.544 | 0.499 | 0.564 | 1.000 |

## Hypothesis 1 (controlled equal weight vs matched Ender)

```
{
  "mean_corr": 0.024933053927693537,
  "corr_ok": true,
  "mean_mmc": 0.010490998884099871,
  "mmc_ok": true,
  "contribution_beyond_ender": 0.00016377514666835713,
  "beyond_ok": true,
  "paired_mmc_improvement": -0.0010625937315521057,
  "paired_ok": false,
  "hypothesis1_passed": false,
  "deployment": false
}
```

## Phase 4 MMC-history gate

```
{
  "required_contiguous_eras": 195,
  "available_contiguous_eras": 95,
  "first": 1133,
  "last": 1227,
  "gate_passed": false,
  "mmc_weight_optimization": "deferred"
}
```

## Compute cost

| Block | Seconds | Peak RSS GB | Train rows | Per-model fit seconds |
|---|---:|---:|---:|---|
| 1133 | 2095.7 | 10.23 | 1,527,361 | {"ender": 402.9, "teager2b": 393.4, "jasper": 379.8, "rowan": 377.2, "ender_resid": 369.0} |
| 1159 | 2383.3 | 10.91 | 1,571,454 | {"ender": 460.8, "teager2b": 417.6, "jasper": 470.7, "rowan": 451.3, "ender_resid": 401.7} |
| 1185 | 2559.9 | 10.91 | 1,609,522 | {"ender": 451.2, "teager2b": 515.8, "jasper": 513.4, "rowan": 457.5, "ender_resid": 448.2} |
| 1211 | 2541.2 | 11.03 | 1,656,713 | {"ender": 500.8, "teager2b": 478.1, "jasper": 458.4, "rowan": 465.9, "ender_resid": 439.9} |

Stopped for review before Phase 4. No deployment, upload or live change.
