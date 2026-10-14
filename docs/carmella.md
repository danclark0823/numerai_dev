# dc_carmella — portfolio-residual LightGBM results

Status: passed its frozen development screen, 2026-10-13. Not deployed. dc_carmella
would be a separate fifth model; the four live models are unchanged. Named for
Carmella, a warmage listed at https://spellmongershall.com/characters. The
[frozen plan](carmella_plan.md) specifies the design and screen.

## What we tried

LightGBM with Tyndal's exact parameters, trained on target_ender_60 residualized
per era against both benchmarks plus a forward-chained Rondal prediction. Each
26-era block was predicted by ridge fitted only on eras before block_start − 13,
after 104 warm-up eras. A matched Tyndal recreation (benchmarks only) trained on
identical rows: every fourth eligible era up to each outer embargo. Rondal was
recreated walk-forward and matched `dc_taren_full_v1` exactly.

## Results

Run `artifacts/dc_carmella_full_v1` took 5,945 seconds with 9.8 GB peak RSS.
Forward-chained Rondal scored CORR 0.029 on its own training eras, so a real
signal was removed.

| Model | CORR, 90 eras | MMC, 1133–1162 | BMC |
|---|---:|---:|---:|
| Rondal recreation | 0.0242 | 0.0114 | 0.0093 |
| Tyndal recreation | 0.0343 | 0.0159 | 0.0111 |
| Rondal + Tyndal blend | 0.0319 | 0.0150 | 0.0112 |
| **Carmella** | 0.0338 | 0.0143 | 0.0073 |

| Window | Carmella CORR | Beyond blend | Beyond Tyndal | Corr with Tyndal | Corr with Rondal |
|---|---:|---:|---:|---:|---:|
| 850 | 0.0466 | +0.0155 | +0.0063 | 0.86 | 0.45 |
| 1000 | 0.0307 | +0.0067 | +0.0019 | 0.82 | 0.43 |
| 1133 | 0.0240 | +0.0089 | +0.0037 | 0.82 | 0.40 |

All four screen criteria passed. Contribution beyond the blend and beyond Tyndal
was positive in all three windows, MMC was nonnegative, and CORR was positive in
every window.

## How strong is this?

It is a modest, consistent result, not a large one.

- Carmella is close to Tyndal: per-era rank correlation 0.82–0.86. Its
  contribution beyond Tyndal is small, 0.002–0.006.
- Its MMC is below the Tyndal recreation's (0.0143 vs 0.0159). The paired
  difference is −0.0017 with exploratory interval [−0.0074, +0.0039], so the
  data cannot separate them. The screen only requires Carmella to add to the
  portfolio, not to beat Tyndal.
- Its correlation with the meta model (0.26) is higher than Tyndal's (0.22) and
  Rondal's (0.09).
- Paired CORR differences against Tyndal straddle zero in every window.
- These windows have been inspected by every prior experiment. MMC covers 30 eras.

## Next step under the plan

A pass warrants a separate frozen replication on fresh live eras, before any
full-data fit, hosted package or live model entry. No upload or deployment
followed this run.

## Reproduction note

The run was launched at 19:05 from an implementation whose code is snapshotted
in the artifact's `sources/`. A second, independently written implementation of
the same plan was briefly placed in the working tree but never ran on full data.
It found the existing output directory and stopped, as designed. The repository
files were restored from the snapshots, and their SHA-256 hashes match the
manifest. The config was rebuilt from the manifest. That second launch
overwrote the run's console log `/tmp/carmella_full.log`; the manifest,
metrics, report and experiment-log record are complete.

Verified checks: forward-chain chronology, label parity, identical training rows,
Rondal parity with Taren, complete eras, finite predictions, submission validity,
reload parity, and unchanged production hashes. All 80 repository tests and lint
pass.
