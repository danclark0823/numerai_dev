"""Research-only multi-target ensemble pieces (dc_mavone); see
docs/multi_target_ensemble_plan.md. dc_mavone would be a separate model."""

from __future__ import annotations

import html
import itertools
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

GAP = 13
BLOCK = 26
# Era 1240 is the newest era of the snapshot downloaded 2026-10-10; eras are weekly, so
# era dates are approximate. Era numbers are not tournament round numbers.
REFERENCE_ERA, REFERENCE_DATE = 1240, date(2026, 10, 10)


def era_date(era: int) -> date:
    return REFERENCE_DATE - timedelta(weeks=REFERENCE_ERA - int(era))


def block_schedule(first_start: int, last_start: int, block: int = BLOCK,
                   gap: int = GAP) -> list[dict]:
    """Fixed blocks [start, start + block - 1] for starts up to `last_start`; each is
    predicted by one fit on eras <= fit_end = start - gap - 1, so labels mature first."""
    if block < 1 or gap < 1 or last_start < first_start:
        raise ValueError("Invalid block schedule")
    out, start = [], int(first_start)
    while start <= last_start:
        fit_end = start - gap - 1
        out.append({"start": start, "end": start + block - 1,
                    "fit_end": fit_end, "label_available_by_era": fit_end + gap,
                    "label_available_by_date": era_date(fit_end + gap).isoformat()})
        start += block
    return out


def stride_mask(eras, stride: int, anchor: int) -> np.ndarray:
    """Global stride: era numbers congruent to `anchor`, independent of target or split."""
    if stride < 1:
        raise ValueError("stride must be positive")
    return np.asarray(eras, dtype=int) % stride == anchor % stride


def training_mask(eras, labels, eligible, fit_end: int, stride: int, anchor: int) -> np.ndarray:
    """Rows usable by every member at `fit_end`: eligible, matured, on the global stride,
    with every selected target present (the joint nonmissing intersection)."""
    labels = np.asarray(labels, dtype=float)
    eras = np.asarray(eras, dtype=int)
    if labels.ndim != 2 or len(labels) != len(eras):
        raise ValueError("labels must be rows x targets")
    return (np.asarray(eligible, dtype=bool) & (eras <= fit_end)
            & stride_mask(eras, stride, anchor) & np.isfinite(labels).all(axis=1))


def exclusions(eras, labels: pd.DataFrame, eligible, fit_end: int, stride: int,
               anchor: int) -> pd.DataFrame:
    """Per stride era: candidate rows, rows missing each target, and rows excluded by
    the joint intersection."""
    eras = np.asarray(eras, dtype=int)
    cand = np.asarray(eligible, dtype=bool) & (eras <= fit_end) & stride_mask(eras, stride, anchor)
    sub = labels.loc[cand].isna()
    out = sub.groupby(eras[cand]).sum().astype(int)
    out.columns = [f"missing_{c}" for c in labels.columns]
    out.insert(0, "candidate_rows", pd.Series(eras[cand]).value_counts().sort_index())
    out["excluded_rows"] = sub.any(axis=1).groupby(eras[cand]).sum().astype(int)
    out.index.name = "era"
    return out


def gaussian_rank(values, eras) -> np.ndarray:
    """Tie-preserving per-era Gaussian rank over each complete era."""
    values = np.asarray(values, dtype=float)
    eras = np.asarray(eras)
    if len(values) != len(eras) or not np.isfinite(values).all():
        raise ValueError("gaussian_rank needs complete, finite predictions")
    out = np.empty(len(values))
    for era in pd.unique(eras):
        m = eras == era
        out[m] = norm.ppf((rankdata(values[m]) - 0.5) / m.sum())
    return out


def check_weights(weights: dict, members) -> dict:
    w = {k: float(v) for k, v in weights.items()}
    if set(w) - set(members) or not w:
        raise ValueError("weights reference unknown members")
    if any(v < 0 for v in w.values()) or not np.isclose(sum(w.values()), 1.0):
        raise ValueError("weights must be nonnegative and sum to 1")
    return w


def blend(preds: pd.DataFrame, weights: dict, eras) -> np.ndarray:
    """Gaussian-rank each member per era, take the weighted sum, then apply the final
    per-era submission rank in (0, 1]."""
    w = check_weights(weights, preds.columns)
    eras = np.asarray(eras)
    z = sum(v * gaussian_rank(preds[k].to_numpy(), eras) for k, v in w.items() if v > 0)
    return pd.Series(z).groupby(eras).rank(pct=True).to_numpy()


def incremental_sets(order) -> list[list[str]]:
    """Fixed prefixes of a pre-registered member order."""
    order = list(order)
    return [order[:i] for i in range(1, len(order) + 1)]


def assert_complete(frame: pd.DataFrame, columns, expected_ids: pd.Index):
    """Explicit join check: same row universe, no duplicates, no missing components."""
    if frame.index.has_duplicates or expected_ids.has_duplicates:
        raise ValueError("duplicate row ids")
    if not frame.index.sort_values().equals(expected_ids.sort_values()):
        raise ValueError("row universe differs from expected ids")
    missing = frame[list(columns)].isna().sum()
    if missing.any():
        raise ValueError(f"missing component predictions: {missing[missing > 0].to_dict()}")


def hypothesis1_decision(scores: dict, zero: float = 1e-12) -> dict:
    """Frozen controlled equal-weight test from the plan's Phase 3."""
    corr = scores["corr"]
    mmc = scores.get("mmc")
    beyond = scores["ender_contribution"]
    paired = scores.get("paired_mmc_improvement")
    checks = {
        "mean_corr": corr, "corr_ok": corr > 0,
        "mean_mmc": mmc, "mmc_ok": mmc is not None and mmc >= 0,
        "contribution_beyond_ender": beyond, "beyond_ok": beyond > zero,
        "paired_mmc_improvement": paired,
        "paired_ok": paired is not None and paired > zero,
    }
    passed = all(checks[k] for k in ("corr_ok", "mmc_ok", "beyond_ok", "paired_ok"))
    return {**checks, "hypothesis1_passed": bool(passed), "deployment": False}


def longest_run(eras) -> tuple[int, int, int]:
    """(length, first, last) of the longest run of consecutive era numbers."""
    eras = sorted({int(e) for e in eras})
    if not eras:
        return 0, -1, -1
    best = (1, eras[0], eras[0])
    start = eras[0]
    for prev, cur in itertools.pairwise(eras):
        if cur != prev + 1:
            start = cur
        if cur - start + 1 > best[0]:
            best = (cur - start + 1, start, cur)
    return best


def mmc_gate(meta_eras, selection: int = 104, gap: int = GAP, blocks: int = 3,
             block: int = BLOCK) -> dict:
    """Phase 4 floor: contiguous mature eras with forward predictions and meta-model
    references for one selection window, the gap and three evaluation blocks."""
    need = selection + gap + blocks * block
    length, first, last = longest_run(meta_eras)
    return {"required_contiguous_eras": need, "available_contiguous_eras": length,
            "first": first, "last": last, "gate_passed": length >= need,
            "mmc_weight_optimization": "allowed" if length >= need else "deferred"}


def heatmap_svg(matrix: pd.DataFrame, title: str, cell: int = 64) -> str:
    """Self-contained SVG heatmap for a square correlation matrix (no plotting deps)."""
    labels = [str(c) for c in matrix.columns]
    n, pad = len(labels), 150
    size = pad + n * cell
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size + 20}" '
             f'height="{size + 40}" font-family="sans-serif" font-size="12">'),
             f'<text x="10" y="20" font-size="14">{html.escape(title)}</text>']
    for i, row in enumerate(labels):
        y = 40 + i * cell
        parts.append(f'<text x="{pad - 6}" y="{y + cell / 2 + 4}" text-anchor="end">'
                     f'{html.escape(row)}</text>')
        for j in range(n):
            v = float(matrix.iloc[i, j])
            t = max(-1.0, min(1.0, v))
            r, g, b = ((255, int(255 * (1 - t)), int(255 * (1 - t))) if t >= 0
                       else (int(255 * (1 + t)), int(255 * (1 + t)), 255))
            x = pad + j * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="rgb({r},{g},{b})" stroke="white"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" '
                         f'text-anchor="middle">{v:.2f}</text>')
    for j, col in enumerate(labels):
        x = pad + j * cell + cell / 2
        y = 40 + n * cell + 14
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle">{html.escape(col)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
