"""Research-only portfolio-residual pieces: forward-chained Rondal predictions and the
screen for a learner trained on what both benchmarks and Rondal leave unexplained."""

from __future__ import annotations

import numpy as np

from numerai_dev.taren import INF, EndpointFitter, WeightedRidge


def block_geometry(first_era: int, last_era: int, warmup: int = 104, gap: int = 13,
                   block: int = 26) -> list[tuple[int, int, int]]:
    """[(block_start, block_end, fit_end)] covering eras up to `last_era`. Each block is
    predicted by a fit on eras <= fit_end = block_start - gap - 1; the first block
    starts after `warmup` eras of history plus the embargo."""
    if warmup < 1 or gap < 1 or block < 1:
        raise ValueError("Invalid block design")
    out, start = [], first_era + warmup + gap
    while start <= last_era:
        out.append((start, min(start + block - 1, last_era), start - gap - 1))
        start += block
    return out


class ForwardChain:
    """Streams rows in era order. Rows of a block are predicted only by the ridge fit
    at that block's fit_end; extra endpoints (e.g. outer folds) are fitted too."""

    def __init__(self, n_features: int, blocks, alpha: float, extra_endpoints=()):
        self.blocks = list(blocks)
        self.starts = np.array([b[0] for b in self.blocks])
        endpoints = [b[2] for b in self.blocks] + [int(e) for e in extra_endpoints]
        self.fitter = EndpointFitter(n_features, [INF], endpoints, alpha)

    def advance(self, era: int):
        self.fitter.advance(era)

    def add_rows(self, era: int, x, y):
        self.fitter.add_rows(era, x, y)

    def block_of(self, era: int):
        i = int(np.searchsorted(self.starts, era, side="right")) - 1
        if i < 0 or era > self.blocks[i][1]:
            return None
        return self.blocks[i]

    def model_for(self, era: int) -> WeightedRidge | None:
        """Ridge fitted strictly before `era`'s block embargo, or None during warm-up."""
        block = self.block_of(era)
        if block is None:
            return None
        model = self.fitter.models[block[2]][INF]
        assert model.train_end <= block[2] < block[0]
        return model

    def endpoint_model(self, endpoint: int) -> WeightedRidge:
        return self.fitter.models[int(endpoint)][INF]

    def finish(self):
        return self.fitter.finish()


def forward_chain(x, y, eras, blocks, alpha, extra_endpoints=()):
    """In-memory forward-chained predictions (NaN during warm-up) and the chain."""
    eras = np.asarray(eras, dtype=int)
    y = np.asarray(y, dtype=float)
    chain = ForwardChain(np.asarray(x).shape[1], blocks, alpha, extra_endpoints)
    out = np.full(len(y), np.nan)
    for era in np.unique(eras):
        mask = eras == era
        chain.advance(era)
        model = chain.model_for(era)
        if model is not None:
            out[mask] = model.predict(x[mask])
        chain.add_rows(era, x[mask], y[mask])
    chain.finish()
    return out, chain


def screening_decision(metrics: dict, zero: float = 1e-12) -> dict:
    """Frozen screen for `carmella` (a separate model) against the recreations."""
    windows = [w for w in metrics if w != "all"]
    c = {w: metrics[w]["carmella"] for w in windows}
    portfolio = {w: c[w]["portfolio_contribution"]["mean"] for w in windows}
    tyndal = {w: c[w]["tyndal_contribution"]["mean"] for w in windows}
    mmc = {w: c[w]["mmc"]["mean"] for w in windows if "mmc" in c[w]}
    corr = {w: c[w]["corr"]["mean"] for w in windows}
    checks = {
        "portfolio_contribution_by_window": portfolio,
        "portfolio_ok": sum(v > zero for v in portfolio.values()) >= 2,
        "tyndal_contribution_by_window": tyndal,
        "tyndal_ok": sum(v > zero for v in tyndal.values()) >= 2,
        "mmc_by_window": mmc,
        "mmc_ok": bool(mmc) and all(v >= 0 for v in mmc.values()),
        "corr_by_window": corr,
        "positive_corr_every_window": all(v > 0 for v in corr.values()),
    }
    passed = bool(checks["portfolio_ok"] and checks["tyndal_ok"] and checks["mmc_ok"]
                  and checks["positive_corr_every_window"])
    return {**checks, "screen_passed": passed, "deployment": False, "shelve": not passed}
