"""Residual capacity and fixed-ensemble comparison (dc_rescap); see
docs/residual_capacity_ensemble_plan.md. Research only; a separate model."""

from __future__ import annotations

from numerai_dev.mavone import block_schedule

# Frozen recipes. B is Numerai's published "standard large" parameters
# (docs.numer.ai/numerai-tournament/models, accessed 2026-10-14) plus our objective
# and seed, which the published dict does not specify.
CURRENT = {"objective": "regression", "n_estimators": 2000, "learning_rate": 0.01,
           "max_depth": 5, "num_leaves": 31, "colsample_bytree": 0.1, "random_state": 42}
STANDARD_LARGE = {"objective": "regression", "n_estimators": 20000, "learning_rate": 0.001,
                  "max_depth": 6, "num_leaves": 64, "colsample_bytree": 0.1, "random_state": 42}
OUTPUTS = ["tree_current", "tree_large", "rondal", "blend_current", "blend_large"]
BLENDS = {"blend_current": "tree_current", "blend_large": "tree_large"}
# (candidate, comparator): the predeclared primary comparisons.
COMPARISONS = [("tree_large", "tree_current"),
               ("blend_current", "tree_current"), ("blend_current", "rondal"),
               ("blend_large", "tree_large"), ("blend_large", "rondal")]


def evaluation_blocks(first_start: int, last_start: int, last_era: int, block: int = 26,
                      gap: int = 13) -> list[dict]:
    """Fixed blocks with evaluation clipped at `last_era` (reserved eras excluded)."""
    if last_era > 1227:
        raise ValueError("Evaluation must end by era 1227; later eras are reserved")
    out = []
    for b in block_schedule(first_start, last_start, block, gap):
        if b["start"] > last_era:
            raise ValueError("A block starts after the last evaluation era")
        out.append({**b, "eval_end": min(b["end"], last_era)})
    return out


def check_recipes(recipes: dict, num_threads: int):
    if recipes.get("current", {}).get("params") != {**CURRENT, "num_threads": num_threads,
                                                    "verbose": -1}:
        raise ValueError("Recipe A differs from the frozen current recipe")
    if recipes.get("large", {}).get("params") != {**STANDARD_LARGE, "num_threads": num_threads,
                                                  "verbose": -1}:
        raise ValueError("Recipe B differs from the frozen standard-large recipe")
    if set(recipes) != {"current", "large"}:
        raise ValueError("Exactly two tree recipes are allowed")


def advancement(candidate_corr: float, candidate_mmc: float, comparator_corr: float,
                paired_mmc: float, block_mmc_diffs, corr_ratio: float = 0.9,
                zero: float = 1e-12) -> dict:
    """Predeclared development advancement rule for one candidate vs one comparator."""
    diffs = [float(d) for d in block_mmc_diffs]
    positive_blocks = sum(d > zero for d in diffs)
    anchor_ok = comparator_corr > zero
    corr_floor = corr_ratio * comparator_corr if anchor_ok else None
    checks = {
        "mean_mmc": candidate_mmc, "mmc_positive": candidate_mmc > zero,
        "paired_mmc_gain": paired_mmc, "paired_gain_positive": paired_mmc > zero,
        "positive_blocks": positive_blocks, "blocks": len(diffs),
        "block_majority": bool(diffs) and positive_blocks > len(diffs) / 2,
        "mean_corr": candidate_corr, "comparator_corr": comparator_corr,
        "corr_floor": corr_floor,
        "corr_floor_ok": None if not anchor_ok else candidate_corr >= corr_floor,
        "inadequate_anchor": not anchor_ok,
    }
    passed = (checks["mmc_positive"] and checks["paired_gain_positive"]
              and checks["block_majority"] and checks["corr_floor_ok"] is True)
    return {**checks, "advances": bool(passed), "deployment": False}


def adds_beyond_both(decisions: dict, blend: str) -> bool:
    """A blend adds value beyond both members only if it advances against each one."""
    tree = BLENDS[blend]
    return bool(decisions[f"{blend}_vs_{tree}"]["advances"]
                and decisions[f"{blend}_vs_rondal"]["advances"])
