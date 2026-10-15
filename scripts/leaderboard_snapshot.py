"""Read-only snapshot of public resolved score histories for context (no writes to Numerai).

Model-level: the latest resolved score per round (canon CORR / MMC / BMC) for our live
models and, only where they could overlap, for every model of the named accounts.
Account-level leaderboard rows are kept separate because they aggregate many models and
payouts. Comparisons use only rounds resolved for both sides at the same score day.
Nothing here is compared with backtests, and no architecture is inferred.
"""

import argparse
import json
import time
from pathlib import Path

import numerapi
import pandas as pd
from submit_live import load_env

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ["bpa", "mistreated", "kowalot"]
LIVE = ["lgbm_bag4", "dc_resid_ensemble", "dc_tyndal", "dc_rondal"]
METRICS = ["canon_corr", "canon_mmc", "canon_bmc"]


def scores(napi, owner, model, model_id, last_n):
    rows = []
    for metric in METRICS:
        for r in napi.submission_scores(model_id, display_name=metric, resolved=True,
                                        last_n_rounds=last_n, distinct_on_round=True):
            rows.append({"owner": owner, "model": model, "round": r["roundNumber"],
                         "metric": metric, "day": r["day"], "value": r["value"],
                         "resolve_date": r["resolveDate"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/leaderboard_snapshot_v2")
    ap.add_argument("--last-n-rounds", type=int, default=100)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"{out} exists; snapshots are never overwritten")
    env = load_env()
    public = numerapi.NumerAPI()
    ours = numerapi.NumerAPI(env["NUMERAI_PUBLIC_ID"], env["NUMERAI_SECRET_KEY"]).get_models()
    rows, errors, accounts = [], {}, {}
    for name in LIVE:
        if name not in ours:
            errors[name] = "not on account"
            continue
        rows += scores(public, "ours", name, ours[name], args.last_n_rounds)
        print(f"ours/{name}: {sum(r['model'] == name for r in rows)} resolved scores", flush=True)
    our_rounds = {r["round"] for r in rows}
    for acct in ACCOUNTS:
        models = public.models_of_account(acct)
        accounts[acct] = len(models)
        if not our_rounds:
            continue
        for i, (name, mid) in enumerate(sorted(models.items())):
            try:
                rows += [r for r in scores(public, acct, name, mid, args.last_n_rounds)
                         if r["round"] in our_rounds]
            except Exception as e:  # noqa: BLE001
                errors[f"{acct}/{name}"] = repr(e)
            print(f"{acct}: {i + 1}/{len(models)}", flush=True)
    try:
        board = pd.DataFrame(public.get_account_leaderboard(limit=200))
    except Exception as e:  # noqa: BLE001
        board, errors["account_leaderboard"] = pd.DataFrame(), repr(e)

    df = pd.DataFrame(rows, columns=["owner", "model", "round", "metric", "day", "value",
                                     "resolve_date"])
    out.mkdir(parents=True)
    df.to_parquet(out / "model_scores.parquet")
    board.to_csv(out / "account_leaderboard.csv", index=False)
    targets = {a: board[board.get("username", pd.Series(dtype=str)).str.lower() == a]
               .to_dict("records") for a in ACCOUNTS} if len(board) else {}
    summary = {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "models_per_account": accounts, "errors": errors,
               "our_resolved_rounds": sorted(int(r) for r in our_rounds),
               "account_leaderboard_rows": targets,
               "account_leaderboard_columns": list(board.columns)}
    if our_rounds:
        summary["model_means_on_our_rounds"] = (
            df.groupby(["owner", "model", "metric"]).value.agg(["mean", "count"]).round(5)
            .reset_index().to_dict("records"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
