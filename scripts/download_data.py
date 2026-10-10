"""Download a Numerai dataset version into data/<version>/."""

import argparse
from pathlib import Path

from numerapi import NumerAPI

DEFAULT_FILES = [
    "features.json",
    "train.parquet",
    "validation.parquet",
    "meta_model.parquet",
    "validation_example_preds.parquet",
    "validation_benchmark_models.parquet",
    "train_benchmark_models.parquet",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v5.3")
    parser.add_argument("--out", default="data")
    parser.add_argument("--live", action="store_true", help="also fetch live.parquet")
    args = parser.parse_args()

    napi = NumerAPI()
    out = Path(args.out) / args.version
    out.mkdir(parents=True, exist_ok=True)
    files = DEFAULT_FILES + (["live.parquet"] if args.live else [])
    for name in files:
        dest = out / name
        if dest.exists() and name != "live.parquet":
            print(f"skip {dest} (exists)")
            continue
        print(f"downloading {args.version}/{name}")
        napi.download_dataset(f"{args.version}/{name}", str(dest))


if __name__ == "__main__":
    main()
