#!/usr/bin/env python3
"""Concatenate metrics.csv from multiple training runs (optional CLI helper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Merge outputs/train/*/metrics.csv for comparison.")
    p.add_argument(
        "root",
        nargs="?",
        default="outputs/train",
        help="Directory containing run subfolders (default: outputs/train)",
    )
    args = p.parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("*/metrics.csv"))
    if not paths:
        print("No metrics.csv files found.", file=sys.stderr)
        return 1
    dfs = [pd.read_csv(x) for x in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
