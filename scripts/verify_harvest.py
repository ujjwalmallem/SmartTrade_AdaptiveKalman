#!/usr/bin/env python3
"""Checklist for path-label harvest output (Priority-1 verification)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def verify_harvest(path: Path, *, min_rows: int = 50) -> int:
    if not path.exists():
        print(f"FAIL: missing {path}")
        return 1
    df = pd.read_csv(path)
    print(f"file: {path}")
    print(f"rows: {len(df)}")
    if len(df) < min_rows:
        print(f"FAIL: need ≥{min_rows} rows")
        return 1

    if "feat_half_life" not in df.columns or "label" not in df.columns:
        print("FAIL: missing feat_half_life or label")
        return 1

    hl = df["feat_half_life"].astype(float) * 30.0
    labels = df["label"].astype(int)
    pos_rate = float(labels.mean())
    print(
        f"half_life_bars: min={hl.min():.2f}  med={hl.median():.2f}  max={hl.max():.2f}"
    )
    print(f"labels: {labels.value_counts().to_dict()}  pos_rate={pos_rate:.1%}")
    if "feat_bars_held" in df.columns:
        bh = df["feat_bars_held"].astype(float)
        print(
            f"bars_held: min={bh.min():.1f}  med={bh.median():.1f}  max={bh.max():.1f}"
        )

    ok = True
    if float(hl.min()) < 3.9:
        print("FAIL: half-life floor still collapsed (< ~4 bars)")
        ok = False
    else:
        print("PASS: half-life floor ≥ 4")
    if float(hl.median()) <= 5.0 and float(hl.max()) <= 5.0:
        print("FAIL: half-life has no spread (all ≤ 5)")
        ok = False
    else:
        print("PASS: half-life shows real variation")
    if abs(float(hl.std())) < 1e-9:
        print("FAIL: feat_half_life is constant")
        ok = False
    else:
        print("PASS: feat_half_life not constant")
    if not (0.05 < pos_rate < 0.95):
        print(f"WARN: extreme class balance pos_rate={pos_rate:.1%}")
    else:
        print("PASS: both classes present with usable balance")

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/exit_training_dataset.csv"),
    )
    p.add_argument("--min-rows", type=int, default=50)
    args = p.parse_args(argv)
    return verify_harvest(args.dataset, min_rows=args.min_rows)


if __name__ == "__main__":
    raise SystemExit(main())
