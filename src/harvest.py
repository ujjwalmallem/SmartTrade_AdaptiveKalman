"""
Path-label training harvest (SYSTEM_SPEC §3.1).

Walks Kalman pairs, records per-bar feature rows while a position is open,
labels each bar with ``label_path_bars``, and writes
``results/exit_training_dataset.csv`` — without touching ``paper_trades.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    entry_min_confidence,
    entry_z_threshold,
    load_strategy_config,
    training_min_samples,
)
from src.features import FEATURE_NAMES, extract_feature_vector
from src.half_life import estimate_half_life
from src.kalman import AdaptiveKalmanPairs, KalmanNoiseModel
from src.train_exit_model import label_path_bars


def _entry_direction(z: float, conf: float, z_thr: float, conf_thr: float) -> int:
    if conf <= conf_thr:
        return 0
    if z < -z_thr:
        return 1
    if z > z_thr:
        return -1
    return 0


def _path_rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def harvest_pair_paths(
    df: pd.DataFrame,
    *,
    ticker_a: str,
    ticker_b: str,
    basket: str = "",
    z_entry: float = 2.0,
    min_confidence: float = 0.55,
    stop_loss_z: float = 4.0,
    hl_lookback: int = 80,
    max_half_life_multiplier: float = 2.5,
    absolute_min_bars: int = 5,
    trade_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate entries on ``df`` and emit labeled per-bar training rows.

    Holds until time-stop or hard z-stop (no soft MR) so paths are long enough
    for lookahead labeling.
    """
    if trade_year is None:
        trade_year = int(pd.Timestamp(df.index.max()).year)

    out_rows: List[Dict[str, Any]] = []
    position = 0
    entry_z = 0.0
    entry_idx = 0
    path: List[Dict[str, Any]] = []
    path_hl = 20.0

    for i in range(60, len(df)):
        ts = df.index[i]
        if int(pd.Timestamp(ts).year) != int(trade_year):
            if position != 0 and path:
                # Year boundary: flush open path without labels past year
                position = 0
                path = []
            continue

        row = df.iloc[i]
        z = float(row["zscore"])
        conf = float(row.get("confidence", 0.5))
        vol = float(row.get("spread_vol", 1.0))
        vel = float(row.get("spread_velocity", 0.0))

        if position == 0:
            direction = _entry_direction(z, conf, z_entry, min_confidence)
            if direction == 0:
                continue
            position = direction
            entry_z = z
            entry_idx = i
            spread_hist = df["spread"].iloc[max(0, i - hl_lookback) : i + 1]
            path_hl = estimate_half_life(spread_hist, lookback=hl_lookback)
            path = []

        bars_held = i - entry_idx
        feats = extract_feature_vector(
            entry_z=entry_z,
            current_z=z,
            direction=position,
            vol=vol,
            confidence=conf,
            velocity=vel,
            bars_held=bars_held,
            half_life=path_hl,
        )
        path.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "basket": basket,
            "direction": "LONG_SPREAD" if position == 1 else "SHORT_SPREAD",
            "exit_z": z,
            "pnl_proxy": float(feats[FEATURE_NAMES.index("pnl_proxy")]),
            "half_life_bars": float(path_hl),
            "bars_held": float(bars_held),
            **{name: float(feats[j]) for j, name in enumerate(FEATURE_NAMES)},
        })

        max_bars = int(max(absolute_min_bars, np.ceil(max_half_life_multiplier * path_hl)))
        stop = (
            (position == 1 and z <= -stop_loss_z)
            or (position == -1 and z >= stop_loss_z)
            or bars_held >= max_bars
        )
        if not stop:
            continue

        path_df = _path_rows_to_frame(path)
        labels = label_path_bars(
            path_df,
            pnl_col="pnl_proxy",
            half_life_col="half_life_bars",
            exit_z_col="exit_z",
            direction_col="direction",
            stop_loss_z=stop_loss_z,
            multiplier=max_half_life_multiplier,
        )
        path_df = path_df.copy()
        path_df["label"] = labels.to_numpy()
        # Drop final bar if horizon cannot look ahead (always label_path handles)
        out_rows.extend(path_df.to_dict(orient="records"))
        position = 0
        path = []

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def harvest_training_dataset(
    pair_frames: Dict[str, Tuple[pd.DataFrame, str, str, str]],
    *,
    results_dir: Union[str, Path] = "results",
    cfg: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    data_source: str = "harvest",
) -> Tuple[Path, pd.DataFrame]:
    """
    ``pair_frames``: label → (kalman_df, ticker_a, ticker_b, basket)

    Writes/replaces ``exit_training_dataset.csv`` (feat_* columns + label).
    """
    cfg = cfg or load_strategy_config()
    entry = cfg.get("entry") or {}
    risk = cfg.get("risk_engine") or {}
    z_thr = float(entry.get("z_entry", entry_z_threshold(cfg)))
    conf_thr = float(entry.get("min_confidence", entry_min_confidence(cfg)))
    stop_z = float(risk.get("stop_loss_z", 4.0))
    hl_mult = float(risk.get("max_half_life_multiplier", 2.5))
    abs_min = int(risk.get("absolute_min_bars", 5))

    run_id = run_id or pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    chunks: List[pd.DataFrame] = []
    for label, (df, a, b, basket) in pair_frames.items():
        part = harvest_pair_paths(
            df,
            ticker_a=a,
            ticker_b=b,
            basket=basket,
            z_entry=z_thr,
            min_confidence=conf_thr,
            stop_loss_z=stop_z,
            max_half_life_multiplier=hl_mult,
            absolute_min_bars=abs_min,
        )
        if part.empty:
            continue
        part = part.copy()
        part["run_id"] = run_id
        part["data_source"] = data_source
        part["trade_id"] = range(1, len(part) + 1)
        part["pair"] = label
        chunks.append(part)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "exit_training_dataset.csv"
    if not chunks:
        empty = pd.DataFrame(columns=[
            "run_id", "data_source", "trade_id", "ticker_a", "ticker_b", "basket",
            *[f"feat_{n}" for n in FEATURE_NAMES], "label",
        ])
        empty.to_csv(out, index=False)
        print(f"⚠️  Path harvest produced 0 rows → {out}")
        return out, empty

    raw = pd.concat(chunks, ignore_index=True)
    # Training frame schema expected by train_exit_model / load_training_frame
    framed = pd.DataFrame({
        "run_id": raw["run_id"],
        "data_source": raw["data_source"],
        "trade_id": raw["trade_id"],
        "ticker_a": raw["ticker_a"],
        "ticker_b": raw["ticker_b"],
        "basket": raw["basket"],
        **{f"feat_{n}": raw[n] for n in FEATURE_NAMES},
        "label": raw["label"].astype(int),
    })
    # Light dedupe on feature fingerprint
    fp = [f"feat_{n}" for n in FEATURE_NAMES] + ["label", "ticker_a", "ticker_b"]
    framed = framed.drop_duplicates(subset=fp, keep="last").reset_index(drop=True)
    framed.to_csv(out, index=False)
    pos = float(framed["label"].mean()) if len(framed) else 0.0
    hl = framed["feat_half_life"] * 30.0
    print(
        f"🌾 Path harvest → {out}  rows={len(framed)}  "
        f"pos_rate={pos:.1%}  "
        f"half_life_bars[min/med/max]="
        f"{hl.min():.1f}/{hl.median():.1f}/{hl.max():.1f}"
    )
    floor = training_min_samples(cfg)
    if len(framed) < floor:
        print(f"⚠️  Harvest below training.min_samples={floor}")
    return out, framed
