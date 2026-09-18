"""
Offline labeling + sklearn exit-model training (SYSTEM_SPEC §3).

Artifacts:
  models/logistic_exit_model.pkl
  models/feature_scaler.pkl
  models/model_metadata.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    fit_scaler,
    frame_to_matrix,
    save_scaler,
)
from src.exit_manager import load_strategy_config

DEFAULT_CONFIG = Path("config/strategy_config.yaml")
DEFAULT_RESULTS = Path("results")


def lookahead_horizon(half_life_bars: float, multiplier: float = 2.5) -> int:
    return int(max(1, np.ceil(multiplier * float(half_life_bars))))


def label_path_bars(
    bars: pd.DataFrame,
    *,
    pnl_col: str = "pnl_proxy",
    half_life_col: str = "half_life_bars",
    cost_per_exit: float = 0.0,
    stop_loss_z: float = 4.0,
    exit_z_col: str = "exit_z",
    direction_col: str = "direction",
    multiplier: float = 2.5,
) -> pd.Series:
    """
    Binary labels for each bar t of an active trade (SYSTEM_SPEC §3.1).

    y_t = 1 iff within h ≤ H bars there exists a future point where
    ΔPnL_net > 0 and MAE stays inside the stop-loss band.
    """
    if bars.empty:
        return pd.Series(dtype=int)

    n = len(bars)
    pnl = bars[pnl_col].to_numpy(dtype=float)
    hl = bars[half_life_col].to_numpy(dtype=float) if half_life_col in bars.columns else np.full(n, 8.0)
    z = bars[exit_z_col].to_numpy(dtype=float) if exit_z_col in bars.columns else np.zeros(n)
    if direction_col in bars.columns:
        raw = bars[direction_col]
        direction = np.array([
            1 if str(v).upper() in ("1", "+1", "LONG", "LONG_SPREAD") else -1
            for v in raw
        ], dtype=int)
    else:
        direction = np.ones(n, dtype=int)

    y = np.zeros(n, dtype=int)
    for t in range(n):
        H = lookahead_horizon(hl[t], multiplier=multiplier)
        end = min(n, t + H + 1)
        base = pnl[t]
        ok = False
        for h in range(t + 1, end):
            # adverse excursion along the path t→h
            window_z = z[t : h + 1]
            if direction[t] == 1:
                mae_hit = bool(np.any(window_z <= -stop_loss_z))
            else:
                mae_hit = bool(np.any(window_z >= stop_loss_z))
            if mae_hit:
                break
            delta_net = (pnl[h] - base) - cost_per_exit
            if delta_net > 0:
                ok = True
                break
        y[t] = 1 if ok else 0
    return pd.Series(y, index=bars.index, name="label")


def label_closed_trade_row(
    row: Mapping[str, Any],
    *,
    good_pnl_threshold: float = 0.35,
    stop_loss_z: float = 4.0,
) -> int:
    """
    Fallback label when only closed-trade summaries exist (no per-bar path).

    Approximates §3.1: good if net z-PnL cleared costs/threshold and exit
    did not breach the stop band.
    """
    pnl = float(row.get("pnl_z", row.get("pnl_proxy", 0.0)) or 0.0)
    exit_z = float(row.get("exit_z", 0.0) or 0.0)
    direction = str(row.get("direction", "LONG_SPREAD")).upper()
    sign = 1 if "LONG" in direction or direction in ("1", "+1") else -1
    if sign == 1 and exit_z <= -stop_loss_z:
        return 0
    if sign == -1 and exit_z >= stop_loss_z:
        return 0
    bars = int(row.get("bars_held", 0) or 0)
    if pnl >= good_pnl_threshold:
        return 1
    if pnl > 0.05 and 8 <= bars <= 22:
        return 1
    if bars >= 25 and pnl > -0.6:
        return 1
    return 0


def load_training_frame(
    results_dir: Union[str, Path] = DEFAULT_RESULTS,
) -> pd.DataFrame:
    """Prefer exit_training_dataset.csv; else build from paper_trades.csv."""
    results_dir = Path(results_dir)
    ds = results_dir / "exit_training_dataset.csv"
    trades = results_dir / "paper_trades.csv"
    if ds.exists():
        df = pd.read_csv(ds)
        if "label" not in df.columns:
            raise ValueError(f"{ds} missing label column")
        return df
    if not trades.exists():
        raise FileNotFoundError(
            f"No training data under {results_dir} "
            "(need exit_training_dataset.csv or paper_trades.csv)"
        )
    journal = pd.read_csv(trades)
    if "status" in journal.columns:
        closed = journal[journal["status"].fillna("CLOSED") == "CLOSED"].copy()
    else:
        closed = journal.copy()
    if closed.empty:
        raise ValueError("paper_trades.csv has no CLOSED rows to train on")
    # Map journal feat_* or reconstruct minimal features
    rows = []
    for _, r in closed.iterrows():
        feat = {}
        for n in FEATURE_NAMES:
            key = f"feat_{n}"
            if key in closed.columns and pd.notna(r.get(key)):
                feat[n] = float(r[key])
            elif n in closed.columns and pd.notna(r.get(n)):
                feat[n] = float(r[n])
        if len(feat) < len(FEATURE_NAMES):
            # reconstruct from trade fields
            entry_z = float(r.get("entry_z", 0.0) or 0.0)
            exit_z = float(r.get("exit_z", 0.0) or 0.0)
            pnl = float(r.get("pnl_z", 0.0) or 0.0)
            bars = int(r.get("bars_held", 0) or 0)
            feat = {
                "vol": float(r.get("feat_vol", 1.0) or 1.0),
                "pnl_proxy": pnl,
                "abs_entry_z": abs(entry_z),
                "confidence": float(r.get("feat_confidence", 0.7) or 0.7),
                "exit_z": exit_z,
                "velocity": float(r.get("feat_velocity", 0.0) or 0.0),
                "bars_held": float(bars),
                "half_life": float(r.get("feat_half_life", 20.0 / 30.0) or (20.0 / 30.0)),
            }
        label = r["label"] if "label" in closed.columns and pd.notna(r.get("label")) else label_closed_trade_row(r)
        rows.append({**feat, "label": int(label)})
    return pd.DataFrame(rows)


def train_exit_model(
    results_dir: Union[str, Path] = DEFAULT_RESULTS,
    *,
    config_path: Union[str, Path] = DEFAULT_CONFIG,
    calibrate: bool = True,
    min_samples: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
    penalty: str = "l2",
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Fit StandardScaler + LogisticRegression (+ optional calibration).

    Returns metadata dict and writes artifacts listed in strategy_config.yaml.
    """
    cfg = load_strategy_config(config_path)
    exit_cfg = cfg.get("exit_model") or {}
    risk_cfg = cfg.get("risk_engine") or {}

    model_path = Path(exit_cfg.get("model_path", "models/logistic_exit_model.pkl"))
    scaler_path = Path(exit_cfg.get("scaler_path", "models/feature_scaler.pkl"))
    meta_path = Path(exit_cfg.get("metadata_path", "models/model_metadata.json"))

    df = load_training_frame(results_dir)
    # Ensure feature columns exist as bare names for frame_to_matrix
    for n in FEATURE_NAMES:
        if n not in df.columns and f"feat_{n}" in df.columns:
            df[n] = df[f"feat_{n}"]
    if "label" not in df.columns:
        stop_z = float(risk_cfg.get("stop_loss_z", 4.0))
        df["label"] = [label_closed_trade_row(r, stop_loss_z=stop_z) for _, r in df.iterrows()]

    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    y = df["label"].astype(int).to_numpy()
    if len(y) < min_samples:
        raise ValueError(f"Need at least {min_samples} labeled rows; found {len(y)}")
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels are a single class — collect more diverse exits")

    X = frame_to_matrix(df)
    scaler = fit_scaler(X)
    Xs = scaler.transform(X)

    strat = y if (y.sum() > 1 and (len(y) - y.sum()) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=min(test_size, 0.5), random_state=random_state, stratify=strat
    )

    base = LogisticRegression(
        penalty=penalty if penalty in ("l1", "l2") else "l2",
        C=float(C),
        solver="saga" if penalty == "l1" else "lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    base.fit(X_train, y_train)

    if calibrate and len(X_train) >= 8:
        model: Any = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train, y_train)
    else:
        model = base

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        metrics["roc_auc"] = None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_scaler(scaler, scaler_path)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "schema_version": FEATURE_SCHEMA_VERSION,
        "penalty": penalty,
        "C": C,
        "calibrated": bool(calibrate and len(X_train) >= 8),
        "probability_threshold": float(exit_cfg.get("probability_threshold", 0.68)),
        "metrics": metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Exit model → {model_path}")
    print(f"✅ Scaler     → {scaler_path}")
    print(f"✅ Metadata   → {meta_path}")
    print(f"   samples={metrics['n_samples']}  acc={metrics['accuracy']:.3f}  "
          f"auc={metrics['roc_auc']}")
    return meta


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train StatArb ML exit model (SYSTEM_SPEC)")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l2")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--min-samples", type=int, default=4)
    args = p.parse_args(argv)
    train_exit_model(
        args.results_dir,
        config_path=args.config,
        calibrate=not args.no_calibrate,
        penalty=args.penalty,
        C=args.C,
        min_samples=args.min_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
