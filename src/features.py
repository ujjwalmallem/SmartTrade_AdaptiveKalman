"""
Exit-model feature engineering (SYSTEM_SPEC §2).

Eight direction-symmetric, non-collinear features. Drops raw entry_z,
favorable, and best_fav.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Spec order — do not reorder without retraining
FEATURE_NAMES = [
    "vol",
    "pnl_proxy",
    "abs_entry_z",
    "confidence",
    "exit_z",
    "velocity",
    "bars_held",
    "half_life",
]

DROPPED_FEATURES = ("entry_z", "favorable", "best_fav")

FEATURE_SCHEMA_VERSION = 3  # SYSTEM_SPEC 8-feature set


def pnl_proxy_z(entry_z: float, current_z: float, direction: int) -> float:
    """Direction-aware unrealized PnL in z-units."""
    if int(direction) == 1:
        return float(current_z) - float(entry_z)
    return float(entry_z) - float(current_z)


def extract_feature_dict(
    *,
    entry_z: float,
    current_z: float,
    direction: int,
    vol: float,
    confidence: float,
    velocity: float,
    bars_held: int,
    half_life: float,
) -> Dict[str, float]:
    """Build the canonical 8-feature dict (long/short symmetric on entry depth)."""
    return {
        "vol": float(vol),
        "pnl_proxy": pnl_proxy_z(entry_z, current_z, direction),
        "abs_entry_z": abs(float(entry_z)),
        "confidence": float(confidence),
        "exit_z": float(current_z),
        "velocity": float(velocity),
        "bars_held": float(bars_held),
        "half_life": float(half_life) / 30.0,  # normalized units
    }


def extract_feature_vector(
    *,
    entry_z: float,
    current_z: float,
    direction: int,
    vol: float,
    confidence: float,
    velocity: float,
    bars_held: int,
    half_life: float,
) -> np.ndarray:
    """Return shape (8,) vector in FEATURE_NAMES order."""
    d = extract_feature_dict(
        entry_z=entry_z,
        current_z=current_z,
        direction=direction,
        vol=vol,
        confidence=confidence,
        velocity=velocity,
        bars_held=bars_held,
        half_life=half_life,
    )
    return np.array([d[n] for n in FEATURE_NAMES], dtype=float)


def features_from_row(
    position: Any,
    row: Mapping[str, Any],
    bars_held: int,
    direction: int,
    half_life: float = 20.0,
) -> np.ndarray:
    """
    Extract features from a Kalman bar row + open position.

    `position` must expose `.entry_z`. Velocity prefers z-score change; falls
    back to spread_velocity when z history is unavailable on the row.
    """
    z = float(row["zscore"])
    conf = float(row.get("confidence", 0.5))
    vol = float(row.get("spread_vol", 1.0))
    vel = float(row.get("z_velocity", row.get("spread_velocity", 0.0)))
    return extract_feature_vector(
        entry_z=float(position.entry_z),
        current_z=z,
        direction=int(direction),
        vol=vol,
        confidence=conf,
        velocity=vel,
        bars_held=int(bars_held),
        half_life=float(half_life),
    )


def vector_from_mapping(feat: Mapping[str, Any]) -> np.ndarray:
    """Pull FEATURE_NAMES from a dict / Series; missing keys → NaN."""
    return np.array([float(feat.get(n, np.nan)) for n in FEATURE_NAMES], dtype=float)


def frame_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build X from a training frame.

    Accepts either bare names (`vol`) or journal columns (`feat_vol`).
    """
    cols = []
    for n in FEATURE_NAMES:
        if n in df.columns:
            cols.append(n)
        elif f"feat_{n}" in df.columns:
            cols.append(f"feat_{n}")
        else:
            raise ValueError(
                f"Training frame missing feature {n!r} (also tried feat_{n}). "
                f"Columns: {list(df.columns)}"
            )
    return df[cols].to_numpy(dtype=float)


def fit_scaler(X: np.ndarray) -> StandardScaler:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected X shape (n, {len(FEATURE_NAMES)}), got {X.shape}")
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def save_scaler(scaler: StandardScaler, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def load_scaler(path: Union[str, Path]) -> StandardScaler:
    return joblib.load(Path(path))


def transform_features(
    scaler: StandardScaler,
    X: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return scaler.transform(X)
