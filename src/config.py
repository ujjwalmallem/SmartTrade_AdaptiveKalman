"""Load strategy_config.yaml as the single source of truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path("config/strategy_config.yaml")

_DEFAULTS: Dict[str, Any] = {
    "exit_model": {
        "model_path": "models/logistic_exit_model.pkl",
        "scaler_path": "models/feature_scaler.pkl",
        "metadata_path": "models/model_metadata.json",
        "probability_threshold": 0.68,
        "backend": "sklearn",
    },
    "training": {
        "min_samples": 50,
        "calibrate_min_samples": 8,
        "good_pnl_threshold": 0.35,
        "penalty": "l2",
        "C": 1.0,
    },
    "risk_engine": {
        "max_half_life_multiplier": 2.5,
        "absolute_min_bars": 5,
        "stop_loss_z": 4.0,
        "hard_pnl_stop_dollars": -150.0,
        "soft_mr_long_z": -0.35,
        "soft_mr_short_z": 0.35,
    },
    "entry": {"z_entry": 2.0, "min_confidence": 0.55},
    "execution": {
        "broker": "alpaca_paper",
        "order_type": "market",
        "cost_bps": 4.0,
        "risk_frac": 0.08,
        "capital": 100_000.0,
    },
    "kalman": {"delta": 1e-4, "R_base": 1e-2, "noise_model": "standard"},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_strategy_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULTS, raw)


def exit_threshold(cfg: Optional[Dict[str, Any]] = None) -> float:
    cfg = cfg or load_strategy_config()
    return float((cfg.get("exit_model") or {}).get("probability_threshold", 0.68))


def training_min_samples(cfg: Optional[Dict[str, Any]] = None) -> int:
    cfg = cfg or load_strategy_config()
    return int((cfg.get("training") or {}).get("min_samples", 50))
