"""
Production StatArb exit manager (SYSTEM_SPEC §4).

Deterministic risk safeguards run before the ML probability check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import yaml

from src.features import FEATURE_NAMES, load_scaler, transform_features

DEFAULT_CONFIG_PATH = Path("config/strategy_config.yaml")


@dataclass
class TradeState:
    trade_id: int
    ticker_a: str
    ticker_b: str
    direction: str  # "LONG_SPREAD" | "SHORT_SPREAD" or "+1"/"-1"
    bars_held: int
    half_life_bars: float
    vol: float
    pnl_proxy: float
    entry_z: float
    confidence: float
    exit_z: float
    velocity: float
    half_life: float
    cost_dollars: float = 0.0
    pnl_dollars: float = 0.0

    def direction_sign(self) -> int:
        d = str(self.direction).upper()
        if d in ("1", "+1", "LONG", "LONG_SPREAD"):
            return 1
        if d in ("-1", "SHORT", "SHORT_SPREAD"):
            return -1
        try:
            return 1 if int(float(d)) >= 0 else -1
        except ValueError as exc:
            raise ValueError(f"Unknown direction {self.direction!r}") from exc


def load_strategy_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "exit_model": {"probability_threshold": 0.68},
            "risk_engine": {
                "max_half_life_multiplier": 2.5,
                "absolute_min_bars": 5,
                "stop_loss_z": 4.0,
                "hard_pnl_stop_dollars": -150.0,
            },
        }
    with path.open() as f:
        return yaml.safe_load(f) or {}


def time_stop_bars(
    half_life_bars: float,
    *,
    multiplier: float = 2.5,
    absolute_min_bars: int = 5,
) -> int:
    """Floor at absolute_min_bars, then scale with half-life (SYSTEM_SPEC errata)."""
    return int(max(absolute_min_bars, np.ceil(multiplier * float(half_life_bars))))


class StatArbExitManager:
    """
    Engine risk first, then scaled logistic exit probability.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        exit_threshold: float = 0.68,
        *,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        max_half_life_multiplier: float = 2.5,
        absolute_min_bars: int = 5,
        stop_loss_z: float = 4.0,
        hard_pnl_stop_dollars: float = -150.0,
        model: Any = None,
        scaler: Any = None,
    ):
        cfg = load_strategy_config(config_path)
        exit_cfg = cfg.get("exit_model") or {}
        risk_cfg = cfg.get("risk_engine") or {}

        self.exit_threshold = float(
            exit_threshold
            if exit_threshold is not None
            else exit_cfg.get("probability_threshold", 0.68)
        )
        self.max_half_life_multiplier = float(
            risk_cfg.get("max_half_life_multiplier", max_half_life_multiplier)
        )
        self.absolute_min_bars = int(
            risk_cfg.get("absolute_min_bars", absolute_min_bars)
        )
        self.stop_loss_z = float(risk_cfg.get("stop_loss_z", stop_loss_z))
        self.hard_pnl_stop_dollars = float(
            risk_cfg.get("hard_pnl_stop_dollars", hard_pnl_stop_dollars)
        )
        self.feature_names = list(FEATURE_NAMES)

        model_path = model_path or exit_cfg.get("model_path", "models/logistic_exit_model.pkl")
        scaler_path = scaler_path or exit_cfg.get(
            "scaler_path", "models/feature_scaler.pkl"
        )

        if model is not None:
            self.model = model
        else:
            mp = Path(model_path)
            self.model = joblib.load(mp) if mp.exists() else None

        if scaler is not None:
            self.scaler = scaler
        else:
            sp = Path(scaler_path)
            self.scaler = load_scaler(sp) if sp.exists() else None

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "StatArbExitManager":
        cfg = load_strategy_config(config_path)
        exit_cfg = cfg.get("exit_model") or {}
        return cls(
            model_path=exit_cfg.get("model_path"),
            scaler_path=exit_cfg.get("scaler_path"),
            exit_threshold=float(exit_cfg.get("probability_threshold", 0.68)),
            config_path=config_path,
        )

    def _raw_feature_row(self, state: TradeState) -> np.ndarray:
        return np.array(
            [
                [
                    float(state.vol),
                    float(state.pnl_proxy),
                    abs(float(state.entry_z)),  # symmetry
                    float(state.confidence),
                    float(state.exit_z),
                    float(state.velocity),
                    float(state.bars_held),
                    float(state.half_life),
                ]
            ],
            dtype=float,
        )

    def evaluate_trade(self, state: TradeState) -> Tuple[bool, str, float]:
        # 1. Deterministic time-stop
        max_bars = time_stop_bars(
            state.half_life_bars,
            multiplier=self.max_half_life_multiplier,
            absolute_min_bars=self.absolute_min_bars,
        )
        if int(state.bars_held) >= max_bars:
            return (
                True,
                f"TIME_STOP_EXPIRED (Held {state.bars_held} >= {max_bars} bars)",
                1.0,
            )

        # 2. Hard stop-loss / structural blow-out
        sign = state.direction_sign()
        z = float(state.exit_z)
        if sign == 1 and z <= -self.stop_loss_z:
            return True, f"STOP_LOSS_Z (long exit_z={z:.2f})", 1.0
        if sign == -1 and z >= self.stop_loss_z:
            return True, f"STOP_LOSS_Z (short exit_z={z:.2f})", 1.0
        if float(state.pnl_dollars) <= self.hard_pnl_stop_dollars:
            return (
                True,
                f"HARD_PNL_STOP (pnl=${state.pnl_dollars:.2f} <= {self.hard_pnl_stop_dollars})",
                1.0,
            )

        # 3. ML probability (requires fitted model + scaler)
        if self.model is None or self.scaler is None:
            return False, "HOLD (no exit model loaded)", 0.0

        raw = self._raw_feature_row(state)
        scaled = transform_features(self.scaler, raw)
        proba = self.model.predict_proba(scaled)[0]
        # binary classifier: column 1 = P(exit / class 1)
        prob_exit = float(proba[1] if len(proba) > 1 else proba[0])

        if prob_exit >= self.exit_threshold:
            return (
                True,
                f"ML_EXIT_TRIGGERED (Prob {prob_exit:.3f} >= {self.exit_threshold})",
                prob_exit,
            )
        return (
            False,
            f"HOLD (Prob {prob_exit:.3f} < {self.exit_threshold})",
            prob_exit,
        )
