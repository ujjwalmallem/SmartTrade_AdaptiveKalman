"""Adaptive Kalman pairs filter (extracted from paper_trading_ml_exit)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

class KalmanNoiseModel(str, Enum):
    STANDARD = "standard"
    VOLUME = "volume"        # needs volume series
    PARKINSON = "parkinson"  # needs high & low series


@dataclass
class AdaptiveKalmanPairs:
    """
    Online Kalman filter for pairs: state = [α, β]
    Observation: price_a = α + β * price_b + v

    Adaptive features
    -----------------
    • Process noise Q is scaled by recent innovation magnitude
    • Measurement noise R can be:
        - STANDARD   : fixed (price-scale calibrated)
        - VOLUME     : shrinks when volume is high (more trust)
        - PARKINSON  : grows with high-low range (volatility)
    """
    delta: float = 1e-4                 # base process-noise scale
    R_base: float = 1e-2                # base measurement noise
    adapt_window: int = 20
    min_conf: float = 0.25
    max_conf: float = 0.95
    noise_model: KalmanNoiseModel = KalmanNoiseModel.STANDARD

    # Optional scaling factors for the adaptive modes
    volume_power: float = 0.6
    parkinson_power: float = 1.0
    r_floor: float = 1e-4
    r_ceil: float = 5.0

    # Back-compat alias used by older call sites (R=...)
    R: Optional[float] = None

    def __post_init__(self):
        if self.R is not None:
            self.R_base = float(self.R)
        if isinstance(self.noise_model, str):
            self.noise_model = KalmanNoiseModel(self.noise_model)
        self.reset()

    def reset(self):
        self.x = np.zeros(2)                     # [α, β]
        self.P = np.eye(2) * 1.0
        self.Q_base = self.delta * np.eye(2)
        self.innovations: List[float] = []
        self._ewma_var = 1e-4
        self.history: List[dict] = []

    def _adapt_Q(self) -> np.ndarray:
        if len(self.innovations) < 5:
            return self.Q_base.copy()
        recent = np.array(self.innovations[-self.adapt_window:])
        scale = np.clip(np.std(recent) / (np.mean(np.abs(recent)) + 1e-8), 0.3, 5.0)
        return self.Q_base * scale

    def _adapt_R(
        self,
        volume: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        if self.noise_model == KalmanNoiseModel.STANDARD:
            return float(self.R_base)

        if self.noise_model == KalmanNoiseModel.VOLUME:
            if volume is None or volume <= 0:
                return float(self.R_base)
            # `volume` is expected as relative volume (≈1.0 = typical).
            # Higher relative volume → lower R (more trust in the print).
            vol_factor = 1.0 / (1.0 + (float(volume) ** self.volume_power))
            R = self.R_base * (0.35 + 1.3 * vol_factor)
            return float(np.clip(R, self.r_floor, self.r_ceil))

        if self.noise_model == KalmanNoiseModel.PARKINSON:
            if high is None or low is None or high <= low:
                return float(self.R_base)
            range_proxy = max(np.log(high / low), 1e-6) ** 2
            R = self.R_base * (1.0 + self.parkinson_power * range_proxy * 100)
            return float(np.clip(R, self.r_floor, self.r_ceil))

        return float(self.R_base)

    def update(
        self,
        price_a: float,
        price_b: float,
        volume: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> dict:
        H = np.array([1.0, price_b])

        x_prior = self.x.copy()
        Q = self._adapt_Q()
        P_prior = self.P + Q

        R = self._adapt_R(volume=volume, high=high, low=low, price=price_a)

        y_pred = H @ x_prior
        innov = price_a - y_pred
        S = float(H @ P_prior @ H.T + R)
        K = (P_prior @ H.T) / S

        self.x = x_prior + K * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ P_prior

        self.innovations.append(float(innov))
        if len(self.innovations) > 200:
            self.innovations = self.innovations[-200:]

        self._ewma_var = 0.94 * self._ewma_var + 0.06 * innov**2
        spread_std = float(np.sqrt(self._ewma_var + 1e-8))
        z = innov / spread_std

        conf = 1.0 / (1.0 + np.sqrt(S / max(R, 1e-8)))
        conf = float(np.clip(conf, self.min_conf, self.max_conf))

        out = {
            "alpha": float(self.x[0]),
            "beta": float(self.x[1]),
            "spread": float(innov),
            "spread_std": float(spread_std),
            "zscore": float(z),
            "confidence": conf,
            "innovation": float(innov),
            "R": float(R),
            "kalman_gain_beta": float(K[1]),
        }
        self.history.append(out)
        return out

    def filter_pair(
        self,
        a: pd.Series,
        b: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run the filter over two aligned price series.
        Optional volume / high / low enable the adaptive R modes.
        """
        self.reset()
        a = a.astype(float).dropna()
        b = b.astype(float).dropna()
        common = a.index.intersection(b.index)

        if volume is not None:
            volume = volume.reindex(common).fillna(0)
            # Relative volume vs rolling median → stable VOLUME-mode R scaling
            vol_med = volume.replace(0, np.nan).rolling(20, min_periods=5).median()
            vol_med = vol_med.fillna(volume.replace(0, np.nan).median()).fillna(1.0)
            volume = (volume / vol_med.replace(0, np.nan)).fillna(1.0).clip(0.05, 20.0)
        if high is not None:
            high = high.reindex(common)
        if low is not None:
            low = low.reindex(common)

        a, b = a.loc[common], b.loc[common]

        # Calibrate base noise to absolute price scale
        price_scale = float(max(np.nanmedian(np.abs(a.values)), 1.0))
        self.R_base = max((price_scale ** 2) * 1e-4, self.r_floor)
        self.Q_base = (self.delta * (price_scale ** 2)) * np.eye(2)
        # For adaptive modes, raise ceiling with price scale
        self.r_ceil = max(self.r_ceil, self.R_base * 50)

        rows = []
        prev_spread = 0.0

        for i, ts in enumerate(common):
            vol = float(volume.iloc[i]) if volume is not None else None
            hi = float(high.iloc[i]) if high is not None and pd.notna(high.iloc[i]) else None
            lo = float(low.iloc[i]) if low is not None and pd.notna(low.iloc[i]) else None

            st = self.update(
                price_a=float(a.iloc[i]),
                price_b=float(b.iloc[i]),
                volume=vol,
                high=hi,
                low=lo,
            )
            spread = st["spread"]
            velocity = spread - prev_spread
            prev_spread = spread

            rows.append({
                "price_a": float(a.iloc[i]),
                "price_b": float(b.iloc[i]),
                "alpha": st["alpha"],
                "beta": st["beta"],
                "spread": spread,
                "zscore": st["zscore"],
                "confidence": st["confidence"],
                "spread_velocity": velocity,
                "spread_vol": st["spread_std"],
                "innovation": st["innovation"],
                "R": st["R"],
            })

        df = pd.DataFrame(rows, index=common)

        df["spread_velocity"] = (
            df["spread"].diff().rolling(5, min_periods=1).mean().fillna(0)
        )
        df["spread_vol"] = (
            df["spread"].rolling(15, min_periods=5).std().fillna(df["spread_vol"])
        )
        # Keep online z, but use rolling residual z for trading thresholds (±2)
        df["zscore_online"] = df["zscore"]
        roll_mu = df["spread"].rolling(40, min_periods=10).mean()
        roll_sd = df["spread"].rolling(40, min_periods=10).std()
        df["zscore"] = ((df["spread"] - roll_mu) / roll_sd.replace(0, np.nan)).fillna(0.0)
        roll = df["spread"].rolling(20, min_periods=5).std()
        med = float(roll.median()) if roll.notna().any() else 1.0
        df["confidence"] = (med / (med + roll)).clip(self.min_conf, self.max_conf).fillna(self.min_conf)
        return df



