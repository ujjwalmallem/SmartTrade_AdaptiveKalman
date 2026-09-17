"""
Paper Trading + ML Exit Model Trainer
Ready for Cursor AI
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from enum import Enum
from types import SimpleNamespace
from typing import List, Dict, Iterable, Optional, Sequence, Tuple, Any
import os
import warnings
warnings.filterwarnings("ignore")

from alpaca_paper_broker import (
    AlpacaPaperBroker,
    alpaca_credentials_present,
    fetch_daily_ohlcv as fetch_alpaca_daily_ohlcv,
)

# Default artifact locations (gitignored locally; CI uploads as artifacts)
RESULTS_DIR = Path("results")
TRADES_CSV = RESULTS_DIR / "paper_trades.csv"
DATASET_CSV = RESULTS_DIR / "exit_training_dataset.csv"
MODEL_JSON = RESULTS_DIR / "logistic_exit_model.json"
SETUPS_GLOB = "setups_*.csv"

FEATURE_NAMES = [
    "entry_z", "abs_entry_z", "pnl_proxy", "bars_held", "confidence",
    "velocity", "exit_z", "favorable", "best_fav", "vol",
    "half_life",
]

# ============================================================
# PASTE ALL PREVIOUS CLASSES HERE (or keep them in the same file)
# AdaptiveKalmanPairs + extract_exit_features + LogisticExitModel included below.
# PositionState, generate_training_data helpers, ExitConfig, etc. can be extended here.
# ============================================================

# For this standalone version I include minimal working versions
# so you can run it immediately.

# ============================================================
# TICKER UNIVERSES — Mag7 / Semis / Memory / Hyperscaler
# ============================================================

TICKER_UNIVERSES: Dict[str, List[str]] = {
    "mag7": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    ],
    "semis": [
        "NVDA", "AMD", "AVGO", "TSM", "QCOM", "ASML", "AMAT", "LRCX", "KLAC", "INTC",
    ],
    "memory": [
        "MU", "WDC", "STX", "SNDK",
    ],
    # Cloud / AI infra hyperscalers (often written "hyper scalar")
    "hyperscaler": [
        "AMZN", "MSFT", "GOOGL", "META", "ORCL", "CRM", "IBM",
    ],
}

# Preferred within-basket pairs (high cointegration interest); rest filled combinatorially
PREFERRED_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "mag7": [
        ("AAPL", "MSFT"), ("GOOGL", "META"), ("AMZN", "MSFT"),
        ("NVDA", "TSLA"), ("AAPL", "GOOGL"),
    ],
    "semis": [
        ("NVDA", "AMD"), ("AVGO", "TSM"), ("AMAT", "LRCX"),
        ("KLAC", "LRCX"), ("QCOM", "AVGO"), ("ASML", "TSM"),
    ],
    "memory": [
        ("MU", "WDC"), ("MU", "STX"), ("WDC", "STX"), ("MU", "SNDK"),
    ],
    "hyperscaler": [
        ("AMZN", "MSFT"), ("GOOGL", "META"), ("MSFT", "ORCL"),
        ("AMZN", "GOOGL"), ("ORCL", "CRM"),
    ],
}

# Cross-basket themes (AI supply chain: hyperscaler ↔ semis ↔ memory)
CROSS_BASKET_PAIRS: List[Tuple[str, str, str]] = [
    # (ticker_a, ticker_b, theme_label)
    ("NVDA", "MU", "semis_memory"),
    ("AMD", "MU", "semis_memory"),
    ("NVDA", "MSFT", "semis_hyperscaler"),
    ("NVDA", "AMZN", "semis_hyperscaler"),
    ("AVGO", "META", "semis_hyperscaler"),
    ("MU", "MSFT", "memory_hyperscaler"),
    ("TSM", "AAPL", "semis_mag7"),
]


def all_universe_tickers(baskets: Optional[Iterable[str]] = None) -> List[str]:
    """Unique tickers across selected baskets (default: all four)."""
    keys = list(baskets) if baskets is not None else list(TICKER_UNIVERSES.keys())
    seen = set()
    out: List[str] = []
    for key in keys:
        if key not in TICKER_UNIVERSES:
            raise KeyError(f"Unknown basket '{key}'. Choose from {list(TICKER_UNIVERSES)}")
        for t in TICKER_UNIVERSES[key]:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def ticker_baskets(ticker: str) -> List[str]:
    return [name for name, members in TICKER_UNIVERSES.items() if ticker in members]


@dataclass(frozen=True)
class PairSpec:
    ticker_a: str
    ticker_b: str
    basket: str  # primary basket or cross-theme label

    @property
    def label(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


def build_pair_universe(
    baskets: Optional[Sequence[str]] = None,
    include_cross: bool = True,
    max_pairs_per_basket: int = 8,
) -> List[PairSpec]:
    """
    Build tradable pair list from Mag7, semis, memory, and hyperscaler universes.
    Prefers curated pairs, then fills with combinations up to max_pairs_per_basket.
    """
    keys = list(baskets) if baskets is not None else list(TICKER_UNIVERSES.keys())
    pairs: List[PairSpec] = []
    seen = set()

    def _add(a: str, b: str, basket: str) -> None:
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        pairs.append(PairSpec(ticker_a=a, ticker_b=b, basket=basket))

    for basket in keys:
        preferred = PREFERRED_PAIRS.get(basket, [])
        for a, b in preferred:
            if a in TICKER_UNIVERSES[basket] and b in TICKER_UNIVERSES[basket]:
                _add(a, b, basket)
        # Fill with remaining within-basket combos if under the cap
        members = TICKER_UNIVERSES[basket]
        for a, b in itertools.combinations(members, 2):
            if sum(1 for p in pairs if p.basket == basket) >= max_pairs_per_basket:
                break
            _add(a, b, basket)

    if include_cross:
        allowed = set(all_universe_tickers(keys))
        for a, b, theme in CROSS_BASKET_PAIRS:
            if a in allowed and b in allowed:
                _add(a, b, theme)

    # Round-robin across baskets so scans cover Mag7 / semis / memory / hyperscaler
    # (and cross themes) instead of exhausting one basket first.
    by_basket: Dict[str, List[PairSpec]] = {}
    for p in pairs:
        by_basket.setdefault(p.basket, []).append(p)
    interleaved: List[PairSpec] = []
    buckets = [by_basket[k] for k in by_basket]
    while any(buckets):
        for bucket in buckets:
            if bucket:
                interleaved.append(bucket.pop(0))
    return interleaved


def summarize_universes(pairs: Sequence[PairSpec]) -> None:
    print("Ticker universes")
    print("-" * 60)
    for name, tickers in TICKER_UNIVERSES.items():
        print(f"  {name:12s} ({len(tickers):2d}): {', '.join(tickers)}")
    print(f"\nActive pairs: {len(pairs)}")
    by_basket: Dict[str, List[str]] = {}
    for p in pairs:
        by_basket.setdefault(p.basket, []).append(p.label)
    for basket, labels in by_basket.items():
        print(f"  [{basket}] {', '.join(labels)}")
    print("-" * 60)


# ---------- Minimal required pieces ----------
@dataclass
class PositionState:
    direction: int = 0
    entry_z: float = 0.0
    entry_bar: int = 0
    entry_spread: float = 0.0
    highest_favorable_z: float = 0.0
    current_size: float = 0.0
    bars_held: int = 0

def _align_ohlcv_panels(
    panels: Dict[str, pd.DataFrame],
    n_bars: int,
    min_bars: int,
    trade_year: Optional[int],
    source_label: str,
) -> Dict[str, pd.DataFrame]:
    """Align ticker OHLCV on shared days and restrict to latest trade year + warm-up."""
    closes = pd.DataFrame({t: panels[t]["Close"] for t in panels}).dropna(how="any")
    if closes.empty:
        raise RuntimeError(f"No overlapping trading days across {source_label} tickers")

    closes = closes.tail(max(n_bars, min_bars + 60))
    common_index = closes.index

    latest_year = int(pd.Timestamp(common_index.max()).year)
    year = int(trade_year) if trade_year is not None else latest_year
    if year != latest_year:
        print(f"⚠️  Requested trade_year={year} but latest bar is {latest_year}; using {latest_year}")
        year = latest_year

    in_year = common_index.year == year
    if not in_year.any():
        raise RuntimeError(f"No {source_label} bars in latest year {year}")

    first_trade_i = int(in_year.argmax())
    warm_start = max(0, first_trade_i - 60)
    final_index = common_index[warm_start:]

    aligned: Dict[str, pd.DataFrame] = {}
    for t in list(panels.keys()):
        frame = panels[t].reindex(final_index)
        if "Volume" in frame.columns:
            frame["Volume"] = frame["Volume"].fillna(0.0)
        for col in ("Open", "High", "Low", "Close"):
            if col in frame.columns:
                frame[col] = frame[col].ffill()
        frame.attrs["trade_year"] = year
        aligned[t] = frame

    trade_bars = int((final_index.year == year).sum())
    if trade_bars < 40:
        raise RuntimeError(
            f"Only {trade_bars} bars in trade year {year}; need more latest-year history"
        )

    print(
        f"{source_label} OHLCV ready: {len(aligned)} tickers × {len(final_index)} bars "
        f"[{final_index.min().date()} → {final_index.max().date()}] "
        f"(trades restricted to {year}: {trade_bars} bars)"
    )
    return aligned


def fetch_ohlcv_from_yfinance(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "2y",
    min_bars: int = 80,
    trade_year: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Daily OHLCV via yfinance (backup source)."""
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    print(f"Fetching yfinance OHLCV for {len(tickers)} tickers (period={period})...")

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    panels: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            if "Close" not in cols:
                continue
            df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if df["Close"].dropna().shape[0] >= min_bars:
                panels[t] = df
        except Exception:
            continue

    missing = [t for t in tickers if t not in panels]
    if missing:
        print(f"⚠️  Dropping tickers with no usable yfinance history: {', '.join(missing)}")
    if len(panels) < 2:
        raise RuntimeError(
            "yfinance returned usable OHLCV for fewer than 2 tickers; "
            "cannot build pairs for training"
        )
    return _align_ohlcv_panels(panels, n_bars, min_bars, trade_year, "yfinance")


def fetch_ohlcv_from_alpaca(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "2y",
    min_bars: int = 80,
    trade_year: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Daily OHLCV via Alpaca market data (primary source)."""
    if not alpaca_credentials_present():
        raise RuntimeError("Alpaca credentials missing")
    panels = fetch_alpaca_daily_ohlcv(tickers, period=period, min_bars=min_bars)
    return _align_ohlcv_panels(panels, n_bars, min_bars, trade_year, "alpaca")


def fetch_real_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "2y",
    min_bars: int = 80,
    trade_year: Optional[int] = None,
    data_source: str = "auto",
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Real daily OHLCV — Alpaca primary, yfinance backup (never synthetic).

    data_source:
      - auto     : try Alpaca, fall back to yfinance
      - alpaca   : Alpaca only
      - yfinance : yfinance only

    Returns (panels_dict, source_label) where panels are keyed by ticker with
    Open/High/Low/Close/Volume columns.
    """
    mode = (data_source or "auto").lower().strip()
    errors: List[str] = []

    if mode in ("auto", "alpaca"):
        try:
            panels = fetch_ohlcv_from_alpaca(
                tickers, n_bars=n_bars, period=period, min_bars=min_bars, trade_year=trade_year
            )
            return panels, "alpaca_live"
        except Exception as exc:
            msg = f"Alpaca OHLCV failed: {exc}"
            errors.append(msg)
            print(f"⚠️  {msg}")
            if mode == "alpaca":
                raise

    if mode in ("auto", "yfinance"):
        try:
            panels = fetch_ohlcv_from_yfinance(
                tickers, n_bars=n_bars, period=period, min_bars=min_bars, trade_year=trade_year
            )
            if errors:
                print("ℹ️  Using yfinance backup after Alpaca failure")
            return panels, "yfinance_live"
        except Exception as exc:
            errors.append(f"yfinance OHLCV failed: {exc}")
            if mode == "yfinance":
                raise

    raise RuntimeError(
        "Unable to load OHLCV from Alpaca or yfinance. " + " | ".join(errors)
    )


def ohlcv_field_panels(ticker_panels: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Convenience views: close / high / low / volume as ticker-column DataFrames.
    """
    closes = pd.DataFrame({t: df["Close"] for t, df in ticker_panels.items()})
    highs = pd.DataFrame({t: df["High"] for t, df in ticker_panels.items() if "High" in df.columns})
    lows = pd.DataFrame({t: df["Low"] for t, df in ticker_panels.items() if "Low" in df.columns})
    volumes = pd.DataFrame(
        {t: df["Volume"] for t, df in ticker_panels.items() if "Volume" in df.columns}
    ).fillna(0.0)

    trade_year = None
    for df in ticker_panels.values():
        trade_year = df.attrs.get("trade_year")
        if trade_year is not None:
            break
    if trade_year is not None:
        closes.attrs["trade_year"] = int(trade_year)
    return {"close": closes, "high": highs, "low": lows, "volume": volumes}


def fetch_market_panels_for_universe(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "2y",
    min_bars: int = 80,
    trade_year: Optional[int] = None,
    data_source: str = "auto",
) -> Dict[str, pd.DataFrame]:
    """
    Field-oriented OHLCV panels (close/high/low/volume).
    Thin wrapper over per-ticker fetch_real_prices_for_universe.
    """
    panels, _src = fetch_real_prices_for_universe(
        tickers,
        n_bars=n_bars,
        period=period,
        min_bars=min_bars,
        trade_year=trade_year,
        data_source=data_source,
    )
    return ohlcv_field_panels(panels)


def _load_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int,
    trade_year: Optional[int] = None,
    data_source: str = "auto",
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Load full OHLCV panels (keyed by ticker).
    Alpaca primary, yfinance backup — never synthesizes data.
    Returns (panels_dict, source_label).
    """
    return fetch_real_prices_for_universe(
        tickers, n_bars=n_bars, trade_year=trade_year, data_source=data_source
    )


# ============================================================
# ADAPTIVE KALMAN PAIRS  –  with Zeiierman-style adaptive R
# ============================================================

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



class LogisticExitModel:
    """Minimal L2 logistic regression for exit decisions."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_names: List[str] = list(FEATURE_NAMES)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, reg=0.3, lr=0.1, epochs=400):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            probs = self._sigmoid(logits)
            err = probs - y
            self.weights -= lr * ((X.T @ err) / n + reg * self.weights)
            self.bias -= lr * (err.mean())
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.bias)

    def save(self, path: Path = MODEL_JSON) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": float(self.bias),
            "feature_names": self.feature_names,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = MODEL_JSON) -> "LogisticExitModel":
        payload = json.loads(Path(path).read_text())
        model = cls()
        model.weights = np.array(payload["weights"], dtype=float) if payload["weights"] else None
        model.bias = float(payload["bias"])
        model.feature_names = list(payload.get("feature_names", FEATURE_NAMES))
        return model


def analyze_feature_importance(model, feature_names):
    print("\nFeature importance (|weight|):")
    abs_w = np.abs(model.weights)
    order = np.argsort(-abs_w)
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        print(f"  {name:16s} {model.weights[i]:+.4f}")


# ============================================================
# RESULTS STORAGE + TRAINING FROM HISTORY
# ============================================================

def estimate_half_life(spread: pd.Series, lookback: int = 40) -> float:
    """
    Ornstein-Uhlenbeck style half-life (in bars).
    Returns a large number when the spread is not mean-reverting.
    """
    if len(spread) < lookback + 5:
        return 30.0
    s = spread.iloc[-lookback:].astype(float)
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.iloc[-len(delta):]
    if float(lag.std()) < 1e-8:
        return 30.0
    beta = float(np.polyfit(lag.values, delta.values, 1)[0])
    if beta >= 0:
        return 60.0
    hl = np.log(2) / abs(beta)
    return float(np.clip(hl, 2.0, 60.0))


def extract_exit_features(
    position: PositionState,
    row: pd.Series,
    bars_held: int,
    direction: int,
    half_life: float = 20.0,
) -> np.ndarray:
    """
    Build the 11-d feature vector used by LogisticExitModel.
    Includes OU half-life as a mean-reversion strength signal.
    Order matches FEATURE_NAMES.
    """
    z = float(row["zscore"])
    conf = float(row.get("confidence", 0.5))
    vel = float(row.get("spread_velocity", 0.0))
    vol = float(row.get("spread_vol", 1.0))

    if direction == 1:
        pnl_proxy = z - position.entry_z
        favorable = max(0.0, z - position.entry_z)
    else:
        pnl_proxy = position.entry_z - z
        favorable = max(0.0, position.entry_z - z)

    position.highest_favorable_z = max(position.highest_favorable_z, favorable)

    return np.array([
        position.entry_z,
        abs(position.entry_z),
        pnl_proxy,
        bars_held / 30.0,
        conf,
        vel,
        z,
        favorable,
        position.highest_favorable_z,
        vol,
        float(half_life) / 30.0,
    ], dtype=float)


def trade_to_features(t: "PaperTrade") -> np.ndarray:
    """
    Prefer the exact live feature vector stored at exit.
    Fall back to a reconstructed approximation only if missing
    (e.g. legacy journal rows).
    """
    if t.exit_features is not None:
        vec = np.asarray(t.exit_features, dtype=float).ravel()
        if len(vec) == len(FEATURE_NAMES):
            return vec
        # Pad legacy 10-d vectors with a neutral half-life feature
        if len(vec) == len(FEATURE_NAMES) - 1:
            return np.concatenate([vec, [20.0 / 30.0]])

    pnl = t.pnl_z
    bars_norm = t.bars_held / 30.0
    exit_z = t.exit_z if t.exit_z is not None else 0.0
    fav = max(0.0, pnl)
    best_fav = max(abs(t.entry_z), fav)

    return np.array([
        t.entry_z,
        abs(t.entry_z),
        pnl,
        bars_norm,
        0.70,
        0.0,
        exit_z,
        fav,
        best_fav,
        1.0,
        20.0 / 30.0,
    ], dtype=float)


def trade_to_label(t: PaperTrade, good_pnl_threshold: float = 0.35) -> int:
    """
    Higher-quality binary label for the exit model.

    1 = "good exit" (we should have exited around here)
    0 = "bad / premature / late exit"
    """
    if t.pnl_z >= good_pnl_threshold:
        return 1
    if t.pnl_z > 0.05 and 8 <= t.bars_held <= 22:
        return 1
    if t.bars_held >= 25 and t.pnl_z > -0.6:
        return 1
    return 0


# ============================================================
# RESEARCH MODE — counterfactual setups (does not touch live journal)
# ============================================================

@dataclass
class Setup:
    """One counterfactual z-crossing simulated to completion (research only)."""
    setup_id: str
    run_id: str
    pair: str
    basket: str
    direction: int  # +1 long spread, -1 short
    entry_time: pd.Timestamp
    entry_bar: int
    features: Dict[str, float] = field(default_factory=dict)

    taken: bool = False
    exit_time: Optional[pd.Timestamp] = None
    bars_held: Optional[int] = None
    pnl_z: Optional[float] = None
    pnl_dollars: Optional[float] = None
    exit_reason: Optional[str] = None

    label: Optional[int] = None
    exit_model_version: Optional[str] = None
    data_window: str = "latest_year"  # or "multi_year"

    def to_row(self) -> dict:
        row = asdict(self)
        for k, v in self.features.items():
            row[f"feat_{k}"] = v
        del row["features"]
        for col in ("entry_time", "exit_time"):
            if row[col] is not None:
                row[col] = str(row[col])
        return row


def get_exit_model_version(model: Optional["LogisticExitModel"] = None) -> str:
    """Simple provenance tag for the exit policy used in a research pass."""
    if model is None or getattr(model, "weights", None) is None:
        return "rules_only"
    raw = json.dumps(
        {
            "weights": np.asarray(model.weights, dtype=float).tolist(),
            "bias": float(model.bias) if model.bias is not None else 0.0,
            "features": list(model.feature_names),
        },
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def save_setups(
    setups: List[Setup],
    results_dir: Optional[Path] = None,
    run_id: str = "",
) -> Path:
    """Write setups_*.csv only — never touches paper_trades / training dataset."""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"setups_{run_id}.csv"
    if not setups:
        print("No setups to save.")
        return path
    df = pd.DataFrame([s.to_row() for s in setups])
    df.to_csv(path, index=False)
    print(f"💾 Saved {len(setups)} setups → {path}")
    return path


def _simulate_setups_for_pair(
    df: pd.DataFrame,
    pair: PairSpec,
    model: Optional[LogisticExitModel],
    run_id: str,
    data_window: str = "multi_year",
    ml_threshold: float = 0.62,
    capital: float = 100_000.0,
    risk_frac: float = 0.08,
    cost_bps: float = 4.0,
    z_to_pct: float = 0.01,
    trade_year: Optional[int] = None,
) -> List[Setup]:
    """
    Research mode: walk the window and simulate every valid z-crossing to
    completion. Does not touch PaperTrader / Alpaca / paper_trades.csv.

    Year gate: when data_window == "latest_year", entries outside trade_year
    are skipped (same idea as live/backtest). multi_year skips that gate.
    """
    setups: List[Setup] = []
    position = 0
    entry_idx = 0
    pos_state = PositionState()
    entry_time = None
    entry_z = 0.0
    direction = 0
    features: Dict[str, float] = {}
    exit_model_version = get_exit_model_version(model)
    year = int(trade_year) if trade_year is not None else int(pd.Timestamp(df.index.max()).year)
    latest_only = (data_window or "latest_year").lower().strip() == "latest_year"

    for i in range(60, len(df)):
        row = df.iloc[i]
        z = float(row["zscore"])
        conf = float(row.get("confidence", 0.5))
        time = df.index[i]

        # ----- ENTRY -----
        if position == 0:
            if latest_only and int(pd.Timestamp(time).year) != year:
                continue
            if z < -2.0 and conf > 0.55:
                direction = 1
            elif z > 2.0 and conf > 0.55:
                direction = -1
            else:
                continue

            position = direction
            entry_idx = i
            entry_time = time
            entry_z = z
            pos_state = PositionState(
                direction=direction,
                entry_z=z,
                entry_bar=i,
                entry_spread=float(row["spread"]),
                highest_favorable_z=0.0,
            )
            half_life = estimate_half_life(df["spread"].iloc[: i + 1], lookback=40)
            features = {
                "entry_z": z,
                "abs_entry_z": abs(z),
                "confidence": conf,
                "half_life": float(half_life) / 30.0,
                "vol": float(row.get("spread_vol", 1.0)),
                "velocity": float(row.get("spread_velocity", 0.0)),
            }
            continue

        # ----- EXIT (exact live helpers) -----
        bars_held = i - entry_idx
        pos_state.bars_held = bars_held
        half_life = estimate_half_life(df["spread"].iloc[: i + 1], lookback=40)
        feat_vec = extract_exit_features(
            pos_state, row, bars_held, position, half_life=half_life
        )
        should_exit, ml_proba = should_exit_with_ml(
            position=position,
            z=z,
            bars_held=bars_held,
            features=feat_vec,
            model=model,
            ml_threshold=ml_threshold,
        )
        if not should_exit:
            continue

        if direction == 1:
            pnl_z = z - entry_z
        else:
            pnl_z = entry_z - z
        notional = float(capital) * float(risk_frac)
        gross = pnl_z * z_to_pct * notional
        cost = notional * (cost_bps / 10_000.0)
        pnl_dollars = float(gross - cost)
        exit_reason = (
            "ml"
            if (ml_proba is not None and ml_proba >= ml_threshold)
            else "rules"
        )
        label = trade_to_label(
            SimpleNamespace(pnl_z=pnl_z, bars_held=bars_held)  # type: ignore[arg-type]
        )
        setups.append(
            Setup(
                setup_id=f"{run_id}_{pair.label}_{entry_idx}",
                run_id=run_id,
                pair=pair.label,
                basket=pair.basket,
                direction=direction,
                entry_time=entry_time,
                entry_bar=entry_idx,
                features=dict(features),
                taken=False,
                exit_time=time,
                bars_held=bars_held,
                pnl_z=float(pnl_z),
                pnl_dollars=pnl_dollars,
                exit_reason=exit_reason,
                label=int(label),
                exit_model_version=exit_model_version,
                data_window=data_window,
            )
        )
        position = 0

    return setups


def run_research_setups(
    n_bars: int = 700,
    baskets: Optional[Sequence[str]] = None,
    include_cross: bool = True,
    max_pairs_per_basket: int = 6,
    ml_threshold: float = 0.62,
    noise_model: KalmanNoiseModel | str = KalmanNoiseModel.STANDARD,
    data_source: str = "auto",
    data_window: str = "multi_year",
) -> Tuple[List[Setup], Path]:
    """
    Counterfactual research pass over the pair universe.
    Writes results/setups_<run_id>.csv only (no broker, no paper journal).
    """
    data_window = (data_window or "multi_year").lower().strip()
    if data_window not in ("multi_year", "latest_year"):
        raise ValueError("data_window must be 'multi_year' or 'latest_year'")

    print("Starting Research Setup Pass...")
    print(f"Mode: research (data_window={data_window})")
    print("Simulates every valid z-crossing to completion — no Alpaca / no paper journal\n")

    baskets = list(baskets) if baskets is not None else list(TICKER_UNIVERSES.keys())
    pairs = build_pair_universe(
        baskets=baskets,
        include_cross=include_cross,
        max_pairs_per_basket=max_pairs_per_basket,
    )
    summarize_universes(pairs)

    tickers = all_universe_tickers(baskets)
    panels, source = _load_prices_for_universe(tickers, n_bars, data_source=data_source)
    trade_year = int(
        next(iter(panels.values())).attrs.get(
            "trade_year", pd.Timestamp(next(iter(panels.values())).index.max()).year
        )
    )
    fields = ohlcv_field_panels(panels)
    prices = fields["close"]
    highs = fields["high"]
    lows = fields["low"]
    volumes = fields["volume"]
    if isinstance(noise_model, str):
        noise_model = KalmanNoiseModel(noise_model)

    print(f"\nPrice panel: {prices.shape[1]} tickers × {prices.shape[0]} bars  [source: {source}]")
    print(f"Kalman R mode: {noise_model.value}")
    if data_window == "latest_year":
        print(f"Research entries: {trade_year} only\n")
    else:
        print("Research entries: all bars in loaded panel (no year gate)\n")

    model = None
    if MODEL_JSON.exists():
        try:
            model = LogisticExitModel.load(MODEL_JSON)
            if model.weights is None or len(model.weights) != len(FEATURE_NAMES):
                model = None
            else:
                print(f"✅ Loaded exit model from {MODEL_JSON}")
        except Exception as exc:
            print(f"⚠️ Could not load model ({exc}); rules-only research")
            model = None
    else:
        print("ℹ️  No saved exit model — rules-only research exits")

    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    kf = AdaptiveKalmanPairs(delta=1e-4, R_base=1e-2, noise_model=noise_model)
    all_setups: List[Setup] = []

    for pair in pairs:
        if pair.ticker_a not in prices.columns or pair.ticker_b not in prices.columns:
            continue
        vol_a = volumes[pair.ticker_a] if pair.ticker_a in volumes.columns else None
        hi_a = highs[pair.ticker_a] if pair.ticker_a in highs.columns else None
        lo_a = lows[pair.ticker_a] if pair.ticker_a in lows.columns else None
        df = kf.filter_pair(
            prices[pair.ticker_a],
            prices[pair.ticker_b],
            volume=vol_a,
            high=hi_a,
            low=lo_a,
        )
        print(f"\n--- Research {pair.label} [{pair.basket}] ---")
        pair_setups = _simulate_setups_for_pair(
            df=df,
            pair=pair,
            model=model,
            run_id=run_id,
            data_window=data_window,
            ml_threshold=ml_threshold,
            trade_year=trade_year,
        )
        print(f"   setups: {len(pair_setups)}")
        all_setups.extend(pair_setups)

    path = save_setups(all_setups, results_dir=RESULTS_DIR, run_id=run_id)
    if all_setups:
        labels = [s.label for s in all_setups if s.label is not None]
        pnls = [s.pnl_z for s in all_setups if s.pnl_z is not None]
        print(f"\nResearch summary: {len(all_setups)} setups")
        if labels:
            print(f"  label=1 rate: {np.mean(labels):.1%}")
        if pnls:
            print(f"  mean pnl_z:   {np.mean(pnls):+.3f}")
        print(f"  model tag:    {get_exit_model_version(model)}")
    return all_setups, path


def closed_trades_to_frame(
    closed_trades: Sequence[PaperTrade], run_id: str = "", data_source: str = ""
) -> pd.DataFrame:
    rows = []
    for t in closed_trades:
        feat = trade_to_features(t) if t.status == "CLOSED" else None
        row = {
            "run_id": run_id,
            "data_source": data_source,
            "trade_id": t.trade_id,
            "ticker_a": t.ticker_a,
            "ticker_b": t.ticker_b,
            "basket": t.basket,
            "direction": t.direction,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_z": t.entry_z,
            "exit_z": t.exit_z,
            "entry_spread": t.entry_spread,
            "exit_spread": t.exit_spread,
            "bars_held": t.bars_held,
            "pnl_z": t.pnl_z,
            "notional": t.notional,
            "pnl_dollars": t.pnl_dollars,
            "cost_dollars": t.cost_dollars,
            "broker": t.broker,
            "qty_a": t.qty_a,
            "qty_b": t.qty_b,
            "alpaca_order_ids": json.dumps(t.alpaca_order_ids or []),
            "ml_proba_at_exit": t.ml_proba_at_exit,
            "status": t.status,
            "label": trade_to_label(t) if t.status == "CLOSED" else None,
            "exit_features_json": (
                json.dumps(np.asarray(t.exit_features, dtype=float).tolist())
                if t.exit_features is not None else None
            ),
        }
        if feat is not None:
            for name, val in zip(FEATURE_NAMES, feat):
                row[f"feat_{name}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _trade_fingerprint_cols() -> List[str]:
    return ["ticker_a", "ticker_b", "direction", "entry_time", "exit_time", "broker"]


def _is_wash_closed_row(df: pd.DataFrame) -> pd.Series:
    """Same-bar CLOSED with zero pnl_z (failed/instant reverse, not a real round-trip)."""
    if df.empty:
        return pd.Series(dtype=bool)
    status = df["status"].astype(str).str.upper() if "status" in df.columns else pd.Series([""] * len(df))
    entry = pd.to_datetime(df["entry_time"], format="mixed", errors="coerce")
    exit_ = pd.to_datetime(df["exit_time"], format="mixed", errors="coerce")
    pnl = pd.to_numeric(df["pnl_z"], errors="coerce") if "pnl_z" in df.columns else pd.Series(np.nan, index=df.index)
    same_bar = entry.notna() & exit_.notna() & (entry == exit_)
    zero_pnl = pnl.fillna(0.0).abs() < 1e-12
    return status.eq("CLOSED") & same_bar & zero_pnl


def dedupe_journal_rows(
    journal: pd.DataFrame,
    drop_wash: bool = True,
) -> pd.DataFrame:
    """
    Deduplicate journal rows for sim *and* alpaca.

    - drop_wash=True: remove CLOSED wash trades (entry_time == exit_time, pnl_z ≈ 0)
      so a later OPEN for the same exposure is kept (matches brokerage).
    - For a real close (entry != exit), CLOSED outranks OPEN on the same pair/entry.
    - Repeated sim backtest fingerprints collapse to one row.
    """
    if journal is None or journal.empty:
        return journal
    df = journal.copy()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    if drop_wash and {"entry_time", "exit_time", "status"}.issubset(df.columns):
        wash = _is_wash_closed_row(df)
        n_wash = int(wash.sum())
        if n_wash:
            print(f"⚠️  Dropped {n_wash} wash CLOSED rows (entry==exit, pnl_z≈0)")
            df = df.loc[~wash].copy()

    # Collapse identical fingerprints (includes exit_time when present)
    fp = [c for c in _trade_fingerprint_cols() if c in df.columns]
    if fp:
        df = df.drop_duplicates(subset=fp, keep="last")

    # Same pair + entry_time: prefer real CLOSED over OPEN; then latest run_id
    key = [c for c in ("ticker_a", "ticker_b", "entry_time") if c in df.columns]
    if len(key) == 3 and "status" in df.columns:
        status = df["status"].astype(str).str.upper()
        df = df.assign(_rank=np.where(status.eq("CLOSED"), 1, 0))
        sort_cols = key + ["_rank"]
        if "run_id" in df.columns:
            sort_cols.append("run_id")
        df = df.sort_values(sort_cols, kind="mergesort")
        df = df.drop_duplicates(subset=key, keep="last").drop(columns=["_rank"])

    sort_cols = [c for c in ("entry_time", "run_id", "trade_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def save_paper_results(
    closed_trades: Sequence[PaperTrade],
    results_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    data_source: str = "",
) -> Tuple[Path, Path]:
    """
    Append closed trades to a journal CSV and write/append the training dataset.
    Prior-year windows (e.g. 2025) are purged so only the latest year remains.
    Wash CLOSED rows and duplicate OPEN adopts are dropped so training stays clean.
    Returns (trades_csv_path, dataset_csv_path).
    """
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    year = _latest_allowed_trade_year()

    frame = closed_trades_to_frame(closed_trades, run_id=run_id, data_source=data_source)
    trades_path = results_dir / "paper_trades.csv"
    dataset_path = results_dir / "exit_training_dataset.csv"

    # Full journal (append → year filter for ML → dedupe wash/OPEN spam)
    if trades_path.exists():
        prev = pd.read_csv(trades_path)
        journal = pd.concat([prev, frame], ignore_index=True)
    else:
        journal = frame
    journal = filter_trades_to_latest_year(
        journal, trade_year=year, require_exit_in_year=False
    )
    journal = dedupe_journal_rows(journal, drop_wash=True)
    journal.to_csv(trades_path, index=False)

    # Training dataset = feature columns + label (closed trades only)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    closed_frame = frame.copy()
    if "status" in closed_frame.columns:
        closed_frame = closed_frame[closed_frame["status"].fillna("CLOSED") == "CLOSED"]
    closed_frame = closed_frame.dropna(subset=["label"]) if "label" in closed_frame.columns else closed_frame
    # OPEN-only saves (live adopt / overnight hold) have no feat_* columns yet
    for col in feat_cols:
        if col not in closed_frame.columns:
            closed_frame[col] = np.nan
    want_cols = ["run_id", "data_source", "trade_id", "ticker_a", "ticker_b", "basket", *feat_cols, "label"]
    for col in want_cols:
        if col not in closed_frame.columns:
            closed_frame[col] = np.nan
    ds = closed_frame[want_cols]
    # Drop rows missing any feature (e.g. open broker mirrors)
    ds = ds.dropna(subset=feat_cols, how="any")
    # Also drop wash closed rows from the newly appended training frame
    if not frame.empty and "status" in frame.columns:
        wash_keys = frame.loc[_is_wash_closed_row(frame), ["run_id", "trade_id"]]
        if not wash_keys.empty and {"run_id", "trade_id"}.issubset(ds.columns):
            ds = ds.merge(wash_keys.drop_duplicates(), on=["run_id", "trade_id"], how="left", indicator=True)
            ds = ds.loc[ds["_merge"] == "left_only"].drop(columns=["_merge"])
    if dataset_path.exists():
        prev_ds = pd.read_csv(dataset_path)
        ds = pd.concat([prev_ds, ds], ignore_index=True)
    if not journal.empty and {"run_id", "trade_id"}.issubset(ds.columns):
        keys = journal[["run_id", "trade_id"]].drop_duplicates()
        ds = ds.merge(keys, on=["run_id", "trade_id"], how="inner")
    elif journal.empty:
        ds = ds.iloc[0:0]
    # Re-dedupe training rows that survived from older polluted journals
    if not ds.empty and {"ticker_a", "ticker_b", "run_id", "trade_id"}.issubset(ds.columns):
        ds = ds.drop_duplicates(subset=["run_id", "trade_id"], keep="last")
    ds.to_csv(dataset_path, index=False)

    print(f"\n💾 Saved {len(closed_trades)} new trades → {trades_path}")
    print(f"💾 Journal rows kept for {year}: {len(journal)}")
    print(f"💾 Training dataset rows: {len(ds)} → {dataset_path}")
    return trades_path, dataset_path


def _latest_allowed_trade_year(now: Optional[pd.Timestamp] = None) -> int:
    """Calendar year used for training/trade windows (never prior years)."""
    return int(pd.Timestamp(now or pd.Timestamp.now("UTC")).year)


def filter_trades_to_latest_year(
    trades: pd.DataFrame,
    trade_year: Optional[int] = None,
    require_exit_in_year: bool = False,
) -> pd.DataFrame:
    """
    Keep rows whose entry falls in trade_year.

    require_exit_in_year=False (default, ML/training): exit may be missing or
    spill into the next calendar year.
    require_exit_in_year=True (year-end reporting): closed trades must also
    exit in the same calendar year.
    """
    if trades.empty:
        return trades
    year = trade_year if trade_year is not None else _latest_allowed_trade_year()
    entry = pd.to_datetime(trades["entry_time"], format="mixed")
    exit_ = pd.to_datetime(trades["exit_time"], format="mixed", errors="coerce")
    open_ok = exit_.isna() & entry.dt.year.eq(year)
    if require_exit_in_year:
        closed_ok = exit_.notna() & entry.dt.year.eq(year) & exit_.dt.year.eq(year)
    else:
        closed_ok = exit_.notna() & entry.dt.year.eq(year)
    mask = open_ok | closed_ok
    kept = trades.loc[mask].copy()
    dropped = len(trades) - len(kept)
    if dropped:
        print(f"⚠️  Dropped {dropped} trades outside latest year {year}")
    return kept


def load_training_dataset(
    dataset_path: Optional[Path] = None,
    require_live: bool = True,
    latest_year_only: bool = True,
    trades_path: Optional[Path] = None,
    trade_year: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(dataset_path) if dataset_path is not None else DATASET_CSV
    trades_path = Path(trades_path) if trades_path is not None else TRADES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No stored dataset at {path}. Run a paper session first to create it."
        )
    ds = pd.read_csv(path)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    # Backfill new features (e.g. half_life) for older journals
    if "feat_half_life" not in ds.columns:
        ds["feat_half_life"] = 20.0 / 30.0
    missing = [c for c in feat_cols + ["label"] if c not in ds.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Training must use real prices only — drop any legacy synthetic rows.
    if require_live and "data_source" in ds.columns:
        before = len(ds)
        src = ds["data_source"].astype(str)
        keep = src.str.startswith("yfinance") | src.str.startswith("alpaca")
        ds = ds[keep].copy()
        dropped = before - len(ds)
        if dropped:
            print(f"⚠️  Dropped {dropped} non-live rows from training dataset")
        if ds.empty:
            raise ValueError(
                "No live (alpaca/yfinance) rows left in the training dataset. "
                "Re-run paper trading to collect real-price samples."
            )

    # Drop any stored windows from prior years (e.g. 2025) — latest year only.
    if latest_year_only:
        year = trade_year if trade_year is not None else _latest_allowed_trade_year()
        tpath = Path(trades_path)
        if tpath.exists() and {"run_id", "trade_id"}.issubset(ds.columns):
            journal = pd.read_csv(tpath)
            journal = filter_trades_to_latest_year(
                journal, trade_year=year, require_exit_in_year=False
            )
            keys = journal[["run_id", "trade_id"]].drop_duplicates()
            before = len(ds)
            ds = ds.merge(keys, on=["run_id", "trade_id"], how="inner")
            dropped = before - len(ds)
            if dropped:
                print(f"⚠️  Dropped {dropped} training rows not in latest year {year}")
        elif "entry_time" in ds.columns and "exit_time" in ds.columns:
            before = len(ds)
            ds = filter_trades_to_latest_year(
                ds, trade_year=year, require_exit_in_year=False
            )
            dropped = before - len(ds)
            if dropped:
                print(f"⚠️  Dropped {dropped} training rows outside year {year}")
        if ds.empty:
            raise ValueError(
                f"No training rows left for latest year {year}. "
                "Re-run paper trading on latest-year windows."
            )

    X = ds[feat_cols].to_numpy(dtype=float)
    y = ds["label"].to_numpy(dtype=float)
    return X, y, ds


def train_from_stored_results(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    reg: float = 0.3,
    min_samples: int = 2,
) -> LogisticExitModel:
    """Fit the exit model on stored live (Alpaca/yfinance) paper trades only."""
    dataset_path = Path(dataset_path) if dataset_path is not None else DATASET_CSV
    model_path = Path(model_path) if model_path is not None else MODEL_JSON
    X, y, ds = load_training_dataset(dataset_path, require_live=True)
    if len(y) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples; found {len(y)} in {dataset_path}")

    print(f"\nTraining from stored results: {len(y)} live samples ({dataset_path})")
    if "data_source" in ds.columns:
        print(f"  Sources: {sorted(ds['data_source'].dropna().astype(str).unique().tolist())}")
    print(f"  Baskets: {sorted(ds['basket'].dropna().unique().tolist()) if 'basket' in ds else 'n/a'}")
    model = LogisticExitModel()
    model.fit(X, y, reg=reg)
    out = model.save(model_path)
    print(f"✅ Model trained and saved → {out}")
    analyze_feature_importance(model, FEATURE_NAMES)
    return model

# ============================================================
# PAPER TRADING ENGINE
# ============================================================

@dataclass
class PaperTrade:
    trade_id: int
    direction: str          # "LONG_SPREAD" or "SHORT_SPREAD"
    entry_time: pd.Timestamp
    entry_z: float
    entry_spread: float
    ticker_a: str = ""
    ticker_b: str = ""
    basket: str = ""
    exit_time: pd.Timestamp = None
    exit_z: float = None
    exit_spread: float = None
    bars_held: int = 0
    pnl_z: float = 0.0
    notional: float = 0.0
    pnl_dollars: float = 0.0
    cost_dollars: float = 0.0
    status: str = "OPEN"
    ml_proba_at_exit: float = None
    # Exact feature vector at exit time (from extract_exit_features)
    exit_features: Optional[np.ndarray] = None
    broker: str = "sim"
    qty_a: float = 0.0
    qty_b: float = 0.0
    alpaca_order_ids: Optional[List[str]] = None


class PaperTrader:
    def __init__(
        self,
        capital=100_000,
        cost_bps: float = 4.0,
        risk_frac: float = 0.08,
        broker: Optional[AlpacaPaperBroker] = None,
        execute_latest_only: bool = True,
        latest_bar: Optional[pd.Timestamp] = None,
    ):
        self.capital = capital
        self.equity = capital
        self.cost_bps = cost_bps          # round-trip cost in basis points
        self.risk_frac = risk_frac
        self.broker = broker
        self.execute_latest_only = bool(execute_latest_only)
        self.latest_bar = pd.Timestamp(latest_bar) if latest_bar is not None else None
        self.trades: List[PaperTrade] = []
        self.current_trade: Optional[PaperTrade] = None
        self.trade_counter = 0

        if self.broker is not None:
            eq = self.broker.get_equity()
            if eq is not None and eq > 0:
                self.capital = float(eq)
                self.equity = float(eq)
                print(f"🏦 Alpaca paper equity synced: ${self.equity:,.2f}")

    def _should_route_to_broker(self, time) -> bool:
        if self.broker is None:
            return False
        if not self.execute_latest_only:
            return True
        if self.latest_bar is None:
            return False
        return pd.Timestamp(time).normalize() == pd.Timestamp(self.latest_bar).normalize()

    def open_trade(
        self,
        direction: int,
        time,
        z,
        spread,
        ticker_a="",
        ticker_b="",
        basket="",
        risk_frac: Optional[float] = None,
        price_a: Optional[float] = None,
        price_b: Optional[float] = None,
    ):
        self.trade_counter += 1
        side = "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD"
        frac = self.risk_frac if risk_frac is None else float(risk_frac)
        notional = float(self.equity * frac)

        trade = PaperTrade(
            trade_id=self.trade_counter,
            direction=side,
            entry_time=time,
            entry_z=z,
            entry_spread=spread,
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            basket=basket,
            notional=notional,
            broker="sim",
        )

        if (
            self._should_route_to_broker(time)
            and ticker_a and ticker_b
            and price_a and price_b
            and price_a > 0 and price_b > 0
        ):
            try:
                result = self.broker.open_pair(
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    direction=side,
                    notional=notional,
                    price_a=float(price_a),
                    price_b=float(price_b),
                )
                trade.broker = self.broker.name
                trade.alpaca_order_ids = result.order_ids
                if len(result.fills) >= 2:
                    trade.qty_a = float(result.fills[0].qty)
                    trade.qty_b = float(result.fills[1].qty)
            except Exception as exc:
                msg = str(exc)
                if "existing Alpaca exposure" in msg or "Skip entry" in msg:
                    print(f"⚠️  {msg}")
                    self.trade_counter -= 1
                    return
                print(f"⚠️  Alpaca entry failed ({exc}); keeping sim journal fill only")

        self.current_trade = trade
        self.trades.append(trade)
        pair = f"{ticker_a}/{ticker_b}" if ticker_a and ticker_b else "PAIR"
        print(f"\n🟢 OPENED Trade #{trade.trade_id} | {side} | {pair} [{basket}] | broker={trade.broker}")
        print(f"   Time: {time.date()} | z={z:.2f} | spread={spread:.3f} | notional=${notional:,.0f}")

    def close_trade(
        self,
        time,
        z,
        spread,
        ml_proba=None,
        features: Optional[np.ndarray] = None,
        bars_held: Optional[int] = None,
        z_to_pct: float = 0.01,
    ) -> bool:
        """
        Close the current trade. Returns True if the journal marks CLOSED.
        Returns False if there is no open trade, or Alpaca exit failed (stays OPEN).
        """
        if self.current_trade is None:
            return False

        t = self.current_trade
        t.exit_time = time
        t.exit_z = z
        t.exit_spread = spread
        if bars_held is not None:
            t.bars_held = max(1, int(bars_held))
        else:
            t.bars_held = max(1, int((time - t.entry_time).days))
        t.ml_proba_at_exit = ml_proba
        t.status = "CLOSED"
        t.exit_features = features.copy() if features is not None else None

        if t.direction == "LONG_SPREAD":
            t.pnl_z = z - t.entry_z
        else:
            t.pnl_z = t.entry_z - z

        # Route exit to Alpaca paper when this bar is eligible
        if self._should_route_to_broker(time) and t.broker.startswith("alpaca"):
            try:
                result = self.broker.close_pair(
                    ticker_a=t.ticker_a,
                    ticker_b=t.ticker_b,
                    qty_a=t.qty_a,
                    qty_b=t.qty_b,
                    direction=t.direction,
                )
                ids = list(t.alpaca_order_ids or [])
                ids.extend(result.order_ids)
                t.alpaca_order_ids = ids
            except Exception as exc:
                # Keep journal OPEN so it still matches brokerage exposure
                print(f"⚠️  Alpaca exit failed ({exc}); leaving trade OPEN in journal")
                t.status = "OPEN"
                t.exit_time = None
                t.exit_z = None
                t.exit_spread = None
                t.pnl_z = None
                t.pnl_dollars = None
                t.cost_dollars = None
                t.ml_proba_at_exit = None
                t.exit_features = None
                return False

        # Dollar PnL: 1 z ≈ z_to_pct of notional, minus round-trip costs
        # (Alpaca fills remain authoritative in the brokerage UI; journal keeps z-scaled $.)
        gross = t.pnl_z * z_to_pct * t.notional
        cost = t.notional * (self.cost_bps / 10_000.0)
        t.cost_dollars = float(cost)
        t.pnl_dollars = float(gross - cost)
        self.equity += t.pnl_dollars
        if self.broker is not None:
            eq = self.broker.get_equity()
            if eq is not None and eq > 0:
                self.equity = float(eq)

        pair = f"{t.ticker_a}/{t.ticker_b}" if t.ticker_a and t.ticker_b else "PAIR"
        print(f"🔴 CLOSED Trade #{t.trade_id} | {t.direction} | {pair}")
        print(
            f"   Time: {time.date()} | z={z:.2f} | PnL(z)={t.pnl_z:+.3f} "
            f"| PnL($)={t.pnl_dollars:+,.0f} | cost=${cost:,.0f} | Bars={t.bars_held}"
        )
        if ml_proba is not None:
            print(f"   ML Exit Prob at close: {ml_proba:.2%}")

        self.current_trade = None
        return True

    def summary(self):
        closed = [t for t in self.trades if t.status == "CLOSED"]
        print("\n" + "=" * 60)
        print("PAPER TRADING JOURNAL")
        print("=" * 60)
        for t in closed:
            pair = f"{t.ticker_a}/{t.ticker_b}" if t.ticker_a else "?"
            print(
                f"Trade #{t.trade_id:2d} | {pair:13s} | {t.basket:18s} | {t.direction:13s} | "
                f"Entry z={t.entry_z:+.2f} → Exit z={t.exit_z:+.2f} | "
                f"PnL(z)={t.pnl_z:+.3f} | PnL($)={t.pnl_dollars:+,.0f} | Held {t.bars_held} bars"
            )

        if closed:
            pnls = [t.pnl_z for t in closed]
            dollar = [t.pnl_dollars for t in closed]
            print(f"\nTotal closed trades: {len(closed)}")
            print(f"Average PnL (z):     {np.mean(pnls):+.3f}")
            print(f"Total PnL ($):       {np.sum(dollar):+,.0f}")
            print(f"Ending equity:       ${self.equity:,.0f}")
            print(f"Win rate:            {np.mean([p > 0 for p in pnls]):.1%}")
            by_basket: Dict[str, List[float]] = {}
            by_basket_d: Dict[str, List[float]] = {}
            for t in closed:
                key = t.basket or "unknown"
                by_basket.setdefault(key, []).append(t.pnl_z)
                by_basket_d.setdefault(key, []).append(t.pnl_dollars)
            print("\nBy basket:")
            for basket, vals in by_basket.items():
                print(
                    f"  {basket:18s} n={len(vals)}  avg PnL(z)={np.mean(vals):+.3f}  "
                    f"PnL($)={np.sum(by_basket_d[basket]):+,.0f}"
                )
        print("=" * 60)
        return closed

# ============================================================
# MAIN PAPER TRADING + TRAINING LOOP
# ============================================================

def should_exit_with_ml(
    position: int,
    z: float,
    bars_held: int,
    features: np.ndarray,
    model: Optional[LogisticExitModel],
    ml_threshold: float = 0.62,
    force_rules: bool = True,
) -> Tuple[bool, Optional[float]]:
    """
    Combine classic mean-reversion / time / stop rules with ML probability.

    Returns
    -------
    (should_exit, ml_proba)
    """
    # ----- Classic safety rules (always available) -----
    rule_exit = False
    if position == 1 and z > -0.35:
        rule_exit = True
    if position == -1 and z < 0.35:
        rule_exit = True
    if bars_held >= 28:                     # hard time stop
        rule_exit = True
    if position == 1 and z < -3.6:          # adverse stop
        rule_exit = True
    if position == -1 and z > 3.6:
        rule_exit = True

    ml_proba = None
    if model is not None and model.weights is not None:
        try:
            ml_proba = float(model.predict_proba(features.reshape(1, -1))[0])
        except Exception:
            ml_proba = None

    # ----- ML override / reinforcement -----
    if ml_proba is not None:
        # High probability → force exit even if rules have not triggered yet
        if ml_proba >= ml_threshold:
            return True, ml_proba
        # Low probability → optionally suppress a soft rule exit
        # (keep hard stops & time stop)
        if force_rules and ml_proba < 0.38 and bars_held < 22:
            if abs(z) < 2.8:          # only suppress mild mean-reversion exits
                return False, ml_proba

    return rule_exit, ml_proba


def _trade_pair_session(
    trader: PaperTrader,
    df: pd.DataFrame,
    pair: PairSpec,
    min_trades: int,
    trades_remaining: int,
    trade_year: Optional[int] = None,
    model: Optional[LogisticExitModel] = None,
    ml_threshold: float = 0.62,
    mode: str = "backtest",
    latest_bar: Optional[pd.Timestamp] = None,
) -> int:
    """
    Run rule-based entries + ML-augmented exits on one pair.
    Returns the number of newly closed trades.

    mode:
      - backtest: scan the full trade-year window (builds ML journal; sim fills)
      - live: warm Kalman on history, enter/exit only on the latest bar
              (this is what places Alpaca paper orders with --broker alpaca)
    """
    position = 0
    entry_idx = 0
    opened = 0
    pos_state = PositionState()
    live = (mode or "backtest").lower().strip() == "live"
    # Fresh Alpaca entries hold overnight; adopted exposure may still exit today.
    fresh_entry_this_bar = False
    latest_norm = (
        pd.Timestamp(latest_bar).normalize()
        if latest_bar is not None
        else pd.Timestamp(df.index.max()).normalize()
    )

    if trade_year is None:
        trade_year = int(pd.Timestamp(df.index.max()).year)

    for i in range(60, len(df)):
        if opened >= trades_remaining:
            break

        row = df.iloc[i]
        z = float(row["zscore"])
        conf = float(row.get("confidence", 0.5))
        time = df.index[i]
        in_trade_year = int(pd.Timestamp(time).year) == int(trade_year)
        is_latest = pd.Timestamp(time).normalize() == latest_norm
        fresh_entry_this_bar = False

        # ---------- ENTRY ----------
        if position == 0:
            if not in_trade_year:
                continue
            # Live mode: ignore historical entry signals; only act on freshest bar
            if live and not is_latest:
                continue

            # Live idempotency: adopt existing Alpaca pair exposure (exit-only)
            if live and trader.broker is not None:
                try:
                    exp = trader.broker.pair_exposure(pair.ticker_a, pair.ticker_b)
                except Exception as exc:
                    print(f"⚠️  Could not read Alpaca exposure for {pair.label}: {exc}")
                    exp = {"flat": True, "direction": 0, "qty_a": 0.0, "qty_b": 0.0, "blocked": False}
                if exp.get("blocked"):
                    print(f"⚠️  Skip {pair.label}: ambiguous open legs on Alpaca")
                    break
                if not exp.get("flat") and exp.get("direction") in (1, -1):
                    position = int(exp["direction"])
                    entry_idx = max(60, i - 1)
                    pos_state = PositionState(
                        direction=position,
                        entry_z=z,
                        entry_bar=entry_idx,
                        entry_spread=float(row["spread"]),
                        highest_favorable_z=0.0,
                    )
                    side = "LONG_SPREAD" if position == 1 else "SHORT_SPREAD"
                    trader.trade_counter += 1
                    mirrored = PaperTrade(
                        trade_id=trader.trade_counter,
                        direction=side,
                        entry_time=time,
                        entry_z=z,
                        entry_spread=float(row["spread"]),
                        ticker_a=pair.ticker_a,
                        ticker_b=pair.ticker_b,
                        basket=pair.basket,
                        notional=0.0,
                        broker=trader.broker.name,
                        qty_a=float(exp["qty_a"]),
                        qty_b=float(exp["qty_b"]),
                        status="OPEN",
                    )
                    trader.current_trade = mirrored
                    trader.trades.append(mirrored)
                    print(
                        f"ℹ️  Adopted open Alpaca {side} on {pair.label} "
                        f"(qty {exp['qty_a']:.0f}/{exp['qty_b']:.0f}) — will not re-enter"
                    )

            # Classic z-score entry with confidence filter (only if still flat)
            if position == 0 and z < -2.0 and conf > 0.55:
                position = 1
                entry_idx = i
                pos_state = PositionState(
                    direction=1,
                    entry_z=z,
                    entry_bar=i,
                    entry_spread=float(row["spread"]),
                    highest_favorable_z=0.0,
                )
                before_n = len(trader.trades)
                trader.open_trade(
                    direction=1,
                    time=time,
                    z=z,
                    spread=row["spread"],
                    ticker_a=pair.ticker_a,
                    ticker_b=pair.ticker_b,
                    basket=pair.basket,
                    risk_frac=0.08,
                    price_a=float(row["price_a"]),
                    price_b=float(row["price_b"]),
                )
                if len(trader.trades) == before_n:
                    position = 0  # broker refused (e.g. existing exposure)
                else:
                    fresh_entry_this_bar = True

            elif position == 0 and z > 2.0 and conf > 0.55:
                position = -1
                entry_idx = i
                pos_state = PositionState(
                    direction=-1,
                    entry_z=z,
                    entry_bar=i,
                    entry_spread=float(row["spread"]),
                    highest_favorable_z=0.0,
                )
                before_n = len(trader.trades)
                trader.open_trade(
                    direction=-1,
                    time=time,
                    z=z,
                    spread=row["spread"],
                    ticker_a=pair.ticker_a,
                    ticker_b=pair.ticker_b,
                    basket=pair.basket,
                    risk_frac=0.08,
                    price_a=float(row["price_a"]),
                    price_b=float(row["price_b"]),
                )
                if len(trader.trades) == before_n:
                    position = 0
                else:
                    fresh_entry_this_bar = True

        # ---------- EXIT (rules + ML + half-life) ----------
        # Use `if` (not elif) so a just-adopted live position can exit this bar
        if position != 0:
            # Live mode only manages the position on the latest bar
            if live and not is_latest:
                continue
            # Live: never exit on the same bar we just entered (avoids wash trades
            # and overnight-hold semantics). Adopted exposure may still exit.
            if live and fresh_entry_this_bar:
                print(
                    f"ℹ️  Live hold overnight on {pair.label} "
                    f"(entered this bar; exit evaluated on next run)"
                )
                break

            bars_held = i - entry_idx
            pos_state.bars_held = bars_held

            # Live half-life of the spread (mean-reversion speed)
            half_life = estimate_half_life(df["spread"].iloc[: i + 1], lookback=40)

            # Exact feature vector the model will see
            features = extract_exit_features(
                position=pos_state,
                row=row,
                bars_held=bars_held,
                direction=position,
                half_life=half_life,
            )

            should_exit, ml_proba = should_exit_with_ml(
                position=position,
                z=z,
                bars_held=bars_held,
                features=features,
                model=model,
                ml_threshold=ml_threshold,
            )

            if should_exit:
                did_close = trader.close_trade(
                    time=time,
                    z=z,
                    spread=row["spread"],
                    ml_proba=ml_proba,
                    features=features,  # exact vector stored on the trade
                    bars_held=bars_held,  # bar count (not calendar days)
                )
                if not did_close:
                    # Broker still holding — stop managing this pair this run
                    break
                position = 0
                opened += 1

                if opened >= trades_remaining:
                    break

    return opened


def build_broker(broker: str = "sim", dry_run: bool = False) -> Optional[AlpacaPaperBroker]:
    """
    broker: 'sim' | 'alpaca'
    Always uses Alpaca *paper* endpoint when alpaca is selected.
    """
    mode = (broker or "sim").lower().strip()
    if mode in ("sim", "none", "local"):
        return None
    if mode not in ("alpaca", "alpaca_paper", "paper"):
        raise ValueError(f"Unknown broker '{broker}'. Use 'sim' or 'alpaca'.")
    if dry_run:
        return AlpacaPaperBroker(paper=True, dry_run=True)
    if not alpaca_credentials_present():
        raise RuntimeError(
            "Alpaca broker requested but credentials are missing. "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET_KEY."
        )
    return AlpacaPaperBroker(paper=True, dry_run=False)


def run_paper_trading_and_train(
    n_bars=600,
    min_trades=3,
    baskets: Optional[Sequence[str]] = None,
    include_cross: bool = True,
    max_pairs_per_basket: int = 6,
    ml_threshold: float = 0.62,
    noise_model: KalmanNoiseModel | str = KalmanNoiseModel.STANDARD,
    broker: str = "sim",
    alpaca_latest_only: bool = True,
    alpaca_dry_run: bool = False,
    data_source: str = "auto",
    mode: str = "backtest",
    data_window: str = "multi_year",
):
    mode = (mode or "backtest").lower().strip()
    if mode not in ("backtest", "live", "research"):
        raise ValueError("mode must be 'backtest', 'live', or 'research'")

    if mode == "research":
        setups, path = run_research_setups(
            n_bars=n_bars,
            baskets=baskets,
            include_cross=include_cross,
            max_pairs_per_basket=max_pairs_per_basket,
            ml_threshold=ml_threshold,
            noise_model=noise_model,
            data_source=data_source,
            data_window=data_window,
        )
        return setups, None, path

    print("Starting Paper Trading Session...")
    print(f"Mode: {mode}")
    if mode == "live":
        print("Live: fresh OHLCV → Kalman warm-up → act only on the latest bar")
        print("Goal: today's signal only (0 trades is OK if no ±2 z setup)\n")
    else:
        print("Backtest: replay history for the ML journal (Alpaca orders only if a fill hits latest bar)")
        print("Goal: Complete at least", min_trades, "round-trip trades\n")

    # 0. Universe — Mag7, semis, memory, hyperscaler
    baskets = list(baskets) if baskets is not None else list(TICKER_UNIVERSES.keys())
    pairs = build_pair_universe(
        baskets=baskets,
        include_cross=include_cross,
        max_pairs_per_basket=max_pairs_per_basket,
    )
    summarize_universes(pairs)

    # 1. Full OHLCV: Alpaca primary / yfinance backup; latest calendar year trades
    tickers = all_universe_tickers(baskets)
    panels, data_source = _load_prices_for_universe(
        tickers, n_bars, data_source=data_source
    )
    trade_year = int(next(iter(panels.values())).attrs.get(
        "trade_year", pd.Timestamp(next(iter(panels.values())).index.max()).year
    ))
    # Convenience field views for Kalman / pair scans
    fields = ohlcv_field_panels(panels)
    prices = fields["close"]
    highs = fields["high"]
    lows = fields["low"]
    volumes = fields["volume"]
    if isinstance(noise_model, str):
        noise_model = KalmanNoiseModel(noise_model)
    print(f"\nPrice panel: {prices.shape[1]} tickers × {prices.shape[0]} bars  [source: {data_source}]")
    print(f"Kalman R mode: {noise_model.value}")
    print(f"Trade windows: {trade_year} only (no prior-year entries)\n")

    # Try to load a previously trained model (rule-only if missing)
    model = None
    if MODEL_JSON.exists():
        try:
            model = LogisticExitModel.load(MODEL_JSON)
            if model.weights is None or len(model.weights) != len(FEATURE_NAMES):
                print(
                    f"⚠️ Saved model feature dim mismatch "
                    f"({None if model.weights is None else len(model.weights)} vs "
                    f"{len(FEATURE_NAMES)}); running rule-only this session"
                )
                model = None
            else:
                print(f"✅ Loaded existing exit model from {MODEL_JSON}")
        except Exception as e:
            print(f"⚠️ Could not load model ({e}); running rule-only this session")
            model = None
    else:
        print("ℹ️  No saved exit model yet — rule-only exits this session")

    # 2–3. Adaptive Kalman filter + paper trader (optional Alpaca paper brokerage)
    broker_client = build_broker(broker, dry_run=alpaca_dry_run)
    latest_bar = pd.Timestamp(prices.index.max())
    if mode == "live":
        alpaca_latest_only = True
    if broker_client is not None:
        print(
            f"🏦 Broker: Alpaca PAPER "
            f"(mode={mode}, execute_latest_only={alpaca_latest_only}, "
            f"latest_bar={latest_bar.date()}, dry_run={alpaca_dry_run})"
        )
    else:
        print("📒 Broker: local simulator (no brokerage orders)")

    trader = PaperTrader(
        broker=broker_client,
        execute_latest_only=alpaca_latest_only,
        latest_bar=latest_bar,
    )
    kf = AdaptiveKalmanPairs(delta=1e-4, R_base=1e-2, noise_model=noise_model)
    pair_frames: Dict[str, pd.DataFrame] = {}
    closed_count = 0

    for pair in pairs:
        if closed_count >= min_trades:
            break
        if pair.ticker_a not in prices.columns or pair.ticker_b not in prices.columns:
            continue

        price_a = prices[pair.ticker_a]
        price_b = prices[pair.ticker_b]
        vol_a = volumes[pair.ticker_a] if pair.ticker_a in volumes.columns else None
        hi_a = highs[pair.ticker_a] if pair.ticker_a in highs.columns else None
        lo_a = lows[pair.ticker_a] if pair.ticker_a in lows.columns else None
        df = kf.filter_pair(
            price_a,
            price_b,
            volume=vol_a,
            high=hi_a,
            low=lo_a,
        )
        pair_frames[pair.label] = df

        print(f"\n--- Scanning {pair.label} [{pair.basket}] (year={trade_year}, R={noise_model.value}) ---")
        opened = _trade_pair_session(
            trader, df, pair,
            min_trades=min_trades,
            trades_remaining=(1 if mode == "backtest" else max(1, min_trades - closed_count)),
            trade_year=trade_year,
            model=model,
            ml_threshold=ml_threshold,
            mode=mode,
            latest_bar=latest_bar,
        )
        closed_count += opened

    # 4. Show journal (latest-year trades only)
    closed_trades = trader.summary()
    closed_trades = [
        t for t in closed_trades
        if int(pd.Timestamp(t.entry_time).year) == trade_year
        and t.exit_time is not None
        and int(pd.Timestamp(t.exit_time).year) == trade_year
    ]
    # Persist still-open Alpaca live entries so the journal matches the brokerage
    open_broker_trades = [
        t for t in trader.trades
        if t.status == "OPEN"
        and str(getattr(t, "broker", "")).startswith("alpaca")
        and int(pd.Timestamp(t.entry_time).year) == trade_year
    ]
    # Use last scanned frame for return compatibility; prefer first if empty
    df_out = next(iter(pair_frames.values())) if pair_frames else prices

    to_save = list(closed_trades) + list(open_broker_trades)

    if len(closed_trades) < 2:
        if mode == "live":
            print(
                "Live session: fewer than 2 closed trades (no/insufficient latest-bar signals). "
                "Open Alpaca entries are still journaled when placed/adopted."
            )
            if to_save:
                save_paper_results(to_save, data_source=data_source)
            return trader, None, df_out
        print("Not enough trades generated. Try increasing n_bars or relaxing entry thresholds.")
        return trader, None, df_out

    # 5. Persist results, then train (from this run + any prior stored history)
    save_paper_results(to_save, data_source=data_source)
    print("\nTraining ML Exit Model on stored paper trades...")
    model = train_from_stored_results()

    return trader, model, df_out


# ============================================================
# RUN IT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper trading + ML exit trainer")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip paper session; retrain from results/exit_training_dataset.csv",
    )
    parser.add_argument("--n-bars", type=int, default=700)
    parser.add_argument("--min-trades", type=int, default=4)
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.62,
        help="ML exit probability threshold for forced exits (default 0.62)",
    )
    parser.add_argument(
        "--noise-model",
        choices=[m.value for m in KalmanNoiseModel],
        default=KalmanNoiseModel.STANDARD.value,
        help="Kalman measurement-noise mode: standard | volume | parkinson",
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "research"],
        default="backtest",
        help=(
            "backtest=replay history for ML journal; "
            "live=act only on latest bar (Alpaca paper); "
            "research=counterfactual setups → results/setups_*.csv (no broker)"
        ),
    )
    parser.add_argument(
        "--data-window",
        choices=["multi_year", "latest_year"],
        default="multi_year",
        help="Research mode only: whether to year-gate entry signals",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "alpaca", "yfinance"],
        default="auto",
        help="OHLCV source: auto (Alpaca then yfinance), alpaca, or yfinance",
    )
    parser.add_argument(
        "--broker",
        choices=["sim", "alpaca"],
        default="sim",
        help="sim = local journal only; alpaca = route latest-bar fills to Alpaca paper",
    )
    parser.add_argument(
        "--alpaca-all-bars",
        action="store_true",
        help="Submit Alpaca paper orders for every simulated fill (dangerous; default is latest bar only)",
    )
    parser.add_argument(
        "--alpaca-dry-run",
        action="store_true",
        help="Build Alpaca order payloads without calling the API (for tests)",
    )
    args = parser.parse_args()

    if args.mode == "live" and args.broker == "sim":
        print("ℹ️  Live mode with --broker sim will not place Alpaca orders. Use --broker alpaca.")

    if args.train_only:
        model = train_from_stored_results()
        print("\n🎯 Retrain complete from stored results.")
    else:
        trader, model, data = run_paper_trading_and_train(
            n_bars=args.n_bars,
            min_trades=args.min_trades,
            baskets=["mag7", "semis", "memory", "hyperscaler"],
            include_cross=True,
            ml_threshold=args.ml_threshold,
            noise_model=args.noise_model,
            broker=args.broker,
            alpaca_latest_only=not args.alpaca_all_bars,
            alpaca_dry_run=args.alpaca_dry_run,
            data_source=args.data_source,
            mode=args.mode,
            data_window=args.data_window,
        )

        if args.mode == "research":
            print("\n🎯 Research setup pass complete.")
            print(f"   Setups: {data}")
            print("Live/backtest journal untouched.")
        elif model is None:
            print("\n⚠️ Paper trading finished without enough trades to train.")
        else:
            print("\n🎯 Paper trading session complete.")
            print(f"   Journal:  {TRADES_CSV}")
            print(f"   Dataset:  {DATASET_CSV}")
            print(f"   Model:    {MODEL_JSON}")
            print("Universe covered: Mag7, semis, memory, hyperscaler.")
            print("Next: python paper_trading_ml_exit.py --train-only")
