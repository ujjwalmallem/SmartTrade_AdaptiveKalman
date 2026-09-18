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

from src.features import (
    FEATURE_NAMES as SPEC_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION as SPEC_FEATURE_SCHEMA_VERSION,
    features_from_row as spec_features_from_row,
    extract_feature_dict,
)
from src.exit_manager import StatArbExitManager, TradeState, time_stop_bars
from src.train_exit_model import train_exit_model as train_sklearn_exit_model
from src.config import load_strategy_config, exit_threshold as config_exit_threshold
from src.kalman import AdaptiveKalmanPairs, KalmanNoiseModel
from src.journal import (
    dedupe_journal_rows,
    filter_journal_by_scope,
    is_wash_closed_row as _is_wash_closed_row,
    trade_fingerprint_cols as _trade_fingerprint_cols,
)
from src.legacy_logistic import LogisticExitModel

# Default artifact locations (gitignored locally; CI uploads as artifacts)
RESULTS_DIR = Path("results")
TRADES_CSV = RESULTS_DIR / "paper_trades.csv"
DATASET_CSV = RESULTS_DIR / "exit_training_dataset.csv"
MODEL_JSON = RESULTS_DIR / "logistic_exit_model.json"
SETUPS_GLOB = "setups_*.csv"

# SYSTEM_SPEC §2 — 8-feature symmetric set (source of truth: src/features.py)
FEATURE_NAMES = list(SPEC_FEATURE_NAMES)
FEATURE_SCHEMA_VERSION = int(SPEC_FEATURE_SCHEMA_VERSION)

# AdaptiveKalmanPairs → src/kalman.py; journal helpers → src/journal.py
# LogisticExitModel → src/legacy_logistic.py (quarantined; not used for exits)

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
# ADAPTIVE KALMAN PAIRS  –  imported from src.kalman
# ============================================================
# AdaptiveKalmanPairs, KalmanNoiseModel available via import above.



# LogisticExitModel imported from src.legacy_logistic (quarantined).



def analyze_feature_importance(model, feature_names):
    print("\nFeature importance (standardized |weight|, odds ratio e^w):")
    abs_w = np.abs(model.weights)
    order = np.argsort(-abs_w)
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        w = float(model.weights[i])
        print(f"  {name:16s} w={w:+.4f}  OR={np.exp(w):.4f}")
    print(f"  {'bias':16s} β0={float(model.bias):+.4f}  P0={1.0/(1.0+np.exp(-float(model.bias))):.1%}")


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
    SYSTEM_SPEC §2 feature vector (delegates to src.features).

    Also tracks highest_favorable_z on the position for journal diagnostics
    (not fed to the classifier — favorable/best_fav are dropped).
    """
    z = float(row["zscore"])
    if direction == 1:
        favorable = max(0.0, z - position.entry_z)
    else:
        favorable = max(0.0, position.entry_z - z)
    position.highest_favorable_z = max(position.highest_favorable_z, favorable)
    return spec_features_from_row(
        position, row, bars_held, direction, half_life=half_life
    )


def trade_to_features(t: "PaperTrade") -> np.ndarray:
    """
    Prefer the exact live feature vector stored at exit.
    Fall back to a reconstructed approximation only if missing
    or if the stored vector is from an older feature schema.
    """
    if t.exit_features is not None:
        vec = np.asarray(t.exit_features, dtype=float).ravel()
        if len(vec) == len(FEATURE_NAMES):
            return vec

    from src.features import extract_feature_vector

    pnl = float(t.pnl_z) if t.pnl_z is not None else 0.0
    exit_z = float(t.exit_z) if t.exit_z is not None else 0.0
    bars = int(t.bars_held or 0)
    direction = 1 if str(getattr(t, "direction", "")).upper().startswith("LONG") else -1
    # Reconstruct entry_z from pnl + exit when possible
    entry_z = float(t.entry_z)
    return extract_feature_vector(
        entry_z=entry_z,
        current_z=exit_z,
        direction=direction,
        vol=1.0,
        confidence=0.70,
        velocity=0.0,
        bars_held=bars,
        half_life=20.0,
    )


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


def get_exit_model_version(model: Optional[Any] = None) -> str:
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
    model: Optional[Any],
    run_id: str,
    data_window: str = "multi_year",
    ml_threshold: float = 0.68,
    capital: float = 100_000.0,
    risk_frac: float = 0.08,
    cost_bps: float = 4.0,
    z_to_pct: float = 0.01,
    trade_year: Optional[int] = None,
    exit_manager: Optional[StatArbExitManager] = None,
) -> List[Setup]:
    """
    Research mode: walk the window and simulate every valid z-crossing to
    completion. Does not touch PaperTrader / Alpaca / paper_trades.csv.

    Year gate: when data_window == "latest_year", entries outside trade_year
    are skipped (same idea as live/backtest). multi_year skips that gate.
    """
    if exit_manager is None:
        raise ValueError("exit_manager is required for research setups")
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
                "vol": float(row.get("spread_vol", 1.0)),
                "pnl_proxy": 0.0,
                "abs_entry_z": abs(z),
                "confidence": conf,
                "exit_z": z,
                "velocity": float(row.get("spread_velocity", 0.0)),
                "bars_held": 0.0,
                "half_life": float(half_life) / 30.0,
            }
            continue

        # ----- EXIT (exact live helpers) -----
        bars_held = i - entry_idx
        pos_state.bars_held = bars_held
        half_life = estimate_half_life(df["spread"].iloc[: i + 1], lookback=40)
        feat_vec = extract_exit_features(
            pos_state, row, bars_held, position, half_life=half_life
        )
        trade_state = build_trade_state(
            pair=pair,
            direction=position,
            pos_state=pos_state,
            row=row,
            bars_held=bars_held,
            half_life=half_life,
        )
        should_exit, ml_proba = should_exit_with_ml(
            position=position,
            z=z,
            bars_held=bars_held,
            features=feat_vec,
            model=None,
            ml_threshold=ml_threshold,
            half_life=half_life,
            exit_manager=exit_manager,
            trade_state=trade_state,
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
    ml_threshold: Optional[float] = None,
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
    if ml_threshold is None:
        ml_threshold = config_exit_threshold()

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

    # Production exit path is StatArbExitManager only (legacy JSON deprecated)
    model = None
    print("ℹ️  Research exits via StatArbExitManager (sklearn artifacts if present)")

    exit_manager = load_exit_manager(ml_threshold=ml_threshold)

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
            exit_manager=exit_manager,
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



# Journal helpers: _trade_fingerprint_cols, _is_wash_closed_row,
# dedupe_journal_rows, filter_journal_by_scope — imported from src.journal.



def save_paper_results(
    closed_trades: Sequence[PaperTrade],
    results_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    data_source: str = "",
    journal_scope: str = "all",
    update_training: bool = True,
) -> Tuple[Path, Path]:
    """
    Append trades to paper_trades.csv and optionally the training dataset.

    journal_scope: all | alpaca | sim | none
      alpaca — only Alpaca paper rows in paper_trades (strips legacy sim history)
      none   — do not write paper_trades.csv
    """
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    year = _latest_allowed_trade_year()
    scope = (journal_scope or "all").lower().strip()

    frame = closed_trades_to_frame(closed_trades, run_id=run_id, data_source=data_source)
    trades_path = results_dir / "paper_trades.csv"
    dataset_path = results_dir / "exit_training_dataset.csv"

    journal = pd.DataFrame()
    if scope not in ("none", "off", "skip"):
        frame_j = filter_journal_by_scope(frame, scope)
        if trades_path.exists():
            prev = pd.read_csv(trades_path)
            prev = filter_journal_by_scope(prev, scope)
            journal = pd.concat([prev, frame_j], ignore_index=True)
        else:
            journal = frame_j
        journal = filter_trades_to_latest_year(
            journal, trade_year=year, require_exit_in_year=False
        )
        journal = dedupe_journal_rows(journal, drop_wash=True)
        journal.to_csv(trades_path, index=False)
        print(f"\n💾 Saved {len(closed_trades)} new trades → {trades_path} (scope={scope})")
        print(f"💾 Journal rows kept for {year}: {len(journal)}")
    else:
        print("\nℹ️  Skipping paper_trades.csv write (journal_scope=none)")

    if not update_training:
        return trades_path, dataset_path

    # Training dataset = feature columns + label (closed trades only)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    closed_frame = frame.copy()
    if "status" in closed_frame.columns:
        closed_frame = closed_frame[closed_frame["status"].fillna("CLOSED") == "CLOSED"]
    # For alpaca journal scope, only train on alpaca closed fills
    if scope == "alpaca" and "broker" in closed_frame.columns:
        closed_frame = closed_frame[
            closed_frame["broker"].astype(str).str.lower().str.startswith("alpaca")
        ]
    closed_frame = closed_frame.dropna(subset=["label"]) if "label" in closed_frame.columns else closed_frame
    for col in feat_cols:
        if col not in closed_frame.columns:
            closed_frame[col] = np.nan
    want_cols = ["run_id", "data_source", "trade_id", "ticker_a", "ticker_b", "basket", *feat_cols, "label"]
    for col in want_cols:
        if col not in closed_frame.columns:
            closed_frame[col] = np.nan
    ds = closed_frame[want_cols]
    ds = ds.dropna(subset=feat_cols, how="any")
    if not frame.empty and "status" in frame.columns:
        wash_keys = frame.loc[_is_wash_closed_row(frame), ["run_id", "trade_id"]]
        if not wash_keys.empty and {"run_id", "trade_id"}.issubset(ds.columns):
            ds = ds.merge(wash_keys.drop_duplicates(), on=["run_id", "trade_id"], how="left", indicator=True)
            ds = ds.loc[ds["_merge"] == "left_only"].drop(columns=["_merge"])
    if dataset_path.exists():
        prev_ds = pd.read_csv(dataset_path)
        ds = pd.concat([prev_ds, ds], ignore_index=True)
    # Tie training rows to journal keys so alpaca scope drops legacy sim labels
    if scope not in ("none", "off", "skip"):
        if not journal.empty and {"run_id", "trade_id"}.issubset(ds.columns):
            keys = journal[["run_id", "trade_id"]].drop_duplicates()
            ds = ds.merge(keys, on=["run_id", "trade_id"], how="inner")
        elif journal.empty:
            # No in-scope journal rows → empty training (e.g. alpaca-only, no fills yet)
            ds = ds.iloc[0:0]
    if not ds.empty and {"run_id", "trade_id"}.issubset(ds.columns):
        ds = ds.drop_duplicates(subset=["run_id", "trade_id"], keep="last")
    ds.to_csv(dataset_path, index=False)
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
    missing = [c for c in feat_cols + ["label"] if c not in ds.columns]
    if missing:
        raise ValueError(
            f"Dataset missing columns {missing} (feature schema v{FEATURE_SCHEMA_VERSION}). "
            "Re-run a paper session to rebuild exit_training_dataset.csv."
        )
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
    min_samples: Optional[int] = None,
) -> None:
    """
    Train the production sklearn exit model (SYSTEM_SPEC).

    Writes models/*.pkl only. Raises if labeled rows < training.min_samples.
    """
    results_dir = Path(dataset_path).parent if dataset_path else RESULTS_DIR
    cfg = load_strategy_config()
    floor = min_samples if min_samples is not None else int(
        (cfg.get("training") or {}).get("min_samples", 50)
    )
    meta = train_sklearn_exit_model(
        results_dir,
        min_samples=floor,
        calibrate=True,
    )
    print(
        f"✅ Sklearn exit model promoted "
        f"(n={meta['metrics']['n_samples']}, "
        f"pos_rate={meta['metrics']['class_balance']['positive_rate']:.1%})"
    )

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

def build_trade_state(
    *,
    pair: PairSpec,
    direction: int,
    pos_state: PositionState,
    row: pd.Series,
    bars_held: int,
    half_life: float,
    trade_id: int = 0,
    cost_dollars: float = 0.0,
    pnl_dollars: float = 0.0,
) -> TradeState:
    """Map Kalman bar + open position → SYSTEM_SPEC TradeState for evaluate_trade."""
    z = float(row["zscore"])
    feats = extract_feature_dict(
        entry_z=float(pos_state.entry_z),
        current_z=z,
        direction=int(direction),
        vol=float(row.get("spread_vol", 1.0)),
        confidence=float(row.get("confidence", 0.5)),
        velocity=float(row.get("z_velocity", row.get("spread_velocity", 0.0))),
        bars_held=int(bars_held),
        half_life=float(half_life),
    )
    return TradeState(
        trade_id=int(trade_id),
        ticker_a=pair.ticker_a,
        ticker_b=pair.ticker_b,
        direction="LONG_SPREAD" if int(direction) == 1 else "SHORT_SPREAD",
        bars_held=int(bars_held),
        half_life_bars=float(half_life),
        vol=feats["vol"],
        pnl_proxy=feats["pnl_proxy"],
        entry_z=float(pos_state.entry_z),
        confidence=feats["confidence"],
        exit_z=feats["exit_z"],
        velocity=feats["velocity"],
        half_life=feats["half_life"],
        cost_dollars=float(cost_dollars),
        pnl_dollars=float(pnl_dollars),
    )


def load_exit_manager(
    ml_threshold: Optional[float] = None,
    config_path: Path | str = "config/strategy_config.yaml",
) -> StatArbExitManager:
    """Load StatArbExitManager from YAML + optional sklearn artifacts."""
    cfg = load_strategy_config(config_path)
    threshold = float(
        ml_threshold if ml_threshold is not None else config_exit_threshold(cfg)
    )
    mgr = StatArbExitManager.from_config(config_path)
    mgr.exit_threshold = threshold
    if mgr.model is not None and mgr.scaler is not None:
        print(
            f"✅ StatArbExitManager ready "
            f"(threshold={mgr.exit_threshold}, time-stop=max("
            f"{mgr.absolute_min_bars}, {mgr.max_half_life_multiplier}×hl))"
        )
    else:
        print(
            "ℹ️  StatArbExitManager active without sklearn weights — "
            "hard time-stop / stop-loss only until models/*.pkl exist "
            f"(need ≥{(cfg.get('training') or {}).get('min_samples', 50)} labeled rows)"
        )
    return mgr


def should_exit_with_ml(
    position: int,
    z: float,
    bars_held: int,
    features: np.ndarray,
    model: Optional[Any] = None,
    ml_threshold: float = 0.68,
    force_rules: bool = True,
    half_life: float = 20.0,
    exit_manager: Optional[StatArbExitManager] = None,
    trade_state: Optional[TradeState] = None,
) -> Tuple[bool, Optional[float]]:
    """
    Exit decision for an open position.

    Requires StatArbExitManager + TradeState (fail-closed). Hard time-stop /
    stop-loss / ML via evaluate_trade. Soft mean-reversion only when the
    manager has no sklearn weights yet (bootstrap until N≥min_samples).

    `model` is ignored (legacy LogisticExitModel path removed).
    """
    if exit_manager is None or trade_state is None:
        raise ValueError(
            "exit_manager and trade_state are required; "
            "legacy LogisticExitModel fallback is disabled"
        )
    if model is not None:
        # Callers may still pass None/legacy; never use it for decisions.
        pass

    should, reason, prob = exit_manager.evaluate_trade(trade_state)
    if should:
        return True, float(prob)
    # Soft mean-reversion only while sklearn weights are missing so
    # positions can still converge before the first train.
    if exit_manager.model is None and force_rules:
        long_z = getattr(exit_manager, "soft_mr_long_z", -0.35)
        short_z = getattr(exit_manager, "soft_mr_short_z", 0.35)
        if position == 1 and z > long_z:
            return True, float(prob)
        if position == -1 and z < short_z:
            return True, float(prob)
    return False, float(prob)


def _trade_pair_session(
    trader: PaperTrader,
    df: pd.DataFrame,
    pair: PairSpec,
    min_trades: int,
    trades_remaining: int,
    trade_year: Optional[int] = None,
    model: Optional[Any] = None,
    ml_threshold: float = 0.68,
    mode: str = "backtest",
    latest_bar: Optional[pd.Timestamp] = None,
    exit_manager: Optional[StatArbExitManager] = None,
) -> int:
    """
    Run rule-based entries + ML-augmented exits on one pair.
    Returns the number of newly closed trades.

    mode:
      - backtest: scan the full trade-year window (builds ML journal; sim fills)
      - live: warm Kalman on history, enter/exit only on the latest bar
              (this is what places Alpaca paper orders with --broker alpaca)
    """
    if exit_manager is None:
        raise ValueError("exit_manager is required for live/backtest exits")
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

            # Mark-to-market $ PnL for hard dollar stop (SYSTEM_SPEC risk_engine)
            open_t = next(
                (t for t in reversed(trader.trades) if t.status == "OPEN"),
                None,
            )
            pnl_dollars = 0.0
            cost_dollars = 0.0
            trade_id = 0
            if open_t is not None:
                trade_id = int(open_t.trade_id)
                cost_dollars = float(open_t.cost_dollars or 0.0)
                notional = float(open_t.notional or 0.0)
                pnl_z = (
                    z - float(pos_state.entry_z)
                    if position == 1
                    else float(pos_state.entry_z) - z
                )
                # Same z→$ mapping as close_trade (approx 1% per z-unit)
                pnl_dollars = pnl_z * 0.01 * notional - cost_dollars

            trade_state = build_trade_state(
                pair=pair,
                direction=position,
                pos_state=pos_state,
                row=row,
                bars_held=bars_held,
                half_life=half_life,
                trade_id=trade_id,
                cost_dollars=cost_dollars,
                pnl_dollars=pnl_dollars,
            )

            should_exit, ml_proba = should_exit_with_ml(
                position=position,
                z=z,
                bars_held=bars_held,
                features=features,
                model=model,
                ml_threshold=ml_threshold,
                half_life=half_life,
                exit_manager=exit_manager,
                trade_state=trade_state,
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
    ml_threshold: Optional[float] = None,
    noise_model: KalmanNoiseModel | str = KalmanNoiseModel.STANDARD,
    broker: str = "sim",
    alpaca_latest_only: bool = True,
    alpaca_dry_run: bool = False,
    data_source: str = "auto",
    mode: str = "backtest",
    data_window: str = "multi_year",
    journal_scope: str = "alpaca",
):
    mode = (mode or "backtest").lower().strip()
    if mode not in ("backtest", "live", "research"):
        raise ValueError("mode must be 'backtest', 'live', or 'research'")

    if ml_threshold is None:
        ml_threshold = config_exit_threshold()

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

    # Production exit path: StatArbExitManager only (SYSTEM_SPEC).
    # Legacy results/logistic_exit_model.json is no longer loaded for live exits.
    model = None
    exit_manager = load_exit_manager(ml_threshold=ml_threshold)

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
            exit_manager=exit_manager,
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
    scope = (journal_scope or "alpaca").lower().strip()

    if len(closed_trades) < 2:
        if mode == "live":
            print(
                "Live session: fewer than 2 closed trades (no/insufficient latest-bar signals). "
                "Open Alpaca entries are still journaled when placed/adopted."
            )
            if to_save:
                save_paper_results(to_save, data_source=data_source, journal_scope=scope)
            return trader, exit_manager, df_out
        print("Not enough trades generated. Try increasing n_bars or relaxing entry thresholds.")
        # Still persist when explicitly journaling (e.g. alpaca scope with open rows)
        if to_save and scope not in ("none", "off", "skip"):
            save_paper_results(to_save, data_source=data_source, journal_scope=scope)
        return trader, exit_manager, df_out

    # 5. Persist results, then train sklearn exit model when enough labels exist
    save_paper_results(to_save, data_source=data_source, journal_scope=scope)
    print("\nTraining ML Exit Model on stored paper trades...")
    try:
        train_from_stored_results()
        # Reload manager so subsequent callers see fresh weights
        exit_manager = load_exit_manager(ml_threshold=ml_threshold)
    except ValueError as e:
        print(f"⚠️ Skipping train this run: {e}")

    return trader, exit_manager, df_out


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
        default=None,
        help=(
            "ML exit probability threshold "
            "(default: config/strategy_config.yaml exit_model.probability_threshold)"
        ),
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
    parser.add_argument(
        "--save-journal",
        choices=["alpaca", "all", "sim", "none"],
        default="alpaca",
        help=(
            "What to keep in paper_trades.csv: "
            "alpaca (default, real paper fills only), all, sim, or none"
        ),
    )
    args = parser.parse_args()

    if args.mode == "live" and args.broker == "sim":
        print("ℹ️  Live mode with --broker sim will not place Alpaca orders. Use --broker alpaca.")
    if args.broker == "sim" and args.save_journal == "alpaca":
        print(
            "ℹ️  --save-journal alpaca with --broker sim writes no sim rows "
            "(journal stays Alpaca-only). Use --save-journal all|sim to keep simulator fills."
        )

    if args.train_only:
        try:
            train_from_stored_results()
            print("\n🎯 Retrain complete → models/logistic_exit_model.pkl")
        except ValueError as e:
            print(f"\n⚠️ Retrain skipped: {e}")
    else:
        thr = args.ml_threshold
        if thr is None:
            thr = config_exit_threshold()
        trader, exit_mgr, data = run_paper_trading_and_train(
            n_bars=args.n_bars,
            min_trades=args.min_trades,
            baskets=["mag7", "semis", "memory", "hyperscaler"],
            include_cross=True,
            ml_threshold=thr,
            noise_model=args.noise_model,
            broker=args.broker,
            alpaca_latest_only=not args.alpaca_all_bars,
            alpaca_dry_run=args.alpaca_dry_run,
            data_source=args.data_source,
            mode=args.mode,
            data_window=args.data_window,
            journal_scope=args.save_journal,
        )

        if args.mode == "research":
            print("\n🎯 Research setup pass complete.")
            print(f"   Setups: {data}")
            print("Live/backtest journal untouched.")
        else:
            print("\n🎯 Paper trading session complete.")
            print(f"   Journal:  {TRADES_CSV}")
            print(f"   Dataset:  {DATASET_CSV}")
            pkl = Path("models/logistic_exit_model.pkl")
            if exit_mgr is not None and exit_mgr.model is not None:
                print(f"   Exit mgr: StatArbExitManager @ {exit_mgr.exit_threshold}")
                print(f"   Model:    {pkl}")
            else:
                print("   Exit mgr: hard stops only (sklearn model not promoted yet)")
            print("Universe covered: Mag7, semis, memory, hyperscaler.")
            print("Next: PYTHONPATH=. python -m src.train_exit_model")
