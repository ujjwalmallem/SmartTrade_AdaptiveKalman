"""
Paper Trading + ML Exit Model Trainer
Ready for Cursor AI
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Iterable, Optional, Sequence, Tuple
import warnings
warnings.filterwarnings("ignore")

# Default artifact locations (gitignored locally; CI uploads as artifacts)
RESULTS_DIR = Path("results")
TRADES_CSV = RESULTS_DIR / "paper_trades.csv"
DATASET_CSV = RESULTS_DIR / "exit_training_dataset.csv"
MODEL_JSON = RESULTS_DIR / "logistic_exit_model.json"

FEATURE_NAMES = [
    "entry_z", "abs_entry_z", "pnl_proxy", "bars_held", "confidence",
    "velocity", "exit_z", "favorable", "best_fav", "vol",
]

# ============================================================
# PASTE ALL PREVIOUS CLASSES HERE (or keep them in the same file)
# Required: AdaptiveKalmanPairs, LogisticExitModel, 
#           PositionState, extract_exit_features, 
#           generate_training_data, ExitConfig, etc.
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

def generate_synthetic_pair(n=800, seed=42, name_a="A", name_b="B"):
    np.random.seed(seed)
    t = np.arange(n)
    true_beta = 1.15 + 0.12 * np.sin(t / 50)
    returns_b = np.random.normal(0.0002, 0.012, n)
    price_b = 100 * np.exp(np.cumsum(returns_b))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.90 * residual[i-1] + np.random.normal(0, 0.7)
    price_a = true_beta * price_b + residual + np.random.normal(0, 0.4, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return (
        pd.Series(price_a, index=idx, name=name_a),
        pd.Series(price_b, index=idx, name=name_b),
    )


def generate_synthetic_prices_for_universe(
    tickers: Sequence[str],
    n: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Correlated synthetic prices for universe tickers.
    Shared market + basket factors make within-basket pairs more cointegrated.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    market = rng.normal(0.00025, 0.01, n)

    basket_factors = {
        name: rng.normal(0.0001, 0.008, n) for name in TICKER_UNIVERSES
    }
    # Stable per-ticker seed offsets so pair generation is reproducible
    ticker_list = list(tickers)
    prices = {}
    for i, ticker in enumerate(ticker_list):
        baskets = ticker_baskets(ticker)
        factor = np.zeros(n)
        for b in baskets:
            factor += basket_factors[b] / max(len(baskets), 1)
        if not baskets:
            factor = rng.normal(0.0, 0.006, n)
        idio = rng.normal(0.00005, 0.012, n)
        # Slightly different loadings by ticker index
        beta_mkt = 0.85 + 0.05 * (i % 5)
        rets = beta_mkt * market + factor + idio
        # AR(1) residual overlay for mean-reverting relative value
        resid = np.zeros(n)
        for t in range(1, n):
            resid[t] = 0.88 * resid[t - 1] + rng.normal(0, 0.004)
        level = 80 + 15 * (i % 7)
        prices[ticker] = level * np.exp(np.cumsum(rets + resid))
    return pd.DataFrame(prices, index=idx)


def fetch_real_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "3y",
) -> pd.DataFrame:
    """
    Real daily close prices for the given tickers via yfinance, aligned on a
    shared trading-day index (only days every ticker has a price for).
    Returns the most recent `n_bars` rows. Raises on any fetch/shape problem
    so the caller can decide how to handle it (see _load_prices_for_universe).
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
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

    closes: Dict[str, pd.Series] = {}
    for t in tickers:
        try:
            series = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        except KeyError:
            continue
        series = series.dropna()
        if len(series) > 0:
            closes[t] = series

    missing = [t for t in tickers if t not in closes]
    if missing:
        print(f"⚠️  No usable close prices from yfinance for: {', '.join(missing)}")
    if not closes:
        raise RuntimeError("yfinance returned no usable close prices for any ticker")

    prices = pd.DataFrame(closes).dropna(how="any")
    if prices.empty:
        raise RuntimeError("No overlapping trading days across fetched tickers")

    return prices.tail(n_bars)


def _load_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int,
    mode: str = "auto",
) -> Tuple[pd.DataFrame, str]:
    """
    mode: "live" (fetch real data, raise on failure), "synthetic" (skip live
    entirely), or "auto" (try live, fall back to synthetic on any failure).
    Returns (prices, source_label) — source_label is recorded on every trade
    row so a synthetic-fallback run is visible in the stored results instead
    of silently looking like live data.
    """
    if mode in ("auto", "live"):
        try:
            prices = fetch_real_prices_for_universe(tickers, n_bars=n_bars)
            if len(prices) < 100:
                raise RuntimeError(f"only {len(prices)} usable bars (<100)")
            return prices, "yfinance_live"
        except Exception as exc:
            if mode == "live":
                raise
            print(f"⚠️  Live data fetch failed ({exc}); falling back to synthetic data.")
    return generate_synthetic_prices_for_universe(tickers, n=n_bars, seed=42), "synthetic_fallback"

# Very simplified Kalman for demo (replace with full AdaptiveKalmanPairs)
class SimpleKalmanPairs:
    def __init__(self):
        self.beta = 1.0
        self.history = []
    
    def filter_pair(self, a, b):
        # Rolling OLS beta so the demo spread is mean-reverting enough for trades
        cov = a.rolling(60).cov(b)
        var = b.rolling(60).var()
        beta = (cov / var).fillna(self.beta)
        spread = a - beta * b
        z = (spread - spread.rolling(40).mean()) / spread.rolling(40).std()
        vol = spread.rolling(20).std()
        vol_med = vol.median()
        conf = (vol_med / (vol_med + vol)).fillna(0.5)
        df = pd.DataFrame({
            "price_a": a,
            "price_b": b,
            "spread": spread,
            "zscore": z,
            "confidence": conf.clip(0.3, 0.95),
            "spread_velocity": spread.diff().rolling(5).mean().fillna(0),
            "spread_vol": spread.rolling(15).std().fillna(1)
        }, index=a.index)
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

def trade_to_features(t: PaperTrade) -> np.ndarray:
    """Build the exit-model feature vector from a closed paper trade."""
    return np.array([
        t.entry_z,
        abs(t.entry_z),
        t.pnl_z,
        t.bars_held / 30.0,
        0.7,   # confidence placeholder until live features are wired
        0.1,   # velocity placeholder
        abs(t.exit_z) if t.exit_z is not None else 0.0,
        abs(t.exit_z) if t.exit_z is not None else 0.0,
        abs(t.entry_z),
        1.0,
    ], dtype=float)


def trade_to_label(t: PaperTrade) -> int:
    """1 = exit was reasonable (profit or time stop)."""
    return 1 if t.pnl_z > 0 or t.bars_held > 20 else 0


def closed_trades_to_frame(
    closed_trades: Sequence[PaperTrade], run_id: str = "", data_source: str = ""
) -> pd.DataFrame:
    rows = []
    for t in closed_trades:
        feat = trade_to_features(t)
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
            "ml_proba_at_exit": t.ml_proba_at_exit,
            "label": trade_to_label(t),
        }
        for name, val in zip(FEATURE_NAMES, feat):
            row[f"feat_{name}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def save_paper_results(
    closed_trades: Sequence[PaperTrade],
    results_dir: Path = RESULTS_DIR,
    run_id: Optional[str] = None,
    data_source: str = "",
) -> Tuple[Path, Path]:
    """
    Append closed trades to a journal CSV and write/append the training dataset.
    Returns (trades_csv_path, dataset_csv_path).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    frame = closed_trades_to_frame(closed_trades, run_id=run_id, data_source=data_source)
    trades_path = results_dir / "paper_trades.csv"
    dataset_path = results_dir / "exit_training_dataset.csv"

    # Full journal (append across sessions)
    if trades_path.exists():
        prev = pd.read_csv(trades_path)
        journal = pd.concat([prev, frame], ignore_index=True)
    else:
        journal = frame
    journal.to_csv(trades_path, index=False)

    # Training dataset = feature columns + label (append)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    ds = frame[["run_id", "data_source", "trade_id", "ticker_a", "ticker_b", "basket", *feat_cols, "label"]]
    if dataset_path.exists():
        prev_ds = pd.read_csv(dataset_path)
        ds = pd.concat([prev_ds, ds], ignore_index=True)
    ds.to_csv(dataset_path, index=False)

    print(f"\n💾 Saved {len(closed_trades)} trades → {trades_path}")
    print(f"💾 Training dataset rows: {len(ds)} → {dataset_path}")
    return trades_path, dataset_path


def load_training_dataset(dataset_path: Path = DATASET_CSV) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No stored dataset at {path}. Run a paper session first to create it."
        )
    ds = pd.read_csv(path)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    missing = [c for c in feat_cols + ["label"] if c not in ds.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    X = ds[feat_cols].to_numpy(dtype=float)
    y = ds["label"].to_numpy(dtype=float)
    return X, y, ds


def train_from_stored_results(
    dataset_path: Path = DATASET_CSV,
    model_path: Path = MODEL_JSON,
    reg: float = 0.3,
    min_samples: int = 2,
) -> LogisticExitModel:
    """Load accumulated paper-trade features and fit the exit model."""
    X, y, ds = load_training_dataset(dataset_path)
    if len(y) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples; found {len(y)} in {dataset_path}")

    print(f"\nTraining from stored results: {len(y)} samples ({dataset_path})")
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
    status: str = "OPEN"
    ml_proba_at_exit: float = None

class PaperTrader:
    def __init__(self, capital=100_000):
        self.capital = capital
        self.trades: List[PaperTrade] = []
        self.current_trade: PaperTrade = None
        self.equity = capital
        self.trade_counter = 0
    
    def open_trade(self, direction: int, time, z, spread, ticker_a="", ticker_b="", basket=""):
        self.trade_counter += 1
        side = "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD"
        trade = PaperTrade(
            trade_id=self.trade_counter,
            direction=side,
            entry_time=time,
            entry_z=z,
            entry_spread=spread,
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            basket=basket,
        )
        self.current_trade = trade
        self.trades.append(trade)
        pair = f"{ticker_a}/{ticker_b}" if ticker_a and ticker_b else "PAIR"
        print(f"\n🟢 OPENED Trade #{trade.trade_id} | {side} | {pair} [{basket}]")
        print(f"   Time: {time.date()} | z={z:.2f} | spread={spread:.3f}")
    
    def close_trade(self, time, z, spread, ml_proba=None):
        if self.current_trade is None:
            return
        
        t = self.current_trade
        t.exit_time = time
        t.exit_z = z
        t.exit_spread = spread
        t.bars_held = (time - t.entry_time).days
        t.ml_proba_at_exit = ml_proba
        t.status = "CLOSED"
        
        if t.direction == "LONG_SPREAD":
            t.pnl_z = z - t.entry_z
        else:
            t.pnl_z = t.entry_z - z
        
        pair = f"{t.ticker_a}/{t.ticker_b}" if t.ticker_a and t.ticker_b else "PAIR"
        print(f"🔴 CLOSED Trade #{t.trade_id} | {t.direction} | {pair}")
        print(f"   Time: {time.date()} | z={z:.2f} | PnL(z)={t.pnl_z:+.3f} | Bars={t.bars_held}")
        if ml_proba is not None:
            print(f"   ML Exit Prob at close: {ml_proba:.2%}")
        
        self.current_trade = None
    
    def summary(self):
        closed = [t for t in self.trades if t.status == "CLOSED"]
        print("\n" + "="*60)
        print("PAPER TRADING JOURNAL")
        print("="*60)
        for t in closed:
            pair = f"{t.ticker_a}/{t.ticker_b}" if t.ticker_a else "?"
            print(f"Trade #{t.trade_id:2d} | {pair:13s} | {t.basket:18s} | {t.direction:13s} | "
                  f"Entry z={t.entry_z:+.2f} → Exit z={t.exit_z:+.2f} | "
                  f"PnL(z)={t.pnl_z:+.3f} | Held {t.bars_held} bars")
        
        if closed:
            pnls = [t.pnl_z for t in closed]
            print(f"\nTotal closed trades: {len(closed)}")
            print(f"Average PnL (z):     {np.mean(pnls):+.3f}")
            print(f"Win rate:            {np.mean([p > 0 for p in pnls]):.1%}")
            by_basket: Dict[str, List[float]] = {}
            for t in closed:
                by_basket.setdefault(t.basket or "unknown", []).append(t.pnl_z)
            print("\nBy basket:")
            for basket, vals in by_basket.items():
                print(f"  {basket:18s} n={len(vals)}  avg PnL(z)={np.mean(vals):+.3f}")
        print("="*60)
        return closed

# ============================================================
# MAIN PAPER TRADING + TRAINING LOOP
# ============================================================

def _trade_pair_session(
    trader: PaperTrader,
    df: pd.DataFrame,
    pair: PairSpec,
    min_trades: int,
    trades_remaining: int,
) -> int:
    """Run rule-based entries/exits on one pair; return number of new closed trades."""
    position = 0
    entry_idx = 0
    opened = 0

    for i in range(60, len(df)):
        if opened >= trades_remaining:
            break
        row = df.iloc[i]
        z = row["zscore"]
        conf = row["confidence"]
        time = df.index[i]

        if position == 0:
            if z < -2.0 and conf > 0.55:
                position = 1
                entry_idx = i
                trader.open_trade(
                    1, time, z, row["spread"],
                    ticker_a=pair.ticker_a, ticker_b=pair.ticker_b, basket=pair.basket,
                )
            elif z > 2.0 and conf > 0.55:
                position = -1
                entry_idx = i
                trader.open_trade(
                    -1, time, z, row["spread"],
                    ticker_a=pair.ticker_a, ticker_b=pair.ticker_b, basket=pair.basket,
                )
        elif position != 0:
            bars_held = i - entry_idx
            should_exit = False

            if position == 1 and z > -0.4:
                should_exit = True
            if position == -1 and z < 0.4:
                should_exit = True
            if bars_held > 25:
                should_exit = True
            if position == 1 and z < -3.5:
                should_exit = True
            if position == -1 and z > 3.5:
                should_exit = True

            if should_exit:
                trader.close_trade(
                    time, z, row["spread"],
                    ml_proba=np.random.uniform(0.55, 0.85),
                )
                position = 0
                opened += 1
                if opened >= trades_remaining:
                    break

    return opened


def run_paper_trading_and_train(
    n_bars=600,
    min_trades=3,
    baskets: Optional[Sequence[str]] = None,
    include_cross: bool = True,
    max_pairs_per_basket: int = 6,
    data_source_mode: str = "auto",
):
    print("Starting Paper Trading Session...")
    print("Goal: Complete at least", min_trades, "round-trip trades\n")

    # 0. Universe — Mag7, semis, memory, hyperscaler
    baskets = list(baskets) if baskets is not None else list(TICKER_UNIVERSES.keys())
    pairs = build_pair_universe(
        baskets=baskets,
        include_cross=include_cross,
        max_pairs_per_basket=max_pairs_per_basket,
    )
    summarize_universes(pairs)

    # 1. Prices for all tickers in the active universe: real (yfinance) by
    # default, falling back to synthetic if the fetch fails (mode="auto").
    tickers = all_universe_tickers(baskets)
    prices, data_source = _load_prices_for_universe(tickers, n_bars, mode=data_source_mode)
    print(f"\nPrice panel: {prices.shape[1]} tickers × {prices.shape[0]} bars  [source: {data_source}]\n")

    # 2–3. Scan pairs with Kalman filter + paper trader
    trader = PaperTrader()
    kf = SimpleKalmanPairs()
    pair_frames: Dict[str, pd.DataFrame] = {}
    closed_count = 0

    for pair in pairs:
        if closed_count >= min_trades:
            break
        if pair.ticker_a not in prices.columns or pair.ticker_b not in prices.columns:
            continue
        price_a = prices[pair.ticker_a]
        price_b = prices[pair.ticker_b]
        df = kf.filter_pair(price_a, price_b)
        pair_frames[pair.label] = df
        print(f"\n--- Scanning {pair.label} [{pair.basket}] ---")
        # At most one round-trip per pair so we rotate across Mag7 / semis / memory / hyperscaler
        opened = _trade_pair_session(
            trader, df, pair,
            min_trades=min_trades,
            trades_remaining=1,
        )
        closed_count += opened

    # 4. Show journal
    closed_trades = trader.summary()
    # Use last scanned frame for return compatibility; prefer first if empty
    df_out = next(iter(pair_frames.values())) if pair_frames else prices

    if len(closed_trades) < 2:
        print("Not enough trades generated. Try increasing n_bars or relaxing entry thresholds.")
        return trader, None, df_out

    # 5. Persist results, then train (from this run + any prior stored history)
    save_paper_results(closed_trades, data_source=data_source)
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
        "--data-source",
        choices=["auto", "live", "synthetic"],
        default="auto",
        help=(
            "auto (default): fetch real prices via yfinance, fall back to "
            "synthetic on failure. live: require real data, error out if "
            "the fetch fails. synthetic: skip live data entirely."
        ),
    )
    args = parser.parse_args()

    if args.train_only:
        model = train_from_stored_results()
        print("\n🎯 Retrain complete from stored results.")
    else:
        trader, model, data = run_paper_trading_and_train(
            n_bars=args.n_bars,
            min_trades=args.min_trades,
            baskets=["mag7", "semis", "memory", "hyperscaler"],
            include_cross=True,
            data_source_mode=args.data_source,
        )

        if model is None:
            print("\n⚠️ Paper trading finished without enough trades to train.")
        else:
            print("\n🎯 Paper trading session complete.")
            print(f"   Journal:  {TRADES_CSV}")
            print(f"   Dataset:  {DATASET_CSV}")
            print(f"   Model:    {MODEL_JSON}")
            print("Universe covered: Mag7, semis, memory, hyperscaler.")
            print("Next: python paper_trading_ml_exit.py --train-only")
