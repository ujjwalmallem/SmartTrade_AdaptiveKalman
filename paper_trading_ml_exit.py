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

def fetch_real_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int = 700,
    period: str = "2y",
    min_bars: int = 80,
    trade_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Real daily close prices via yfinance only (no synthetic data).

    Aligns tickers on shared trading days. Keeps enough history for indicator
    warm-up, but trade windows are restricted to `trade_year` (default: the
    calendar year of the latest bar — e.g. 2026, not prior years).
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
    print(f"Fetching yfinance daily closes for {len(tickers)} tickers (period={period})...")
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
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) >= min_bars:
            closes[t] = series.rename(t)

    missing = [t for t in tickers if t not in closes]
    if missing:
        print(f"⚠️  Dropping tickers with no usable yfinance history: {', '.join(missing)}")
    if len(closes) < 2:
        raise RuntimeError(
            "yfinance returned usable closes for fewer than 2 tickers; "
            "cannot build pairs for training"
        )

    prices = pd.DataFrame(closes).dropna(how="any")
    if prices.empty:
        raise RuntimeError("No overlapping trading days across fetched tickers")

    # Prefer a recent window that still leaves warm-up bars before trade_year.
    prices = prices.tail(max(n_bars, min_bars + 60))

    latest_year = int(pd.Timestamp(prices.index.max()).year)
    year = int(trade_year) if trade_year is not None else latest_year
    if year != latest_year:
        print(f"⚠️  Requested trade_year={year} but latest bar is {latest_year}; using {latest_year}")
        year = latest_year

    in_year = prices.index.year == year
    if not in_year.any():
        raise RuntimeError(f"No yfinance bars in latest year {year}")

    # Keep prior-year warm-up (for rolling z/confidence) + all latest-year bars.
    first_trade_i = int(in_year.argmax())  # first True
    warm_start = max(0, first_trade_i - 60)
    prices = prices.iloc[warm_start:]

    trade_bars = int((prices.index.year == year).sum())
    if trade_bars < 40:
        raise RuntimeError(
            f"Only {trade_bars} bars in trade year {year}; need more latest-year history"
        )

    print(
        f"yfinance panel ready: {prices.shape[1]} tickers × {prices.shape[0]} bars "
        f"[{prices.index.min().date()} → {prices.index.max().date()}] "
        f"(trades restricted to {year}: {trade_bars} bars)"
    )
    prices.attrs["trade_year"] = year
    return prices


def _load_prices_for_universe(
    tickers: Sequence[str],
    n_bars: int,
    trade_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load prices for training. yfinance only — never synthesizes data.
    Trade windows use the latest calendar year only.
    Returns (prices, source_label) for the results journal.
    """
    prices = fetch_real_prices_for_universe(
        tickers, n_bars=n_bars, trade_year=trade_year
    )
    return prices, "yfinance_live"



# ============================================================
# ADAPTIVE KALMAN FILTER FOR PAIRS (real implementation)
# ============================================================

@dataclass
class AdaptiveKalmanPairs:
    """
    Online Kalman filter estimating time-varying intercept (α) and hedge ratio (β).
    State: θ = [α, β]
    Observation: price_a = α + β * price_b + v

    Adaptive process noise: Q is scaled by recent innovation magnitude so the
    filter becomes more responsive in volatile regimes and smoother in calm ones.
    """
    delta: float = 1e-4          # base process-noise scale
    R: float = 1e-2              # observation noise variance
    adapt_window: int = 20      # window for innovation-based adaptation
    min_conf: float = 0.25
    max_conf: float = 0.95

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.x = np.zeros(2)                     # [α, β]
        self.P = np.eye(2) * 1.0                 # state covariance
        self.Q_base = self.delta * np.eye(2)
        self.innovations: List[float] = []
        self.history: List[dict] = []
        self._ewma_var = None

    def _adapt_Q(self) -> np.ndarray:
        """Scale process noise by recent |innovation| (adaptive Kalman)."""
        if len(self.innovations) < 5:
            return self.Q_base.copy()
        recent = np.array(self.innovations[-self.adapt_window:])
        scale = np.clip(np.std(recent) / (np.mean(np.abs(recent)) + 1e-8), 0.3, 5.0)
        return self.Q_base * scale

    def update(self, price_a: float, price_b: float) -> dict:
        """One-step predict + update. Returns current state diagnostics."""
        # Observation matrix H = [1, price_b]
        H = np.array([1.0, price_b])

        # ----- Predict -----
        x_prior = self.x.copy()
        Q = self._adapt_Q()
        P_prior = self.P + Q

        # ----- Update -----
        y_pred = H @ x_prior
        innov = price_a - y_pred
        S = float(H @ P_prior @ H.T + self.R)   # innovation variance
        K = (P_prior @ H.T) / S                 # Kalman gain

        self.x = x_prior + K * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ P_prior

        self.innovations.append(float(innov))
        if len(self.innovations) > 200:
            self.innovations = self.innovations[-200:]

        spread = innov                          # residual = observed − predicted
        # Online estimate of residual std (EWMA)
        if self._ewma_var is None:
            self._ewma_var = max(S, 1e-6)
        else:
            self._ewma_var = 0.94 * self._ewma_var + 0.06 * innov**2
        spread_std = float(np.sqrt(self._ewma_var + 1e-8))

        z = spread / spread_std
        # Confidence: higher when innovation variance is low relative to long-run
        conf = 1.0 / (1.0 + np.sqrt(S))
        conf = float(np.clip(conf, self.min_conf, self.max_conf))

        out = {
            "alpha": float(self.x[0]),
            "beta": float(self.x[1]),
            "spread": float(spread),
            "spread_std": float(spread_std),
            "zscore": float(z),
            "confidence": conf,
            "innovation": float(innov),
            "kalman_gain_beta": float(K[1]),
        }
        self.history.append(out)
        return out

    def filter_pair(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """
        Run the filter over two aligned price series.
        Returns a DataFrame with all diagnostics needed by the trading engine.
        """
        self.reset()
        a = a.astype(float).dropna()
        b = b.astype(float).dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]

        # Calibrate noise to absolute price scale (R=1e-2 is for unit prices).
        price_scale = float(max(np.nanmedian(np.abs(a.values)), 1.0))
        self.R = max((price_scale ** 2) * 1e-4, 1e-6)
        self.Q_base = (self.delta * (price_scale ** 2)) * np.eye(2)

        rows = []
        prev_spread = 0.0

        for ts, pa, pb in zip(common, a.values, b.values):
            st = self.update(float(pa), float(pb))
            spread = st["spread"]
            velocity = spread - prev_spread
            prev_spread = spread

            rows.append({
                "price_a": pa,
                "price_b": pb,
                "alpha": st["alpha"],
                "beta": st["beta"],
                "spread": spread,
                "zscore": st["zscore"],
                "confidence": st["confidence"],
                "spread_velocity": velocity,
                "spread_vol": st["spread_std"],
                "innovation": st["innovation"],
            })

        df = pd.DataFrame(rows, index=common)

        # Extra rolling diagnostics useful for exit features
        df["spread_velocity"] = df["spread"].diff().rolling(5, min_periods=1).mean().fillna(0)
        df["spread_vol"] = df["spread"].rolling(15, min_periods=5).std().fillna(df["spread_vol"])
        # Trading z-score: rolling standardize the Kalman residual (online EWMA z
        # adapts too fast to ever reach classic ±2 entry thresholds).
        roll_mu = df["spread"].rolling(40, min_periods=10).mean()
        roll_sd = df["spread"].rolling(40, min_periods=10).std()
        df["zscore"] = ((df["spread"] - roll_mu) / roll_sd.replace(0, np.nan)).fillna(0.0)
        # Regime confidence from relative residual vol (tradeable scale)
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

def extract_exit_features(
    position: PositionState,
    row: pd.Series,
    bars_held: int,
    direction: int,
) -> np.ndarray:
    """
    Build the 10-dimensional feature vector used by LogisticExitModel.
    All values are computed from live Kalman state + path statistics.
    Order matches FEATURE_NAMES.
    """
    z = float(row["zscore"])
    conf = float(row.get("confidence", 0.5))
    vel = float(row.get("spread_velocity", 0.0))
    vol = float(row.get("spread_vol", 1.0))

    # Current PnL in z-space (positive = favorable)
    if direction == 1:          # long the spread
        pnl_proxy = z - position.entry_z
        favorable = max(0.0, z - position.entry_z)
    else:                       # short the spread
        pnl_proxy = position.entry_z - z
        favorable = max(0.0, position.entry_z - z)

    # Track best favorable excursion
    position.highest_favorable_z = max(position.highest_favorable_z, favorable)
    best_fav = position.highest_favorable_z

    # Normalized bars held (0–1-ish scale)
    bars_norm = bars_held / 30.0

    feat = np.array([
        position.entry_z,           # 0 entry_z
        abs(position.entry_z),      # 1 abs_entry_z
        pnl_proxy,                  # 2 pnl_proxy
        bars_norm,                  # 3 bars_held
        conf,                       # 4 confidence
        vel,                        # 5 velocity
        z,                          # 6 exit_z (current)
        favorable,                  # 7 favorable (current)
        best_fav,                   # 8 best_fav
        vol,                        # 9 vol
    ], dtype=float)
    return feat


def trade_to_features(t: "PaperTrade") -> np.ndarray:
    """
    Reconstruct a reasonable feature vector from a closed PaperTrade
    (used when training from the journal). Some path-dependent fields
    are approximated because the full intra-trade series is not stored.
    """
    pnl = t.pnl_z
    bars_norm = t.bars_held / 30.0
    exit_z = t.exit_z if t.exit_z is not None else 0.0
    fav = max(0.0, pnl)                 # realized favorable excursion proxy
    best_fav = max(abs(t.entry_z), fav) # crude upper bound

    return np.array([
        t.entry_z,
        abs(t.entry_z),
        pnl,
        bars_norm,
        0.70,               # confidence placeholder (could be stored later)
        0.0,                # velocity unknown from journal
        exit_z,
        fav,
        best_fav,
        1.0,                # vol placeholder
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
    Prior-year windows (e.g. 2025) are purged so only the latest year remains.
    Returns (trades_csv_path, dataset_csv_path).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    year = _latest_allowed_trade_year()

    frame = closed_trades_to_frame(closed_trades, run_id=run_id, data_source=data_source)
    trades_path = results_dir / "paper_trades.csv"
    dataset_path = results_dir / "exit_training_dataset.csv"

    # Full journal (append, then keep latest year only)
    if trades_path.exists():
        prev = pd.read_csv(trades_path)
        journal = pd.concat([prev, frame], ignore_index=True)
    else:
        journal = frame
    journal = filter_trades_to_latest_year(journal, trade_year=year)
    journal.to_csv(trades_path, index=False)

    # Training dataset = feature columns + label (append, then align to latest-year journal)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    ds = frame[["run_id", "data_source", "trade_id", "ticker_a", "ticker_b", "basket", *feat_cols, "label"]]
    if dataset_path.exists():
        prev_ds = pd.read_csv(dataset_path)
        ds = pd.concat([prev_ds, ds], ignore_index=True)
    if not journal.empty and {"run_id", "trade_id"}.issubset(ds.columns):
        keys = journal[["run_id", "trade_id"]].drop_duplicates()
        ds = ds.merge(keys, on=["run_id", "trade_id"], how="inner")
    elif journal.empty:
        ds = ds.iloc[0:0]
    ds.to_csv(dataset_path, index=False)

    print(f"\n💾 Saved {len(closed_trades)} new trades → {trades_path}")
    print(f"💾 Journal rows kept for {year}: {len(journal)}")
    print(f"💾 Training dataset rows: {len(ds)} → {dataset_path}")
    return trades_path, dataset_path


def _latest_allowed_trade_year(now: Optional[pd.Timestamp] = None) -> int:
    """Calendar year used for training/trade windows (never prior years)."""
    return int(pd.Timestamp(now or pd.Timestamp.utcnow()).year)


def filter_trades_to_latest_year(
    trades: pd.DataFrame,
    trade_year: Optional[int] = None,
) -> pd.DataFrame:
    """Keep only rows whose entry and exit fall in the latest calendar year."""
    if trades.empty:
        return trades
    year = trade_year if trade_year is not None else _latest_allowed_trade_year()
    entry = pd.to_datetime(trades["entry_time"], format="mixed")
    exit_ = pd.to_datetime(trades["exit_time"], format="mixed")
    mask = entry.dt.year.eq(year) & exit_.dt.year.eq(year)
    kept = trades.loc[mask].copy()
    dropped = len(trades) - len(kept)
    if dropped:
        print(f"⚠️  Dropped {dropped} trades outside latest year {year}")
    return kept


def load_training_dataset(
    dataset_path: Path = DATASET_CSV,
    require_live: bool = True,
    latest_year_only: bool = True,
    trades_path: Path = TRADES_CSV,
    trade_year: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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

    # Training must use real prices only — drop any legacy synthetic rows.
    if require_live and "data_source" in ds.columns:
        before = len(ds)
        ds = ds[ds["data_source"].astype(str).str.startswith("yfinance")].copy()
        dropped = before - len(ds)
        if dropped:
            print(f"⚠️  Dropped {dropped} non-yfinance rows from training dataset")
        if ds.empty:
            raise ValueError(
                "No yfinance rows left in the training dataset. "
                "Re-run paper trading to collect real-price samples."
            )

    # Drop any stored windows from prior years (e.g. 2025) — latest year only.
    if latest_year_only:
        year = trade_year if trade_year is not None else _latest_allowed_trade_year()
        tpath = Path(trades_path)
        if tpath.exists() and {"run_id", "trade_id"}.issubset(ds.columns):
            journal = pd.read_csv(tpath)
            journal = filter_trades_to_latest_year(journal, trade_year=year)
            keys = journal[["run_id", "trade_id"]].drop_duplicates()
            before = len(ds)
            ds = ds.merge(keys, on=["run_id", "trade_id"], how="inner")
            dropped = before - len(ds)
            if dropped:
                print(f"⚠️  Dropped {dropped} training rows not in latest year {year}")
        elif "entry_time" in ds.columns and "exit_time" in ds.columns:
            before = len(ds)
            ds = filter_trades_to_latest_year(ds, trade_year=year)
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
    dataset_path: Path = DATASET_CSV,
    model_path: Path = MODEL_JSON,
    reg: float = 0.3,
    min_samples: int = 2,
) -> LogisticExitModel:
    """Fit the exit model on stored yfinance-backed paper trades only."""
    X, y, ds = load_training_dataset(dataset_path, require_live=True)
    if len(y) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples; found {len(y)} in {dataset_path}")

    print(f"\nTraining from stored results: {len(y)} yfinance samples ({dataset_path})")
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
    trade_year: Optional[int] = None,
) -> int:
    """Run rule-based entries/exits on one pair; return number of new closed trades.

    Entries are allowed only in `trade_year` (latest calendar year). Prior-year
    bars may exist for indicator warm-up but never open a 2025 (or older) window.
    While in a position, richer Kalman/path exit features are computed each bar
    (ready for ML-driven exits).
    """
    position = 0
    entry_idx = 0
    opened = 0
    pos_state = PositionState()
    if trade_year is None:
        trade_year = int(pd.Timestamp(df.index.max()).year)

    for i in range(60, len(df)):
        if opened >= trades_remaining:
            break
        row = df.iloc[i]
        z = row["zscore"]
        conf = row["confidence"]
        time = df.index[i]
        in_trade_year = int(pd.Timestamp(time).year) == int(trade_year)

        if position == 0:
            if not in_trade_year:
                continue
            if z < -2.0 and conf > 0.55:
                position = 1
                entry_idx = i
                pos_state = PositionState(
                    direction=1,
                    entry_z=float(z),
                    entry_bar=i,
                    entry_spread=float(row["spread"]),
                    highest_favorable_z=0.0,
                )
                trader.open_trade(
                    1, time, z, row["spread"],
                    ticker_a=pair.ticker_a, ticker_b=pair.ticker_b, basket=pair.basket,
                )
            elif z > 2.0 and conf > 0.55:
                position = -1
                entry_idx = i
                pos_state = PositionState(
                    direction=-1,
                    entry_z=float(z),
                    entry_bar=i,
                    entry_spread=float(row["spread"]),
                    highest_favorable_z=0.0,
                )
                trader.open_trade(
                    -1, time, z, row["spread"],
                    ticker_a=pair.ticker_a, ticker_b=pair.ticker_b, basket=pair.basket,
                )
        elif position != 0:
            bars_held = i - pos_state.entry_bar
            pos_state.bars_held = bars_held
            features = extract_exit_features(pos_state, row, bars_held, position)
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
                # Soft exit score from live features (pnl_proxy + confidence);
                # swap for LogisticExitModel.predict_proba when a trained model
                # is passed into the session.
                ml_proba = float(
                    1.0 / (1.0 + np.exp(-(0.8 * features[2] + 0.5 * (features[4] - 0.5))))
                )
                ml_proba = float(np.clip(ml_proba, 0.05, 0.95))
                trader.close_trade(
                    time, z, row["spread"],
                    ml_proba=ml_proba,
                )
                position = 0
                pos_state = PositionState()
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

    # 1. Prices: yfinance only; trades restricted to latest calendar year
    tickers = all_universe_tickers(baskets)
    prices, data_source = _load_prices_for_universe(tickers, n_bars)
    trade_year = int(prices.attrs.get("trade_year", pd.Timestamp(prices.index.max()).year))
    print(f"\nPrice panel: {prices.shape[1]} tickers × {prices.shape[0]} bars  [source: {data_source}]")
    print(f"Trade windows: {trade_year} only (no prior-year entries)\n")

    # 2–3. Adaptive Kalman filter + paper trader
    trader = PaperTrader()
    kf = AdaptiveKalmanPairs(delta=1e-4, R=1e-2)
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
        print(f"\n--- Scanning {pair.label} [{pair.basket}] (year={trade_year}) ---")
        # At most one round-trip per pair so we rotate across Mag7 / semis / memory / hyperscaler
        opened = _trade_pair_session(
            trader, df, pair,
            min_trades=min_trades,
            trades_remaining=1,
            trade_year=trade_year,
        )
        closed_count += opened

    # 4. Show journal (latest-year trades only)
    closed_trades = trader.summary()
    closed_trades = [
        t for t in closed_trades
        if int(pd.Timestamp(t.entry_time).year) == trade_year
        and int(pd.Timestamp(t.exit_time).year) == trade_year
    ]
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
