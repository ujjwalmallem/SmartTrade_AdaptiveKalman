"""
Share-level pairs backtest with real dollar P&L.

Answers what the live journal cannot: does a given signal/sizing config make
money after costs, out of sample?

Mechanics (no look-ahead):
  - signal computed on the close of day t, filled at the OPEN of day t+1
  - integer share legs (min 1), same gross notional per trade as live
  - costs per side on traded notional + borrow on the short leg
  - positions marked to market daily for Sharpe / drawdown

Usage:
  python research/backtest_pairs.py                # yfinance data
  python research/backtest_pairs.py --synthetic    # offline smoke test
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.half_life import estimate_half_life  # noqa: E402
from src.kalman import AdaptiveKalmanPairs  # noqa: E402

CAPITAL = 100_000.0
GROSS_PER_TRADE = 8_000.0          # live: risk_frac 0.08 × $100k, split across legs
COST_BPS_PER_SIDE = 5.0            # commission-free, but spread + slippage on market orders
BORROW_BPS_PER_YEAR = 50.0         # general-collateral borrow on the short leg
STOP_Z = 4.0
LEVEL_WINDOW = 120
COINT_WINDOW = 250
COINT_T_CRIT = -3.34               # Engle-Granger 5% critical value, 2 variables
OOS_START = "2024-10-01"


@dataclass(frozen=True)
class Config:
    signal: str        # "innov" (live Kalman innovation z) | "level" (log-price spread z)
    entry_z: float
    exit_z: float
    min_conf: float
    hedge: str         # "model" (hedge ratio) | "dollar" (equal $ legs, live behaviour)
    coint: bool        # require Engle-Granger cointegration on the prior 250 days
    window: int = LEVEL_WINDOW   # level-signal lookback
    delay: int = 1     # fill at open of day t+delay (live 5-bar lookback ≈ delay up to 5)
    exclusive: bool = False      # at most one open pair per ticker (Alpaca nets per symbol)

    @property
    def name(self) -> str:
        return (f"{self.signal}{self.window if self.signal == 'level' else ''}|in{self.entry_z}"
                f"|out{self.exit_z}|conf{self.min_conf}|{self.hedge}|coint{'Y' if self.coint else 'N'}"
                f"|d{self.delay}|{'excl' if self.exclusive else 'shared'}")


# --------------------------------------------------------------------------- data

def load_yfinance(tickers: List[str], period: str = "6y") -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers=tickers, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t][["Open", "Close"]].dropna()
        except KeyError:
            continue
        if len(df) > 300:
            out[t] = df
    missing = sorted(set(tickers) - set(out))
    if missing:
        print(f"⚠️  no usable history: {missing}")
    return out


def load_synthetic(tickers: List[str], n: int = 1500, seed: int = 7) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-25", periods=n)
    market = np.cumsum(rng.normal(0.0003, 0.011, n))
    out = {}
    resid = np.zeros(n)
    for k, t in enumerate(tickers):
        for i in range(1, n):
            resid[i] = 0.93 * resid[i - 1] + rng.normal(0, 0.006)
        logp = np.log(50 + 25 * (k % 9)) + market * (0.8 + 0.05 * (k % 5)) + resid
        close = np.exp(logp)
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        out[t] = pd.DataFrame({"Open": open_, "Close": close}, index=idx)
    return out


# ------------------------------------------------------------------------ signals

def innov_signal(ca: pd.Series, cb: pd.Series) -> pd.DataFrame:
    """Live signal: z of Kalman innovation. Confidence recomputed causally
    (live uses a full-history median, which leaks future volatility)."""
    df = AdaptiveKalmanPairs(delta=1e-4, R_base=1e-2).filter_pair(ca, cb)
    roll = df["spread"].rolling(20, min_periods=5).std()
    med = roll.expanding(min_periods=5).median()
    df["confidence"] = (med / (med + roll)).clip(0.25, 0.95).fillna(0.25)
    # price-level hedge: 1 share A vs beta shares B
    df["share_ratio"] = df["beta"]
    return df[["zscore", "confidence", "spread", "share_ratio"]]


def level_signal(ca: pd.Series, cb: pd.Series, window: int = LEVEL_WINDOW) -> pd.DataFrame:
    """Classic pairs signal: rolling OLS of log prices, z of the residual."""
    la, lb = np.log(ca), np.log(cb)
    mb = lb.rolling(window).mean()
    ma = la.rolling(window).mean()
    cov = (la * lb).rolling(window).mean() - ma * mb
    var = (lb * lb).rolling(window).mean() - mb * mb
    beta = cov / var
    alpha = ma - beta * mb
    resid = la - alpha - beta * lb
    sd = resid.rolling(window).std()
    z = resid / sd
    # log-price beta is a dollar hedge ratio → shares_b/shares_a = beta * pA / pB
    share_ratio = beta * ca / cb
    return pd.DataFrame({"zscore": z, "confidence": 1.0, "spread": resid,
                         "share_ratio": share_ratio}, index=ca.index)


def eg_tstat(la: np.ndarray, lb: np.ndarray) -> Tuple[float, float]:
    """Engle-Granger step 2 ADF t-stat (no lags) and hedge slope."""
    X = np.column_stack([np.ones(len(lb)), lb])
    coef, *_ = np.linalg.lstsq(X, la, rcond=None)
    e = la - X @ coef
    de, e1 = np.diff(e), e[:-1]
    X2 = np.column_stack([np.ones(len(e1)), e1])
    c2, *_ = np.linalg.lstsq(X2, de, rcond=None)
    r = de - X2 @ c2
    s2 = (r @ r) / (len(de) - 2)
    cov = s2 * np.linalg.inv(X2.T @ X2)
    return float(c2[1] / math.sqrt(cov[1, 1])), float(coef[1])


# ----------------------------------------------------------------------- simulate

@dataclass
class Trade:
    pair: str
    direction: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares_a: int
    shares_b: int
    pnl: float
    cost: float
    bars: int
    reason: str


class PairBook:
    """Per-pair arrays on the portfolio calendar plus open-position state."""

    def __init__(self, a: str, b: str, prices: Dict[str, pd.DataFrame], sig: pd.DataFrame,
                 cal: pd.DatetimeIndex):
        self.a, self.b, self.label = a, b, f"{a}/{b}"
        r = lambda s: s.reindex(cal).to_numpy(dtype=float)  # noqa: E731
        self.oa, self.ob = r(prices[a]["Open"]), r(prices[b]["Open"])
        self.ca, self.cb = r(prices[a]["Close"]), r(prices[b]["Close"])
        self.z, self.conf = r(sig["zscore"]), r(sig["confidence"])
        self.ratio = r(sig["share_ratio"])
        self.spread = sig["spread"].reindex(cal)
        self.la, self.lb = np.log(self.ca), np.log(self.cb)
        self.first = int(np.argmax(np.isfinite(self.ca) & np.isfinite(self.cb)))
        self.coint_cache: Dict[int, bool] = {}
        self.reset()

    def reset(self) -> None:
        self.pos = self.sa = self.sb = 0
        self.entry_i = self.max_bars = 0
        self.entry_val = self.entry_cost = 0.0
        self.pending: Optional[Tuple[str, int, int]] = None   # (kind, direction, execute_at)
        self.reason = ""

    def valid(self, i: int) -> bool:
        return all(np.isfinite(x[i]) for x in (self.oa, self.ob, self.ca, self.cb))

    def cointegrated(self, i: int) -> bool:
        ok = self.coint_cache.get(i)
        if ok is None:
            lo = i - COINT_WINDOW + 1
            if lo < self.first:
                ok = False
            else:
                t, b = eg_tstat(self.la[lo: i + 1], self.lb[lo: i + 1])
                ok = t < COINT_T_CRIT and b > 0
            self.coint_cache[i] = ok
        return ok


def simulate_portfolio(books: List[PairBook], cfg: Config, cal: pd.DatetimeIndex
                       ) -> Tuple[List[Trade], pd.Series]:
    """Day loop over all pairs, in live scan order, so shared tickers interact."""
    for bk in books:
        bk.reset()
    daily = np.zeros(len(cal))
    trades: List[Trade] = []
    busy: Dict[str, str] = {}          # ticker -> pair label holding it (exclusive mode)
    warm = max(cfg.window if cfg.signal == "level" else 60, 60) + 1

    for i in range(1, len(cal)):
        for bk in books:
            if not bk.valid(i) or i < bk.first + warm:
                continue
            # 1) execute scheduled orders at today's open
            if bk.pending is not None and bk.pending[2] <= i:
                kind, d, _ = bk.pending
                bk.pending = None
                if kind == "open":
                    r = bk.ratio[i - 1]
                    if cfg.hedge == "dollar":
                        qa = max(1, int(GROSS_PER_TRADE / 2 / bk.oa[i]))
                        qb = max(1, int(GROSS_PER_TRADE / 2 / bk.ob[i]))
                    else:
                        qa = max(1, int(GROSS_PER_TRADE / (bk.oa[i] + r * bk.ob[i])))
                        qb = max(1, int(round(r * qa)))
                    bk.sa, bk.sb = d * qa, -d * qb
                    bk.entry_val = bk.sa * bk.oa[i] + bk.sb * bk.ob[i]
                    bk.entry_cost = (abs(bk.sa) * bk.oa[i] + abs(bk.sb) * bk.ob[i]) * COST_BPS_PER_SIDE / 1e4
                    daily[i] += bk.sa * (bk.ca[i] - bk.oa[i]) + bk.sb * (bk.cb[i] - bk.ob[i]) - bk.entry_cost
                    bk.pos, bk.entry_i = d, i
                    hl = estimate_half_life(bk.spread.iloc[: i].dropna(), lookback=80)
                    bk.max_bars = max(5, math.ceil(2.5 * hl))
                    continue
                exit_cost = (abs(bk.sa) * bk.oa[i] + abs(bk.sb) * bk.ob[i]) * COST_BPS_PER_SIDE / 1e4
                short_notional = abs(bk.sa) * bk.oa[i] if bk.sa < 0 else abs(bk.sb) * bk.ob[i]
                borrow = short_notional * BORROW_BPS_PER_YEAR / 1e4 * (i - bk.entry_i) / 252
                daily[i] += (bk.sa * (bk.oa[i] - bk.ca[i - 1]) + bk.sb * (bk.ob[i] - bk.cb[i - 1])
                             - exit_cost - borrow)
                exit_val = bk.sa * bk.oa[i] + bk.sb * bk.ob[i]
                trades.append(Trade(bk.label, bk.pos, cal[bk.entry_i], cal[i], bk.sa, bk.sb,
                                    exit_val - bk.entry_val - bk.entry_cost - exit_cost - borrow,
                                    bk.entry_cost + exit_cost + borrow, i - bk.entry_i, bk.reason))
                bk.pos = bk.sa = bk.sb = 0
                for t in (bk.a, bk.b):
                    if busy.get(t) == bk.label:
                        del busy[t]

            # 2) mark open position
            if bk.pos != 0:
                daily[i] += bk.sa * (bk.ca[i] - bk.ca[i - 1]) + bk.sb * (bk.cb[i] - bk.cb[i - 1])

            z = bk.z[i]
            if i == len(cal) - 1 or not np.isfinite(z) or bk.pending is not None:
                continue

            # 3) decide at today's close
            if bk.pos != 0:
                held = i - bk.entry_i
                if (bk.pos == 1 and z >= -cfg.exit_z) or (bk.pos == -1 and z <= cfg.exit_z):
                    bk.pending, bk.reason = ("close", bk.pos, i + 1), "revert"
                elif abs(z) >= STOP_Z:
                    bk.pending, bk.reason = ("close", bk.pos, i + 1), "stop"
                elif held >= bk.max_bars:
                    bk.pending, bk.reason = ("close", bk.pos, i + 1), "time"
                continue
            if abs(z) < cfg.entry_z or bk.conf[i] < cfg.min_conf:
                continue
            if not np.isfinite(bk.ratio[i]) or bk.ratio[i] <= 0:
                continue
            if cfg.exclusive and (bk.a in busy or bk.b in busy):
                continue
            if cfg.coint and not bk.cointegrated(i):
                continue
            bk.pending = ("open", 1 if z < 0 else -1, i + cfg.delay)
            if cfg.exclusive:
                busy[bk.a] = busy[bk.b] = bk.label

    return trades, pd.Series(daily, index=cal)


# ------------------------------------------------------------------------ metrics

def summarize(daily: pd.Series, trades: List[Trade]) -> Dict[str, float]:
    ret = daily / CAPITAL
    sharpe = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0
    eq = CAPITAL + daily.cumsum()
    dd = float((eq - eq.cummax()).min())
    pnls = [t.pnl for t in trades]
    return {
        "pnl": round(float(daily.sum()), 0),
        "sharpe": round(sharpe, 2),
        "max_dd": round(dd, 0),
        "trades": len(trades),
        "win_rate": round(float(np.mean([p > 0 for p in pnls])), 3) if pnls else 0.0,
        "avg_trade": round(float(np.mean(pnls)), 1) if pnls else 0.0,
        "avg_bars": round(float(np.mean([t.bars for t in trades])), 1) if trades else 0.0,
    }


LIVE_LIKE = Config("innov", 1.5, 0.5, 0.40, "dollar", False, delay=1, exclusive=False)


def build_grid() -> List[Config]:
    grid = [LIVE_LIKE,
            Config("innov", 1.5, 0.5, 0.40, "dollar", False, delay=3, exclusive=False),
            Config("innov", 2.5, 0.5, 0.0, "dollar", False, delay=1, exclusive=True)]
    for w, ez, xz, excl, dl in itertools.product([60, 120, 250], [2.0, 2.5, 3.0], [0.0, 0.5],
                                                 [False, True], [1, 3]):
        grid.append(Config("level", ez, xz, 0.0, "dollar", False, window=w, delay=dl, exclusive=excl))
    return grid


def run(prices: Dict[str, pd.DataFrame], pairs: List[Tuple[str, str]], out_dir: Path) -> None:
    grid = build_grid()
    pairs = [(a, b) for a, b in pairs if a in prices and b in prices]
    cal = pd.DatetimeIndex(sorted(set().union(*[prices[t].index for t in {x for p in pairs for x in p}])))
    signals: Dict[Tuple[str, int], List[PairBook]] = {}

    def books_for(cfg: Config) -> List[PairBook]:
        key = (cfg.signal, cfg.window if cfg.signal == "level" else 0)
        if key not in signals:
            bks = []
            for a, b in pairs:
                common = prices[a].index.intersection(prices[b].index)
                ca, cb = prices[a]["Close"].loc[common], prices[b]["Close"].loc[common]
                sig = innov_signal(ca, cb) if cfg.signal == "innov" else level_signal(ca, cb, cfg.window)
                bks.append(PairBook(a, b, prices, sig, cal))
            signals[key] = bks
        return signals[key]

    print(f"pairs with data: {len(pairs)}  configs: {len(grid)}  calendar: {cal[0].date()} → {cal[-1].date()}")
    oos = pd.Timestamp(OOS_START)
    rows, pair_rows, yearly_rows = [], [], []
    for cfg in grid:
        trades, daily = simulate_portfolio(books_for(cfg), cfg, cal)
        is_m = summarize(daily[daily.index < oos], [t for t in trades if t.entry_date < oos])
        oo_m = summarize(daily[daily.index >= oos], [t for t in trades if t.entry_date >= oos])
        all_m = summarize(daily, trades)
        rows.append({"config": cfg.name, **asdict(cfg),
                     **{f"is_{k}": v for k, v in is_m.items()},
                     **{f"oos_{k}": v for k, v in oo_m.items()},
                     **{f"all_{k}": v for k, v in all_m.items()}})
        by_year = daily.groupby(daily.index.year).sum().round(0)
        yearly_rows.append({"config": cfg.name, **{str(y): v for y, v in by_year.items()}})
        for lbl in sorted({t.pair for t in trades}):
            pt = [t for t in trades if t.pair == lbl]
            pair_rows.append({"config": cfg.name, "pair": lbl,
                              "is_pnl": round(sum(t.pnl for t in pt if t.entry_date < oos), 0),
                              "oos_pnl": round(sum(t.pnl for t in pt if t.entry_date >= oos), 0),
                              "trades": len(pt)})

    res = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False)
    yearly = pd.DataFrame(yearly_rows).set_index("config")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "grid_results.csv", index=False)
    yearly.to_csv(out_dir / "yearly_pnl.csv")
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_results.csv", index=False)

    cols = ["config", "is_pnl", "is_sharpe", "is_trades", "oos_pnl", "oos_sharpe", "oos_trades",
            "oos_win_rate", "oos_avg_trade", "all_pnl", "all_sharpe", "all_max_dd"]
    level = res[res["signal"] == "level"]
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_rows", 200):
        print("\n=== INNOVATION-SIGNAL REFERENCE ROWS (live-like first) ===")
        print(res[res["signal"] == "innov"][cols].to_string(index=False))
        print(f"\n=== LEVEL SIGNAL, ALL CONFIGS BY IN-SAMPLE SHARPE (IS < {OOS_START} <= OOS) ===")
        print(level[cols].to_string(index=False))
        print("\n=== ROBUSTNESS: fraction of level configs profitable ===")
        for part in ("is", "oos", "all"):
            print(f"  {part}: {(level[f'{part}_pnl'] > 0).mean():.0%} of {len(level)} configs")
        print("\n=== LEVER EFFECTS (median all-period P&L across level configs) ===")
        for lever in ("window", "entry_z", "exit_z", "exclusive", "delay"):
            print(f"  {lever}: " + ", ".join(f"{k}→{v:,.0f}" for k, v in
                                             level.groupby(lever)["all_pnl"].median().items()))
        top = list(level["config"].head(5)) + [LIVE_LIKE.name]
        print("\n=== YEARLY P&L: top-5 level by IS Sharpe + live-like ===")
        print(yearly.loc[top].to_string())
    summary = {"oos_start": OOS_START, "rows": res.to_dict(orient="records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="research_out")
    args = ap.parse_args()

    import paper_trading_ml_exit as live

    specs = live.build_pair_universe(max_pairs_per_basket=6)
    pairs = [(p.ticker_a, p.ticker_b) for p in specs]
    tickers = sorted({t for pr in pairs for t in pr})
    prices = load_synthetic(tickers) if args.synthetic else load_yfinance(tickers)
    if prices:
        first = min(df.index.min() for df in prices.values())
        last = max(df.index.max() for df in prices.values())
        print(f"data: {len(prices)} tickers, {first.date()} → {last.date()}")
    run(prices, pairs, Path(args.out))


if __name__ == "__main__":
    main()
