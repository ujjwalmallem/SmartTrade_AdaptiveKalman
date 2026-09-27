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

    @property
    def name(self) -> str:
        return (f"{self.signal}|in{self.entry_z}|out{self.exit_z}|conf{self.min_conf}"
                f"|{self.hedge}|coint{'Y' if self.coint else 'N'}")


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


def simulate_pair(pair: str, px_a: pd.DataFrame, px_b: pd.DataFrame, sig: pd.DataFrame,
                  cfg: Config, coint_cache: Dict[int, bool]) -> Tuple[List[Trade], pd.Series]:
    idx = sig.index
    oa, ob = px_a["Open"].reindex(idx).to_numpy(), px_b["Open"].reindex(idx).to_numpy()
    ca, cb = px_a["Close"].reindex(idx).to_numpy(), px_b["Close"].reindex(idx).to_numpy()
    z = sig["zscore"].to_numpy()
    conf = sig["confidence"].to_numpy()
    ratio = sig["share_ratio"].to_numpy()
    spread = sig["spread"]
    la, lb = np.log(ca), np.log(cb)

    daily = np.zeros(len(idx))
    trades: List[Trade] = []
    pos = 0
    sa = sb = 0
    entry_i = 0
    max_bars = 0
    pending: Optional[Tuple[str, int]] = None   # ("open"/"close", direction)
    reason = ""
    entry_cost = 0.0
    entry_val = 0.0
    warm = max(LEVEL_WINDOW, 60) + 1

    for i in range(warm, len(idx)):
        # 1) execute yesterday's decision at today's open
        if pending is not None:
            kind, d = pending
            pending = None
            if kind == "open":
                r = ratio[i - 1]
                if cfg.hedge == "dollar":
                    leg = GROSS_PER_TRADE / 2
                    qa = max(1, int(leg / oa[i]))
                    qb = max(1, int(leg / ob[i]))
                else:
                    # gross = qa*pA + r*qa*pB
                    qa = max(1, int(GROSS_PER_TRADE / (oa[i] + r * ob[i])))
                    qb = max(1, int(round(r * qa)))
                sa, sb = d * qa, -d * qb
                entry_val = sa * oa[i] + sb * ob[i]
                entry_cost = (abs(sa) * oa[i] + abs(sb) * ob[i]) * COST_BPS_PER_SIDE / 1e4
                daily[i] += sa * (ca[i] - oa[i]) + sb * (cb[i] - ob[i]) - entry_cost
                pos, entry_i = d, i
                hl = estimate_half_life(spread.iloc[: i], lookback=80)
                max_bars = max(5, math.ceil(2.5 * hl))
                continue
            else:  # close
                exit_cost = (abs(sa) * oa[i] + abs(sb) * ob[i]) * COST_BPS_PER_SIDE / 1e4
                short_notional = abs(sa) * oa[i] if sa < 0 else abs(sb) * ob[i]
                borrow = short_notional * BORROW_BPS_PER_YEAR / 1e4 * (i - entry_i) / 252
                daily[i] += sa * (oa[i] - ca[i - 1]) + sb * (ob[i] - cb[i - 1]) - exit_cost - borrow
                exit_val = sa * oa[i] + sb * ob[i]
                trades.append(Trade(pair, pos, idx[entry_i], idx[i], sa, sb,
                                    exit_val - entry_val - entry_cost - exit_cost - borrow,
                                    entry_cost + exit_cost + borrow, i - entry_i, reason))
                pos, sa, sb = 0, 0, 0

        # 2) mark open position
        if pos != 0:
            daily[i] += sa * (ca[i] - ca[i - 1]) + sb * (cb[i] - cb[i - 1])

        if i == len(idx) - 1 or not np.isfinite(z[i]):
            continue

        # 3) decide at today's close
        if pos != 0:
            held = i - entry_i
            if (pos == 1 and z[i] >= -cfg.exit_z) or (pos == -1 and z[i] <= cfg.exit_z):
                pending, reason = ("close", pos), "revert"
            elif abs(z[i]) >= STOP_Z:
                pending, reason = ("close", pos), "stop"
            elif held >= max_bars:
                pending, reason = ("close", pos), "time"
        else:
            if abs(z[i]) < cfg.entry_z or conf[i] < cfg.min_conf:
                continue
            if not np.isfinite(ratio[i]) or ratio[i] <= 0:
                continue
            if cfg.coint:
                ok = coint_cache.get(i)
                if ok is None:
                    t, b = eg_tstat(la[i - COINT_WINDOW + 1: i + 1], lb[i - COINT_WINDOW + 1: i + 1]) \
                        if i >= COINT_WINDOW else (0.0, 0.0)
                    ok = t < COINT_T_CRIT and b > 0
                    coint_cache[i] = ok
                if not ok:
                    continue
            pending = ("open", 1 if z[i] < 0 else -1)

    return trades, pd.Series(daily, index=idx)


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


def run(prices: Dict[str, pd.DataFrame], pairs: List[Tuple[str, str]], out_dir: Path) -> None:
    grid: List[Config] = []
    for ez, xz, hedge, co in itertools.product([1.5, 2.0, 2.5], [0.0, 0.5], ["model", "dollar"], [False, True]):
        for mc in (0.0, 0.40):
            grid.append(Config("innov", ez, xz, mc, hedge, co))
        grid.append(Config("level", ez, xz, 0.0, hedge, co))

    signals = {}
    for a, b in pairs:
        if a not in prices or b not in prices:
            continue
        common = prices[a].index.intersection(prices[b].index)
        ca, cb = prices[a]["Close"].loc[common], prices[b]["Close"].loc[common]
        signals[(a, b, "innov")] = innov_signal(ca, cb)
        signals[(a, b, "level")] = level_signal(ca, cb)
    used_pairs = sorted({(a, b) for a, b, _ in signals})
    print(f"pairs with data: {len(used_pairs)}  configs: {len(grid)}")

    oos = pd.Timestamp(OOS_START)
    rows, pair_rows = [], []
    coint_caches: Dict[Tuple[str, str], Dict[int, bool]] = {}
    for cfg in grid:
        all_daily, all_trades = [], []
        for a, b in used_pairs:
            sig = signals[(a, b, cfg.signal)]
            cache = coint_caches.setdefault((a, b, cfg.signal), {})
            trades, daily = simulate_pair(f"{a}/{b}", prices[a], prices[b], sig, cfg, cache)
            all_daily.append(daily)
            all_trades.extend(trades)
            if cfg.signal in ("innov", "level"):
                is_t = [t for t in trades if t.entry_date < oos]
                oo_t = [t for t in trades if t.entry_date >= oos]
                pair_rows.append({"config": cfg.name, "pair": f"{a}/{b}",
                                  "is_pnl": round(sum(t.pnl for t in is_t), 0), "is_trades": len(is_t),
                                  "oos_pnl": round(sum(t.pnl for t in oo_t), 0), "oos_trades": len(oo_t)})
        daily = pd.concat(all_daily, axis=1).fillna(0.0).sum(axis=1).sort_index()
        is_d, oo_d = daily[daily.index < oos], daily[daily.index >= oos]
        is_m = summarize(is_d, [t for t in all_trades if t.entry_date < oos])
        oo_m = summarize(oo_d, [t for t in all_trades if t.entry_date >= oos])
        rows.append({"config": cfg.name, **asdict(cfg),
                     **{f"is_{k}": v for k, v in is_m.items()},
                     **{f"oos_{k}": v for k, v in oo_m.items()}})

    res = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "grid_results.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_results.csv", index=False)

    cols = ["config", "is_pnl", "is_sharpe", "is_trades", "is_win_rate",
            "oos_pnl", "oos_sharpe", "oos_trades", "oos_win_rate", "oos_avg_trade", "oos_max_dd"]
    live_like = res[res["config"] == Config("innov", 1.5, 0.5, 0.40, "dollar", False).name]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== LIVE-LIKE CONFIG (innov z, entry 1.5, conf 0.40, equal-dollar legs, no coint gate) ===")
        print(live_like[cols].to_string(index=False))
        print(f"\n=== TOP 15 BY IN-SAMPLE SHARPE (IS < {OOS_START} <= OOS) ===")
        print(res[cols].head(15).to_string(index=False))
        print("\n=== TOP 10 BY OOS SHARPE (for reference only — do NOT select on this) ===")
        print(res.sort_values("oos_sharpe", ascending=False)[cols].head(10).to_string(index=False))
        best = res.iloc[0]["config"]
        pr = pd.DataFrame(pair_rows)
        pr = pr[pr["config"] == best].sort_values("is_pnl", ascending=False)
        print(f"\n=== PER-PAIR, best IS config: {best} ===")
        print(pr.to_string(index=False))
    summary = {"oos_start": OOS_START, "best_is_config": res.iloc[0].to_dict(),
               "live_like": live_like.iloc[0].to_dict() if len(live_like) else None}
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
