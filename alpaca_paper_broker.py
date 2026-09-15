"""
Alpaca paper-brokerage adapter for SmartTrade pairs.

Uses the official alpaca-py TradingClient against the paper endpoint.
Credentials (never commit these):
  ALPACA_API_KEY / ALPACA_API_SECRET_KEY
  or APCA_API_KEY_ID / APCA_API_SECRET_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val.strip()
    return None


def alpaca_credentials_present() -> bool:
    key = _env_first("ALPACA_API_KEY", "APCA_API_KEY_ID")
    secret = _env_first("ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY")
    return bool(key and secret)


@dataclass
class BrokerFill:
    symbol: str
    side: str
    qty: float
    order_id: str = ""
    status: str = ""
    notional: float = 0.0


@dataclass
class PairOrderResult:
    direction: str
    fills: List[BrokerFill] = field(default_factory=list)
    raw_orders: List[Any] = field(default_factory=list)

    @property
    def order_ids(self) -> List[str]:
        return [f.order_id for f in self.fills if f.order_id]


class AlpacaPaperBroker:
    """
    Thin wrapper around Alpaca paper trading.

    LONG_SPREAD  → buy ticker_a, sell ticker_b
    SHORT_SPREAD → sell ticker_a, buy ticker_b
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        dry_run: bool = False,
        client: Any = None,
    ):
        self.paper = bool(paper)
        self.dry_run = bool(dry_run)
        self._client = client

        if self._client is None and not self.dry_run:
            key = api_key or _env_first("ALPACA_API_KEY", "APCA_API_KEY_ID")
            secret = secret_key or _env_first("ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY")
            if not key or not secret:
                raise RuntimeError(
                    "Alpaca credentials missing. Set ALPACA_API_KEY and "
                    "ALPACA_API_SECRET_KEY (paper keys from app.alpaca.markets)."
                )
            from alpaca.trading.client import TradingClient

            # Force paper endpoint — never route pair paper trades to live.
            self._client = TradingClient(key, secret, paper=True)

    @property
    def name(self) -> str:
        return "alpaca_paper" if self.paper else "alpaca_live"

    def get_equity(self) -> Optional[float]:
        if self.dry_run:
            return None
        try:
            acct = self._client.get_account()
            return float(acct.equity)
        except Exception as exc:
            print(f"⚠️  Alpaca get_account failed: {exc}")
            return None

    def _qty_for_leg(self, notional_leg: float, price: float) -> float:
        if price <= 0:
            raise ValueError("price must be positive to size Alpaca orders")
        # Whole shares for shorting compatibility; min 1 share.
        return float(max(1, int(notional_leg / price)))

    def _market_order(self, symbol: str, qty: float, side: str):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        if self.dry_run:
            return {
                "id": f"dry-{symbol}-{side}",
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "status": "accepted",
            }
        return self._client.submit_order(req)

    def open_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        direction: str,
        notional: float,
        price_a: float,
        price_b: float,
    ) -> PairOrderResult:
        """
        Submit both legs of a pairs entry to Alpaca paper.
        direction: LONG_SPREAD | SHORT_SPREAD
        Refuses to open if either leg already has an open position.
        """
        exp = self.pair_exposure(ticker_a, ticker_b)
        if not exp["flat"] or exp["blocked"]:
            raise RuntimeError(
                f"Skip entry {ticker_a}/{ticker_b}: existing Alpaca exposure "
                f"(qty_a={exp['qty_a']}, qty_b={exp['qty_b']}, blocked={exp['blocked']})"
            )
        leg = float(notional) / 2.0
        qty_a = self._qty_for_leg(leg, float(price_a))
        qty_b = self._qty_for_leg(leg, float(price_b))

        if direction == "LONG_SPREAD":
            side_a, side_b = "buy", "sell"
        elif direction == "SHORT_SPREAD":
            side_a, side_b = "sell", "buy"
        else:
            raise ValueError(f"Unknown direction: {direction}")

        result = PairOrderResult(direction=direction)
        for symbol, qty, side in (
            (ticker_a, qty_a, side_a),
            (ticker_b, qty_b, side_b),
        ):
            order = self._market_order(symbol, qty, side)
            oid = getattr(order, "id", None) or (order.get("id") if isinstance(order, dict) else "")
            status = getattr(order, "status", None) or (
                order.get("status") if isinstance(order, dict) else ""
            )
            result.fills.append(
                BrokerFill(
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    order_id=str(oid),
                    status=str(status),
                    notional=float(qty) * (float(price_a) if symbol == ticker_a else float(price_b)),
                )
            )
            result.raw_orders.append(order)

        print(
            f"   🏦 Alpaca paper entry [{direction}] "
            f"{side_a} {qty_a:.0f} {ticker_a} / {side_b} {qty_b:.0f} {ticker_b}"
        )
        return result

    def cancel_open_orders(self, *symbols: str) -> int:
        """Cancel open orders for the given symbols (avoids wash-trade blocks)."""
        if self.dry_run or self._client is None:
            return 0
        cancelled = 0
        want = {s.upper() for s in symbols if s}
        try:
            orders = list(self._client.get_orders(status="open") or [])
        except TypeError:
            # Older alpaca-py: GetOrdersRequest
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus

                orders = list(
                    self._client.get_orders(
                        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
                    )
                    or []
                )
            except Exception as exc:
                print(f"⚠️  Alpaca list open orders failed: {exc}")
                return 0
        except Exception as exc:
            print(f"⚠️  Alpaca list open orders failed: {exc}")
            return 0
        for order in orders:
            sym = str(getattr(order, "symbol", "") or "").upper()
            if want and sym not in want:
                continue
            oid = getattr(order, "id", None)
            if not oid:
                continue
            try:
                self._client.cancel_order_by_id(oid)
                cancelled += 1
            except Exception as exc:
                print(f"⚠️  Alpaca cancel {sym} order {oid} failed: {exc}")
        return cancelled

    def close_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        qty_a: float,
        qty_b: float,
        direction: str,
    ) -> PairOrderResult:
        """
        Flatten both legs (reverse of entry). Falls back to close_position when qty unknown.
        Raises RuntimeError if either leg fails (so the journal can stay OPEN).
        """
        # Pending entry orders of the opposite side trigger Alpaca wash-trade rejects.
        n_cancel = self.cancel_open_orders(ticker_a, ticker_b)
        if n_cancel:
            print(f"   🏦 Cancelled {n_cancel} open order(s) before exit")

        if direction == "LONG_SPREAD":
            # entry was buy A / sell B → exit sell A / buy B
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        result = PairOrderResult(direction=direction)
        failures: List[str] = []
        for symbol, qty, side in (
            (ticker_a, qty_a, side_a),
            (ticker_b, qty_b, side_b),
        ):
            try:
                if qty and qty > 0:
                    order = self._market_order(symbol, float(qty), side)
                elif not self.dry_run:
                    order = self._client.close_position(symbol)
                else:
                    order = {"id": f"dry-close-{symbol}", "status": "accepted"}
                oid = getattr(order, "id", None) or (
                    order.get("id") if isinstance(order, dict) else ""
                )
                status = getattr(order, "status", None) or (
                    order.get("status") if isinstance(order, dict) else ""
                )
                result.fills.append(
                    BrokerFill(
                        symbol=symbol,
                        side=side,
                        qty=float(qty or 0),
                        order_id=str(oid),
                        status=str(status),
                    )
                )
                result.raw_orders.append(order)
            except Exception as exc:
                msg = f"{symbol}: {exc}"
                print(f"⚠️  Alpaca close {msg}")
                failures.append(msg)

        if failures:
            raise RuntimeError(
                f"Alpaca exit incomplete for {ticker_a}/{ticker_b}: " + "; ".join(failures)
            )

        print(f"   🏦 Alpaca paper exit flattened {ticker_a}/{ticker_b}")
        return result

    def list_open_positions(self) -> Sequence[Any]:
        if self.dry_run:
            return []
        return list(self._client.get_all_positions())

    def position_signed_qty(self, symbol: str) -> float:
        """Positive = long, negative = short, 0 = flat."""
        if self.dry_run:
            return 0.0
        try:
            pos = self._client.get_open_position(symbol)
            qty = float(getattr(pos, "qty", 0) or 0)
            side = str(getattr(pos, "side", "")).lower()
            if side == "short" or qty < 0:
                return -abs(qty)
            return abs(qty)
        except Exception:
            return 0.0

    def pair_exposure(self, ticker_a: str, ticker_b: str) -> dict:
        """
        Infer whether a pairs position is already open on Alpaca.

        Returns keys:
          flat | direction (1/-1) | qty_a | qty_b | blocked (ambiguous legs)
        """
        qa = self.position_signed_qty(ticker_a)
        qb = self.position_signed_qty(ticker_b)
        if qa == 0.0 and qb == 0.0:
            return {"flat": True, "direction": 0, "qty_a": 0.0, "qty_b": 0.0, "blocked": False}
        # LONG_SPREAD: +A / -B ; SHORT_SPREAD: -A / +B
        if qa > 0 and qb < 0:
            return {"flat": False, "direction": 1, "qty_a": abs(qa), "qty_b": abs(qb), "blocked": False}
        if qa < 0 and qb > 0:
            return {"flat": False, "direction": -1, "qty_a": abs(qa), "qty_b": abs(qb), "blocked": False}
        return {
            "flat": False,
            "direction": 0,
            "qty_a": abs(qa),
            "qty_b": abs(qb),
            "blocked": True,
        }

    def legs_are_busy(self, ticker_a: str, ticker_b: str) -> bool:
        exp = self.pair_exposure(ticker_a, ticker_b)
        return (not exp["flat"]) or exp["blocked"]


def _period_to_start(period: str) -> datetime:
    """Map yfinance-style period strings to a UTC start timestamp."""
    now = datetime.now(timezone.utc)
    mapping = {
        "1y": 365,
        "2y": 730,
        "5y": 365 * 5,
        "6mo": 182,
        "3mo": 90,
        "1mo": 31,
    }
    days = mapping.get(str(period).lower(), 730)
    return now - timedelta(days=days)


def fetch_daily_ohlcv(
    tickers: Sequence[str],
    period: str = "2y",
    min_bars: int = 80,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Batch-fetch daily OHLCV from Alpaca market data.

    Returns {ticker: DataFrame[Open, High, Low, Close, Volume]} (tz-naive index).
    Uses IEX feed (free / paper-friendly). Raises on hard failures.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise RuntimeError("No tickers requested for Alpaca OHLCV")

    key = api_key or _env_first("ALPACA_API_KEY", "APCA_API_KEY_ID")
    secret = secret_key or _env_first("ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials missing for market data")

    client = StockHistoricalDataClient(key, secret)
    start = _period_to_start(period)
    end = datetime.now(timezone.utc)

    print(
        f"Fetching Alpaca OHLCV for {len(tickers)} tickers "
        f"(feed=IEX, timeframe=1Day, period={period})..."
    )
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req)
    raw = bars.df
    if raw is None or raw.empty:
        raise RuntimeError("Alpaca returned no bar data")

    panels: Dict[str, pd.DataFrame] = {}

    # alpaca-py returns MultiIndex (symbol, timestamp) or single-symbol flat index
    if isinstance(raw.index, pd.MultiIndex):
        symbols = raw.index.get_level_values(0).unique().tolist()
        for sym in symbols:
            try:
                frame = raw.xs(sym, level=0).copy()
            except Exception:
                continue
            panels[sym] = _normalize_alpaca_frame(frame, min_bars=min_bars)
    else:
        # Single symbol response
        sym = tickers[0]
        panels[sym] = _normalize_alpaca_frame(raw.copy(), min_bars=min_bars)

    panels = {k: v for k, v in panels.items() if v is not None and not v.empty}
    missing = [t for t in tickers if t not in panels]
    if missing:
        print(f"⚠️  Dropping tickers with no usable Alpaca history: {', '.join(missing)}")
    if len(panels) < 2:
        raise RuntimeError(
            "Alpaca returned usable OHLCV for fewer than 2 tickers; "
            "cannot build pairs for training"
        )
    return panels


def _normalize_alpaca_frame(frame: pd.DataFrame, min_bars: int = 80) -> Optional[pd.DataFrame]:
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    cols = {c: rename[c] for c in frame.columns if c in rename}
    if "Close" not in cols.values() and "close" not in frame.columns:
        return None
    out = frame.rename(columns=cols)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    # tz-naive dates for alignment with the rest of the pipeline
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if "Close" not in out.columns or out["Close"].dropna().shape[0] < min_bars:
        return None
    return out
