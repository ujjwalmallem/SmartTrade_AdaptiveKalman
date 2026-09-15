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
from typing import Any, Dict, List, Optional, Sequence


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
        """
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
        """
        if direction == "LONG_SPREAD":
            # entry was buy A / sell B → exit sell A / buy B
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        result = PairOrderResult(direction=direction)
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
                print(f"⚠️  Alpaca close {symbol} failed: {exc}")

        print(f"   🏦 Alpaca paper exit flattened {ticker_a}/{ticker_b}")
        return result

    def list_open_positions(self) -> Sequence[Any]:
        if self.dry_run:
            return []
        return list(self._client.get_all_positions())
