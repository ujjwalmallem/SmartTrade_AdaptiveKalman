"""Broker safety, journal ownership, level-signal exits and equity logging."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import paper_trading_ml_exit as m
from alpaca_paper_broker import AlpacaPaperBroker
from src.exit_manager import StatArbExitManager
from src.signals import level_rule_exit, level_signal


class NoPosition(Exception):
    status_code = 404


class FakeClient:
    """Minimal alpaca TradingClient stand-in with per-symbol net positions."""

    def __init__(self, positions=None, fail_symbols=(), read_error=None):
        self.positions = dict(positions or {})
        self.fail_symbols = set(fail_symbols)
        self.read_error = read_error
        self.orders = []
        self.closed = []

    def get_open_position(self, symbol):
        if self.read_error is not None:
            raise self.read_error
        qty = self.positions.get(symbol, 0)
        if qty == 0:
            raise NoPosition("position does not exist")
        return SimpleNamespace(qty=str(abs(qty)), side="long" if qty > 0 else "short")

    def submit_order(self, req):
        sym = req.symbol
        if sym in self.fail_symbols:
            raise RuntimeError("wash trade rejected")
        side = str(req.side).lower()
        sign = 1 if "buy" in side else -1
        self.positions[sym] = self.positions.get(sym, 0) + sign * int(req.qty)
        self.orders.append((sym, sign * int(req.qty)))
        return SimpleNamespace(id=f"oid-{len(self.orders)}", status="accepted")

    def get_orders(self, status=None):
        return []

    def close_position(self, symbol):
        self.closed.append(symbol)
        self.positions[symbol] = 0
        return SimpleNamespace(id=f"close-{symbol}", status="accepted")

    def get_account(self):
        return SimpleNamespace(equity="101234.5", last_equity="100000", cash="90000",
                               long_market_value="8000", short_market_value="-7900")

    def get_all_positions(self):
        return [SimpleNamespace(unrealized_pl="12.5"), SimpleNamespace(unrealized_pl="-2.5")]


class TestBrokerSafety(unittest.TestCase):
    def test_second_leg_failure_unwinds_first_leg(self):
        client = FakeClient(fail_symbols={"MSFT"})
        br = AlpacaPaperBroker(client=client)
        with self.assertRaises(RuntimeError):
            br.open_pair("AAPL", "MSFT", "LONG_SPREAD", 8000, 200.0, 400.0)
        self.assertEqual(client.closed, ["AAPL"])
        self.assertEqual(client.positions.get("AAPL", 0), 0)

    def test_first_leg_failure_places_nothing(self):
        client = FakeClient(fail_symbols={"AAPL"})
        br = AlpacaPaperBroker(client=client)
        with self.assertRaises(RuntimeError):
            br.open_pair("AAPL", "MSFT", "LONG_SPREAD", 8000, 200.0, 400.0)
        self.assertEqual(client.orders, [])
        self.assertEqual(client.closed, [])

    def test_no_position_reads_as_flat(self):
        br = AlpacaPaperBroker(client=FakeClient())
        self.assertEqual(br.position_signed_qty("AAPL"), 0.0)

    def test_api_error_is_not_treated_as_flat(self):
        br = AlpacaPaperBroker(client=FakeClient(read_error=RuntimeError("429 rate limited")))
        with self.assertRaises(RuntimeError):
            br.position_signed_qty("AAPL")
        with self.assertRaises(RuntimeError):
            br.open_pair("AAPL", "MSFT", "LONG_SPREAD", 8000, 200.0, 400.0)


def _flat_frame(z_last: float, n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02", periods=n, freq="B")
    z = np.zeros(n)
    z[-1] = z_last
    return pd.DataFrame({
        "zscore": z, "confidence": 1.0, "spread": np.linspace(0, 1, n),
        "spread_velocity": 0.0, "spread_vol": 1.0, "price_a": 200.0, "price_b": 400.0,
    }, index=idx)


def _run_live(trader, df, pair):
    m._trade_pair_session(
        trader, df, pair, min_trades=1, trades_remaining=1, trade_year=2026,
        mode="live", latest_bar=df.index[-1],
        exit_manager=StatArbExitManager(model=None, scaler=None),
    )


class TestLiveOwnership(unittest.TestCase):
    def test_pair_sharing_ticker_with_open_pair_is_skipped(self):
        client = FakeClient(positions={"AMZN": 10, "META": -3})
        br = AlpacaPaperBroker(client=client)
        trader = m.PaperTrader(broker=br, execute_latest_only=True)
        trader.claim_tickers("AMZN/META", "AMZN", "META")
        df = _flat_frame(-3.5)
        trader.latest_bar = df.index[-1]
        _run_live(trader, df, m.PairSpec("AVGO", "META", "semis_hyperscaler"))
        self.assertEqual(trader.trades, [])
        self.assertEqual(client.orders, [])

    def test_journal_adoption_uses_true_entry_and_caps_qty(self):
        # META short also backs another pair: Alpaca shows 5, this pair owns 2.
        client = FakeClient(positions={"AMZN": 14, "META": -5})
        br = AlpacaPaperBroker(client=client)
        trader = m.PaperTrader(broker=br, execute_latest_only=True)
        df = _flat_frame(-1.0)
        trader.latest_bar = df.index[-1]
        trader.journal_open = {"AMZN/META": {
            "ticker_a": "AMZN", "ticker_b": "META", "direction": 1,
            "entry_time": df.index[-6], "entry_z": -3.2, "entry_spread": 0.1,
            "qty_a": 14.0, "qty_b": 2.0, "order_ids": ["a", "b"],
        }}
        _run_live(trader, df, m.PairSpec("AMZN", "META", "hyperscaler"))
        t = trader.trades[0]
        self.assertEqual(pd.Timestamp(t.entry_time), df.index[-6])
        self.assertAlmostEqual(t.entry_z, -3.2)
        self.assertEqual((t.qty_a, t.qty_b), (14.0, 2.0))
        self.assertEqual(t.alpaca_order_ids, ["a", "b"])

    def test_load_open_journal_collapses_repeated_snapshots(self):
        rows = [
            dict(run_id="r1", status="OPEN", broker="alpaca_paper", ticker_a="MU", ticker_b="WDC",
                 direction="SHORT_SPREAD", entry_time="2026-09-24 04:00:00", entry_z=3.1,
                 entry_spread=0.2, qty_a=3.0, qty_b=8.0, alpaca_order_ids=json.dumps(["x", "y"])),
            dict(run_id="r2", status="OPEN", broker="alpaca_paper", ticker_a="MU", ticker_b="WDC",
                 direction="SHORT_SPREAD", entry_time="2026-09-24 04:00:00", entry_z=3.1,
                 entry_spread=0.2, qty_a=3.0, qty_b=8.0, alpaca_order_ids=json.dumps(["x", "y"])),
            dict(run_id="r0", status="CLOSED", broker="alpaca_paper", ticker_a="NVDA", ticker_b="AMD",
                 direction="LONG_SPREAD", entry_time="2026-09-21", entry_z=-3.0,
                 entry_spread=0.0, qty_a=1.0, qty_b=1.0, alpaca_order_ids="[]"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_trades.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            j = m.load_open_journal(path)
        self.assertEqual(list(j), ["MU/WDC"])
        self.assertEqual((j["MU/WDC"]["qty_a"], j["MU/WDC"]["qty_b"]), (3.0, 8.0))
        self.assertEqual(j["MU/WDC"]["direction"], -1)


class TestLevelSignal(unittest.TestCase):
    def test_level_signal_is_causal(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2024-01-01", periods=400)
        a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
        b = pd.Series(80 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
        full = level_signal(a, b, 120)["zscore"]
        cut = level_signal(a.iloc[:300], b.iloc[:300], 120)["zscore"]
        pd.testing.assert_series_equal(full.iloc[:300], cut, check_names=False)

    def test_rule_exit(self):
        self.assertTrue(level_rule_exit(1, -0.4, 3, 20, exit_z=0.5, stop_z=4.0)[0])
        self.assertFalse(level_rule_exit(1, -1.5, 3, 20, exit_z=0.5, stop_z=4.0)[0])
        self.assertTrue(level_rule_exit(-1, 4.2, 3, 20, exit_z=0.5, stop_z=4.0)[0])
        self.assertTrue(level_rule_exit(-1, 2.0, 20, 20, exit_z=0.5, stop_z=4.0)[0])


class TestEquitySnapshot(unittest.TestCase):
    def test_snapshot_appends_account_truth(self):
        br = AlpacaPaperBroker(client=FakeClient())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpaca_equity.csv"
            m.record_account_snapshot(br, path)
            m.record_account_snapshot(br, path)
            df = pd.read_csv(path)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df["day_pnl"].iloc[0], 1234.5)
        self.assertAlmostEqual(df["unrealized_pl"].iloc[0], 10.0)

    def test_snapshot_skips_dry_run(self):
        self.assertIsNone(m.record_account_snapshot(AlpacaPaperBroker(dry_run=True)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
