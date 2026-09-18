"""Additional coverage requested by code review: Kalman, journal, broker dry-run."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import paper_trading_ml_exit as m
from alpaca_paper_broker import AlpacaPaperBroker
from src.config import exit_threshold, load_strategy_config, training_min_samples
from src.exit_manager import TradeState, time_stop_bars


class TestConfigSSOT(unittest.TestCase):
    def test_threshold_and_min_samples_from_yaml(self):
        cfg = load_strategy_config()
        self.assertAlmostEqual(exit_threshold(cfg), 0.68)
        self.assertGreaterEqual(training_min_samples(cfg), 50)
        self.assertEqual(cfg["exit_model"]["backend"], "sklearn")


class TestHalfLifeFields(unittest.TestCase):
    def test_trade_state_keeps_raw_and_normalized_distinct(self):
        pos = m.PositionState(direction=1, entry_z=-2.0, entry_bar=0, entry_spread=1.0)
        row = pd.Series({
            "zscore": -0.5, "confidence": 0.7,
            "spread_velocity": 0.0, "spread_vol": 1.0,
        })
        pair = m.PairSpec(ticker_a="AAPL", ticker_b="MSFT", basket="mag7")
        state = m.build_trade_state(
            pair=pair, direction=1, pos_state=pos, row=row,
            bars_held=10, half_life=20.0,
        )
        self.assertEqual(state.half_life_bars, 20.0)
        self.assertAlmostEqual(state.half_life, 20.0 / 30.0)
        self.assertEqual(time_stop_bars(state.half_life_bars), 50)


class TestKalmanSynthetic(unittest.TestCase):
    def test_mean_reverting_pair_produces_finite_z(self):
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        # Cointegrated: B ≈ 2*A + noise
        a = 100 + np.cumsum(rng.normal(0, 0.5, n))
        b = 2.0 * a + rng.normal(0, 1.0, n)
        kf = m.AdaptiveKalmanPairs(noise_model=m.KalmanNoiseModel.STANDARD)
        df = kf.filter_pair(pd.Series(a, index=idx), pd.Series(b, index=idx))
        self.assertIn("zscore", df.columns)
        self.assertTrue(np.isfinite(df["zscore"].iloc[60:]).all())
        self.assertTrue(np.isfinite(df["beta"].iloc[60:]).all())
        # β should stabilize to a non-trivial hedge ratio
        self.assertGreater(abs(float(df["beta"].iloc[-20:].mean())), 0.2)


class TestJournalDedupe(unittest.TestCase):
    def test_wash_and_open_dedupe(self):
        df = pd.DataFrame([
            {
                "run_id": "a", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "basket": "semis", "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15", "exit_time": "2026-09-15",
                "broker": "alpaca_paper", "status": "CLOSED", "pnl_z": 0.0,
            },
            {
                "run_id": "b", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "basket": "semis", "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
            {
                "run_id": "c", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "basket": "semis", "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
        ])
        out = m.dedupe_journal_rows(df, drop_wash=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["status"], "OPEN")


class TestBrokerDryRun(unittest.TestCase):
    def test_dry_run_open_blocked_when_exposed(self):
        br = AlpacaPaperBroker(paper=True, dry_run=True)
        br.pair_exposure = lambda a, b: {
            "flat": False, "direction": -1, "qty_a": 21.0, "qty_b": 11.0, "blocked": False,
        }
        with self.assertRaises(RuntimeError):
            br.open_pair(
                "QCOM", "AVGO", direction="SHORT_SPREAD",
                notional=8000.0, price_a=180.0, price_b=340.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
