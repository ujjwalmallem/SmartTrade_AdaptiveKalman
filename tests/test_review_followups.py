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
from src.exit_manager import StatArbExitManager, TradeState, time_stop_bars


class TestConfigSSOT(unittest.TestCase):
    def test_threshold_and_min_samples_from_yaml(self):
        cfg = load_strategy_config()
        self.assertAlmostEqual(exit_threshold(cfg), 0.68)
        self.assertGreaterEqual(training_min_samples(cfg), 50)
        self.assertEqual(cfg["exit_model"]["backend"], "sklearn")

    def test_entry_and_kalman_from_yaml(self):
        from src.config import (
            entry_z_threshold,
            entry_min_confidence,
            execution_risk_frac,
            kalman_settings,
        )
        cfg = load_strategy_config()
        self.assertAlmostEqual(entry_z_threshold(cfg), 1.75)
        self.assertAlmostEqual(entry_min_confidence(cfg), 0.45)
        self.assertAlmostEqual(execution_risk_frac(cfg), 0.08)
        k = kalman_settings(cfg)
        self.assertAlmostEqual(k["delta"], 1e-4)
        self.assertAlmostEqual(k["R_base"], 1e-2)

    def test_cli_imports_src_kalman(self):
        from src import kalman as kmod
        self.assertIs(m.AdaptiveKalmanPairs, kmod.AdaptiveKalmanPairs)
        self.assertIs(m.KalmanNoiseModel, kmod.KalmanNoiseModel)
        kf = m.build_kalman()
        self.assertIsInstance(kf, kmod.AdaptiveKalmanPairs)

    def test_entry_direction_respects_thresholds(self):
        self.assertEqual(m.entry_direction(-2.1, 0.6, z_entry=2.0, min_confidence=0.55), 1)
        self.assertEqual(m.entry_direction(2.1, 0.6, z_entry=2.0, min_confidence=0.55), -1)
        self.assertEqual(m.entry_direction(-2.1, 0.4, z_entry=2.0, min_confidence=0.55), 0)
        self.assertEqual(m.entry_direction(-1.5, 0.9, z_entry=2.0, min_confidence=0.55), 0)


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


class TestNaNSafeEvaluate(unittest.TestCase):
    def test_non_finite_features_hold(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 8))
        y = (X[:, 1] > 0).astype(int)
        model = LogisticRegression(max_iter=200).fit(X, y)
        scaler = StandardScaler().fit(X)
        mgr = StatArbExitManager(
            model=model, scaler=scaler, exit_threshold=0.01,
        )
        state = TradeState(
            trade_id=1, ticker_a="QCOM", ticker_b="AVGO",
            direction="SHORT_SPREAD", bars_held=2, half_life_bars=20.0,
            vol=float("nan"), pnl_proxy=0.1, entry_z=2.0, confidence=0.7,
            exit_z=1.0, velocity=0.0, half_life=20.0 / 30.0,
        )
        should, reason, prob = mgr.evaluate_trade(state)
        self.assertFalse(should)
        self.assertIn("non-finite", reason.lower())
        self.assertEqual(prob, 0.0)


class TestFailClosedExit(unittest.TestCase):
    def test_should_exit_raises_without_manager(self):
        with self.assertRaises(ValueError):
            m.should_exit_with_ml(
                position=1, z=-0.2, bars_held=5,
                features=np.zeros(len(m.FEATURE_NAMES)), model=None,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
