"""Smoke / unit tests for AdaptiveKalmanPairs + exit features + latest-year trading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import paper_trading_ml_exit as m


class TestAdaptiveKalmanPairs(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 180
        idx = pd.date_range("2026-01-02", periods=n, freq="B")
        beta = 1.2
        b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
        resid = np.zeros(n)
        for i in range(1, n):
            resid[i] = 0.9 * resid[i - 1] + rng.normal(0, 0.5)
        a = 2.0 + beta * b + resid
        self.a = pd.Series(a, index=idx, name="A")
        self.b = pd.Series(b, index=idx, name="B")

    def test_filter_pair_columns_and_finite(self):
        kf = m.AdaptiveKalmanPairs(delta=1e-4, R=1e-2)
        df = kf.filter_pair(self.a, self.b)
        required = {
            "price_a", "price_b", "alpha", "beta", "spread", "zscore",
            "confidence", "spread_velocity", "spread_vol", "innovation",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), len(self.a))
        self.assertTrue(np.isfinite(df["zscore"]).all())
        self.assertTrue(np.isfinite(df["confidence"]).all())
        self.assertTrue((df["confidence"] >= kf.min_conf).all())
        self.assertTrue((df["confidence"] <= kf.max_conf).all())
        # Hedge ratio should be in a sensible neighborhood of true β≈1.2
        self.assertGreater(df["beta"].iloc[-1], 0.5)
        self.assertLess(df["beta"].iloc[-1], 2.0)

    def test_reset_clears_state_between_pairs(self):
        kf = m.AdaptiveKalmanPairs()
        d1 = kf.filter_pair(self.a, self.b)
        hist_len = len(kf.history)
        d2 = kf.filter_pair(self.a, self.b)
        self.assertEqual(len(d1), len(d2))
        self.assertEqual(len(kf.history), hist_len)  # reset each run


class TestExitFeatures(unittest.TestCase):
    def test_extract_exit_features_shape_and_order(self):
        pos = m.PositionState(direction=1, entry_z=-2.0, entry_bar=10, entry_spread=1.0)
        row = pd.Series({
            "zscore": -0.5,
            "confidence": 0.7,
            "spread_velocity": 0.1,
            "spread_vol": 1.5,
        })
        feat = m.extract_exit_features(pos, row, bars_held=12, direction=1)
        self.assertEqual(feat.shape, (len(m.FEATURE_NAMES),))
        self.assertEqual(feat[0], -2.0)          # entry_z
        self.assertEqual(feat[1], 2.0)           # abs_entry_z
        self.assertAlmostEqual(feat[2], 1.5)     # pnl_proxy = -0.5 - (-2.0)
        self.assertAlmostEqual(feat[3], 12 / 30)
        self.assertEqual(feat[4], 0.7)
        self.assertEqual(feat[6], -0.5)          # current z
        self.assertGreaterEqual(pos.highest_favorable_z, 1.5)

    def test_trade_to_features_matches_feature_names(self):
        trade = m.PaperTrade(
            trade_id=1,
            direction="LONG_SPREAD",
            entry_time=pd.Timestamp("2026-04-01"),
            entry_z=-2.2,
            entry_spread=1.0,
            exit_time=pd.Timestamp("2026-04-10"),
            exit_z=-0.3,
            bars_held=9,
            pnl_z=1.9,
            status="CLOSED",
        )
        feat = m.trade_to_features(trade)
        self.assertEqual(len(feat), len(m.FEATURE_NAMES))
        self.assertTrue(np.isfinite(feat).all())


class TestLatestYearFilter(unittest.TestCase):
    def test_filter_drops_2025_keeps_2026(self):
        df = pd.DataFrame({
            "entry_time": ["2025-06-01", "2026-03-01", "2026-05-01"],
            "exit_time": ["2025-06-10", "2026-03-08", "2026-05-12"],
            "trade_id": [1, 2, 3],
        })
        kept = m.filter_trades_to_latest_year(df, trade_year=2026)
        self.assertEqual(len(kept), 2)
        self.assertListEqual(kept["trade_id"].tolist(), [2, 3])


class TestEndToEndYfinance(unittest.TestCase):
    def test_paper_session_latest_year_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Point artifact paths at temp dir for this process
            m.RESULTS_DIR = tmp_path
            m.TRADES_CSV = tmp_path / "paper_trades.csv"
            m.DATASET_CSV = tmp_path / "exit_training_dataset.csv"
            m.MODEL_JSON = tmp_path / "logistic_exit_model.json"

            trader, model, _ = m.run_paper_trading_and_train(
                n_bars=400,
                min_trades=2,
                baskets=["mag7", "semis", "memory", "hyperscaler"],
                include_cross=True,
            )
            self.assertIsNotNone(trader)
            closed = [t for t in trader.trades if t.status == "CLOSED"]
            self.assertGreaterEqual(len(closed), 2)
            for t in closed:
                self.assertEqual(pd.Timestamp(t.entry_time).year, 2026)
                self.assertEqual(pd.Timestamp(t.exit_time).year, 2026)
            self.assertTrue((tmp_path / "paper_trades.csv").exists())
            journal = pd.read_csv(tmp_path / "paper_trades.csv")
            self.assertTrue((journal["data_source"] == "yfinance_live").all())
            years = pd.to_datetime(journal["entry_time"], format="mixed").dt.year.unique().tolist()
            self.assertEqual(years, [2026])
            self.assertIsNotNone(model)
            self.assertIsNotNone(model.weights)
            self.assertEqual(len(model.weights), len(m.FEATURE_NAMES))
            self.assertTrue((tmp_path / "logistic_exit_model.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
