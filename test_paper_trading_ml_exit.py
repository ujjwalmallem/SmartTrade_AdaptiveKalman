"""Smoke / unit tests for AdaptiveKalmanPairs + exit features + latest-year trading."""

from __future__ import annotations

import json
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
            "confidence", "spread_velocity", "spread_vol", "innovation", "R",
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

    def test_volume_mode_adapts_R(self):
        rng = np.random.default_rng(1)
        vol = pd.Series(
            rng.integers(5e5, 8e6, size=len(self.a)).astype(float),
            index=self.a.index,
        )
        kf = m.AdaptiveKalmanPairs(noise_model=m.KalmanNoiseModel.VOLUME)
        df = kf.filter_pair(self.a, self.b, volume=vol)
        self.assertIn("R", df.columns)
        self.assertTrue(np.isfinite(df["R"]).all())
        self.assertGreater(df["R"].std(), 0.0)

    def test_parkinson_mode_adapts_R(self):
        rng = np.random.default_rng(2)
        noise = rng.uniform(0.2, 2.5, size=len(self.a))
        high = self.a + noise
        low = self.a - noise
        kf = m.AdaptiveKalmanPairs(noise_model=m.KalmanNoiseModel.PARKINSON)
        df = kf.filter_pair(self.a, self.b, high=high, low=low)
        self.assertIn("R", df.columns)
        self.assertTrue(np.isfinite(df["R"]).all())
        self.assertGreater(df["R"].max(), df["R"].min())

    def test_noise_model_string_accepted(self):
        kf = m.AdaptiveKalmanPairs(noise_model="standard")
        self.assertEqual(kf.noise_model, m.KalmanNoiseModel.STANDARD)


class TestExitFeatures(unittest.TestCase):
    def test_extract_exit_features_shape_and_order(self):
        pos = m.PositionState(direction=1, entry_z=-2.0, entry_bar=10, entry_spread=1.0)
        row = pd.Series({
            "zscore": -0.5,
            "confidence": 0.7,
            "spread_velocity": 0.1,
            "spread_vol": 1.5,
        })
        feat = m.extract_exit_features(pos, row, bars_held=12, direction=1, half_life=15.0)
        self.assertEqual(feat.shape, (len(m.FEATURE_NAMES),))
        self.assertEqual(feat[0], -2.0)          # entry_z
        self.assertEqual(feat[1], 2.0)           # abs_entry_z
        self.assertAlmostEqual(feat[2], 1.5)     # pnl_proxy = -0.5 - (-2.0)
        self.assertAlmostEqual(feat[3], 12 / 30)
        self.assertEqual(feat[4], 0.7)          # raw Kalman confidence
        self.assertEqual(feat[6], -0.5)          # current z
        self.assertAlmostEqual(feat[10], 15.0 / 30.0)  # half_life
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

    def test_trade_to_features_prefers_stored_exit_features(self):
        live = np.arange(len(m.FEATURE_NAMES), dtype=float) + 0.25
        trade = m.PaperTrade(
            trade_id=2,
            direction="SHORT_SPREAD",
            entry_time=pd.Timestamp("2026-04-01"),
            entry_z=2.1,
            entry_spread=1.0,
            exit_time=pd.Timestamp("2026-04-05"),
            exit_z=0.4,
            bars_held=4,
            pnl_z=1.7,
            status="CLOSED",
            exit_features=live,
        )
        feat = m.trade_to_features(trade)
        np.testing.assert_allclose(feat, live)

    def test_close_trade_stores_exit_features_and_frame_json(self):
        trader = m.PaperTrader(capital=100_000, cost_bps=4.0, risk_frac=0.08)
        trader.open_trade(1, pd.Timestamp("2026-05-01"), -2.0, 1.0, "AAPL", "MSFT", "mag7")
        self.assertAlmostEqual(trader.trades[0].notional, 8000.0)
        live = np.linspace(0.1, 1.0, len(m.FEATURE_NAMES))
        trader.close_trade(
            pd.Timestamp("2026-05-05"), -0.4, 0.8,
            ml_proba=0.77, features=live, bars_held=4,
        )
        t = trader.trades[0]
        self.assertIsNotNone(t.exit_features)
        np.testing.assert_allclose(t.exit_features, live)
        self.assertEqual(t.bars_held, 4)
        self.assertAlmostEqual(t.pnl_z, 1.6)
        self.assertGreater(t.pnl_dollars, 0.0)
        self.assertAlmostEqual(t.cost_dollars, 8000.0 * 4.0 / 10000.0)
        frame = m.closed_trades_to_frame([t], run_id="test", data_source="yfinance_live")
        self.assertIn("exit_features_json", frame.columns)
        self.assertIn("pnl_dollars", frame.columns)
        self.assertIn("notional", frame.columns)
        parsed = json.loads(frame.loc[0, "exit_features_json"])
        np.testing.assert_allclose(parsed, live)
        for i, name in enumerate(m.FEATURE_NAMES):
            self.assertAlmostEqual(frame.loc[0, f"feat_{name}"], live[i])


class TestShouldExitWithML(unittest.TestCase):
    def _features(self, pnl=0.5, conf=0.7):
        # Aligns with FEATURE_NAMES length
        feat = np.zeros(len(m.FEATURE_NAMES))
        feat[2] = pnl
        feat[4] = conf
        return feat

    def test_rule_exit_without_model(self):
        should, proba = m.should_exit_with_ml(
            position=1, z=-0.2, bars_held=5,
            features=self._features(), model=None,
        )
        self.assertTrue(should)
        self.assertIsNone(proba)

    def test_hard_time_stop(self):
        should, _ = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=28,
            features=self._features(), model=None,
        )
        self.assertTrue(should)

    def test_ml_force_exit(self):
        model = m.LogisticExitModel()
        # Craft weights so predict_proba is high for any finite feature vector
        model.weights = np.zeros(len(m.FEATURE_NAMES))
        model.bias = 3.0  # sigmoid(3) ≈ 0.95
        should, proba = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=3,
            features=self._features(), model=model, ml_threshold=0.62,
        )
        self.assertTrue(should)
        self.assertIsNotNone(proba)
        self.assertGreaterEqual(proba, 0.62)

    def test_ml_suppresses_soft_rule_exit(self):
        model = m.LogisticExitModel()
        model.weights = np.zeros(len(m.FEATURE_NAMES))
        model.bias = -3.0  # sigmoid(-3) ≈ 0.05
        should, proba = m.should_exit_with_ml(
            position=1, z=-0.2, bars_held=5,  # would be soft rule exit
            features=self._features(), model=model, ml_threshold=0.62,
        )
        self.assertFalse(should)
        self.assertLess(proba, 0.38)



class TestTradeLabels(unittest.TestCase):
    def _trade(self, pnl_z, bars_held):
        return m.PaperTrade(
            trade_id=1,
            direction="LONG_SPREAD",
            entry_time=pd.Timestamp("2026-04-01"),
            entry_z=-2.0,
            entry_spread=1.0,
            exit_time=pd.Timestamp("2026-04-10"),
            exit_z=-0.5,
            bars_held=bars_held,
            pnl_z=pnl_z,
            status="CLOSED",
        )

    def test_strong_profit_is_good(self):
        self.assertEqual(m.trade_to_label(self._trade(0.5, 3)), 1)

    def test_small_profit_reasonable_hold(self):
        self.assertEqual(m.trade_to_label(self._trade(0.1, 12)), 1)

    def test_defensive_time_stop(self):
        self.assertEqual(m.trade_to_label(self._trade(-0.4, 26)), 1)

    def test_loss_is_bad(self):
        self.assertEqual(m.trade_to_label(self._trade(-0.8, 5)), 0)


class TestHalfLife(unittest.TestCase):
    def test_mean_reverting_series_has_finite_half_life(self):
        rng = np.random.default_rng(0)
        n = 120
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, 0.3)
        hl = m.estimate_half_life(pd.Series(x), lookback=40)
        self.assertGreaterEqual(hl, 2.0)
        self.assertLessEqual(hl, 60.0)

    def test_trending_series_returns_long_half_life(self):
        x = pd.Series(np.linspace(0, 10, 80) + np.random.default_rng(1).normal(0, 0.01, 80))
        hl = m.estimate_half_life(x, lookback=40)
        self.assertGreaterEqual(hl, 30.0)


class TestOHLCVLoader(unittest.TestCase):
    def test_fetch_returns_per_ticker_ohlcv(self):
        panels = m.fetch_real_prices_for_universe(["AAPL", "MSFT", "NVDA"], n_bars=400)
        self.assertGreaterEqual(len(panels), 2)
        for ticker, df in panels.items():
            self.assertIn("Close", df.columns)
            self.assertTrue({"High", "Low", "Volume"}.issubset(df.columns))
            self.assertGreater(len(df), 40)
            self.assertIn("trade_year", df.attrs)
            self.assertEqual(int(df.attrs["trade_year"]), 2026)
        # All tickers share the same index
        idxs = [tuple(df.index) for df in panels.values()]
        self.assertTrue(all(ix == idxs[0] for ix in idxs))

    def test_field_panels_helper(self):
        panels = m.fetch_real_prices_for_universe(["AAPL", "MSFT"], n_bars=400)
        fields = m.ohlcv_field_panels(panels)
        self.assertEqual(set(fields), {"close", "high", "low", "volume"})
        self.assertTrue(fields["close"].columns.equals(pd.Index(list(panels.keys()))))
        self.assertEqual(fields["close"].attrs.get("trade_year"), 2026)

    def test_load_prices_returns_ticker_panels(self):
        panels, source = m._load_prices_for_universe(["AAPL", "MSFT", "GOOGL"], n_bars=400)
        self.assertEqual(source, "yfinance_live")
        self.assertNotIn("close", panels)  # ticker-keyed, not field-keyed
        self.assertIn("AAPL", panels)
        self.assertIn("Close", panels["AAPL"].columns)


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
            self.assertIn("half_life", m.FEATURE_NAMES)
            self.assertTrue((tmp_path / "logistic_exit_model.json").exists())
            for t in closed:
                self.assertGreater(t.notional, 0.0)
                self.assertIsNotNone(t.pnl_dollars)
                self.assertGreaterEqual(t.cost_dollars, 0.0)
                self.assertEqual(len(t.exit_features), len(m.FEATURE_NAMES))
            self.assertIn("pnl_dollars", journal.columns)
            self.assertIn("feat_half_life", pd.read_csv(tmp_path / "exit_training_dataset.csv").columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
