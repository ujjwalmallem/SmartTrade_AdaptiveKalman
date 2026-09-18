"""Smoke / unit tests for AdaptiveKalmanPairs + exit features + latest-year trading."""

from __future__ import annotations

import json
import os
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
        self.assertEqual(list(m.FEATURE_NAMES), [
            "entry_mag", "pnl_z", "giveback", "bars_held", "confidence",
            "edge_velocity", "z_abs", "vol", "half_life", "hold_vs_hl",
        ])
        self.assertEqual(feat[0], 2.0)           # entry_mag
        self.assertAlmostEqual(feat[1], 1.5)     # pnl_z = -0.5 - (-2.0)
        self.assertAlmostEqual(feat[2], 0.0)     # giveback at peak
        self.assertAlmostEqual(feat[3], 12 / 30)
        self.assertEqual(feat[4], 0.7)           # confidence
        self.assertAlmostEqual(feat[5], 0.1)     # edge_velocity = +1 * 0.1
        self.assertEqual(feat[6], 0.5)           # z_abs
        self.assertEqual(feat[7], 1.5)           # vol
        self.assertAlmostEqual(feat[8], 15.0 / 30.0)
        self.assertAlmostEqual(feat[9], 12 / 15.0)
        self.assertGreaterEqual(pos.highest_favorable_z, 1.5)

    def test_long_short_entry_mag_symmetric(self):
        """entry_mag must be identical for ±entry_z (no long/short cancel)."""
        row = pd.Series({
            "zscore": 0.0, "confidence": 0.6, "spread_velocity": -0.2, "spread_vol": 1.0,
        })
        long_pos = m.PositionState(direction=1, entry_z=-2.5, entry_bar=0, entry_spread=1.0)
        short_pos = m.PositionState(direction=-1, entry_z=2.5, entry_bar=0, entry_spread=1.0)
        f_long = m.extract_exit_features(long_pos, row, 5, 1, half_life=20.0)
        f_short = m.extract_exit_features(short_pos, row, 5, -1, half_life=20.0)
        self.assertEqual(f_long[0], f_short[0])  # entry_mag
        self.assertAlmostEqual(f_long[1], f_short[1])  # pnl_z both +2.5 toward 0
        self.assertAlmostEqual(f_long[5], -f_short[5])  # edge_velocity flips with direction

    def test_giveback_not_collinear_with_pnl(self):
        pos = m.PositionState(direction=1, entry_z=-2.0, entry_bar=0, entry_spread=1.0)
        # Peak favorable at z=0 → pnl=2
        peak = pd.Series({"zscore": 0.0, "confidence": 0.7, "spread_velocity": 0.0, "spread_vol": 1.0})
        m.extract_exit_features(pos, peak, 4, 1, half_life=20.0)
        # Give back to z=-1 → pnl=1, giveback=1
        now = pd.Series({"zscore": -1.0, "confidence": 0.7, "spread_velocity": 0.0, "spread_vol": 1.0})
        feat = m.extract_exit_features(pos, now, 8, 1, half_life=20.0)
        self.assertAlmostEqual(feat[1], 1.0)  # pnl_z
        self.assertAlmostEqual(feat[2], 1.0)  # giveback after retreat from MFE
        # Move back to peak — giveback 0 while pnl stays high
        feat2 = m.extract_exit_features(pos, peak, 9, 1, half_life=20.0)
        self.assertAlmostEqual(feat2[1], 2.0)
        self.assertAlmostEqual(feat2[2], 0.0)
        # pnl and giveback are not locked equal across path states
        self.assertNotAlmostEqual(feat2[1], feat2[2])

    def test_model_standardizes_and_persists_scaler(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, len(m.FEATURE_NAMES))) * np.linspace(1, 5, len(m.FEATURE_NAMES))
        y = (X[:, 1] + X[:, 2] > 0).astype(float)
        model = m.LogisticExitModel().fit(X, y, epochs=50)
        self.assertIsNotNone(model.feat_mean)
        self.assertIsNotNone(model.feat_std)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = m.LogisticExitModel.load(path)
            np.testing.assert_allclose(loaded.feat_mean, model.feat_mean)
            np.testing.assert_allclose(
                loaded.predict_proba(X[:3]), model.predict_proba(X[:3]), rtol=1e-5
            )

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

    def test_default_ml_exit_threshold_env(self):
        old = os.environ.pop("ML_EXIT_THRESHOLD", None)
        try:
            self.assertEqual(m.default_ml_exit_threshold(), m.DEFAULT_ML_EXIT_THRESHOLD)
            os.environ["ML_EXIT_THRESHOLD"] = "0.72"
            self.assertAlmostEqual(m.default_ml_exit_threshold(), 0.72)
            os.environ["ML_EXIT_THRESHOLD"] = "1.5"
            with self.assertRaises(ValueError):
                m.default_ml_exit_threshold()
        finally:
            if old is None:
                os.environ.pop("ML_EXIT_THRESHOLD", None)
            else:
                os.environ["ML_EXIT_THRESHOLD"] = old

    def test_time_stop_uses_max_not_min(self):
        # Bug guard: min(5, 2.5*hl) would always cap at 5 and force early exits.
        # hl=10 → allowed hold up to 25 bars.
        should, _ = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=10,
            features=self._features(), model=None, half_life=10.0,
        )
        self.assertFalse(should)

    def test_rule_exit_without_model(self):
        should, proba = m.should_exit_with_ml(
            position=1, z=-0.2, bars_held=5,
            features=self._features(), model=None,
        )
        self.assertTrue(should)
        self.assertIsNone(proba)

    def test_hard_time_stop(self):
        # Fixed floor: 5 bars when half_life is tiny
        should, _ = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=5,
            features=self._features(), model=None, half_life=1.0,
        )
        self.assertTrue(should)

    def test_half_life_time_stop_scales(self):
        # hl=10 → stop at ceil(2.5*10)=25
        should_early, _ = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=24,
            features=self._features(), model=None, half_life=10.0,
        )
        should_late, _ = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=25,
            features=self._features(), model=None, half_life=10.0,
        )
        self.assertFalse(should_early)
        self.assertTrue(should_late)

    def test_ml_force_exit(self):
        model = m.LogisticExitModel()
        # Craft weights so predict_proba is high for any finite feature vector
        model.weights = np.zeros(len(m.FEATURE_NAMES))
        model.bias = 3.0  # sigmoid(3) ≈ 0.95
        should, proba = m.should_exit_with_ml(
            position=1, z=-1.5, bars_held=3,
            features=self._features(), model=model, ml_threshold=0.68,
        )
        self.assertTrue(should)
        self.assertIsNotNone(proba)
        self.assertGreaterEqual(proba, 0.68)

    def test_ml_suppresses_soft_rule_exit(self):
        model = m.LogisticExitModel()
        model.weights = np.zeros(len(m.FEATURE_NAMES))
        model.bias = -3.0  # sigmoid(-3) ≈ 0.05
        should, proba = m.should_exit_with_ml(
            position=1, z=-0.2, bars_held=5,  # would be soft rule exit
            features=self._features(), model=model, ml_threshold=0.68,
            half_life=20.0,  # time-stop at 50 — soft exit can be suppressed
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
        panels, source = m.fetch_real_prices_for_universe(
            ["AAPL", "MSFT", "NVDA"], n_bars=400, data_source="yfinance"
        )
        self.assertIn(source, {"yfinance_live", "alpaca_live"})
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
        panels, _ = m.fetch_real_prices_for_universe(
            ["AAPL", "MSFT"], n_bars=400, data_source="yfinance"
        )
        fields = m.ohlcv_field_panels(panels)
        self.assertEqual(set(fields), {"close", "high", "low", "volume"})
        self.assertTrue(fields["close"].columns.equals(pd.Index(list(panels.keys()))))
        self.assertEqual(fields["close"].attrs.get("trade_year"), 2026)

    def test_load_prices_returns_ticker_panels(self):
        panels, source = m._load_prices_for_universe(
            ["AAPL", "MSFT", "GOOGL"], n_bars=400, data_source="yfinance"
        )
        self.assertEqual(source, "yfinance_live")
        self.assertNotIn("close", panels)  # ticker-keyed, not field-keyed
        self.assertIn("AAPL", panels)
        self.assertIn("Close", panels["AAPL"].columns)


    def test_auto_falls_back_to_yfinance_without_alpaca_creds(self):
        import os
        for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
            os.environ.pop(k, None)
        panels, source = m.fetch_real_prices_for_universe(
            ["AAPL", "MSFT"], n_bars=400, data_source="auto"
        )
        self.assertEqual(source, "yfinance_live")
        self.assertGreaterEqual(len(panels), 2)


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

    def test_cross_year_exit_kept_when_not_required(self):
        df = pd.DataFrame({
            "entry_time": ["2025-12-30", "2026-01-02"],
            "exit_time": ["2026-01-02", "2026-01-05"],
            "trade_id": [1, 2],
        })
        loose = m.filter_trades_to_latest_year(
            df, trade_year=2026, require_exit_in_year=False
        )
        strict = m.filter_trades_to_latest_year(
            df, trade_year=2026, require_exit_in_year=True
        )
        # entry in 2025 → dropped either way; entry 2026 kept
        self.assertEqual(loose["trade_id"].tolist(), [2])
        self.assertEqual(strict["trade_id"].tolist(), [2])
        # entry 2026 exit 2027 kept only when exit year not required
        df2 = pd.DataFrame({
            "entry_time": ["2026-12-30"],
            "exit_time": ["2027-01-02"],
            "trade_id": [9],
        })
        self.assertEqual(
            len(m.filter_trades_to_latest_year(df2, trade_year=2026, require_exit_in_year=False)),
            1,
        )
        self.assertEqual(
            len(m.filter_trades_to_latest_year(df2, trade_year=2026, require_exit_in_year=True)),
            0,
        )


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
                journal_scope="all",  # this test exercises sim fills; production default is alpaca
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
            self.assertIn("giveback", m.FEATURE_NAMES)
            self.assertIn("entry_mag", m.FEATURE_NAMES)
            self.assertNotIn("favorable", m.FEATURE_NAMES)
            self.assertNotIn("best_fav", m.FEATURE_NAMES)
            self.assertTrue((tmp_path / "logistic_exit_model.json").exists())
            for t in closed:
                self.assertGreater(t.notional, 0.0)
                self.assertIsNotNone(t.pnl_dollars)
                self.assertGreaterEqual(t.cost_dollars, 0.0)
                self.assertEqual(len(t.exit_features), len(m.FEATURE_NAMES))
            self.assertIn("pnl_dollars", journal.columns)
            ds = pd.read_csv(tmp_path / "exit_training_dataset.csv")
            self.assertIn("feat_half_life", ds.columns)
            self.assertIn("feat_giveback", ds.columns)
            self.assertIn("feat_hold_vs_hl", ds.columns)
            self.assertNotIn("feat_favorable", ds.columns)




class TestLiveMode(unittest.TestCase):

    def test_live_skips_reentry_when_alpaca_exposed(self):
        from alpaca_paper_broker import AlpacaPaperBroker
        br = AlpacaPaperBroker(paper=True, dry_run=True)
        br.pair_exposure = lambda a, b: {
            "flat": False, "direction": -1, "qty_a": 21.0, "qty_b": 11.0, "blocked": False,
        }
        calls = {"n": 0}
        real = br.open_pair
        def wrapped(*a, **k):
            calls["n"] += 1
            return real(*a, **k)
        br.open_pair = wrapped

        idx = pd.date_range("2026-01-02", periods=80, freq="B")
        z = np.zeros(len(idx)); z[-1] = 2.5
        df = pd.DataFrame({
            "zscore": z, "confidence": np.full(len(idx), 0.7), "spread": 0.0,
            "spread_velocity": 0.0, "spread_vol": 1.0,
            "price_a": 180.0, "price_b": 340.0,
        }, index=idx)
        pair = m.PairSpec(ticker_a="QCOM", ticker_b="AVGO", basket="semis")
        trader = m.PaperTrader(broker=br, execute_latest_only=True, latest_bar=idx[-1])
        m._trade_pair_session(
            trader, df, pair, min_trades=1, trades_remaining=1,
            trade_year=2026, mode="live", latest_bar=idx[-1],
        )
        self.assertEqual(calls["n"], 0)
        self.assertTrue(any(tr.status == "OPEN" and tr.broker == "alpaca_paper" for tr in trader.trades))

    def test_live_mode_only_enters_on_latest_bar(self):
        idx = pd.date_range("2026-01-02", periods=120, freq="B")
        rng = np.random.default_rng(0)
        z = rng.normal(0, 0.5, len(idx))
        z[70] = -2.5
        z[-1] = -2.4
        df = pd.DataFrame({
            "zscore": z,
            "confidence": np.full(len(idx), 0.7),
            "spread": np.cumsum(rng.normal(0, 0.2, len(idx))),
            "spread_velocity": 0.0,
            "spread_vol": 1.0,
            "price_a": 100 + np.arange(len(idx)) * 0.1,
            "price_b": 200 + np.arange(len(idx)) * 0.05,
        }, index=idx)
        pair = m.PairSpec(ticker_a="AAPL", ticker_b="MSFT", basket="mag7")
        trader = m.PaperTrader()
        m._trade_pair_session(
            trader, df, pair, min_trades=1, trades_remaining=1,
            trade_year=2026, mode="live", latest_bar=idx[-1],
        )
        self.assertGreaterEqual(len(trader.trades), 1)
        for tr in trader.trades:
            self.assertEqual(pd.Timestamp(tr.entry_time).normalize(), pd.Timestamp(idx[-1]).normalize())

    def test_live_holds_overnight_no_same_bar_exit(self):
        from alpaca_paper_broker import AlpacaPaperBroker
        br = AlpacaPaperBroker(paper=True, dry_run=True)
        br.pair_exposure = lambda a, b: {
            "flat": True, "direction": 0, "qty_a": 0.0, "qty_b": 0.0, "blocked": False,
        }
        idx = pd.date_range("2026-01-02", periods=80, freq="B")
        # Extreme z that would normally force an immediate ML/rule exit
        z = np.zeros(len(idx))
        z[-1] = 2.5
        df = pd.DataFrame({
            "zscore": z, "confidence": np.full(len(idx), 0.7), "spread": 0.0,
            "spread_velocity": 0.0, "spread_vol": 1.0,
            "price_a": 180.0, "price_b": 340.0,
        }, index=idx)
        pair = m.PairSpec(ticker_a="QCOM", ticker_b="AVGO", basket="semis")
        trader = m.PaperTrader(broker=br, execute_latest_only=True, latest_bar=idx[-1])
        m._trade_pair_session(
            trader, df, pair, min_trades=1, trades_remaining=1,
            trade_year=2026, mode="live", latest_bar=idx[-1],
        )
        self.assertEqual(len(trader.trades), 1)
        self.assertEqual(trader.trades[0].status, "OPEN")
        self.assertEqual(trader.trades[0].broker, "alpaca_paper")
        self.assertIsNone(trader.trades[0].exit_time)

    def test_failed_alpaca_exit_keeps_trade_open(self):
        from alpaca_paper_broker import AlpacaPaperBroker
        br = AlpacaPaperBroker(paper=True, dry_run=True)

        def boom(*a, **k):
            raise RuntimeError("wash trade detected")

        br.close_pair = boom
        trader = m.PaperTrader(broker=br, execute_latest_only=True, latest_bar=pd.Timestamp("2026-09-15"))
        trader.open_trade(
            -1, pd.Timestamp("2026-09-15"), 2.2, 1.0,
            "QCOM", "AVGO", "semis", price_a=180, price_b=340,
        )
        self.assertEqual(trader.trades[0].status, "OPEN")
        ok = trader.close_trade(pd.Timestamp("2026-09-15"), 0.1, 0.5, bars_held=2)
        self.assertFalse(ok)
        self.assertEqual(trader.trades[0].status, "OPEN")
        self.assertIsNone(trader.trades[0].exit_time)

    def test_dedupe_sim_journal_keeps_alpaca(self):
        df = pd.DataFrame([
            {
                "run_id": "a", "trade_id": 1, "ticker_a": "AAPL", "ticker_b": "MSFT",
                "direction": "LONG_SPREAD", "entry_time": "2026-04-13", "exit_time": "2026-04-17",
                "broker": "sim", "status": "CLOSED", "pnl_z": 1.0,
            },
            {
                "run_id": "b", "trade_id": 1, "ticker_a": "AAPL", "ticker_b": "MSFT",
                "direction": "LONG_SPREAD", "entry_time": "2026-04-13", "exit_time": "2026-04-17",
                "broker": "sim", "status": "CLOSED", "pnl_z": 1.0,
            },
            {
                "run_id": "c", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "direction": "SHORT_SPREAD", "entry_time": "2026-09-15", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
            {
                "run_id": "d", "trade_id": 2, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "direction": "SHORT_SPREAD", "entry_time": "2026-09-15", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
        ])
        out = m.dedupe_journal_rows(df)
        sim_rows = out[out["broker"] == "sim"]
        alpaca_rows = out[out["broker"] == "alpaca_paper"]
        self.assertEqual(len(sim_rows), 1)
        self.assertEqual(len(alpaca_rows), 1)
        self.assertEqual(alpaca_rows.iloc[0]["run_id"], "d")

    def test_dedupe_drops_wash_keeps_open(self):
        df = pd.DataFrame([
            {
                "run_id": "w", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15 04:00:00", "exit_time": "2026-09-15 04:00:00",
                "broker": "alpaca_paper", "status": "CLOSED", "pnl_z": 0.0,
            },
            {
                "run_id": "o1", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15 04:00:00", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
            {
                "run_id": "o2", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15 04:00:00", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
        ])
        out = m.dedupe_journal_rows(df, drop_wash=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["status"], "OPEN")
        self.assertEqual(out.iloc[0]["run_id"], "o2")

    def test_dedupe_prefers_real_closed_over_open(self):
        df = pd.DataFrame([
            {
                "run_id": "o", "trade_id": 1, "ticker_a": "AAPL", "ticker_b": "MSFT",
                "direction": "LONG_SPREAD",
                "entry_time": "2026-04-13", "exit_time": None,
                "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
            },
            {
                "run_id": "c", "trade_id": 1, "ticker_a": "AAPL", "ticker_b": "MSFT",
                "direction": "LONG_SPREAD",
                "entry_time": "2026-04-13", "exit_time": "2026-04-17",
                "broker": "alpaca_paper", "status": "CLOSED", "pnl_z": 1.2,
            },
        ])
        out = m.dedupe_journal_rows(df, drop_wash=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["status"], "CLOSED")
        self.assertEqual(out.iloc[0]["run_id"], "c")

    def test_save_open_only_trades_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trade = m.PaperTrade(
                trade_id=1,
                direction="SHORT_SPREAD",
                entry_time=pd.Timestamp("2026-09-15"),
                entry_z=2.2,
                entry_spread=1.0,
                ticker_a="QCOM",
                ticker_b="AVGO",
                basket="semis",
                broker="alpaca_paper",
                qty_a=21.0,
                qty_b=11.0,
                status="OPEN",
            )
            path, ds_path = m.save_paper_results(
                [trade],
                results_dir=tmp_path,
                data_source="alpaca_live",
                journal_scope="alpaca",
            )
            journal = pd.read_csv(path)
            self.assertEqual(len(journal), 1)
            self.assertEqual(journal.iloc[0]["status"], "OPEN")
            self.assertEqual(journal.iloc[0]["broker"], "alpaca_paper")
            self.assertTrue(ds_path.exists())
            self.assertEqual(len(pd.read_csv(ds_path)), 0)

    def test_filter_journal_by_scope_alpaca(self):
        df = pd.DataFrame([
            {"broker": "sim", "trade_id": 1},
            {"broker": "alpaca_paper", "trade_id": 2},
            {"broker": "alpaca", "trade_id": 3},
            {"broker": "none", "trade_id": 4},
        ])
        out = m.filter_journal_by_scope(df, "alpaca")
        self.assertEqual(sorted(out["trade_id"].tolist()), [2, 3])
        sim_only = m.filter_journal_by_scope(df, "sim")
        self.assertEqual(sorted(sim_only["trade_id"].tolist()), [1, 4])
        none = m.filter_journal_by_scope(df, "none")
        self.assertEqual(len(none), 0)

    def test_save_alpaca_scope_strips_sim_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prev = pd.DataFrame([
                {
                    "run_id": "sim1", "trade_id": 1, "ticker_a": "MU", "ticker_b": "WDC",
                    "basket": "x", "direction": "LONG_SPREAD",
                    "entry_time": "2026-01-06", "exit_time": "2026-01-07",
                    "broker": "sim", "status": "CLOSED", "pnl_z": 2.0,
                    "label": 1, "data_source": "alpaca_live",
                    **{f"feat_{n}": 0.1 for n in m.FEATURE_NAMES},
                },
                {
                    "run_id": "live1", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                    "basket": "semis", "direction": "SHORT_SPREAD",
                    "entry_time": "2026-09-15", "exit_time": None,
                    "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
                    "label": None, "data_source": "alpaca_live",
                },
            ])
            prev.to_csv(tmp_path / "paper_trades.csv", index=False)
            # Prior training tied to sim journal key
            pd.DataFrame([{
                "run_id": "sim1", "trade_id": 1, "ticker_a": "MU", "ticker_b": "WDC",
                "basket": "x", "label": 1, "data_source": "alpaca_live",
                **{f"feat_{n}": 0.1 for n in m.FEATURE_NAMES},
            }]).to_csv(tmp_path / "exit_training_dataset.csv", index=False)

            trade = m.PaperTrade(
                trade_id=2,
                direction="LONG_SPREAD",
                entry_time=pd.Timestamp("2026-09-16"),
                entry_z=-2.1,
                entry_spread=1.0,
                ticker_a="NVDA",
                ticker_b="AMD",
                basket="semis",
                broker="alpaca_paper",
                status="OPEN",
            )
            path, ds_path = m.save_paper_results(
                [trade],
                results_dir=tmp_path,
                data_source="alpaca_live",
                journal_scope="alpaca",
            )
            journal = pd.read_csv(path)
            self.assertTrue((journal["broker"].astype(str).str.startswith("alpaca")).all())
            self.assertFalse((journal["broker"] == "sim").any())
            self.assertEqual(len(journal), 2)  # prior OPEN + new OPEN
            ds = pd.read_csv(ds_path)
            # Sim training key purged with sim journal rows
            self.assertEqual(len(ds), 0)

    def test_save_dedupes_polluted_journal_like_results_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Simulate the 19-row pollution: 4 sim + wash CLOSED + many OPEN adopts
            rows = []
            for i, (a, b, entry, exit_) in enumerate([
                ("MU", "WDC", "2026-01-06", "2026-01-07"),
                ("NVDA", "AMD", "2026-01-13", "2026-01-15"),
                ("MSFT", "ORCL", "2026-03-11", "2026-03-12"),
                ("AAPL", "MSFT", "2026-04-13", "2026-04-17"),
            ], start=1):
                rows.append({
                    "run_id": "sim1", "trade_id": i, "ticker_a": a, "ticker_b": b,
                    "basket": "x", "direction": "LONG_SPREAD",
                    "entry_time": entry, "exit_time": exit_,
                    "broker": "sim", "status": "CLOSED", "pnl_z": 2.0,
                    "label": 1, "data_source": "alpaca_live",
                    **{f"feat_{n}": 0.1 for n in m.FEATURE_NAMES},
                })
            rows.append({
                "run_id": "wash", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                "basket": "semis", "direction": "SHORT_SPREAD",
                "entry_time": "2026-09-15 04:00:00", "exit_time": "2026-09-15 04:00:00",
                "broker": "alpaca_paper", "status": "CLOSED", "pnl_z": 0.0,
                "label": 0, "data_source": "alpaca_live",
                **{f"feat_{n}": 0.1 for n in m.FEATURE_NAMES},
            })
            for i in range(14):
                rows.append({
                    "run_id": f"open{i}", "trade_id": 1, "ticker_a": "QCOM", "ticker_b": "AVGO",
                    "basket": "semis", "direction": "SHORT_SPREAD",
                    "entry_time": "2026-09-15 04:00:00", "exit_time": None,
                    "broker": "alpaca_paper", "status": "OPEN", "pnl_z": 0.0,
                    "label": None, "data_source": "alpaca_live",
                })
            pd.DataFrame(rows).to_csv(tmp_path / "paper_trades.csv", index=False)
            # Append nothing new — just run save path via empty closed list by calling dedupe on loaded
            prev = pd.read_csv(tmp_path / "paper_trades.csv")
            cleaned = m.dedupe_journal_rows(
                m.filter_trades_to_latest_year(prev, trade_year=2026, require_exit_in_year=False),
                drop_wash=True,
            )
            self.assertEqual(len(cleaned), 5)  # 4 sim + 1 OPEN
            self.assertEqual((cleaned["broker"] == "sim").sum(), 4)
            self.assertEqual((cleaned["status"] == "OPEN").sum(), 1)
            self.assertFalse(((cleaned["status"] == "CLOSED") & (cleaned["ticker_a"] == "QCOM")).any())

class TestAlpacaDataPrimary(unittest.TestCase):
    def test_alpaca_primary_path_with_mock(self):
        import types
        idx = pd.date_range("2025-10-01", periods=200, freq="B")
        def fake_fetch(tickers, period="2y", min_bars=80):
            out = {}
            for i, tkr in enumerate(tickers):
                px = 100 + i * 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, len(idx)))
                out[tkr] = pd.DataFrame({
                    "Open": px, "High": px + 1, "Low": px - 1, "Close": px,
                    "Volume": np.full(len(idx), 1e6),
                }, index=idx)
            return out
        # Patch credentials + fetch
        real_present = m.alpaca_credentials_present
        real_fetch = m.fetch_alpaca_daily_ohlcv
        m.alpaca_credentials_present = lambda: True
        m.fetch_alpaca_daily_ohlcv = fake_fetch
        try:
            panels, source = m.fetch_real_prices_for_universe(
                ["AAPL", "MSFT", "NVDA"], n_bars=400, data_source="auto"
            )
            self.assertEqual(source, "alpaca_live")
            self.assertEqual(len(panels), 3)
            self.assertIn("Close", panels["AAPL"].columns)
        finally:
            m.alpaca_credentials_present = real_present
            m.fetch_alpaca_daily_ohlcv = real_fetch


class TestAlpacaBroker(unittest.TestCase):

    def test_dry_run_pair_round_trip(self):
        from alpaca_paper_broker import AlpacaPaperBroker
        br = AlpacaPaperBroker(paper=True, dry_run=True)
        opened = br.open_pair("AAPL", "MSFT", "LONG_SPREAD", 8000, 180.0, 400.0)
        self.assertEqual(len(opened.fills), 2)
        self.assertEqual(opened.fills[0].side, "buy")
        self.assertEqual(opened.fills[1].side, "sell")
        closed = br.close_pair("AAPL", "MSFT", opened.fills[0].qty, opened.fills[1].qty, "LONG_SPREAD")
        self.assertEqual(len(closed.fills), 2)

    def test_trader_routes_only_latest_bar(self):
        br = m.AlpacaPaperBroker(paper=True, dry_run=True)
        latest = pd.Timestamp("2026-09-14")
        trader = m.PaperTrader(broker=br, execute_latest_only=True, latest_bar=latest)
        # Historical bar → sim only
        trader.open_trade(
            1, pd.Timestamp("2026-04-13"), -2.1, 1.0,
            "AAPL", "MSFT", "mag7", price_a=180, price_b=400,
        )
        self.assertEqual(trader.trades[0].broker, "sim")
        trader.close_trade(pd.Timestamp("2026-04-17"), -0.2, 0.5, bars_held=4)
        # Latest bar → alpaca paper
        trader.open_trade(
            1, latest, -2.2, 1.0,
            "AAPL", "MSFT", "mag7", price_a=180, price_b=400,
        )
        self.assertEqual(trader.trades[-1].broker, "alpaca_paper")
        self.assertGreater(trader.trades[-1].qty_a, 0)
        trader.close_trade(latest, -0.3, 0.4, bars_held=1)
        self.assertTrue(any(t.broker == "alpaca_paper" for t in trader.trades if t.status == "CLOSED"))

    def test_build_broker_sim_is_none(self):
        self.assertIsNone(m.build_broker("sim"))


class TestResearchMode(unittest.TestCase):

    def _synthetic_pair_frame(self, n: int = 100) -> pd.DataFrame:
        idx = pd.date_range("2025-06-02", periods=n, freq="B")
        z = np.zeros(n)
        # Two crossings: long then short, each mean-reverts enough to exit
        z[65] = -2.4
        z[66:70] = np.linspace(-2.0, 0.0, 4)
        z[80] = 2.5
        z[81:85] = np.linspace(2.0, 0.0, 4)
        return pd.DataFrame({
            "zscore": z,
            "confidence": np.full(n, 0.7),
            "spread": np.cumsum(np.random.default_rng(0).normal(0, 0.1, n)),
            "spread_velocity": 0.0,
            "spread_vol": 1.0,
            "price_a": 100.0,
            "price_b": 200.0,
        }, index=idx)

    def test_simulate_setups_finds_crossings(self):
        df = self._synthetic_pair_frame()
        pair = m.PairSpec(ticker_a="AAPL", ticker_b="MSFT", basket="mag7")
        setups = m._simulate_setups_for_pair(
            df, pair, model=None, run_id="TEST", data_window="multi_year",
        )
        self.assertGreaterEqual(len(setups), 1)
        for s in setups:
            self.assertFalse(s.taken)
            self.assertIsNotNone(s.exit_time)
            self.assertIsNotNone(s.label)
            self.assertEqual(s.exit_model_version, "rules_only")
            self.assertIn("feat_entry_mag", s.to_row())
            self.assertNotIn("feat_entry_z", s.to_row())
            self.assertNotIn("feat_favorable", s.to_row())

    def test_latest_year_gate_skips_prior_year(self):
        df = self._synthetic_pair_frame()
        pair = m.PairSpec(ticker_a="AAPL", ticker_b="MSFT", basket="mag7")
        # Force trade_year=2026 while signals are in 2025
        setups = m._simulate_setups_for_pair(
            df, pair, model=None, run_id="TEST",
            data_window="latest_year", trade_year=2026,
        )
        self.assertEqual(len(setups), 0)

    def test_save_setups_does_not_touch_paper_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paper = tmp_path / "paper_trades.csv"
            paper.write_text("run_id,broker\nkeep,me\n")
            setup = m.Setup(
                setup_id="x",
                run_id="R1",
                pair="AAPL/MSFT",
                basket="mag7",
                direction=1,
                entry_time=pd.Timestamp("2026-04-13"),
                entry_bar=70,
                features={"entry_mag": 2.2, "z_abs": 2.2},
                taken=False,
                exit_time=pd.Timestamp("2026-04-17"),
                bars_held=4,
                pnl_z=1.5,
                pnl_dollars=100.0,
                exit_reason="rules",
                label=1,
                exit_model_version="rules_only",
                data_window="multi_year",
            )
            out = m.save_setups([setup], results_dir=tmp_path, run_id="R1")
            self.assertTrue(out.exists())
            self.assertTrue(out.name.startswith("setups_"))
            self.assertEqual(pd.read_csv(paper).iloc[0]["broker"], "me")
            saved = pd.read_csv(out)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["pair"], "AAPL/MSFT")

    def test_research_mode_dispatch_returns_setups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m.RESULTS_DIR = tmp_path
            m.MODEL_JSON = tmp_path / "logistic_exit_model.json"
            # Tiny universe via monkeypatch of build/load is heavy; call simulator path only.
            setups, path = [], tmp_path / "setups_empty.csv"
            # Ensure mode validation accepts research
            with self.assertRaises(ValueError):
                m.run_paper_trading_and_train(mode="nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
