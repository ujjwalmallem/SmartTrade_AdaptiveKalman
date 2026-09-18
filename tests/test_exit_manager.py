"""Unit tests for StatArb exit manager + feature schema (SYSTEM_SPEC)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.exit_manager import StatArbExitManager, TradeState, time_stop_bars
from src.features import (
    FEATURE_NAMES,
    DROPPED_FEATURES,
    extract_feature_vector,
    fit_scaler,
    save_scaler,
)
from src.train_exit_model import label_path_bars, train_exit_model


class TestFeatures(unittest.TestCase):
    def test_feature_order_and_drops(self):
        self.assertEqual(
            FEATURE_NAMES,
            [
                "vol", "pnl_proxy", "abs_entry_z", "confidence",
                "exit_z", "velocity", "bars_held", "half_life",
            ],
        )
        for name in DROPPED_FEATURES:
            self.assertNotIn(name, FEATURE_NAMES)

    def test_long_short_abs_entry_symmetric(self):
        long_v = extract_feature_vector(
            entry_z=-2.5, current_z=0.0, direction=1,
            vol=1.0, confidence=0.7, velocity=0.1, bars_held=4, half_life=20.0,
        )
        short_v = extract_feature_vector(
            entry_z=2.5, current_z=0.0, direction=-1,
            vol=1.0, confidence=0.7, velocity=0.1, bars_held=4, half_life=20.0,
        )
        self.assertEqual(long_v[2], short_v[2])  # abs_entry_z
        self.assertAlmostEqual(long_v[1], short_v[1])  # pnl_proxy both +2.5


class TestTimeStop(unittest.TestCase):
    def test_uses_max_not_min(self):
        # hl=10 → 25 bars; min(5,25)=5 would be wrong
        self.assertEqual(time_stop_bars(10.0), 25)
        self.assertEqual(time_stop_bars(1.0), 5)  # floor


class TestStatArbExitManager(unittest.TestCase):
    def _state(self, **kwargs) -> TradeState:
        base = dict(
            trade_id=1,
            ticker_a="QCOM",
            ticker_b="AVGO",
            direction="SHORT_SPREAD",
            bars_held=2,
            half_life_bars=10.0,
            vol=1.2,
            pnl_proxy=0.5,
            entry_z=2.4,
            confidence=0.7,
            exit_z=1.0,
            velocity=-0.1,
            half_life=10.0 / 30.0,
            cost_dollars=3.0,
            pnl_dollars=40.0,
        )
        base.update(kwargs)
        return TradeState(**base)

    def test_time_stop_triggers(self):
        mgr = StatArbExitManager(model=None, scaler=None, exit_threshold=0.68)
        should, reason, prob = mgr.evaluate_trade(self._state(bars_held=25, half_life_bars=10.0))
        self.assertTrue(should)
        self.assertIn("TIME_STOP", reason)
        self.assertEqual(prob, 1.0)

    def test_stop_loss_z(self):
        mgr = StatArbExitManager(model=None, scaler=None)
        should, reason, _ = mgr.evaluate_trade(
            self._state(direction="LONG_SPREAD", exit_z=-4.2, bars_held=1, half_life_bars=20.0)
        )
        self.assertTrue(should)
        self.assertIn("STOP_LOSS", reason)

    def test_ml_exit_with_toy_model(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, len(FEATURE_NAMES)))
        y = (X[:, 1] > 0).astype(int)
        y[0] = 0
        y[1] = 1
        scaler = fit_scaler(X)
        model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
        mgr = StatArbExitManager(
            model=model, scaler=scaler, exit_threshold=0.50,
            absolute_min_bars=5, max_half_life_multiplier=2.5,
        )
        # Force high pnl_proxy-ish via raw features the model saw
        should, reason, prob = mgr.evaluate_trade(
            self._state(bars_held=2, half_life_bars=20.0, pnl_proxy=3.0)
        )
        self.assertIsInstance(should, bool)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_hold_without_model(self):
        mgr = StatArbExitManager(model=None, scaler=None)
        should, reason, prob = mgr.evaluate_trade(
            self._state(bars_held=2, half_life_bars=20.0)
        )
        self.assertFalse(should)
        self.assertIn("no exit model", reason)

    def test_paper_loop_hooks_evaluate_trade(self):
        """Open-position exits go through StatArbExitManager.evaluate_trade."""
        import paper_trading_ml_exit as m

        calls = {"n": 0}
        real_eval = StatArbExitManager.evaluate_trade

        def wrapped(self, state):
            calls["n"] += 1
            return real_eval(self, state)

        StatArbExitManager.evaluate_trade = wrapped  # type: ignore
        try:
            mgr = StatArbExitManager(model=None, scaler=None, exit_threshold=0.68)
            pos = m.PositionState(direction=-1, entry_z=2.4, entry_bar=0, entry_spread=1.0)
            row = pd.Series({
                "zscore": 1.0, "confidence": 0.7,
                "spread_velocity": -0.1, "spread_vol": 1.2,
            })
            pair = m.PairSpec(ticker_a="QCOM", ticker_b="AVGO", basket="semis")
            state = m.build_trade_state(
                pair=pair, direction=-1, pos_state=pos, row=row,
                bars_held=2, half_life=20.0,
            )
            should, proba = m.should_exit_with_ml(
                position=-1, z=1.0, bars_held=2,
                features=np.zeros(len(m.FEATURE_NAMES)),
                model=None, ml_threshold=0.68, half_life=20.0,
                exit_manager=mgr, trade_state=state,
            )
            self.assertEqual(calls["n"], 1)
            self.assertIsInstance(should, bool)
            self.assertIsNotNone(proba)
        finally:
            StatArbExitManager.evaluate_trade = real_eval  # type: ignore


class TestTrainExitModel(unittest.TestCase):
    def test_label_path_bars_lookahead(self):
        df = pd.DataFrame({
            "pnl_proxy": [0.0, 0.2, 0.5, 0.1],
            "half_life_bars": [2.0, 2.0, 2.0, 2.0],
            "exit_z": [-2.0, -1.0, -0.2, -0.5],
            "direction": ["LONG_SPREAD"] * 4,
        })
        y = label_path_bars(df, multiplier=2.5, stop_loss_z=4.0)
        self.assertEqual(len(y), 4)
        self.assertEqual(int(y.iloc[0]), 1)  # sees +0.5 within horizon

    def test_train_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results = tmp_path / "results"
            results.mkdir()
            models = tmp_path / "models"
            models.mkdir()
            # Balanced synthetic dataset
            rows = []
            for i in range(12):
                rows.append({
                    "vol": 1.0 + 0.1 * i,
                    "pnl_proxy": 1.5 if i % 2 == 0 else -0.2,
                    "abs_entry_z": 2.2,
                    "confidence": 0.7,
                    "exit_z": 0.1 if i % 2 == 0 else -1.0,
                    "velocity": 0.0,
                    "bars_held": 5 + i,
                    "half_life": 0.5,
                    "label": 1 if i % 2 == 0 else 0,
                })
            pd.DataFrame(rows).to_csv(results / "exit_training_dataset.csv", index=False)
            cfg = tmp_path / "cfg.yaml"
            cfg.write_text(
                "\n".join([
                    "exit_model:",
                    f"  model_path: '{models / 'logistic_exit_model.pkl'}'",
                    f"  scaler_path: '{models / 'feature_scaler.pkl'}'",
                    f"  metadata_path: '{models / 'model_metadata.json'}'",
                    "  probability_threshold: 0.68",
                    "risk_engine:",
                    "  max_half_life_multiplier: 2.5",
                    "  absolute_min_bars: 5",
                    "  stop_loss_z: 4.0",
                    "  hard_pnl_stop_dollars: -150.0",
                ])
            )
            meta = train_exit_model(
                results, config_path=cfg, calibrate=False, min_samples=4
            )
            self.assertTrue(Path(meta["model_path"]).exists())
            self.assertTrue(Path(meta["scaler_path"]).exists())
            self.assertEqual(meta["feature_names"], list(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
