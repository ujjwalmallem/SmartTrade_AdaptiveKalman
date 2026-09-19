"""Tests for path-label harvest + half-life module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.half_life import estimate_half_life
from src.harvest import harvest_pair_paths, harvest_training_dataset


class TestHalfLifeModule(unittest.TestCase):
    def test_floor_and_ceil(self):
        # Near-white-noise → fast MR → clipped to min_hl
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200))
        hl = estimate_half_life(x, lookback=80, min_hl=4.0, max_hl=60.0)
        self.assertGreaterEqual(hl, 4.0)
        self.assertLessEqual(hl, 60.0)

    def test_short_series_returns_default(self):
        hl = estimate_half_life(pd.Series([1.0, 2.0, 1.5]), lookback=80, default_hl=20.0)
        self.assertEqual(hl, 20.0)

    def test_explosive_phi_returns_max(self):
        # Explicitly explosive AR(1) φ>1 → no stationary MR → max_hl
        rng = np.random.default_rng(1)
        n = 200
        x = np.zeros(n)
        x[0] = 0.1
        for i in range(1, n):
            x[i] = 1.08 * x[i - 1] + rng.normal(0, 0.05)
        hl = estimate_half_life(pd.Series(x), lookback=80, max_hl=60.0)
        self.assertEqual(hl, 60.0)

    def test_constant_series_returns_default(self):
        hl = estimate_half_life(pd.Series(np.ones(100)), lookback=80, default_hl=20.0)
        self.assertEqual(hl, 20.0)


class TestPathHarvest(unittest.TestCase):
    def _mr_pair_frame(self, n: int = 180) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2026-01-02", periods=n, freq="B")
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.4)
        # Inject a few z crossings via residual z construction
        z = spread / (np.std(spread) + 1e-6)
        z[70] = -2.5
        z[71:85] = np.linspace(-2.0, 0.5, 14)
        z[100] = 2.4
        z[101:115] = np.linspace(2.0, -0.3, 14)
        return pd.DataFrame({
            "zscore": z,
            "confidence": np.full(n, 0.7),
            "spread": spread,
            "spread_velocity": np.r_[0.0, np.diff(spread)],
            "spread_vol": np.full(n, float(np.std(spread) + 0.1)),
        }, index=idx)

    def test_harvest_pair_emits_labeled_bars_with_hl_variation(self):
        df = self._mr_pair_frame()
        out = harvest_pair_paths(
            df, ticker_a="AAPL", ticker_b="MSFT", basket="mag7",
            z_entry=2.0, min_confidence=0.55,
        )
        self.assertGreater(len(out), 5)
        self.assertIn("label", out.columns)
        self.assertIn("half_life_bars", out.columns)
        self.assertTrue(set(out["label"].unique()).issubset({0, 1}))
        # Should not be glued to the old 2-bar floor
        self.assertGreaterEqual(float(out["half_life_bars"].min()), 4.0)

    def test_harvest_training_dataset_writes_feat_columns(self):
        df = self._mr_pair_frame()
        with tempfile.TemporaryDirectory() as tmp:
            path, framed = harvest_training_dataset(
                {"AAPL/MSFT": (df, "AAPL", "MSFT", "mag7")},
                results_dir=tmp,
                data_source="test",
            )
            self.assertTrue(path.exists())
            self.assertGreater(len(framed), 0)
            self.assertIn("feat_half_life", framed.columns)
            self.assertIn("label", framed.columns)
            # Normalized feature should vary or at least exceed old 2/30 constant
            hl_feat = framed["feat_half_life"]
            self.assertGreaterEqual(float(hl_feat.min()) * 30.0, 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
