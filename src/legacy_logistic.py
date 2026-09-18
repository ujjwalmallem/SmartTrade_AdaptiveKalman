"""
LEGACY — in-process logistic exit model (JSON artifacts).

Production exits use StatArbExitManager + models/*.pkl.
Kept only for reading old results/logistic_exit_model.json and unit tests.
Do not wire this into live / backtest / research exit decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

DEFAULT_LEGACY_MODEL_JSON = Path("results") / "logistic_exit_model.json"


class LogisticExitModel:
    """Deprecated JSON logistic — quarantine module only."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.schema_version: int = FEATURE_SCHEMA_VERSION

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.feat_mean is None or self.feat_std is None:
            return X
        return (X - self.feat_mean) / self.feat_std

    def fit(self, X, y, reg=0.3, lr=0.1, epochs=400):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.feat_mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.feat_std = np.where(std < 1e-8, 1.0, std)
        Xs = self._transform(X)
        self.weights = np.zeros(d)
        self.bias = 0.0
        self.schema_version = FEATURE_SCHEMA_VERSION
        self.feature_names = list(FEATURE_NAMES)
        for _ in range(epochs):
            logits = Xs @ self.weights + self.bias
            probs = self._sigmoid(logits)
            err = probs - y
            self.weights -= lr * ((Xs.T @ err) / n + reg * self.weights)
            self.bias -= lr * (err.mean())
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._sigmoid(self._transform(X) @ self.weights + self.bias)

    def save(self, path: Path = DEFAULT_LEGACY_MODEL_JSON) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": float(self.bias),
            "feature_names": self.feature_names,
            "feat_mean": self.feat_mean.tolist() if self.feat_mean is not None else None,
            "feat_std": self.feat_std.tolist() if self.feat_std is not None else None,
            "schema_version": int(self.schema_version),
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_LEGACY_MODEL_JSON) -> "LogisticExitModel":
        payload = json.loads(Path(path).read_text())
        model = cls()
        model.weights = (
            np.array(payload["weights"], dtype=float) if payload["weights"] else None
        )
        model.bias = float(payload["bias"])
        model.feature_names = list(payload.get("feature_names", FEATURE_NAMES))
        mean = payload.get("feat_mean")
        std = payload.get("feat_std")
        model.feat_mean = np.array(mean, dtype=float) if mean is not None else None
        model.feat_std = np.array(std, dtype=float) if std is not None else None
        model.schema_version = int(payload.get("schema_version", 1))
        return model
