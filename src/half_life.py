"""Ornstein–Uhlenbeck / AR(1) half-life estimation for pairs spreads."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray]


def estimate_half_life(
    spread: ArrayLike,
    lookback: int = 80,
    *,
    min_hl: float = 4.0,
    max_hl: float = 60.0,
    default_hl: float = 20.0,
) -> float:
    """
    AR(1) half-life in bars: x_t = c + φ x_{t-1} + ε.

    half_life = -ln(2) / ln(φ) for 0 < φ < 1.

    Notes
    -----
    - Default lookback=80 reduces finite-sample downward bias vs 40 on daily bars.
    - Floor min_hl=4 so time-stop is not glued to absolute_min_bars=5 for every pair.
    - Non-mean-reverting / explosive φ → max_hl (slow exit).
    """
    s = pd.Series(spread, dtype=float).dropna()
    if len(s) < max(lookback, 20):
        return float(default_hl)

    window = s.iloc[-int(lookback) :]
    y = window.iloc[1:].to_numpy(dtype=float)
    x = window.iloc[:-1].to_numpy(dtype=float)
    if len(y) < 15:
        return float(default_hl)
    if float(np.std(x)) < 1e-10:
        return float(default_hl)

    # OLS with intercept
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float(default_hl)
    phi = float(coef[1])

    if not np.isfinite(phi):
        return float(default_hl)
    if phi <= 0.0 or phi >= 1.0:
        # No stationary mean reversion on this window
        return float(max_hl)

    hl = float(-np.log(2.0) / np.log(phi))
    if not np.isfinite(hl) or hl <= 0:
        return float(default_hl)
    return float(np.clip(hl, min_hl, max_hl))
