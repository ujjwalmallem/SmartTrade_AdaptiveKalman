"""Entry/exit signals for pairs, shared by live trading and research backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def level_signal(close_a: pd.Series, close_b: pd.Series, window: int = 120) -> pd.DataFrame:
    """
    Classic pairs signal: rolling OLS of log prices over ``window`` bars,
    z-score of today's residual against that window. Causal (trailing only).

    Unlike the Kalman innovation z (a one-day surprise), this measures how far
    the spread you actually hold sits from its recent equilibrium, which is
    what a hedged position earns from as it reverts.
    """
    la, lb = np.log(close_a.astype(float)), np.log(close_b.astype(float))
    ma, mb = la.rolling(window).mean(), lb.rolling(window).mean()
    cov = (la * lb).rolling(window).mean() - ma * mb
    var = (lb * lb).rolling(window).mean() - mb * mb
    beta = cov / var
    resid = la - (ma - beta * mb) - beta * lb
    z = resid / resid.rolling(window).std()
    return pd.DataFrame(
        {
            "zscore": z,
            "spread": resid,
            "beta": beta,
            # log-price beta is a dollar hedge ratio → shares_b/shares_a
            "share_ratio": beta * close_a / close_b,
        },
        index=close_a.index,
    )


def level_rule_exit(
    position: int,
    z: float,
    bars_held: int,
    max_bars: int,
    exit_z: float,
    stop_z: float,
) -> tuple[bool, str]:
    """Exit rules validated by research/backtest_pairs.py for the level signal."""
    if (position == 1 and z >= -exit_z) or (position == -1 and z <= exit_z):
        return True, f"REVERTED (z={z:+.2f})"
    if abs(z) >= stop_z:
        return True, f"STOP_LOSS_Z (z={z:+.2f})"
    if bars_held >= max_bars:
        return True, f"TIME_STOP ({bars_held} >= {max_bars} bars)"
    return False, f"HOLD (z={z:+.2f}, {bars_held}/{max_bars} bars)"
