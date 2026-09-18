#!/usr/bin/env python3
"""
Dry-run verification: replay journal rows through StatArbExitManager.evaluate_trade.

Validates:
  1. Long/short symmetry for identical |z_entry|
  2. Time-stop at max(5, ceil(2.5 * half_life_bars))  — NOT min(...)
  3. Feature vectors finite (no NaN/Inf) after extract + optional scale
  4. Decision string + prob_exit logged every evaluation
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exit_manager import StatArbExitManager, TradeState, time_stop_bars
from src.features import FEATURE_NAMES, extract_feature_vector, fit_scaler, transform_features


def _finite(vec: np.ndarray) -> bool:
    return bool(np.isfinite(vec).all())


def symmetry_check(mgr: StatArbExitManager) -> dict:
    """Identical |z_entry| → same abs_entry_z feature → same ML geometry."""
    long_state = TradeState(
        trade_id=1, ticker_a="QCOM", ticker_b="AVGO", direction="LONG_SPREAD",
        bars_held=2, half_life_bars=20.0,
        vol=1.2, pnl_proxy=1.0, entry_z=-2.4, confidence=0.7,
        exit_z=-0.5, velocity=0.1, half_life=20.0 / 30.0, pnl_dollars=50.0,
    )
    short_state = TradeState(
        trade_id=2, ticker_a="QCOM", ticker_b="AVGO", direction="SHORT_SPREAD",
        bars_held=2, half_life_bars=20.0,
        vol=1.2, pnl_proxy=1.0, entry_z=2.4, confidence=0.7,
        exit_z=0.5, velocity=-0.1, half_life=20.0 / 30.0, pnl_dollars=50.0,
    )
    # Feature rows used by the manager (abs_entry_z forced)
    long_row = mgr._raw_feature_row(long_state)[0]
    short_row = mgr._raw_feature_row(short_state)[0]
    abs_idx = FEATURE_NAMES.index("abs_entry_z")
    pnl_idx = FEATURE_NAMES.index("pnl_proxy")
    ok_abs = math.isclose(long_row[abs_idx], short_row[abs_idx], rel_tol=1e-9)
    ok_pnl = math.isclose(long_row[pnl_idx], short_row[pnl_idx], rel_tol=1e-9)

    should_l, reason_l, prob_l = mgr.evaluate_trade(long_state)
    should_s, reason_s, prob_s = mgr.evaluate_trade(short_state)
    # With a fitted model, probs should match when abs features match and
    # exit_z is mirrored into the same |z| geometry for the toy scaler.
    return {
        "abs_entry_z_equal": ok_abs,
        "pnl_proxy_equal": ok_pnl,
        "long": (should_l, reason_l, prob_l),
        "short": (should_s, reason_s, prob_s),
        "prob_delta": abs(prob_l - prob_s),
    }


def time_stop_boundary_check(mgr: StatArbExitManager) -> dict:
    """Confirm max(5, ceil(2.5*hl)) — reject the min(...) mis-spec."""
    cases = []
    for hl, bars, expect_exit in (
        (1.0, 4, False),   # floor 5 → hold
        (1.0, 5, True),    # floor 5 → stop
        (10.0, 24, False), # ceil(25)=25 → hold
        (10.0, 25, True),  # stop
        (10.0, 5, False),  # min(5,25)=5 would wrongly stop; max keeps HOLD
    ):
        stop_at = time_stop_bars(hl)
        state = TradeState(
            trade_id=0, ticker_a="QCOM", ticker_b="AVGO", direction="SHORT_SPREAD",
            bars_held=bars, half_life_bars=hl,
            vol=1.0, pnl_proxy=0.2, entry_z=2.2, confidence=0.7,
            exit_z=1.5, velocity=0.0, half_life=hl / 30.0, pnl_dollars=10.0,
        )
        should, reason, prob = mgr.evaluate_trade(state)
        cases.append({
            "hl": hl, "bars_held": bars, "stop_at": stop_at,
            "expect_exit": expect_exit, "got_exit": should, "reason": reason, "prob": prob,
            "ok": should == expect_exit,
        })
    # Explicitly prove min(...) is wrong for hl=10, bars=5
    min_would_stop = 5 >= min(5, int(np.ceil(2.5 * 10.0)))
    max_holds = not (5 >= time_stop_bars(10.0))
    return {
        "cases": cases,
        "all_ok": all(c["ok"] for c in cases),
        "min_formula_would_stop_at_5_for_hl10": min_would_stop,
        "max_formula_holds_at_5_for_hl10": max_holds,
    }


def replay_journal(path: Path, mgr: StatArbExitManager) -> pd.DataFrame:
    """Map historical journal rows → evaluate_trade dry-run."""
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        entry_z = float(r.get("entry_z", 0.0) or 0.0)
        exit_z_raw = r.get("exit_z", np.nan)
        exit_z = float(exit_z_raw) if pd.notna(exit_z_raw) else entry_z
        bars = int(r.get("bars_held", 0) or 0)
        # Prefer feat_half_life (normalized) → recover bars; else default 20
        feat_hl = r.get("feat_half_life", np.nan)
        if pd.notna(feat_hl) and float(feat_hl) > 0:
            hl_bars = float(feat_hl) * 30.0
            hl_feat = float(feat_hl)
        else:
            hl_bars = 20.0
            hl_feat = 20.0 / 30.0
        direction = str(r.get("direction", "LONG_SPREAD"))
        vol = float(r.get("feat_vol", 1.0) or 1.0) if pd.notna(r.get("feat_vol", 1.0)) else 1.0
        conf = float(r.get("feat_confidence", 0.7) or 0.7) if pd.notna(r.get("feat_confidence", 0.7)) else 0.7
        vel = float(r.get("feat_velocity", 0.0) or 0.0) if pd.notna(r.get("feat_velocity", 0.0)) else 0.0
        pnl_raw = r.get("pnl_z", r.get("feat_pnl_proxy", 0.0))
        pnl_proxy = float(pnl_raw) if pd.notna(pnl_raw) else 0.0
        # OPEN rows: use entry as current z
        if str(r.get("status", "CLOSED")).upper() == "OPEN":
            exit_z = entry_z
            bars = max(bars, 1)
            pnl_proxy = 0.0
        # Sanitize
        for name, val in (("vol", vol), ("conf", conf), ("vel", vel), ("pnl", pnl_proxy),
                          ("entry_z", entry_z), ("exit_z", exit_z)):
            if not np.isfinite(val):
                raise ValueError(f"non-finite {name} on row {i}")

        vec = extract_feature_vector(
            entry_z=entry_z, current_z=exit_z,
            direction=1 if "LONG" in direction.upper() else -1,
            vol=vol, confidence=conf, velocity=vel,
            bars_held=bars, half_life=hl_bars,
        )
        finite = _finite(vec)

        state = TradeState(
            trade_id=int(r.get("trade_id", i) or 0),
            ticker_a=str(r.get("ticker_a", "")),
            ticker_b=str(r.get("ticker_b", "")),
            direction=direction,
            bars_held=bars,
            half_life_bars=hl_bars,
            vol=vol,
            pnl_proxy=pnl_proxy,
            entry_z=entry_z,
            confidence=conf,
            exit_z=exit_z,
            velocity=vel,
            half_life=hl_feat,
            pnl_dollars=float(r.get("pnl_dollars", 0.0) or 0.0),
            cost_dollars=float(r.get("cost_dollars", 0.0) or 0.0),
        )
        should, reason, prob = mgr.evaluate_trade(state)
        decision = "EXIT" if should else "HOLD"
        # Parse trigger family
        if reason.startswith("TIME_STOP"):
            family = "TIME_STOP"
        elif reason.startswith("STOP_LOSS") or reason.startswith("HARD_PNL"):
            family = "STOP"
        elif reason.startswith("ML_EXIT"):
            family = "ML_EXIT"
        else:
            family = "HOLD"

        print(
            f"  [{r.get('ticker_a')}/{r.get('ticker_b')}] bars={bars} hl={hl_bars:.1f} "
            f"status={r.get('status')} → {decision} prob_exit={prob:.4f} | {reason}"
        )
        rows.append({
            "pair": f"{r.get('ticker_a')}/{r.get('ticker_b')}",
            "broker": r.get("broker"),
            "status": r.get("status"),
            "bars_held": bars,
            "half_life_bars": hl_bars,
            "time_stop_at": time_stop_bars(hl_bars),
            "finite_features": finite,
            "should_exit": should,
            "decision": decision,
            "family": family,
            "prob_exit": prob,
            "reason": reason,
            "held_past_5": bars > 5,
            "would_violate_max_stop": bars >= time_stop_bars(hl_bars) and not should
                if family == "HOLD" else False,
        })
    return pd.DataFrame(rows)


def qcom_avgo_telemetry(mgr: StatArbExitManager) -> None:
    """Simulate bar-close telemetry for the active QCOM/AVGO short."""
    print("\n=== 3) QCOM/AVGO short-spread telemetry (simulated bar close) ===")
    # Adopt-style OPEN from journal if present
    entry_z = 2.2
    for bars, z in ((1, 2.1), (2, 1.8), (3, 1.2), (5, 0.9), (25, 0.4)):
        vec = extract_feature_vector(
            entry_z=entry_z, current_z=z, direction=-1,
            vol=1.1, confidence=0.72, velocity=-0.05,
            bars_held=bars, half_life=20.0,
        )
        assert _finite(vec), "NaN/Inf in feature vector"
        if mgr.scaler is not None:
            scaled = transform_features(mgr.scaler, vec)
            assert _finite(scaled), "NaN/Inf after scaler"
        state = TradeState(
            trade_id=1, ticker_a="QCOM", ticker_b="AVGO", direction="SHORT_SPREAD",
            bars_held=bars, half_life_bars=20.0,
            vol=1.1, pnl_proxy=float(entry_z - z), entry_z=entry_z,
            confidence=0.72, exit_z=z, velocity=-0.05, half_life=20.0 / 30.0,
            pnl_dollars=40.0,
        )
        should, reason, prob = mgr.evaluate_trade(state)
        tag = "EXIT" if should else "HOLD"
        print(
            f"  bar_close bars_held={bars:2d} z={z:+.2f} "
            f"prob_exit={prob:.4f} decision={tag} | {reason}"
        )


def main() -> int:
    print("=== 1) Unit & boundary tests (pytest) — already run separately ===")
    print("=== 2) Historical log replay through evaluate_trade() ===\n")

    # Toy model so ML path is exercisable even without promoted artifacts
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = (X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)
    y[0], y[1] = 0, 1
    scaler = fit_scaler(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    mgr = StatArbExitManager(
        model=model, scaler=scaler, exit_threshold=0.68,
        absolute_min_bars=5, max_half_life_multiplier=2.5,
    )

    sym = symmetry_check(mgr)
    print("Symmetry (|z_entry|):")
    print(f"  abs_entry_z equal: {sym['abs_entry_z_equal']}")
    print(f"  pnl_proxy equal:   {sym['pnl_proxy_equal']}")
    print(f"  long  → {sym['long'][1]}  (prob={sym['long'][2]:.4f})")
    print(f"  short → {sym['short'][1]}  (prob={sym['short'][2]:.4f})")
    print(f"  |Δprob|: {sym['prob_delta']:.6f}")

    ts = time_stop_boundary_check(mgr)
    print("\nTime-stop boundaries (max, not min):")
    for c in ts["cases"]:
        mark = "OK" if c["ok"] else "FAIL"
        print(
            f"  [{mark}] hl={c['hl']} bars={c['bars_held']} stop_at={c['stop_at']} "
            f"expect_exit={c['expect_exit']} got={c['got_exit']} | {c['reason']}"
        )
    print(
        f"  min(5, 2.5*10) would stop at bars=5? {ts['min_formula_would_stop_at_5_for_hl10']}"
    )
    print(
        f"  max(5, 2.5*10) holds at bars=5?     {ts['max_formula_holds_at_5_for_hl10']}"
    )
    print(
        "  NOTE: positions MAY be held past 5 bars when half_life is large "
        "(policy is max floor, not a 5-bar hard cap)."
    )

    journal = ROOT / "results" / "paper_trades.csv"
    if journal.exists():
        print(f"\nReplaying {journal} …")
        out = replay_journal(journal, mgr)
        n_past_5 = int(out["held_past_5"].sum()) if len(out) else 0
        n_bad = int((~out["finite_features"]).sum()) if len(out) else 0
        print(f"\n  rows={len(out)}  non-finite_features={n_bad}  rows_with_bars>5={n_past_5}")
        print(f"  exit families: {out['family'].value_counts().to_dict() if len(out) else {}}")
        out_path = Path("/opt/cursor/artifacts/exit_replay_summary.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
    else:
        print("\nNo results/paper_trades.csv — skip journal replay.")

    qcom_avgo_telemetry(mgr)

    ok = (
        sym["abs_entry_z_equal"]
        and sym["pnl_proxy_equal"]
        and ts["all_ok"]
        and ts["max_formula_holds_at_5_for_hl10"]
    )
    print("\n=== VERDICT ===")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
