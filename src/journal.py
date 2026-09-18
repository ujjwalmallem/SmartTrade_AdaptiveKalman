"""Journal wash filters, dedupe, and broker-scope helpers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def trade_fingerprint_cols() -> List[str]:
    return ["ticker_a", "ticker_b", "direction", "entry_time", "exit_time", "broker"]


def is_wash_closed_row(df: pd.DataFrame) -> pd.Series:
    """Same-bar CLOSED with zero pnl_z (failed/instant reverse, not a real round-trip)."""
    if df.empty:
        return pd.Series(dtype=bool)
    status = (
        df["status"].astype(str).str.upper()
        if "status" in df.columns
        else pd.Series([""] * len(df))
    )
    entry = pd.to_datetime(df["entry_time"], format="mixed", errors="coerce")
    exit_ = pd.to_datetime(df["exit_time"], format="mixed", errors="coerce")
    pnl = (
        pd.to_numeric(df["pnl_z"], errors="coerce")
        if "pnl_z" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    same_bar = entry.notna() & exit_.notna() & (entry == exit_)
    zero_pnl = pnl.fillna(0.0).abs() < 1e-12
    return status.eq("CLOSED") & same_bar & zero_pnl


def dedupe_journal_rows(
    journal: pd.DataFrame,
    drop_wash: bool = True,
) -> pd.DataFrame:
    """
    Deduplicate journal rows for sim *and* alpaca.

    - drop_wash=True: remove CLOSED wash trades (entry_time == exit_time, pnl_z ≈ 0)
      so a later OPEN for the same exposure is kept (matches brokerage).
    - For a real close (entry != exit), CLOSED outranks OPEN on the same pair/entry.
    - Repeated sim backtest fingerprints collapse to one row.
    """
    if journal is None or journal.empty:
        return journal
    df = journal.copy()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    if drop_wash and {"entry_time", "exit_time", "status"}.issubset(df.columns):
        wash = is_wash_closed_row(df)
        n_wash = int(wash.sum())
        if n_wash:
            print(f"⚠️  Dropped {n_wash} wash CLOSED rows (entry==exit, pnl_z≈0)")
            df = df.loc[~wash].copy()

    fp = [c for c in trade_fingerprint_cols() if c in df.columns]
    if fp:
        df = df.drop_duplicates(subset=fp, keep="last")

    key = [c for c in ("ticker_a", "ticker_b", "entry_time") if c in df.columns]
    if len(key) == 3 and "status" in df.columns:
        status = df["status"].astype(str).str.upper()
        df = df.assign(_rank=np.where(status.eq("CLOSED"), 1, 0))
        sort_cols = key + ["_rank"]
        if "run_id" in df.columns:
            sort_cols.append("run_id")
        df = df.sort_values(sort_cols, kind="mergesort")
        df = df.drop_duplicates(subset=key, keep="last").drop(columns=["_rank"])

    sort_cols = [c for c in ("entry_time", "run_id", "trade_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def filter_journal_by_scope(journal: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    """
    scope:
      all     — keep every broker
      alpaca  — keep broker starting with 'alpaca' only (real paper fills)
      sim     — keep local simulator rows only
      none    — empty frame (caller should skip writing)
    """
    scope = (scope or "all").lower().strip()
    if journal is None or journal.empty:
        return journal if journal is not None else pd.DataFrame()
    if scope in ("none", "off", "skip"):
        return journal.iloc[0:0].copy()
    if scope == "all":
        return journal
    broker = (
        journal["broker"].astype(str)
        if "broker" in journal.columns
        else pd.Series([""] * len(journal))
    )
    if scope == "alpaca":
        return journal.loc[broker.str.lower().str.startswith("alpaca")].copy()
    if scope == "sim":
        return journal.loc[broker.str.lower().isin(["sim", "none", "local", ""])].copy()
    raise ValueError(f"Unknown journal scope '{scope}'. Use all|alpaca|sim|none.")
