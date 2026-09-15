# Accumulated paper-trading results

This orphan branch stores CI/session artifacts only (not application code).

Reset on 2026-09-15 for the Alpaca-primary era:
- Cleared prior yfinance/sim journals (labels were nearly all `1`, many duplicate CI runs)
- Schema headers retained so appends keep working
- Model removed; next successful paper session will retrain

Sources going forward: prefer `alpaca_live` (yfinance backup is fine).
Trade windows: latest calendar year only.
