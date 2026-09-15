# Accumulated paper-trading results

This orphan branch stores CI/session artifacts only (not application code).

Reset on 2026-09-15 (post live-idempotent merge):
- Cleared thin/noisy journal (sim bootstrap + duplicate OPEN QCOM/AVGO adopts)
- Cleared training rows (only 5 labels, nearly all `1`) and removed the exit model
- Schema headers retained so CI appends keep working

Alpaca paper account is the source of truth for open positions.
Next successful closed live fills (`broker=alpaca_paper`) will rebuild the ML journal.

Sources: prefer `alpaca_live` (yfinance backup is fine).
Trade windows: latest calendar year only.
