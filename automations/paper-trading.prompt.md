# Paper trading + ML exit (scheduled run)

You are running a scheduled paper-trading session for Mag7, semis, memory, and hyperscaler pairs.

## Steps

1. Confirm you are on the latest `main` (or the automation’s configured branch).
2. Install deps if needed: `pip install -r requirements.txt`
3. Run **live Alpaca paper** (real brokerage fills → journal). Requires
   `ALPACA_API_KEY` + `ALPACA_API_SECRET_KEY` secrets:
   ```bash
   python paper_trading_ml_exit.py --mode live --broker alpaca --data-source auto --save-journal alpaca --min-trades 1
   ```
   Do **not** use `--broker sim` or `--save-journal all` on scheduled runs — the journal must stay Alpaca-only.
4. Verify artifacts exist under `results/` (journal may be OPEN-only until a real close):
   - `paper_trades.csv`
   - `exit_training_dataset.csv` (may be empty until closed Alpaca fills exist)
   - `logistic_exit_model.json` (optional until enough labeled closes)
5. Optionally retrain from accumulated **Alpaca** history:
   ```bash
   python paper_trading_ml_exit.py --train-only
   ```
   If fewer than 2 labeled samples exist, report that clearly (do not invent labels).
6. Summarize in your reply:
   - Number of closed / open Alpaca trades this run
   - Baskets covered (mag7 / semis / memory / hyperscaler)
   - Win rate / avg PnL(z) if available
   - Whether model training succeeded
   - Confirm `data_source` is live (`alpaca_live` or `yfinance_live` backup)
   - Confirm trade entry/exit dates are in the **latest calendar year only** (no 2025 or earlier windows)
   - Confirm journal brokers are Alpaca-only (no `sim` rows)
   - Any errors or empty-result conditions

## Rules

- Do **not** open a PR unless the run fails and you made a real code fix.
- Prefer **Alpaca OHLCV** (`--data-source auto`); yfinance is backup only. Never invent prices or fills.
- Trade windows must be in the **latest year only**; drop/ignore any prior-year (e.g. 2025) results.
- Journal scope is **alpaca** — never append simulator fills on scheduled runs.
- If fewer than 2 closed trades are generated, report that clearly and stop (no fake training).
- Keep the reply short and operational.
