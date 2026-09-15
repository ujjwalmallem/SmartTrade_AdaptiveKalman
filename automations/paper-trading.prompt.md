# Paper trading + ML exit (scheduled run)

You are running a scheduled paper-trading session for Mag7, semis, memory, and hyperscaler pairs.

## Steps

1. Confirm you are on the latest `main` (or the automation’s configured branch).
2. Install deps if needed: `pip install -r requirements.txt`
3. Run the trainer (yfinance only — no synthetic prices):
   ```bash
   python paper_trading_ml_exit.py
   ```
4. Verify artifacts exist under `results/`:
   - `paper_trades.csv`
   - `exit_training_dataset.csv`
   - `logistic_exit_model.json`
5. Optionally retrain from accumulated history:
   ```bash
   python paper_trading_ml_exit.py --train-only
   ```
6. Summarize in your reply:
   - Number of closed trades this run
   - Baskets covered (mag7 / semis / memory / hyperscaler)
   - Win rate / avg PnL(z) if available
   - Whether model training succeeded
   - Confirm `data_source` is `yfinance_live` (required for training)
   - Any errors or empty-result conditions

## Rules

- Do **not** open a PR unless the run fails and you made a real code fix.
- Prices must come from **yfinance only**. The script has no synthetic fallback — if the fetch fails, the run must fail. Never invent prices or fills.
- If fewer than 2 trades are generated, report that clearly and stop (no fake training).
- Keep the reply short and operational.
