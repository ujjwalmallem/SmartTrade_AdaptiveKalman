# Accumulated paper-trading results

This orphan branch stores CI/session artifacts only (not application code).

Contents:
- `paper_trades.csv` — Alpaca live journal (grows slowly; one bar/day)
- `exit_training_dataset.csv` — path-label harvest for ML training
- `model_metadata.json` — last train metrics / coefficients
- `logistic_exit_model.pkl` / `feature_scaler.pkl` — sklearn weights restored into `models/` before live

Path harvest (`alpaca_live_path_harvest`) is the training SSOT until the live journal has enough closed fills.

Alpaca paper account is the source of truth for open positions.
