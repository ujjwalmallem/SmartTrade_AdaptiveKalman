# SmartTrade_AdaptiveKalman

Paper trading + ML exit trainer for **Mag7 / semis / memory / hyperscaler** pairs.

## Run locally

```bash
pip install -r requirements.txt
python paper_trading_ml_exit.py
python paper_trading_ml_exit.py --train-only   # retrain from results/
```

Artifacts land in `results/` (gitignored):
- `paper_trades.csv`
- `exit_training_dataset.csv`
- `logistic_exit_model.json`

## Scheduler: Cursor Automations (primary)

Use **Cursor Automations** for market-hour runs (more controllable than GitHub `schedule`).

Setup guide + paste-ready prompt: [`automations/README.md`](automations/README.md)

Create 6 weekday automations during regular hours (10:05–15:05 ET), cron in UTC, prompt from [`automations/paper-trading.prompt.md`](automations/paper-trading.prompt.md).

## Backup: GitHub Actions

`.github/workflows/paper-trading.yml` still runs on push/PR and a soft UTC cron, and uploads `results/` as artifacts.
