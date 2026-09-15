# SmartTrade_AdaptiveKalman

Paper trading + ML exit trainer for **Mag7 / semis / memory / hyperscaler** pairs.

## Run locally

```bash
pip install -r requirements.txt
python paper_trading_ml_exit.py
python paper_trading_ml_exit.py --noise-model volume      # volume-adjusted R
python paper_trading_ml_exit.py --noise-model parkinson   # Parkinson high-low R
python paper_trading_ml_exit.py --train-only   # retrain from results/
```

**Data:** OHLCV from **yfinance only** (no synthetic). Close is always used; High/Low/Volume enable adaptive Kalman R modes. Trade windows and training samples are restricted to the **latest calendar year** (e.g. 2026) — prior years like 2025 are excluded.

**Kalman R modes** (`--noise-model`):
- `standard` — fixed (price-scale calibrated) measurement noise
- `volume` — lower R when relative volume is high
- `parkinson` — higher R when the high–low range is wide

Artifacts land in `results/` (gitignored):
- `paper_trades.csv`
- `exit_training_dataset.csv`
- `logistic_exit_model.json`

## Scheduler: Cursor Automations (primary)

Use **Cursor Automations** for market-hour runs (more controllable than GitHub `schedule`).

Setup guide + paste-ready prompt: [`automations/README.md`](automations/README.md)

Create 6 weekday automations (pre / RTH / post), cron in UTC, prompt from [`automations/paper-trading.prompt.md`](automations/paper-trading.prompt.md).

## Backup: GitHub Actions

`.github/workflows/paper-trading.yml` still runs on push/PR and external `workflow_dispatch`, and uploads `results/` as artifacts.
