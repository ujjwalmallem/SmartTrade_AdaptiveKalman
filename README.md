# SmartTrade_AdaptiveKalman

Paper trading + ML exit trainer for **Mag7 / semis / memory / hyperscaler** pairs.

## Run locally

```bash
pip install -r requirements.txt
python paper_trading_ml_exit.py                          # local sim journal
python paper_trading_ml_exit.py --broker alpaca          # Alpaca *paper* for latest-bar fills
python paper_trading_ml_exit.py --noise-model volume
python paper_trading_ml_exit.py --train-only
```

**Brokerage:** default is a local simulator. With `--broker alpaca`, entry/exit on the **latest bar only** are submitted to the Alpaca **paper** API (never live). Set:

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET_KEY=...
# (aliases APCA_API_KEY_ID / APCA_API_SECRET_KEY also work)
```

Historical bars still drive Kalman / ML training in-process; only the most recent signal date is routed to Alpaca so replay does not spam orders.

**Data:** per-ticker OHLCV from **yfinance only** (no synthetic). Each ticker panel includes Open/High/Low/Close/Volume; High/Low/Volume feed adaptive Kalman R modes. Trade windows and training samples are restricted to the **latest calendar year** (e.g. 2026) — prior years like 2025 are excluded.

**Kalman R modes** (`--noise-model`):
- `standard` — fixed (price-scale calibrated) measurement noise
- `volume` — lower R when relative volume is high
- `parkinson` — higher R when the high–low range is wide

Artifacts land in `results/` (gitignored):
- `paper_trades.csv` (includes z-PnL, $-PnL after costs, notional)
- `exit_training_dataset.csv` (smarter exit labels + `half_life` feature)
- `logistic_exit_model.json`

## Scheduler: Cursor Automations (primary)

Use **Cursor Automations** for market-hour runs (more controllable than GitHub `schedule`).

Setup guide + paste-ready prompt: [`automations/README.md`](automations/README.md)

Create 6 weekday automations (pre / RTH / post), cron in UTC, prompt from [`automations/paper-trading.prompt.md`](automations/paper-trading.prompt.md).

## Backup: GitHub Actions

`.github/workflows/paper-trading.yml` still runs on push/PR and external `workflow_dispatch`, and uploads `results/` as artifacts.
