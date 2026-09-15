# SmartTrade_AdaptiveKalman

Paper trading + ML exit trainer for **Mag7 / semis / memory / hyperscaler** pairs.

## Run locally

```bash
pip install -r requirements.txt
python paper_trading_ml_exit.py --mode backtest --broker sim   # replay history → ML journal
python paper_trading_ml_exit.py --mode live --broker alpaca    # fresh data → latest-bar Alpaca paper orders
python paper_trading_ml_exit.py --data-source auto             # Alpaca primary, yfinance backup
python paper_trading_ml_exit.py --data-source yfinance         # force Yahoo backup
python paper_trading_ml_exit.py --noise-model volume
python paper_trading_ml_exit.py --train-only
```

**Brokerage:** default is a local simulator. With `--broker alpaca`, entry/exit on the **latest bar only** are submitted to the Alpaca **paper** API (never live).

### GitHub Secrets (recommended for CI)

1. Open the repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Add exactly these names (paper keys only — never live keys):
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET_KEY`
3. Live mode is **idempotent**: if QCOM/AVGO (or any pair) is already open on Alpaca, the next run adopts it for exit management and will **not** stack duplicate entries.
4. Fresh live entries **hold overnight** (no same-bar exit) so Alpaca is not asked to reverse a just-submitted pair (wash-trade). Failed exits leave the journal **OPEN**.
5. The CSV journal dedupes repeated `broker=sim` backtest replays; look for `broker=alpaca_paper` rows for real paper fills.

CI (with GitHub Secrets) runs two steps when Alpaca keys exist:
   1. `--mode backtest --broker sim` → builds the ML journal from history
   2. `--mode live --broker alpaca` → places paper orders only if today's latest bar has a signal

Do **not** put keys in the repo, `.env` commits, workflow logs, or PR text.

### Local env

```bash
cp .env.example .env   # then edit; .env is gitignored
export ALPACA_API_KEY=...
export ALPACA_API_SECRET_KEY=...
# (aliases APCA_API_KEY_ID / APCA_API_SECRET_KEY also work)
python paper_trading_ml_exit.py --broker alpaca
```

Historical bars still drive Kalman / ML training in-process; only the most recent signal date is routed to Alpaca so replay does not spam orders.

**Data:** per-ticker OHLCV from **Alpaca first** (IEX daily bars), with **yfinance as backup** if Alpaca credentials/data fail. Never synthetic. Each ticker panel includes Open/High/Low/Close/Volume; High/Low/Volume feed adaptive Kalman R modes. Trade windows and training samples are restricted to the **latest calendar year** (e.g. 2026) — prior years like 2025 are excluded.

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

`.github/workflows/paper-trading.yml` runs on push/PR and external `workflow_dispatch`, uploads `results/` as artifacts, and uses GitHub Secrets for Alpaca paper when configured (see above).
