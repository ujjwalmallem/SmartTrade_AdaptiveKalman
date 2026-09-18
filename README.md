# SmartTrade_AdaptiveKalman

Paper trading + ML exit trainer for **Mag7 / semis / memory / hyperscaler** pairs.

## Run locally

```bash
pip install -r requirements.txt
python paper_trading_ml_exit.py --mode live --broker alpaca --save-journal alpaca   # real Alpaca paper journal
python paper_trading_ml_exit.py --mode backtest --broker sim --save-journal all     # optional local sim history
python paper_trading_ml_exit.py --mode research                # counterfactual setups → results/setups_*.csv
python paper_trading_ml_exit.py --mode research --data-window latest_year
python paper_trading_ml_exit.py --data-source auto             # Alpaca primary, yfinance backup
python paper_trading_ml_exit.py --data-source yfinance         # force Yahoo backup
python paper_trading_ml_exit.py --noise-model volume
python paper_trading_ml_exit.py --train-only
# SYSTEM_SPEC sklearn exit model (writes models/*.pkl)
PYTHONPATH=. python -m src.train_exit_model --results-dir results
```

**Exit stack (see `SYSTEM_SPEC.md`):** Kalman state → `StatArbExitManager` (hard time-stop / stop-loss / ML ≥ 0.68 from `config/strategy_config.yaml`). Sklearn artifacts under `models/*.pkl`. Live/backtest/research **require** the manager (fail-closed). Soft MR only when sklearn weights are missing. Legacy JSON logistic lives in `src/legacy_logistic.py` for old artifacts/tests only. Training requires ≥50 labeled rows by default.

**Modes:**
- `live` — latest-bar Alpaca paper orders only (preferred path for real paper fills)
- `backtest` — historical replay; use `--save-journal all|sim` if you want simulator rows in the CSV
- `research` — simulate every valid z-crossing to completion into `results/setups_*.csv` (no broker, does not touch the live journal)

**Journal (`--save-journal`):** default **`alpaca`** — `paper_trades.csv` keeps Alpaca paper fills only (legacy sim rows are stripped on save). Use `all` / `sim` for local experiments, or `none` to skip the journal write.

**Brokerage:** default is a local simulator. With `--broker alpaca`, entry/exit on the **latest bar only** are submitted to the Alpaca **paper** API (never live).

### GitHub Secrets (recommended for CI)

1. Open the repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Add exactly these names (paper keys only — never live keys):
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET_KEY`
3. Live mode is **idempotent**: if QCOM/AVGO (or any pair) is already open on Alpaca, the next run adopts it for exit management and will **not** stack duplicate entries.
4. Fresh live entries **hold overnight** (no same-bar exit) so Alpaca is not asked to reverse a just-submitted pair (wash-trade). Failed exits leave the journal **OPEN**.
5. The CSV journal drops wash CLOSED rows (`entry==exit`, `pnl_z≈0`) and dedupes repeated OPEN adopts for the same pair/entry (keeps one live OPEN, or a real CLOSED when present). Year filter for training keeps `entry` in the latest year without requiring `exit` in-year.

CI (with GitHub Secrets) runs **live Alpaca only**:
   - `--mode live --broker alpaca --save-journal alpaca` → places paper orders if today's latest bar has a signal; journal is real paper fills only (no sim backtest step)

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
- `exit_training_dataset.csv` (SYSTEM_SPEC 8-feature set + labels)
- `models/logistic_exit_model.pkl` + `models/feature_scaler.pkl` (sklearn exit path)
- `logistic_exit_model.json`

## Scheduler: Cursor Automations (primary)

Use **Cursor Automations** for market-hour runs (more controllable than GitHub `schedule`).

Setup guide + paste-ready prompt: [`automations/README.md`](automations/README.md)

Create 6 weekday automations (pre / RTH / post), cron in UTC, prompt from [`automations/paper-trading.prompt.md`](automations/paper-trading.prompt.md).

## Backup: GitHub Actions

`.github/workflows/paper-trading.yml` runs on push/PR and external `workflow_dispatch`, uploads `results/` as artifacts, and uses GitHub Secrets for Alpaca paper when configured (see above).
