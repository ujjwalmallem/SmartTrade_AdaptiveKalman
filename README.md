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

## Scheduler: Cloudflare Cron → GitHub Actions (primary)

Use a **Cloudflare Worker** cron to dispatch the GitHub Action on US extended-hours weekdays.

Full setup: [`cloudflare/README.md`](cloudflare/README.md)

```bash
cd cloudflare
npm install
npx wrangler login
npm run secret:github    # fine-grained PAT with Actions write
npm run deploy
```

6 weekday crons fire `workflow_dispatch` on `.github/workflows/paper-trading.yml`.

## Other options

- **Cursor Automations** (Cloud Agent runs the script): [`automations/README.md`](automations/README.md)
- **GitHub `schedule`**: kept as a soft weekday backup only
