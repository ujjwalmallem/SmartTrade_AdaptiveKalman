# Cursor Automations — Paper Trading

Primary scheduler for Mag7 / semis / memory / hyperscaler paper trading + ML exit training.

Create these at: https://cursor.com/automations

## Quick setup

1. Open **Automations** → **New**
2. **Trigger:** Scheduled → Custom cron (UTC)
3. **Repository:** attach this repo
4. **Prompt:** paste from [`paper-trading.prompt.md`](./paper-trading.prompt.md)
5. Save / activate
6. Repeat for each cron below (6 weekday runs)

## Schedule (US regular market hours only)

NYSE/Nasdaq RTH is **9:30–16:00 ET**, weekdays. No pre-market or after-hours runs.

Times are Eastern. Cron is UTC for **EDT (UTC−4)**. In **EST (UTC−5)** add +1 hour to each UTC value.

| Session | ET | UTC (EDT) | Cron |
|---|---|---|---|
| Open | 10:05 | 14:05 | `5 14 * * 1-5` |
| Mid-morning | 11:05 | 15:05 | `5 15 * * 1-5` |
| Midday | 12:05 | 16:05 | `5 16 * * 1-5` |
| Early afternoon | 13:05 | 17:05 | `5 17 * * 1-5` |
| Mid afternoon | 14:05 | 18:05 | `5 18 * * 1-5` |
| Before close | 15:05 | 19:05 | `5 19 * * 1-5` |

One-automation alternative: `5 14,15,16,17,18,19 * * 1-5`

Suggested automation names:
- `paper-trading-rth-1005`
- `paper-trading-rth-1105`
- `paper-trading-rth-1205`
- `paper-trading-rth-1305`
- `paper-trading-rth-1405`
- `paper-trading-rth-1505`

## Notes

- Automations can run **late**, never early
- Weekends skipped; exchange holidays are not skipped automatically
- Each run uses Cloud Agent quota
- GitHub Action `schedule` can stay as a soft backup; Automations is the primary clock
- Do **not** open a PR on every run — only when the prompt finds a real failure worth fixing
