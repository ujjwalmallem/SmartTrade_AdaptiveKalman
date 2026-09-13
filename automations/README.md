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

## Schedule (US extended hours)

Times are Eastern. Cron is UTC for **EDT (UTC−4)**. In **EST (UTC−5)** add +1 hour to each UTC value.

| Session | ET | UTC (EDT) | Cron |
|---|---|---|---|
| Pre-market | 4:05 | 08:05 | `5 8 * * 1-5` |
| Pre-market | 7:05 | 11:05 | `5 11 * * 1-5` |
| Regular | 10:05 | 14:05 | `5 14 * * 1-5` |
| Regular | 13:05 | 17:05 | `5 17 * * 1-5` |
| Post-market | 16:05 | 20:05 | `5 20 * * 1-5` |
| Post-market | 19:05 | 23:05 | `5 23 * * 1-5` |

Suggested automation names:
- `paper-trading-pre-0405`
- `paper-trading-pre-0705`
- `paper-trading-rth-1005`
- `paper-trading-rth-1305`
- `paper-trading-post-1605`
- `paper-trading-post-1905`

## Notes

- Automations can run **late**, never early
- Weekends skipped; exchange holidays are not skipped automatically
- Each run uses Cloud Agent quota
- GitHub Action `schedule` can stay as a soft backup; Automations is the primary clock
- Do **not** open a PR on every run — only when the prompt finds a real failure worth fixing
