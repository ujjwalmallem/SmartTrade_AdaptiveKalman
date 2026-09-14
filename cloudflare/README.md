# Cloudflare Cron → GitHub Actions

Primary market-hours scheduler: a **Cloudflare Worker** with cron triggers that
calls GitHub `workflow_dispatch` on `.github/workflows/paper-trading.yml`.

## Why this

- Cloudflare cron is more reliable than GitHub’s native `schedule`
- Keeps the actual work in GitHub Actions (Python + artifacts)
- Easy weekday / extended-hours timing (UTC)

## One-time setup

### 1. GitHub token

Create a **fine-grained PAT** (or classic with `repo` + `workflow`) with:

- Repository access: this repo
- Permissions: **Actions: Read and write**, **Contents: Read**

### 2. Enable workflow_dispatch

Already enabled on `paper-trading.yml`. Confirm in the Actions UI that
**Run workflow** appears.

### 3. Install & configure Worker

```bash
cd cloudflare
npm install
npx wrangler login
```

Edit `wrangler.toml` `[vars]` if owner/repo/ref differ.

```bash
npm run secret:github          # paste the PAT when prompted
npm run secret:dispatch        # optional: secret for manual POST /run
npm run deploy
```

### 4. Verify

- Worker logs: `npm run tail`
- Manual trigger (optional):
  ```bash
  curl -X POST "https://<worker>.workers.dev/run" \
    -H "Authorization: Bearer $DISPATCH_SECRET"
  ```
- Or in GitHub: Actions → Paper Trading CI → Run workflow

## Schedule (UTC ≈ ET under EDT)

| Cron (UTC) | Approx ET | Session |
|---|---|---|
| `5 8 * * 1-5` | 4:05 | Pre |
| `5 11 * * 1-5` | 7:05 | Pre |
| `5 14 * * 1-5` | 10:05 | RTH |
| `5 17 * * 1-5` | 13:05 | RTH |
| `5 20 * * 1-5` | 16:05 | Post |
| `5 23 * * 1-5` | 19:05 | Post |

Cloudflare crons are UTC-only. Shift +1 hour UTC in winter (EST) if you need exact ET.

## Files

- `src/index.js` — cron + optional `/run` HTTP dispatch
- `wrangler.toml` — crons and repo vars
- `package.json` — wrangler scripts
