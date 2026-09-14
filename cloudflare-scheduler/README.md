# Cloudflare scheduler for paper-trading.yml

A small Cloudflare Worker whose Cron Trigger calls the GitHub API to fire
the repo's `paper-trading.yml` workflow (`workflow_dispatch`). This is now
the sole scheduler for that workflow — `paper-trading.yml` no longer has
its own `schedule:` block, so nothing runs until this Worker (or a manual
`workflow_dispatch`/push/PR) triggers it.

The Worker does **not** run the Python trading/ML code itself — Workers
don't have a numpy/pandas-capable Python runtime. It only pokes GitHub to
start the job, which still runs on GitHub Actions.

## Setup

1. Create a GitHub token that can dispatch workflows on
   `ujjwalmallem/smarttrade_adaptivekalman`:
   - Fine-grained PAT scoped to this repo, with **Actions: read and write**
     permission, or
   - A classic PAT with the `repo` scope (or `public_repo` if the repo is
     public) and `workflow` scope.

2. Install deps and authenticate wrangler:

   ```bash
   cd cloudflare-scheduler
   npm install
   npx wrangler login
   ```

3. Store the token as a Worker secret (never commit it):

   ```bash
   npx wrangler secret put GITHUB_TOKEN
   ```

4. Adjust `wrangler.toml` if needed:
   - `[triggers].crons` — when the Worker fires (UTC, standard 5-field cron).
   - `[vars]` — owner/repo/workflow file/ref, if they ever change.

5. Deploy:

   ```bash
   npm run deploy
   ```

6. Sanity check without waiting for the cron:

   ```bash
   curl -X POST https://<your-worker-subdomain>.workers.dev/
   ```

   Then check the Actions tab for a new `workflow_dispatch` run.

## Important

Since GitHub Actions' own `schedule:` trigger was removed, **if this
Worker is undeployed, paused, or its `GITHUB_TOKEN` secret expires, no
scheduled runs will happen at all** — only push/PR/manual runs. Keep an
eye on the Worker (e.g. `wrangler tail`, or Cloudflare's dashboard logs)
to confirm it's actually firing on schedule.
