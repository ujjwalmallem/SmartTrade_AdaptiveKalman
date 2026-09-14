# Cloudflare scheduler for paper-trading.yml

A small Cloudflare Worker whose Cron Trigger calls the GitHub API to fire
the repo's `paper-trading.yml` workflow (`workflow_dispatch`), instead of
(or in addition to) relying on GitHub Actions' own `schedule:` trigger.

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

## Avoiding double runs

`paper-trading.yml` still has its own `schedule:` cron block. If this
Worker's cron fires at the same times, the workflow will run twice per
slot. Either:

- stagger the two schedules, or
- delete/comment out the `schedule:` block in `.github/workflows/paper-trading.yml`
  once you're confident the Cloudflare-driven trigger is reliable.
