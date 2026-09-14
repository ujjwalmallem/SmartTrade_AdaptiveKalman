/**
 * Cloudflare Worker: cron → GitHub Actions workflow_dispatch
 *
 * Dispatches .github/workflows/paper-trading.yml on US extended-hours
 * weekday crons (UTC approximating Eastern EDT).
 *
 * Required secrets (wrangler secret put ...):
 *   GITHUB_TOKEN  — fine-grained PAT with Actions: Read and write
 *
 * Required vars (wrangler.toml [vars] or dashboard):
 *   GH_OWNER      — e.g. ujjwalmallem
 *   GH_REPO       — e.g. SmartTrade_AdaptiveKalman
 *   WORKFLOW_FILE — paper-trading.yml
 *   GH_REF        — main
 */

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(dispatchWorkflow(env, event.cron));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return json({ ok: true, service: "paper-trading-dispatcher" });
    }
    if (url.pathname === "/run" && request.method === "POST") {
      const auth = request.headers.get("Authorization") || "";
      if (!env.DISPATCH_SECRET || auth !== `Bearer ${env.DISPATCH_SECRET}`) {
        return json({ error: "unauthorized" }, 401);
      }
      const result = await dispatchWorkflow(env, "manual");
      return json(result, result.ok ? 200 : 502);
    }
    return json({ error: "not found" }, 404);
  },
};

async function dispatchWorkflow(env, cronLabel) {
  const owner = env.GH_OWNER;
  const repo = env.GH_REPO;
  const workflow = env.WORKFLOW_FILE || "paper-trading.yml";
  const ref = env.GH_REF || "main";
  const token = env.GITHUB_TOKEN;

  if (!owner || !repo || !token) {
    console.error("Missing GH_OWNER, GH_REPO, or GITHUB_TOKEN");
    return { ok: false, error: "missing_config" };
  }

  const endpoint =
    `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflow}/dispatches`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "cloudflare-paper-trading-dispatcher",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ref,
      inputs: {
        reason: `cloudflare-cron:${cronLabel}`,
      },
    }),
  });

  if (res.status === 204) {
    console.log(`Dispatched ${workflow} on ${owner}/${repo}@${ref} (${cronLabel})`);
    return { ok: true, status: 204, cron: cronLabel, ref };
  }

  const body = await res.text();
  console.error(`GitHub dispatch failed: ${res.status} ${body}`);
  return { ok: false, status: res.status, body, cron: cronLabel };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
