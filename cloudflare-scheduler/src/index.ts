export interface Env {
  GITHUB_TOKEN: string;
  GITHUB_OWNER: string;
  GITHUB_REPO: string;
  GITHUB_WORKFLOW_FILE: string;
  GITHUB_REF: string;
}

export default {
  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(dispatchWorkflow(env));
  },

  // GET for a health check, POST for a manual trigger while testing.
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "POST") {
      try {
        await dispatchWorkflow(env);
        return new Response("Workflow dispatched.\n", { status: 200 });
      } catch (err) {
        return new Response(`Dispatch failed: ${(err as Error).message}\n`, { status: 502 });
      }
    }
    return new Response(
      "SmartTrade paper-trading scheduler is running. POST here to trigger a run manually.\n",
      { status: 200 },
    );
  },
};

async function dispatchWorkflow(env: Env): Promise<void> {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/${env.GITHUB_WORKFLOW_FILE}/dispatches`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "smarttrade-cloudflare-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GITHUB_REF }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GitHub workflow_dispatch failed (${res.status}): ${body}`);
  }
}
