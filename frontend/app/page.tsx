"use client";

import { useEffect, useState } from "react";

type StatusPayload = {
  bot_status?: string;
  latest_update_timestamp?: string | null;
  total_trades?: number;
  latest_balance?: number | string | null;
  error?: string;
};

export default function Home() {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/dashboard/status", { cache: "no-store" });
        const json = (await res.json()) as StatusPayload;
        if (!res.ok) {
          const msg =
            typeof json.error === "string"
              ? json.error
              : `HTTP ${res.status}`;
          setErr(msg);
          setData(json);
        } else if (!cancelled) {
          setData(json);
        }
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Request failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <main>
        <h1>TJTB Live Dashboard</h1>
        <p className="loading">Loading status…</p>
      </main>
    );
  }

  return (
    <main>
      <h1>TJTB Live Dashboard</h1>
      {err ? (
        <p className="error">{err}</p>
      ) : null}
      <dl>
        <div>
          <dt>bot_status</dt>
          <dd>{data?.bot_status ?? "—"}</dd>
        </div>
        <div>
          <dt>latest_update_timestamp</dt>
          <dd>{data?.latest_update_timestamp ?? "—"}</dd>
        </div>
        <div>
          <dt>total_trades</dt>
          <dd>{data?.total_trades ?? "—"}</dd>
        </div>
        <div>
          <dt>latest_balance</dt>
          <dd>{data?.latest_balance ?? "—"}</dd>
        </div>
      </dl>
    </main>
  );
}
