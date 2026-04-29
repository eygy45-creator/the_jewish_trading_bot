import { NextResponse } from "next/server";

/** Server-side proxy: adds Bearer token; never expose token to the browser. */
export async function GET() {
  const base = process.env.TJTB_API_BASE_URL?.replace(/\/$/, "");
  const token = process.env.API_TOKEN;
  if (!base || !token) {
    return NextResponse.json(
      { error: "Missing TJTB_API_BASE_URL or API_TOKEN on server" },
      { status: 500 }
    );
  }

  const url = `${base}/api/status`;
  let res: Response;
  try {
    res = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "Upstream fetch failed" }, { status: 502 });
  }

  const text = await res.text();
  let body: unknown;
  try {
    body = JSON.parse(text) as unknown;
  } catch {
    body = { raw: text };
  }

  return NextResponse.json(body, { status: res.status });
}
