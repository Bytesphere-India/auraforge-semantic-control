/**
 * Outbound forwarding for the Jev shadow tap.
 *
 * The tap is a transparent proxy: the caller's body is forwarded byte-for-byte
 * and the upstream status/headers/body are returned unchanged. Caller headers
 * (trace ids, idempotency keys, accept, ...) are forwarded too, except hop-by-hop
 * and transport-controlled ones; the tap keeps the caller's `Content-Type` and
 * only defaults `Accept` to `application/json` when the caller sent none. The
 * one header it injects is the host-side `Authorization`, and a caller-supplied
 * `Authorization` is never forwarded.
 *
 * The transport is injectable so the whole tap is testable against a local fake
 * upstream with no external network.
 */

/** A lower-cased header map safe to forward. */
export type HeaderMap = Record<string, string>;

/**
 * Request headers never forwarded: hop-by-hop headers, the ones the transport
 * sets itself (`host`, `content-length`, `content-type`, `accept`,
 * `authorization`), the internal `x-caller` bookkeeping header, and
 * `accept-encoding` (the response body is decoded before we re-encode it, so we
 * must not let the upstream compress).
 */
const REQUEST_SKIP = new Set([
  "host",
  "connection",
  "keep-alive",
  "proxy-connection",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "content-length",
  "authorization",
  "x-caller",
  "accept-encoding",
  "expect",
]);

/** Response headers never forwarded: hop-by-hop plus framing ones we re-derive. */
const RESPONSE_SKIP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "content-encoding",
  "content-length",
]);

/** Keep the caller's headers that may be forwarded unchanged. */
export function forwardableRequestHeaders(headers: Record<string, string | string[] | undefined> | undefined): HeaderMap {
  const out: HeaderMap = {};
  if (!headers) return out;
  for (const [key, value] of Object.entries(headers)) {
    const name = key.toLowerCase();
    if (REQUEST_SKIP.has(name) || value === undefined) continue;
    out[name] = Array.isArray(value) ? value.join(", ") : value;
  }
  return out;
}

/** Keep the upstream response headers that may be returned unchanged. */
export function forwardableResponseHeaders(headers: Headers): HeaderMap {
  const out: HeaderMap = {};
  headers.forEach((value, key) => {
    const name = key.toLowerCase();
    if (!RESPONSE_SKIP.has(name)) out[name] = value;
  });
  return out;
}

/**
 * Persist only the media type of a `content-type` value. MIME parameters are
 * caller-controlled and can smuggle credentials (e.g.
 * `application/json; api_key=...`), so they are dropped; anything that is not a
 * clean `type/subtype` falls back to `application/json`.
 */
export function mediaType(value: string): string {
  const bare = (value.split(";")[0] ?? "").trim().toLowerCase();
  return /^[-a-z0-9!#$&^_.+]+\/[-a-z0-9!#$&^_.+]+$/.test(bare) ? bare : "application/json";
}

export interface JevForwardInput {
  url: string;
  /** exact request body bytes to forward */
  body: string;
  contentType: string;
  /** caller headers already filtered by `forwardableRequestHeaders` */
  headers: HeaderMap;
  /** host-side key; null when unavailable (the call is then not authenticated) */
  apiKey: string | null;
  timeoutMs: number;
}

export interface JevForwardResult {
  status: number;
  contentType: string;
  /** upstream response headers already filtered by `forwardableResponseHeaders` */
  headers: HeaderMap;
  body: string;
  latencyMs: number;
  /** short code when the transport itself failed (not an upstream HTTP status) */
  error: string | null;
}

export type JevTransport = (input: JevForwardInput) => Promise<JevForwardResult>;

/**
 * Parse and minimally validate an upstream URL: http(s) only, a non-empty
 * hostname, and no embedded credentials. Address-class policy (loopback /
 * link-local / metadata ranges) lives in `config.ts`, the service's env gate.
 */
export function parseJevUpstream(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`JEV upstream is not a valid URL: ${JSON.stringify(raw)}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`JEV upstream must use http or https, got ${JSON.stringify(url.protocol)}`);
  }
  if (url.hostname.length === 0) throw new Error("JEV upstream must have a hostname");
  if (url.username.length > 0 || url.password.length > 0) {
    throw new Error("JEV upstream must not embed credentials");
  }
  return url;
}

/** Default transport: Node's global fetch with an abort-based timeout. */
export const fetchJevTransport: JevTransport = async (input) => {
  const started = performance.now();
  const headers: Record<string, string> = {
    ...input.headers,
    "content-type": input.contentType || "application/json",
  };
  if (!("accept" in headers)) headers.accept = "application/json";
  if (input.apiKey) headers.authorization = `Bearer ${input.apiKey}`;

  try {
    const response = await fetch(input.url, {
      method: "POST",
      headers,
      body: input.body,
      // never follow a redirect: a 3xx `Location` could point at loopback or
      // cloud metadata and would bypass the startup address-class gate
      redirect: "manual",
      signal: AbortSignal.timeout(input.timeoutMs),
    });
    const body = await response.text();
    return {
      status: response.status,
      contentType: response.headers.get("content-type") ?? "application/json",
      headers: forwardableResponseHeaders(response.headers),
      body,
      latencyMs: performance.now() - started,
      error: null,
    };
  } catch (error) {
    const timedOut = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    return {
      status: timedOut ? 504 : 502,
      contentType: "application/json; charset=utf-8",
      headers: {},
      body: JSON.stringify({
        error: timedOut ? "jev upstream timeout" : "jev upstream unreachable",
        code: timedOut ? "JEV_UPSTREAM_TIMEOUT" : "JEV_UPSTREAM_ERROR",
      }),
      latencyMs: performance.now() - started,
      error: timedOut ? "JEV_UPSTREAM_TIMEOUT" : "JEV_UPSTREAM_ERROR",
    };
  }
};
