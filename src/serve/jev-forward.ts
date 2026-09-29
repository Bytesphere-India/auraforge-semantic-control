/**
 * Outbound forwarding for the Jev shadow tap.
 *
 * The tap is a transparent proxy: the caller's body is forwarded byte-for-byte
 * and the upstream status/body are returned unchanged. The only injected header
 * is the host-side `Authorization`; a caller-supplied one is never forwarded.
 *
 * The transport is injectable so the whole tap is testable against a local fake
 * upstream with no external network.
 */
export interface JevForwardInput {
  url: string;
  /** exact request body bytes to forward */
  body: string;
  contentType: string;
  /** host-side key; null when unavailable (the call is then not authenticated) */
  apiKey: string | null;
  timeoutMs: number;
}

export interface JevForwardResult {
  status: number;
  contentType: string;
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
    "content-type": input.contentType || "application/json",
    accept: "application/json",
  };
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
      body,
      latencyMs: performance.now() - started,
      error: null,
    };
  } catch (error) {
    const timedOut = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    return {
      status: timedOut ? 504 : 502,
      contentType: "application/json; charset=utf-8",
      body: JSON.stringify({
        error: timedOut ? "jev upstream timeout" : "jev upstream unreachable",
        code: timedOut ? "JEV_UPSTREAM_TIMEOUT" : "JEV_UPSTREAM_ERROR",
      }),
      latencyMs: performance.now() - started,
      error: timedOut ? "JEV_UPSTREAM_TIMEOUT" : "JEV_UPSTREAM_ERROR",
    };
  }
};
