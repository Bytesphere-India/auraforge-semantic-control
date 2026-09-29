/**
 * Outbound HTTP transport for the Qwen4b Parallel Decision shadow.
 *
 * Qwen4b is an **external local service** (the llama.cpp Parallel Decision
 * server on the GPU, `POST /v1/decision`); `laya-serve` itself stays CPU-only
 * and only ever reads a response here. The transport is injectable so the whole
 * shadow is testable against a fake upstream with no real network.
 *
 * The URL parser deliberately does *not* apply the loopback policy: that is the
 * service's env gate and lives in `config.ts` (which owns `isLoopbackHost`).
 * Keeping the policy out of this module avoids an import cycle.
 */

export interface Qwen4bForwardInput {
  url: string;
  /** exact JSON body to POST */
  body: string;
  timeoutMs: number;
}

export interface Qwen4bForwardResult {
  /** HTTP status, or null when the transport itself failed */
  status: number | null;
  /** raw response body text (may be empty on transport failure) */
  body: string;
  latencyMs: number;
  /** short machine code when the transport failed; never a raw stack trace */
  error: string | null;
}

export type Qwen4bTransport = (input: Qwen4bForwardInput) => Promise<Qwen4bForwardResult>;

/**
 * Parse and minimally validate the Qwen4b endpoint: http(s) only, a non-empty
 * hostname, no embedded credentials and no fragment. Address-class policy
 * (loopback-only) is enforced at startup by `config.ts`.
 */
export function parseQwen4bUrl(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`QWEN4B endpoint is not a valid URL: ${JSON.stringify(raw)}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`QWEN4B endpoint must use http or https, got ${JSON.stringify(url.protocol)}`);
  }
  if (url.hostname.length === 0) throw new Error("QWEN4B endpoint must have a hostname");
  if (url.username.length > 0 || url.password.length > 0) {
    throw new Error("QWEN4B endpoint must not embed credentials");
  }
  if (url.hash.length > 0) throw new Error("QWEN4B endpoint must not carry a URL fragment");
  return url;
}

/** Default transport: Node's global fetch with an abort-based timeout. */
export const fetchQwen4bTransport: Qwen4bTransport = async (input) => {
  const started = performance.now();
  try {
    const response = await fetch(input.url, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: input.body,
      // never follow a redirect: a 3xx `Location` could point at loopback-adjacent
      // or metadata space and would bypass the startup address-class gate
      redirect: "manual",
      signal: AbortSignal.timeout(input.timeoutMs),
    });
    const body = await response.text();
    return { status: response.status, body, latencyMs: performance.now() - started, error: null };
  } catch (error) {
    const timedOut = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    return {
      status: null,
      body: "",
      latencyMs: performance.now() - started,
      error: timedOut ? "QWEN4B_TIMEOUT" : "QWEN4B_UNREACHABLE",
    };
  }
};
