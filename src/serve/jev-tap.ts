/**
 * The Jev shadow tap.
 *
 * A caller POSTs to `/jev/api/alpha/decisions`; the tap
 *
 *   1. forwards the body byte-for-byte to upstream Jev and returns the upstream
 *      status/body unchanged and immediately (only the host-side Authorization
 *      header is added), then
 *   2. runs the same payload through the local Laya engine on a serial queue
 *      (no short timeout) and appends one joined record to
 *      `jev-laya-pairs.jsonl` with the payload, Jev's reply and Laya's reply.
 *
 * Everything that exists only to prepare the shadow (JSON decode, SHA-256,
 * request validation and request-hash computation) runs *after* the response,
 * inside `runShadow`, not on the forward path: the caller's Jev response has no
 * added latency and can never be failed by shadow-prep. A prep failure is caught
 * and recorded as `SHADOW_PREP_ERROR`.
 *
 * Dispatch is a *handoff*: `handle()` returns the response plus a `commit()`
 * callback. The HTTP layer calls `commit()` only from the response `finish`/
 * `close` events, so the shadow is scheduled strictly after the response is on
 * the wire and `drain()` can account for an in-flight handoff during shutdown.
 * A slow or failing Laya can never delay or fail the Jev response.
 *
 * The tap fails closed when no host key is configured: it refuses to forward and
 * returns `503 JEV_KEY_MISSING`. The key is never logged or persisted.
 */
import { createHash, randomUUID } from "node:crypto";
import { redactSecret, sanitizeForStorage, secretFingerprint } from "./jev-key.js";
import { parseJevUpstream, type JevForwardResult, type JevTransport } from "./jev-forward.js";
import { computeRequestHash, validateDecisionRequest, type ServeLimits, type ValidationResult } from "./protocol.js";
import { SerialQueue } from "./serial-queue.js";
import { recordShadowAnswers, type JevLayaPairLaya, type JevLayaPairsLog, type ShadowLog } from "./shadow.js";
import type { DecisionEngine, EngineIdentity } from "./types.js";

export interface JevTapStats {
  enabled: boolean;
  upstream: string;
  key_present: boolean;
  /** non-reversible; safe to expose */
  key_fingerprint: string | null;
  calls: number;
  forwarded_ok: number;
  forwarded_error: number;
  shadows_ok: number;
  shadows_error: number;
  queued: number;
  queue_max: number;
  last_error: string | null;
}

/** What the HTTP layer returns to the caller: the upstream response, unchanged. */
export interface TapResponse {
  status: number;
  contentType: string;
  body: string;
}

/**
 * The response plus the deferred shadow dispatch. `commit()` is idempotent and
 * must be called once the response has finished (or its connection closed).
 */
export interface TapHandoff extends TapResponse {
  commit: () => void;
}

export interface JevTap {
  handle(body: string, caller: string | null, contentType: string): Promise<TapHandoff>;
  stats(): JevTapStats;
  /** resolves when every queued Laya shadow and every uncommitted handoff is settled */
  drain(): Promise<void>;
}

export interface JevTapOptions {
  upstreamUrl: string;
  apiKey: string | null;
  transport: JevTransport;
  timeoutMs: number;
  engine: DecisionEngine;
  identity: EngineIdentity;
  shadow: ShadowLog;
  pairs: JevLayaPairsLog;
  limits: ServeLimits;
  /** max queued Laya shadows; overflow still writes a pair with laya.error QUEUE_FULL */
  queueMax?: number;
  now?: () => number;
  clock?: () => Date;
  newId?: () => string;
  /** shadow-only: decode/validate a body; injectable so tests can prove it can never fail the forward */
  validateRequest?: typeof validateDecisionRequest;
  /** shadow-only: compute the canonical request hash; injectable for the same reason */
  hashRequest?: typeof computeRequestHash;
}

/** Only what the outbound forward needs — no shadow-prep work runs before the response. */
interface ShadowArgs {
  requestId: string;
  receivedAt: string;
  caller: string | null;
  contentType: string;
  body: string;
  forward: JevForwardResult;
}

/** Shadow-only decode/validate/hash, computed post-response inside `runShadow`. */
interface ShadowPrep {
  parsed: unknown;
  payloadSha: string;
  requestHash: string;
  validation: ValidationResult | null;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/**
 * Persist only the media type of a `content-type` value. MIME parameters are
 * caller-controlled and can smuggle credentials (e.g.
 * `application/json; api_key=...`), so they are dropped; anything that is not a
 * clean `type/subtype` falls back to `application/json`.
 */
function mediaType(value: string): string {
  const bare = (value.split(";")[0] ?? "").trim().toLowerCase();
  return /^[-a-z0-9!#$&^_.+]+\/[-a-z0-9!#$&^_.+]+$/.test(bare) ? bare : "application/json";
}

/** Synthesized upstream view when the tap refuses to forward (no host key). */
function keyMissingForward(): JevForwardResult {
  return {
    status: 503,
    contentType: "application/json; charset=utf-8",
    body: JSON.stringify({ error: "jev key is not configured", code: "JEV_KEY_MISSING" }),
    latencyMs: 0,
    error: "JEV_KEY_MISSING",
  };
}

export function createJevTap(options: JevTapOptions): JevTap {
  const upstreamUrl = parseJevUpstream(options.upstreamUrl).toString();
  const now = options.now ?? (() => performance.now());
  const clock = options.clock ?? (() => new Date());
  const newId = options.newId ?? (() => randomUUID());
  const queueMax = options.queueMax ?? 256;
  const queue = new SerialQueue(queueMax);
  const validateRequest = options.validateRequest ?? validateDecisionRequest;
  const hashRequest = options.hashRequest ?? computeRequestHash;
  const counters = { calls: 0, forwarded_ok: 0, forwarded_error: 0, shadows_ok: 0, shadows_error: 0, last_error: null as string | null };

  // In-flight handoffs between `handle()` returning and `commit()` being called.
  let handoffs = 0;
  let handoffWaiters: Array<() => void> = [];
  const handoffStart = (): void => {
    handoffs += 1;
  };
  const handoffDone = (): void => {
    if (handoffs > 0) handoffs -= 1;
    if (handoffs === 0) {
      const waiters = handoffWaiters;
      handoffWaiters = [];
      for (const wait of waiters) wait();
    }
  };
  const waitForHandoffs = (): Promise<void> => (handoffs === 0 ? Promise.resolve() : new Promise<void>((resolve) => handoffWaiters.push(resolve)));

  /** Shadow-only prep. Never called on the forward path; may throw, and the caller of this must isolate it. */
  const prepareShadow = (body: string): ShadowPrep => {
    const parsed = parseJson(body);
    const payloadSha = sha256(body);
    const validation = parsed === undefined ? null : validateRequest(parsed, options.limits);
    const requestHash = validation !== null && validation.ok ? hashRequest(options.identity, validation.state, validation.questions) : payloadSha;
    return { parsed, payloadSha, requestHash, validation };
  };

  /** A record can always be written even if `prepareShadow` throws. */
  const fallbackPrep = (body: string): ShadowPrep => {
    const payloadSha = sha256(body);
    return { parsed: undefined, payloadSha, requestHash: payloadSha, validation: null };
  };

  const writePair = (args: ShadowArgs, prep: ShadowPrep, laya: JevLayaPairLaya): void => {
    const replyParsed = parseJson(args.forward.body);
    const replyRecord = asRecord(replyParsed);
    const usage = replyRecord && "usage" in replyRecord ? sanitizeForStorage(replyRecord.usage, options.apiKey) : null;
    options.pairs.record({
      schema: 1,
      kind: "jev_laya_pair",
      ts: clock().toISOString(),
      received_at: args.receivedAt,
      request_id: args.requestId,
      caller: args.caller === null ? null : redactSecret(args.caller, options.apiKey),
      request_hash: prep.requestHash,
      payload_sha256: prep.payloadSha,
      request: prep.parsed === undefined ? redactSecret(args.body, options.apiKey) : sanitizeForStorage(prep.parsed, options.apiKey),
      forwarded: { url: upstreamUrl, content_type: mediaType(args.contentType) },
      jev: {
        status: args.forward.status,
        ok: args.forward.error === null && args.forward.status >= 200 && args.forward.status < 300,
        content_type: mediaType(args.forward.contentType),
        latency_ms: Math.max(0, Math.round(args.forward.latencyMs)),
        reply:
          args.forward.error === null && replyParsed !== undefined
            ? sanitizeForStorage(replyParsed, options.apiKey)
            : redactSecret(args.forward.body, options.apiKey),
        model: typeof replyRecord?.model === "string" ? replyRecord.model : null,
        usage,
        error: args.forward.error,
      },
      laya,
    });
  };

  const errorLaya = (started: number, error: string): JevLayaPairLaya => ({
    status: "error",
    latency_ms: Math.max(0, Math.round(now() - started)),
    model: options.identity.model,
    model_sha256: options.identity.modelSha256,
    answers: null,
    usage: null,
    error,
  });

  const failShadow = (args: ShadowArgs, prep: ShadowPrep, started: number, code: string): void => {
    counters.shadows_error += 1;
    counters.last_error = code;
    writePair(args, prep, errorLaya(started, code));
  };

  const runShadow = async (args: ShadowArgs): Promise<void> => {
    const started = now();
    let prep: ShadowPrep;
    try {
      prep = prepareShadow(args.body);
    } catch {
      // Shadow prep can never fail the (already written) Jev response.
      failShadow(args, fallbackPrep(args.body), started, "SHADOW_PREP_ERROR");
      return;
    }

    if (args.forward.error === "JEV_KEY_MISSING") {
      // Fail closed: the call was refused, so Laya must not run either.
      failShadow(args, prep, started, "JEV_KEY_MISSING");
      return;
    }
    if (prep.validation === null) {
      failShadow(args, prep, started, "INVALID_JSON");
      return;
    }
    if (!prep.validation.ok) {
      failShadow(args, prep, started, prep.validation.code);
      return;
    }

    const { state, questions } = prep.validation;
    try {
      const result = await options.engine.systemOne(state, questions);
      const latencyMs = Math.max(0, now() - started);
      recordShadowAnswers(options.shadow, options.identity, {
        requestId: args.requestId,
        requestHash: prep.requestHash,
        questions,
        answers: result.answers,
        latencyMs,
        inputTokens: result.usage.input_tokens,
        error: "MISSING_ANSWER",
      });
      counters.shadows_ok += 1;
      writePair(args, prep, {
        status: "ok",
        latency_ms: Math.round(latencyMs),
        model: options.identity.model,
        model_sha256: options.identity.modelSha256,
        answers: sanitizeForStorage(result.answers, options.apiKey) as JevLayaPairLaya["answers"],
        usage: result.usage,
        error: null,
      });
    } catch {
      const latencyMs = Math.max(0, now() - started);
      recordShadowAnswers(options.shadow, options.identity, {
        requestId: args.requestId,
        requestHash: prep.requestHash,
        questions,
        answers: null,
        latencyMs,
        inputTokens: 0,
        error: "ENGINE_ERROR",
      });
      failShadow(args, prep, started, "ENGINE_ERROR");
    }
  };

  const makeHandoff = (args: ShadowArgs): TapHandoff => {
    handoffStart();
    let committed = false;
    const commit = (): void => {
      if (committed) return;
      committed = true;
      if (!queue.push(() => runShadow(args))) {
        let prep: ShadowPrep;
        try {
          prep = prepareShadow(args.body);
        } catch {
          prep = fallbackPrep(args.body);
        }
        failShadow(args, prep, now(), "QUEUE_FULL");
      }
      handoffDone();
    };
    return { status: args.forward.status, contentType: args.forward.contentType, body: args.forward.body, commit };
  };

  const handle = async (body: string, caller: string | null, contentType: string): Promise<TapHandoff> => {
    counters.calls += 1;
    // Forward-path only: nothing here decodes, hashes, validates or hashes the
    // body, so shadow prep cannot add latency to or fail the Jev response.
    const args: ShadowArgs = {
      requestId: newId(),
      receivedAt: clock().toISOString(),
      caller,
      contentType: contentType || "application/json",
      body,
      forward: keyMissingForward(),
    };

    if (options.apiKey === null) {
      counters.forwarded_error += 1;
      counters.last_error = "JEV_KEY_MISSING";
      return makeHandoff(args);
    }

    args.forward = await options.transport({
      url: upstreamUrl,
      body,
      contentType: args.contentType,
      apiKey: options.apiKey,
      timeoutMs: options.timeoutMs,
    });
    if (args.forward.error === null) counters.forwarded_ok += 1;
    else counters.forwarded_error += 1;
    return makeHandoff(args);
  };

  return {
    handle,
    stats: () => ({
      enabled: true,
      upstream: upstreamUrl,
      key_present: options.apiKey !== null,
      key_fingerprint: options.apiKey !== null ? secretFingerprint(options.apiKey) : null,
      calls: counters.calls,
      forwarded_ok: counters.forwarded_ok,
      forwarded_error: counters.forwarded_error,
      shadows_ok: counters.shadows_ok,
      shadows_error: counters.shadows_error,
      queued: queue.pending,
      queue_max: queueMax,
      last_error: counters.last_error,
    }),
    drain: async () => {
      await waitForHandoffs();
      await queue.idle();
    },
  };
}
