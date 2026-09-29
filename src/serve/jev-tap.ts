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
 * Laya is scheduled with `setImmediate`, i.e. after the response turn, and a
 * slow or failing Laya can never delay or fail the Jev response. The Jev key is
 * never logged or persisted.
 */
import { createHash, randomUUID } from "node:crypto";
import { redactSecret, sanitizeForStorage, secretFingerprint } from "./jev-key.js";
import type { JevForwardResult, JevTransport } from "./jev-forward.js";
import { computeRequestHash, validateDecisionRequest, type ServeLimits, type ValidatedDecisionRequest } from "./protocol.js";
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

export interface JevTap {
  handle(body: string, caller: string | null, contentType: string): Promise<TapResponse>;
  stats(): JevTapStats;
  /** resolves when every queued Laya shadow has finished */
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
}

interface ShadowArgs {
  requestId: string;
  receivedAt: string;
  caller: string | null;
  contentType: string;
  body: string;
  parsed: unknown;
  payloadSha: string;
  requestHash: string;
  validation: ValidatedDecisionRequest | { ok: false; code: string } | null;
  forward: JevForwardResult;
}

/** Serial task queue: Laya shadows never overlap, and overflow is observable. */
class SerialQueue {
  private tail: Promise<void> = Promise.resolve();
  private depth = 0;

  constructor(private readonly max: number) {}

  get pending(): number {
    return this.depth;
  }

  /** returns false when the queue is full */
  push(task: () => Promise<void>): boolean {
    if (this.depth >= this.max) return false;
    this.depth += 1;
    this.tail = this.tail
      .then(task)
      .catch(() => {
        // a task must never break the chain; task bodies handle their own errors
      })
      .finally(() => {
        this.depth -= 1;
      });
    return true;
  }

  idle(): Promise<void> {
    return this.tail;
  }
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

export function createJevTap(options: JevTapOptions): JevTap {
  const now = options.now ?? (() => performance.now());
  const clock = options.clock ?? (() => new Date());
  const newId = options.newId ?? (() => randomUUID());
  const queueMax = options.queueMax ?? 256;
  const queue = new SerialQueue(queueMax);
  const counters = { calls: 0, forwarded_ok: 0, forwarded_error: 0, shadows_ok: 0, shadows_error: 0, last_error: null as string | null };

  const writePair = (args: ShadowArgs, laya: JevLayaPairLaya): void => {
    const replyParsed = parseJson(args.forward.body);
    const replyRecord = asRecord(replyParsed);
    const usage = replyRecord && "usage" in replyRecord ? sanitizeForStorage(replyRecord.usage, options.apiKey) : null;
    options.pairs.record({
      schema: 1,
      kind: "jev_laya_pair",
      ts: clock().toISOString(),
      received_at: args.receivedAt,
      request_id: args.requestId,
      caller: args.caller,
      request_hash: args.requestHash,
      payload_sha256: args.payloadSha,
      request: args.parsed === undefined ? redactSecret(args.body, options.apiKey) : sanitizeForStorage(args.parsed, options.apiKey),
      forwarded: { url: options.upstreamUrl, content_type: args.contentType },
      jev: {
        status: args.forward.status,
        ok: args.forward.error === null && args.forward.status >= 200 && args.forward.status < 300,
        content_type: args.forward.contentType,
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

  const runShadow = async (args: ShadowArgs): Promise<void> => {
    const started = now();
    if (args.validation === null) {
      counters.shadows_error += 1;
      counters.last_error = "INVALID_JSON";
      writePair(args, errorLaya(started, "INVALID_JSON"));
      return;
    }
    if (!args.validation.ok) {
      counters.shadows_error += 1;
      counters.last_error = args.validation.code;
      writePair(args, errorLaya(started, args.validation.code));
      return;
    }

    const { state, questions } = args.validation;
    try {
      const result = await options.engine.systemOne(state, questions);
      const latencyMs = Math.max(0, now() - started);
      recordShadowAnswers(options.shadow, options.identity, {
        requestId: args.requestId,
        requestHash: args.requestHash,
        questions,
        answers: result.answers,
        latencyMs,
        inputTokens: result.usage.input_tokens,
        error: "MISSING_ANSWER",
      });
      counters.shadows_ok += 1;
      writePair(args, {
        status: "ok",
        latency_ms: Math.round(latencyMs),
        model: options.identity.model,
        model_sha256: options.identity.modelSha256,
        answers: result.answers,
        usage: result.usage,
        error: null,
      });
    } catch {
      const latencyMs = Math.max(0, now() - started);
      recordShadowAnswers(options.shadow, options.identity, {
        requestId: args.requestId,
        requestHash: args.requestHash,
        questions,
        answers: null,
        latencyMs,
        inputTokens: 0,
        error: "ENGINE_ERROR",
      });
      counters.shadows_error += 1;
      counters.last_error = "ENGINE_ERROR";
      writePair(args, errorLaya(started, "ENGINE_ERROR"));
    }
  };

  const handle = async (body: string, caller: string | null, contentType: string): Promise<TapResponse> => {
    counters.calls += 1;
    const requestId = newId();
    const receivedAt = clock().toISOString();
    const parsed = parseJson(body);
    const payloadSha = sha256(body);
    const validation = parsed === undefined ? null : validateDecisionRequest(parsed, options.limits);
    const requestHash = validation !== null && validation.ok ? computeRequestHash(options.identity, validation.state, validation.questions) : payloadSha;

    const forward = await options.transport({
      url: options.upstreamUrl,
      body,
      contentType: contentType || "application/json",
      apiKey: options.apiKey,
      timeoutMs: options.timeoutMs,
    });
    if (forward.error === null) counters.forwarded_ok += 1;
    else counters.forwarded_error += 1;

    const args: ShadowArgs = {
      requestId,
      receivedAt,
      caller,
      contentType: contentType || "application/json",
      body,
      parsed,
      payloadSha,
      requestHash,
      validation,
      forward,
    };

    // Schedule the Laya shadow after this turn so the Jev response is on the
    // wire first; the shadow never participates in the response path.
    setImmediate(() => {
      if (!queue.push(() => runShadow(args))) {
        counters.shadows_error += 1;
        counters.last_error = "QUEUE_FULL";
        writePair(args, errorLaya(now(), "QUEUE_FULL"));
      }
    });

    return { status: forward.status, contentType: forward.contentType, body: forward.body };
  };

  return {
    handle,
    stats: () => ({
      enabled: true,
      upstream: options.upstreamUrl,
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
    drain: () => queue.idle(),
  };
}
