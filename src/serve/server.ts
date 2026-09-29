/**
 * The `laya-serve` HTTP layer.
 *
 * Bound to loopback only (the bind is validated in config.ts). Two endpoints:
 *   GET  /health                 -> readiness, identity and shadow-log counters
 *   POST /api/alpha/decisions    -> Jev-compatible typed questions (also /decisions)
 *
 * The engine and shadow log are injected, so the whole HTTP surface is testable
 * offline with a stub engine and a temp shadow file. No outbound network calls
 * are made by this module.
 */
import { randomUUID } from "node:crypto";
import http from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Answer } from "../types.js";
import type { JevTap, TapHandoff } from "./jev-tap.js";
import { buildDecisionResponse, computeRequestHash, validateDecisionRequest, type ServeLimits } from "./protocol.js";
import { recordShadowAnswers, type ShadowLog } from "./shadow.js";
import type { DecisionEngine, EngineIdentity, ErrorResponseBody } from "./types.js";

export interface ServeConfig {
  host: string;
  port: number;
  maxBodyBytes: number;
  limits: ServeLimits;
  /** process start time (Date.now()) used for uptime */
  startedAtMs: number;
}

export interface ServerDeps {
  engine: DecisionEngine;
  identity: EngineIdentity;
  shadow: ShadowLog;
  config: ServeConfig;
  /** monotonic milliseconds; injectable for deterministic latency in tests */
  now?: () => number;
  /** Jev shadow tap; when absent the tap route reports 503 */
  jevTap?: JevTap;
}

const DECISION_PATHS = new Set(["/api/alpha/decisions", "/decisions", "/v1/decisions"]);
const JEV_TAP_PATHS = new Set(["/jev/api/alpha/decisions", "/jev/decisions"]);

function headerValue(value: string | string[] | undefined): string | null {
  const raw = Array.isArray(value) ? value[0] : value;
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8", "content-length": Buffer.byteLength(payload) });
  res.end(payload);
}

function sendError(res: ServerResponse, status: number, code: string, message: string): void {
  const body: ErrorResponseBody = { error: message, code };
  sendJson(res, status, body);
}

function normalizePath(rawUrl: string): string {
  const pathname = new URL(rawUrl, "http://127.0.0.1").pathname;
  if (pathname.length > 1 && pathname.endsWith("/")) return pathname.replace(/\/+$/, "");
  return pathname;
}

type BodyResult = { ok: true; text: string } | { ok: false; status: number; code: string; message: string };

async function readBody(req: IncomingMessage, maxBytes: number): Promise<BodyResult> {
  const chunks: Buffer[] = [];
  let size = 0;
  let tooLarge = false;
  // Drain the rest of the body even after exceeding the cap: the caller then
  // gets a real 413 instead of a connection reset. `requestTimeout` bounds a
  // client that would otherwise stream forever.
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    size += buf.length;
    if (size > maxBytes) {
      tooLarge = true;
      chunks.length = 0;
      continue;
    }
    if (!tooLarge) chunks.push(buf);
  }
  if (tooLarge) return { ok: false, status: 413, code: "BODY_TOO_LARGE", message: `request body exceeds ${maxBytes} bytes` };
  return { ok: true, text: Buffer.concat(chunks).toString("utf8") };
}

export function createServer(deps: ServerDeps): http.Server {
  const { engine, identity, shadow, config, jevTap } = deps;
  const now = deps.now ?? (() => performance.now());
  let requests = 0;
  let errors = 0;

  const health = (): Record<string, unknown> => ({
    status: "ok",
    engine: identity.engine,
    engine_version: identity.engineVersion,
    model: identity.model,
    model_sha256: identity.modelSha256,
    model_data_sha256: identity.modelDataSha256,
    model_dir: identity.modelDir,
    calibration_status: identity.calibrationStatus,
    execution_providers: identity.executionProviders,
    pid: process.pid,
    uptime_s: Math.max(0, (Date.now() - config.startedAtMs) / 1000),
    rss_mb: Math.round((process.memoryUsage().rss / 1048576) * 10) / 10,
    requests,
    errors,
    shadow_log: shadow.stats(),
    jev_tap: jevTap ? jevTap.stats() : null,
  });

  const handleDecision = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const body = await readBody(req, config.maxBodyBytes);
    if (!body.ok) {
      errors += 1;
      return sendError(res, body.status, body.code, body.message);
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(body.text);
    } catch {
      errors += 1;
      return sendError(res, 400, "INVALID_JSON", "request body was not valid JSON");
    }

    const validated = validateDecisionRequest(parsed, config.limits);
    if (!validated.ok) {
      errors += 1;
      return sendError(res, validated.status, validated.code, validated.message);
    }

    const requestId = randomUUID();
    const requestHash = computeRequestHash(identity, validated.state, validated.questions);
    const started = now();
    requests += 1;

    let answers: Record<string, Answer>;
    let inputTokens: number;
    try {
      const result = await engine.systemOne(validated.state, validated.questions);
      answers = result.answers;
      inputTokens = result.usage.input_tokens;
    } catch {
      // Detail goes to stderr only: the shadow log and HTTP body must never
      // carry raw input text.
      const latencyMs = Math.max(0, now() - started);
      errors += 1;
      recordShadowAnswers(shadow, identity, {
        requestId,
        requestHash,
        questions: validated.questions,
        answers: null,
        latencyMs,
        inputTokens: 0,
        error: "ENGINE_ERROR",
      });
      return sendError(res, 422, "ENGINE_ERROR", "the model could not answer this request; see service logs");
    }

    const latencyMs = Math.max(0, now() - started);
    const response = buildDecisionResponse({
      identity,
      requestId,
      requestHash,
      latencyMs: Math.round(latencyMs * 100) / 100,
      answers,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    });

    recordShadowAnswers(shadow, identity, {
      requestId,
      requestHash,
      questions: validated.questions,
      answers,
      latencyMs,
      inputTokens,
      error: "MISSING_ANSWER",
    });

    sendJson(res, 200, response);
  };

  const handleJevTap = async (req: IncomingMessage, res: ServerResponse, tap: JevTap): Promise<void> => {
    const body = await readBody(req, config.maxBodyBytes);
    if (!body.ok) {
      errors += 1;
      return sendError(res, body.status, body.code, body.message);
    }
    const caller = headerValue(req.headers["x-caller"]);
    const contentType = headerValue(req.headers["content-type"]) ?? "application/json";
    requests += 1;

    let forwarded: TapHandoff;
    try {
      forwarded = await tap.handle(body.text, caller, contentType);
    } catch {
      errors += 1;
      return sendError(res, 502, "JEV_TAP_ERROR", "the Jev shadow tap failed to forward the request");
    }
    // Dispatch the Laya shadow only once the response is on the wire (or the
    // connection closed), so a slow shadow never sits in the response path and
    // `drain()` can see the in-flight handoff during shutdown. `commit` is
    // idempotent, so registering both events is safe.
    res.once("finish", forwarded.commit);
    res.once("close", forwarded.commit);
    res.writeHead(forwarded.status, { "content-type": forwarded.contentType, "content-length": Buffer.byteLength(forwarded.body) });
    res.end(forwarded.body);
  };

  const handle = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const pathname = normalizePath(req.url ?? "/");
    if ((req.method === "GET" || req.method === "HEAD") && (pathname === "/health" || pathname === "/healthz")) {
      return sendJson(res, 200, health());
    }
    if (JEV_TAP_PATHS.has(pathname)) {
      if (req.method !== "POST") {
        res.setHeader("allow", "POST");
        return sendError(res, 405, "METHOD_NOT_ALLOWED", "use POST for the Jev shadow tap");
      }
      if (!jevTap) return sendError(res, 503, "JEV_TAP_DISABLED", "the Jev shadow tap is not configured");
      return handleJevTap(req, res, jevTap);
    }
    if (DECISION_PATHS.has(pathname)) {
      if (req.method !== "POST") {
        res.setHeader("allow", "POST");
        return sendError(res, 405, "METHOD_NOT_ALLOWED", "use POST for decision requests");
      }
      return handleDecision(req, res);
    }
    return sendError(res, 404, "NOT_FOUND", "unknown path");
  };

  const server = http.createServer((req, res) => {
    void handle(req, res).catch(() => {
      errors += 1;
      if (!res.headersSent) sendError(res, 500, "INTERNAL_ERROR", "internal error");
      else res.end();
    });
  });

  server.requestTimeout = 60_000;
  server.headersTimeout = 10_000;
  server.keepAliveTimeout = 5_000;
  return server;
}
