/**
 * Append-only shadow logs for `laya-serve`.
 *
 * Two files are written:
 *
 * - `laya-decisions.jsonl` (as before): one line per answered question with the
 *   request hash, question type, answer, probabilities and latency. The raw
 *   request text is an input to the request hash only and is NEVER written here.
 * - `jev-laya-pairs.jsonl` (the Jev shadow tap): one joined line per Jev call
 *   with the full request payload, Jev's reply and Laya's reply.
 *
 * Audit writes must not be able to take down the serve path, so append failures
 * are swallowed and reported through `stats()` (`/health`) instead.
 */
import { appendFileSync, closeSync, mkdirSync, openSync } from "node:fs";
import path from "node:path";
import type { Answer, Question } from "../types.js";
import type { EngineIdentity } from "./types.js";

export interface JsonlLogStats {
  enabled: boolean;
  path: string | null;
  writable: boolean;
  lines: number;
  errors: number;
  last_error: string | null;
}

export type ShadowStats = JsonlLogStats;
export type JevLayaPairsStats = JsonlLogStats;

export interface ShadowEntry {
  requestId: string;
  requestHash: string;
  questionId: string;
  questionType: "noul" | "choice" | "score";
  /** null when the engine failed for this request */
  answer: Answer | null;
  latencyMs: number;
  inputTokens: number;
  status: "ok" | "error";
  /** short machine code only ("ENGINE_ERROR", ...); never raw input text */
  error: string | null;
}

const round4 = (x: number): number => Math.round(x * 1e4) / 1e4;

/** Project a typed answer onto the training-data fields (never the criteria text). */
export function extractShadowAnswer(answer: Answer): { answer: string | number; probabilities: Record<string, number> | null; confidence: number | null } {
  switch (answer.type) {
    case "choice":
      return { answer: answer.choice, probabilities: answer.probabilities, confidence: answer.confidence };
    case "score":
      return { answer: answer.score, probabilities: answer.probabilities, confidence: answer.confidence };
    case "noul":
      return { answer: answer.noul, probabilities: { false: round4(1 - answer.noul), true: answer.noul }, confidence: null };
  }
}

/** Shared append-only JSONL writer: failures are counted, never thrown. */
class JsonlAppendLog {
  private lines = 0;
  private errors = 0;
  private writable = false;
  private lastError: string | null = null;

  constructor(
    protected readonly filePath: string | null,
    private readonly label: string,
    protected readonly now: () => Date,
  ) {}

  /** Create the parent directory and prove the file is appendable. Safe to call once at startup. */
  open(): JsonlLogStats {
    if (this.filePath === null) return this.stats();
    try {
      mkdirSync(path.dirname(this.filePath), { recursive: true });
      closeSync(openSync(this.filePath, "a"));
      this.writable = true;
    } catch (error) {
      this.writable = false;
      this.lastError = errorText(error);
    }
    return this.stats();
  }

  protected append(record: unknown): void {
    if (this.filePath === null) return;
    let line: string;
    try {
      line = JSON.stringify(record);
    } catch (error) {
      this.errors += 1;
      this.lastError = errorText(error);
      return;
    }
    try {
      appendFileSync(this.filePath, line + "\n", { encoding: "utf8", flag: "a" });
      this.lines += 1;
      this.writable = true;
    } catch (error) {
      this.errors += 1;
      this.writable = false;
      this.lastError = errorText(error);
      process.stderr.write(`laya-serve: ${this.label} append failed for ${this.filePath}: ${this.lastError}\n`);
    }
  }

  stats(): JsonlLogStats {
    return {
      enabled: this.filePath !== null,
      path: this.filePath,
      writable: this.writable,
      lines: this.lines,
      errors: this.errors,
      last_error: this.lastError,
    };
  }
}

function errorText(error: unknown): string {
  return `${error instanceof Error ? error.name : "Error"}: ${String(error instanceof Error ? error.message : error)}`.slice(0, 300);
}

export class ShadowLog extends JsonlAppendLog {
  constructor(filePath: string | null, now: () => Date = () => new Date()) {
    super(filePath, "shadow", now);
  }

  /** Append one line per (request, question). Never throws. */
  record(identity: EngineIdentity, entry: ShadowEntry): void {
    if (this.filePath === null) return;
    const fields = entry.answer === null ? { answer: null, probabilities: null, confidence: null } : extractShadowAnswer(entry.answer);
    this.append({
      schema: 1,
      ts: this.now().toISOString(),
      engine: identity.engine,
      engine_version: identity.engineVersion,
      model: identity.model,
      model_sha256: identity.modelSha256,
      calibration_status: identity.calibrationStatus,
      request_id: entry.requestId,
      request_hash: entry.requestHash,
      question_id: entry.questionId,
      question_type: entry.questionType,
      answer: fields.answer,
      probabilities: fields.probabilities,
      confidence: fields.confidence,
      latency_ms: Math.max(0, Math.round(entry.latencyMs)),
      input_tokens: entry.inputTokens,
      status: entry.status,
      error: entry.error,
    });
  }
}

/** One Jev call's Laya view inside a joined pair record. */
export interface JevLayaPairLaya {
  status: "ok" | "error";
  latency_ms: number;
  model: string;
  model_sha256: string;
  answers: Record<string, Answer> | null;
  usage: { input_tokens: number; output_tokens: number } | null;
  error: string | null;
}

/** One Jev call's upstream view inside a joined pair record. */
export interface JevLayaPairJev {
  status: number;
  ok: boolean;
  content_type: string;
  latency_ms: number;
  reply: unknown;
  model: string | null;
  usage: unknown;
  error: string | null;
}

/**
 * The joined record: full request payload, Jev's full reply and Laya's full
 * reply, keyed by caller and request hash. Credentials never appear here.
 */
export interface JevLayaPairRecord {
  schema: 1;
  kind: "jev_laya_pair";
  ts: string;
  received_at: string;
  request_id: string;
  caller: string | null;
  request_hash: string;
  payload_sha256: string;
  request: unknown;
  forwarded: { url: string; content_type: string };
  jev: JevLayaPairJev;
  laya: JevLayaPairLaya;
}

export class JevLayaPairsLog extends JsonlAppendLog {
  constructor(filePath: string | null, now: () => Date = () => new Date()) {
    super(filePath, "jev-laya pair", now);
  }

  /** Append one joined record per Jev call. Never throws. */
  record(record: JevLayaPairRecord): void {
    this.append(record);
  }
}

/**
 * Write Laya's per-question lines into `laya-decisions.jsonl`, exactly as the
 * direct decision route does. `answers === null` records a machine-code error
 * for every question instead.
 */
export function recordShadowAnswers(
  shadow: ShadowLog,
  identity: EngineIdentity,
  params: {
    requestId: string;
    requestHash: string;
    questions: Record<string, Question>;
    answers: Record<string, Answer> | null;
    latencyMs: number;
    inputTokens: number;
    /** recorded for questions with no answer (failure path) */
    error: string;
  },
): void {
  for (const [questionId, question] of Object.entries(params.questions)) {
    const answer = params.answers?.[questionId] ?? null;
    shadow.record(identity, {
      requestId: params.requestId,
      requestHash: params.requestHash,
      questionId,
      questionType: answer?.type ?? question.type,
      answer,
      latencyMs: params.latencyMs,
      inputTokens: params.inputTokens,
      status: answer ? "ok" : "error",
      error: answer ? null : params.error,
    });
  }
}
