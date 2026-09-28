/**
 * Append-only shadow log for `laya-serve`.
 *
 * Every answered question becomes one JSONL line: the request hash, question
 * type, answer, probabilities and latency. The raw request text (state,
 * evidence, instructions) is an input to the request hash only and is NEVER
 * written here. This file is Laya's training data; learning is offline and
 * qualified, never online.
 *
 * Auditing must not be able to take down the serve path, so append failures are
 * swallowed and reported through {@link ShadowLog.stats} (`/health`) instead.
 */
import { appendFileSync, closeSync, mkdirSync, openSync } from "node:fs";
import path from "node:path";
import type { Answer } from "../types.js";
import type { EngineIdentity } from "./types.js";

export interface ShadowStats {
  enabled: boolean;
  path: string | null;
  writable: boolean;
  lines: number;
  errors: number;
  last_error: string | null;
}

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

export class ShadowLog {
  private lines = 0;
  private errors = 0;
  private writable = false;
  private lastError: string | null = null;

  constructor(
    private readonly filePath: string | null,
    private readonly now: () => Date = () => new Date(),
  ) {}

  /** Create the parent directory and prove the file is appendable. Safe to call once at startup. */
  open(): ShadowStats {
    if (this.filePath === null) return this.stats();
    try {
      mkdirSync(path.dirname(this.filePath), { recursive: true });
      closeSync(openSync(this.filePath, "a"));
      this.writable = true;
    } catch (error) {
      this.writable = false;
      this.lastError = `${error instanceof Error ? error.name : "Error"}: ${String(error instanceof Error ? error.message : error)}`.slice(0, 300);
    }
    return this.stats();
  }

  /** Append one line per (request, question). Never throws. */
  record(identity: EngineIdentity, entry: ShadowEntry): void {
    if (this.filePath === null) return;
    const fields = entry.answer === null ? { answer: null, probabilities: null, confidence: null } : extractShadowAnswer(entry.answer);
    const line = JSON.stringify({
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
    try {
      appendFileSync(this.filePath, line + "\n", { encoding: "utf8", flag: "a" });
      this.lines += 1;
      this.writable = true;
    } catch (error) {
      this.errors += 1;
      this.writable = false;
      this.lastError = `${error instanceof Error ? error.name : "Error"}: ${String(error instanceof Error ? error.message : error)}`.slice(0, 300);
      process.stderr.write(`laya-serve: shadow append failed for ${this.filePath}: ${this.lastError}\n`);
    }
  }

  stats(): ShadowStats {
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
