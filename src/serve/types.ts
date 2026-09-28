/**
 * Types for `laya-serve`, the local always-on shadow decision service.
 *
 * The wire contract deliberately mirrors TypeSafe Jev's OpenRouter
 * `/api/alpha/decisions` typed-question API so a caller can switch engine by
 * URL alone. The engine identity, the pinned model hash and the (deliberately
 * unqualified) calibration status are added to Jev's response envelope, never
 * mixed into `answers`.
 */
import type { Answer, Question, SystemOneResult } from "../types.js";

/** Jev's response envelope has `answers`; everything else is Laya identity metadata. */
export interface EngineIdentity {
  /** engine family, constant "laya" */
  engine: string;
  /** engine package version (from package.json) */
  engineVersion: string;
  /** stable, human-readable model identity, e.g. "laya-base-fp32" */
  model: string;
  /** absolute directory the ONNX bundle was loaded from */
  modelDir: string;
  /** sha256 of laya.onnx (graph). Matches AF-LAYA-BASELINE-001 for the pinned bundle. */
  modelSha256: string;
  /** sha256 of laya.onnx.data (weights), when computed */
  modelDataSha256: string | null;
  /** the model has NOT been calibrated for decision acceptance; never claim otherwise */
  calibrationStatus: "UNQUALIFIED";
  /** requested execution providers; CPU only by policy */
  executionProviders: string[];
}

/** The minimal engine surface the HTTP layer depends on. `Laya` satisfies it. */
export interface DecisionEngine {
  systemOne(state: unknown, questions: Record<string, Question>): Promise<SystemOneResult<Record<string, Question>>>;
}

/** Successful `/api/alpha/decisions` body: a Jev-compatible superset. */
export interface DecisionResponseBody {
  engine: string;
  engine_version: string;
  model: string;
  model_sha256: string;
  model_data_sha256: string | null;
  model_dir: string;
  calibration_status: "UNQUALIFIED";
  provider: "local-laya";
  request_id: string;
  request_hash: string;
  latency_ms: number;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** Error body. Never carries raw input text, a stack trace, or a secret. */
export interface ErrorResponseBody {
  error: string;
  code: string;
}
