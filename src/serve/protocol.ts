/**
 * Pure request validation + response construction for `laya-serve`.
 *
 * No I/O, no model: everything here is unit-testable offline. The accepted body
 * is TypeSafe Jev's `/api/alpha/decisions` shape
 *
 *     { "model"?: string, "questions": { "<id>": Question }, "state": <json> }
 *
 * where a Question is `{ type: "noul" | "choice" | "score", instructions, criteria }`.
 */
import { createHash } from "node:crypto";
import type { Answer, ChoiceQuestion, NoulQuestion, Question, ScoreQuestion } from "../types.js";
import type { DecisionResponseBody, EngineIdentity } from "./types.js";

export interface ServeLimits {
  /** maximum questions in one request (batched in a single forward pass) */
  maxQuestions: number;
  /** maximum characters of `instructions` (string length, or serialized object length) */
  maxInstructionsChars: number;
  /** maximum options in a choice, or levels in a score */
  maxCriteria: number;
  /** maximum characters of the serialized `state` */
  maxStateChars: number;
  /** maximum characters of a question id */
  maxQuestionIdChars: number;
}

export const DEFAULT_LIMITS: ServeLimits = {
  maxQuestions: 32,
  maxInstructionsChars: 20_000,
  maxCriteria: 64,
  maxStateChars: 400_000,
  maxQuestionIdChars: 128,
};

export interface ValidationError {
  ok: false;
  /** HTTP status the caller should send */
  status: number;
  /** stable machine code, e.g. INVALID_QUESTION */
  code: string;
  message: string;
}

export interface ValidatedDecisionRequest {
  ok: true;
  state: unknown;
  questions: Record<string, Question>;
  /** the `model` the caller asked for, if any; served regardless (engine is chosen by URL) */
  requestedModel: string | null;
}

export type ValidationResult = ValidatedDecisionRequest | ValidationError;

/** Internal: a validated value, or the error that stopped validation. */
type Checked<T> = { ok: true; value: T } | ValidationError;

function fail(status: number, code: string, message: string): ValidationError {
  return { ok: false, status, code, message };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function serializedLength(value: unknown): number {
  try {
    return JSON.stringify(value)?.length ?? 0;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

function checkInstructions(raw: Record<string, unknown>, qid: string, limits: ServeLimits): Checked<string | object> {
  const instructions = raw.instructions;
  if (typeof instructions !== "string" && !isPlainObject(instructions)) {
    return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: instructions must be a string or an object`);
  }
  if (serializedLength(instructions) > limits.maxInstructionsChars) {
    return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: instructions exceed ${limits.maxInstructionsChars} characters`);
  }
  return { ok: true, value: instructions };
}

function checkChoice(raw: Record<string, unknown>, qid: string, limits: ServeLimits): Checked<ChoiceQuestion> {
  const criteria = raw.criteria;
  let normalized: Record<string, string | null> | string[];
  if (Array.isArray(criteria)) {
    if (criteria.length < 2 || criteria.length > limits.maxCriteria || criteria.some((c) => typeof c !== "string" || c.length === 0)) {
      return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: choice criteria must be 2..${limits.maxCriteria} non-empty strings`);
    }
    normalized = criteria as string[];
  } else if (isPlainObject(criteria)) {
    const entries = Object.entries(criteria);
    if (entries.length < 2 || entries.length > limits.maxCriteria) {
      return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: choice criteria must have 2..${limits.maxCriteria} options`);
    }
    for (const [key, value] of entries) {
      if (key.length === 0 || (value !== null && typeof value !== "string")) {
        return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: choice option ${JSON.stringify(key)} must map to a string or null`);
      }
    }
    normalized = Object.fromEntries(entries) as Record<string, string | null>;
  } else {
    return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: choice requires criteria (an option map or a list of options)`);
  }
  const instructions = checkInstructions(raw, qid, limits);
  if (!instructions.ok) return instructions;
  return { ok: true, value: { type: "choice", instructions: instructions.value, criteria: normalized } };
}

function checkScore(raw: Record<string, unknown>, qid: string, limits: ServeLimits): Checked<ScoreQuestion> {
  const criteria = raw.criteria;
  if (
    !Array.isArray(criteria) ||
    criteria.length < 2 ||
    criteria.length > limits.maxCriteria ||
    criteria.some((c) => typeof c !== "string" || c.length === 0)
  ) {
    return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: score criteria must be 2..${limits.maxCriteria} non-empty level labels`);
  }
  const instructions = checkInstructions(raw, qid, limits);
  if (!instructions.ok) return instructions;
  return { ok: true, value: { type: "score", instructions: instructions.value, criteria: criteria as string[] } };
}

function checkNoul(raw: Record<string, unknown>, qid: string, limits: ServeLimits): Checked<NoulQuestion> {
  const instructions = checkInstructions(raw, qid, limits);
  if (!instructions.ok) return instructions;
  const criteria = raw.criteria;
  if (criteria !== undefined && !isPlainObject(criteria)) {
    return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: noul criteria must be an object with optional true/false strings`);
  }
  const normalized: { true?: string; false?: string } = {};
  if (isPlainObject(criteria)) {
    for (const key of ["true", "false"] as const) {
      const value = criteria[key];
      if (value !== undefined && typeof value !== "string") {
        return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: noul criteria.${key} must be a string`);
      }
      if (typeof value === "string") normalized[key] = value;
    }
  }
  const question: NoulQuestion =
    Object.keys(normalized).length > 0
      ? { type: "noul", instructions: instructions.value, criteria: normalized }
      : { type: "noul", instructions: instructions.value };
  return { ok: true, value: question };
}

function checkQuestion(value: unknown, qid: string, limits: ServeLimits): Checked<Question> {
  if (!isPlainObject(value)) return fail(400, "INVALID_QUESTION", `question ${JSON.stringify(qid)}: must be an object`);
  switch (value.type) {
    case "choice":
      return checkChoice(value, qid, limits);
    case "score":
      return checkScore(value, qid, limits);
    case "noul":
      return checkNoul(value, qid, limits);
    default:
      return fail(400, "UNSUPPORTED_TYPE", `question ${JSON.stringify(qid)}: type must be one of "noul", "choice", "score"`);
  }
}

/** Validate and normalize a decoded request body. Pure; throws for nothing. */
export function validateDecisionRequest(body: unknown, limits: ServeLimits = DEFAULT_LIMITS): ValidationResult {
  if (!isPlainObject(body)) return fail(400, "INVALID_BODY", "request body must be a JSON object");
  if (!Object.prototype.hasOwnProperty.call(body, "state")) return fail(400, "INVALID_STATE", "request body must carry a `state`");

  const state = body.state;
  if (serializedLength(state) > limits.maxStateChars) {
    return fail(400, "INVALID_STATE", `state exceeds ${limits.maxStateChars} characters`);
  }

  const rawQuestions = body.questions;
  if (!isPlainObject(rawQuestions)) return fail(400, "MISSING_QUESTIONS", "request body must carry a `questions` object");
  const ids = Object.keys(rawQuestions);
  if (ids.length === 0) return fail(400, "MISSING_QUESTIONS", "`questions` must contain at least one question");
  if (ids.length > limits.maxQuestions) return fail(400, "TOO_MANY_QUESTIONS", `at most ${limits.maxQuestions} questions are allowed per request`);

  const questions: Record<string, Question> = {};
  for (const qid of ids) {
    if (qid.length === 0 || qid.length > limits.maxQuestionIdChars) {
      return fail(400, "INVALID_QUESTION", `question id must be 1..${limits.maxQuestionIdChars} characters`);
    }
    const question = checkQuestion(rawQuestions[qid], qid, limits);
    if (!question.ok) return question;
    questions[qid] = question.value;
  }

  const requestedModel = typeof body.model === "string" && body.model.length > 0 ? body.model : null;
  return { ok: true, state, questions, requestedModel };
}

/**
 * Deterministic JSON with object keys sorted recursively. Used only for hashing,
 * so neither key order nor whitespace can change a request's identity.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") return Number.isFinite(value) ? JSON.stringify(value) : "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return "{" + entries.map(([k, v]) => JSON.stringify(k) + ":" + canonicalJson(v)).join(",") + "}";
  }
  return JSON.stringify(String(value));
}

/**
 * The shadow record: sha256 over the canonical (engine, model, state, questions).
 * The raw input text is an input to the hash and is never stored.
 */
export function computeRequestHash(
  identity: Pick<EngineIdentity, "engine" | "model" | "modelSha256">,
  state: unknown,
  questions: Record<string, Question>,
): string {
  return createHash("sha256")
    .update(canonicalJson({ v: 1, engine: identity.engine, model: identity.model, model_sha256: identity.modelSha256, state, questions }))
    .digest("hex");
}

export interface ResponseInputs {
  identity: EngineIdentity;
  requestId: string;
  requestHash: string;
  latencyMs: number;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** Build the success body: Jev's `answers`/`usage` plus Laya identity and latency. */
export function buildDecisionResponse(inputs: ResponseInputs): DecisionResponseBody {
  return {
    engine: inputs.identity.engine,
    engine_version: inputs.identity.engineVersion,
    model: inputs.identity.model,
    model_sha256: inputs.identity.modelSha256,
    model_data_sha256: inputs.identity.modelDataSha256,
    model_dir: inputs.identity.modelDir,
    calibration_status: inputs.identity.calibrationStatus,
    provider: "local-laya",
    request_id: inputs.requestId,
    request_hash: inputs.requestHash,
    latency_ms: inputs.latencyMs,
    answers: inputs.answers,
    usage: inputs.usage,
  };
}
