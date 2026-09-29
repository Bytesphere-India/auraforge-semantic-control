/**
 * Pure Qwen4b Parallel Decision adapter: request rendering and response
 * normalization. No I/O, no transport, no timer.
 *
 * Qwen4b is a local GPU llama.cpp **Parallel Decision** server
 * (`POST /v1/decision`). This module turns a Jev-shaped typed-question payload
 * into Qwen4b's native wire format and normalizes the reply back into shadow
 * answers, so the same bounded question semantics that reach Jev and Laya also
 * reach Qwen4b.
 *
 * `noul` uses Medhika's native Parallel Decision format **exactly**, one call per
 * question, with semantic parity to the Decision Fabric Qwen adapter: the
 * question, the TRUE criterion, the FALSE criterion and the bounded evidence all
 * reach the engine.
 *
 *     {
 *       "instructions": "<question>\n\nTRUE:\n<true>\n\nFALSE:\n<false>\n\nClassify only from the supplied evidence.",
 *       "schema": { "result": { "type": "boolean", "description": "Does the evidence support the predicate?" } },
 *       "contexts": ["<canonical JSON of state>"],
 *       "mode": "tree",
 *       "cache_prompt": true
 *     }
 *
 * `choice` / `score` use the brief's older draft mapping, one batched call for
 * all non-`noul` questions, with the exact draft schema
 * (`{"type":"object","properties":{<qid>: <per type>},"required":[all qids]}`)
 * and the state plus every question id/type/instructions/criteria in `contexts`.
 * The reply is read from `results[0].fields.<qid> = {value, probability}` (the
 * per-question field map), with the one-level wrapper shape also recognised for
 * robustness; the mapping and its limits are documented in `docs/laya-serve.md`.
 *
 * Probability normalization: the endpoint reports the probability of the
 * *selected* value, never blindly P(TRUE). `true + p` -> `p_true = p`;
 * `false + p` -> `p_true = 1 - p`. A probability that is not finite or is outside
 * `[0, 1]`, a missing/contradictory result, or an unparseable body becomes an
 * error record and never an answer.
 */
import type { Question } from "../types.js";
import { canonicalJson } from "./protocol.js";

export const QWEN4B_CLASSIFY_ONLY = "Classify only from the supplied evidence.";
export const QWEN4B_BATCH_INSTRUCTIONS = "Answer every question strictly from the supplied evidence.";
export const QWEN4B_BOOLEAN_DESCRIPTION = "Does the evidence support the predicate?";
export const QWEN4B_MODE = "tree";
export const QWEN4B_CACHE_PROMPT = true;

/** A noul answer projected from the native Boolean decision. `noul` is P(true). */
export interface Qwen4bNoulAnswer {
  type: "noul";
  noul: number;
  p_true: number;
  p_false: number;
}

/**
 * A choice answer. The native enum field returns only the *selected* option and
 * its probability, so no full distribution is claimed (unlike Laya's answer).
 */
export interface Qwen4bChoiceAnswer {
  type: "choice";
  choice: string;
  /** probability the endpoint reported for the selected option (uncalibrated) */
  probability: number;
}

/** A score answer projected from per-level Boolean support, normalized to a level distribution. */
export interface Qwen4bScoreAnswer {
  type: "score";
  /** expected level index (0 = lowest) */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  /** 1 - normalized entropy of the level distribution */
  confidence: number;
}

export type Qwen4bAnswer = Qwen4bNoulAnswer | Qwen4bChoiceAnswer | Qwen4bScoreAnswer;

/** One raw HTTP exchange with Qwen4b, kept for audit/provenance. */
export interface Qwen4bExchange {
  question_ids: string[];
  request: unknown;
  /** HTTP status, or null when the transport failed */
  status: number | null;
  /** parsed JSON, or the raw body text when unparseable */
  response: unknown;
  error: string | null;
  latency_ms: number;
}

/** The `qwen4b` member of one joined `jev-laya-pairs.jsonl` record. */
export interface Qwen4bShadowResult {
  status: "ok" | "error";
  latency_ms: number;
  /** the model identity the server returned, as provenance only */
  model: string | null;
  /** per-question answers, all-or-nothing: null on any error */
  answers: Record<string, Qwen4bAnswer> | null;
  /** every raw exchange (request + response), even on failure */
  raw: Qwen4bExchange[];
  error: string | null;
}

/** A normalized-response failure; `code` is the machine code stored on the record. */
export class Qwen4bError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "Qwen4bError";
  }
}

function invalid(code: string, message: string): Qwen4bError {
  return new Qwen4bError(`QWEN4B_${code}`, message);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The support weight for one Boolean decision. The endpoint reports the
 * probability of the *selected* value, so P(supported) is `p` for `true` and
 * `1 - p` for `false`; without a probability, support is the hard `1`/`0`.
 */
function supportProbability(value: boolean, probability: unknown): number {
  if (isFiniteNumber(probability)) return value ? probability : 1 - probability;
  return value ? 1 : 0;
}

/** The model identity the server returned, as provenance only. */
export function modelFromResponse(response: unknown): string | null {
  const model = asRecord(response)?.model;
  return typeof model === "string" && model.length > 0 ? model : null;
}

/** Deterministic question text: a string verbatim, an object via canonical JSON. */
export function renderQuestionText(instructions: string | object): string {
  return typeof instructions === "string" ? instructions : canonicalJson(instructions);
}

/**
 * A caller-supplied field must never be able to inject the fixed `TRUE:`/`FALSE:`
 * section headers of the rendered Boolean question (or the classifier
 * terminator); fail closed rather than ask a spoofed question. The header gate
 * anchors to any line start (`m` covers `\n`, `\r`, `\u2028` and `\u2029`) and
 * tolerates any Unicode horizontal whitespace (NBSP included) before/after the
 * keyword; the classifier sentence is matched case-insensitively with optional
 * terminal punctuation.
 */
const STRUCTURAL_HEADER = /^[^\S\n\r\u2028\u2029]*(?:TRUE|FALSE)[^\S\n\r\u2028\u2029]*:/imu;
const CLASSIFIER_TERMINATOR = /classify\s+only\s+from\s+the\s+supplied\s+evidence\.?/i;

function assertNoInjection(label: string, text: string): void {
  if (STRUCTURAL_HEADER.test(text)) {
    throw invalid("STRUCTURAL_HEADER", `${label} contains a TRUE:/FALSE: section header`);
  }
  if (CLASSIFIER_TERMINATOR.test(text)) {
    throw invalid("STRUCTURAL_HEADER", `${label} contains the classifier terminator sentence`);
  }
}

/**
 * The deterministic four-part Boolean rendering: question, TRUE criterion, FALSE
 * criterion, classifier line. A criterion that is absent is omitted rather than
 * sent as an empty section.
 */
export function renderNoulInstructions(question: Extract<Question, { type: "noul" }>): string {
  const questionText = renderQuestionText(question.instructions);
  assertNoInjection("instructions", questionText);
  const parts = [questionText];
  const trueCriterion = question.criteria?.true;
  const falseCriterion = question.criteria?.false;
  if (typeof trueCriterion === "string" && trueCriterion.length > 0) {
    assertNoInjection("criteria.true", trueCriterion);
    parts.push(`TRUE:\n${trueCriterion}`);
  }
  if (typeof falseCriterion === "string" && falseCriterion.length > 0) {
    assertNoInjection("criteria.false", falseCriterion);
    parts.push(`FALSE:\n${falseCriterion}`);
  }
  parts.push(QWEN4B_CLASSIFY_ONLY);
  return parts.join("\n\n");
}

/** Build the exact native Parallel Decision body for one noul question. */
export function buildNoulRequest(state: unknown, question: Extract<Question, { type: "noul" }>): Record<string, unknown> {
  return {
    instructions: renderNoulInstructions(question),
    schema: { result: { type: "boolean", description: QWEN4B_BOOLEAN_DESCRIPTION } },
    contexts: [canonicalJson(state)],
    mode: QWEN4B_MODE,
    cache_prompt: QWEN4B_CACHE_PROMPT,
  };
}

/**
 * Normalize one native Boolean decision. `results[0].decision.result` and
 * `results[0].fields.result.value` must both be booleans and must agree; the
 * reported probability is the probability of the selected value and is never
 * stored as P(true) blindly.
 */
export function parseNoulResponse(raw: unknown): Qwen4bNoulAnswer {
  const root = asRecord(raw);
  if (!root) throw invalid("MALFORMED_RESPONSE", "response was not a JSON object");
  const results = root.results;
  if (!Array.isArray(results) || results.length === 0) throw invalid("EMPTY_RESULT", "response carries no results");
  const first = asRecord(results[0]);
  if (!first) throw invalid("MALFORMED_RESPONSE", "results[0] must be an object");

  const decision = asRecord(first.decision);
  if (!decision || typeof decision.result !== "boolean") {
    throw invalid("SCHEMA_VIOLATION", "results[0].decision.result must be a boolean");
  }
  const fields = asRecord(first.fields);
  if (!fields) throw invalid("SCHEMA_VIOLATION", "results[0].fields is missing");
  const resultField = asRecord(fields.result);
  if (!resultField) throw invalid("SCHEMA_VIOLATION", "results[0].fields.result is missing");
  const value = resultField.value;
  if (typeof value !== "boolean") throw invalid("SCHEMA_VIOLATION", "results[0].fields.result.value must be a boolean");
  if (value !== decision.result) {
    throw invalid("CONTRADICTORY_RESULT", "results[0].decision.result and results[0].fields.result.value disagree");
  }
  const probability = resultField.probability;
  if (!isFiniteNumber(probability)) throw invalid("PROBABILITY_INVALID", "probability must be a finite number");
  if (probability < 0 || probability > 1) throw invalid("PROBABILITY_OUT_OF_RANGE", `probability ${probability} is outside [0, 1]`);
  const pTrue = value ? probability : 1 - probability;
  return { type: "noul", noul: pTrue, p_true: pTrue, p_false: 1 - pTrue };
}

/** The choice option keys, from either the option map or the option list. */
export function choiceKeys(question: Extract<Question, { type: "choice" }>): string[] {
  return Array.isArray(question.criteria) ? [...question.criteria] : Object.keys(question.criteria);
}

/** Per-type field descriptor for the batched (choice/score) schema field map. */
function batchDescriptor(question: Question): unknown {
  if (question.type === "choice") {
    return { type: "string", enum: choiceKeys(question) };
  }
  if (question.type === "score") {
    const properties: Record<string, unknown> = {};
    for (const level of question.criteria) {
      properties[level] = { type: "boolean", description: `Does the evidence support level ${JSON.stringify(level)}?` };
    }
    return { type: "object", properties };
  }
  throw invalid("UNSUPPORTED_TYPE", `question type ${JSON.stringify(question.type)} is not batched`);
}

/** The deterministic batched evidence context: canonical state plus the questions. */
export function renderBatchContext(state: unknown, questions: Array<[string, Question]>): string {
  const lines = questions.map(
    ([qid, question]) =>
      `${qid}\n  type: ${question.type}\n  instructions: ${renderQuestionText(question.instructions)}\n  criteria: ${canonicalJson(question.criteria ?? null)}`,
  );
  return `${canonicalJson(state)}\n\nQuestions:\n${lines.join("\n\n")}`;
}

/**
 * Build the batched call for every `choice`/`score` question, or `null` when the
 * request has none. `schema` is the brief's exact draft wrapper
 * (`{"type":"object","properties":{<qid>: <per type>},"required":[all qids]}`).
 */
export function buildBatchRequest(state: unknown, questions: Array<[string, Question]>): Record<string, unknown> | null {
  if (questions.length === 0) return null;
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const [qid, question] of questions) {
    properties[qid] = batchDescriptor(question);
    required.push(qid);
  }
  return {
    instructions: QWEN4B_BATCH_INSTRUCTIONS,
    schema: { type: "object", properties, required },
    contexts: [renderBatchContext(state, questions)],
    mode: QWEN4B_MODE,
    cache_prompt: QWEN4B_CACHE_PROMPT,
  };
}

function parseChoiceField(qid: string, field: Record<string, unknown>, keys: string[]): Qwen4bChoiceAnswer {
  const value = field.value;
  if (typeof value !== "string" || !keys.includes(value)) {
    throw invalid("SCHEMA_VIOLATION", `fields.${qid}.value must be one of the declared options`);
  }
  const probability = field.probability;
  if (!isFiniteNumber(probability)) throw invalid("PROBABILITY_INVALID", `fields.${qid}.probability must be a finite number`);
  if (probability < 0 || probability > 1) {
    throw invalid("PROBABILITY_OUT_OF_RANGE", `fields.${qid}.probability ${probability} is outside [0, 1]`);
  }
  return { type: "choice", choice: value, probability };
}

/**
 * Parse the score field. The native endpoint may return per-level fields nested
 * under the question field (`fields.<qid>.<level> = {value, probability}`) or a
 * single object value (`fields.<qid>.value = {<level>: bool}` with an optional
 * `probabilities` map). Both are accepted; per-level support weights are
 * normalized into a level distribution, and a field with no supported level fails
 * closed.
 */
function parseScoreField(qid: string, field: Record<string, unknown>, levels: string[]): Qwen4bScoreAnswer {
  const weights: Record<string, number> = {};

  const nested = levels.every((level) => {
    const entry = asRecord(field[level]);
    return entry !== null && typeof entry.value === "boolean";
  });
  if (nested) {
    for (const level of levels) {
      const entry = asRecord(field[level]) as Record<string, unknown>;
      const value = entry.value as boolean;
      const probability = entry.probability;
      if (probability !== undefined && (!isFiniteNumber(probability) || probability < 0 || probability > 1)) {
        throw invalid("PROBABILITY_OUT_OF_RANGE", `fields.${qid}.${level}.probability is not a finite probability`);
      }
      weights[level] = supportProbability(value, probability);
    }
  } else {
    const valueObject = asRecord(field.value);
    if (!valueObject) throw invalid("SCHEMA_VIOLATION", `fields.${qid}.value must be an object of level booleans`);
    const probabilities = asRecord(field.probabilities) ?? {};
    for (const level of levels) {
      const value = valueObject[level];
      if (typeof value !== "boolean") throw invalid("SCHEMA_VIOLATION", `fields.${qid}.value.${level} must be a boolean`);
      const probability = probabilities[level];
      if (probability !== undefined && (!isFiniteNumber(probability) || probability < 0 || probability > 1)) {
        throw invalid("PROBABILITY_OUT_OF_RANGE", `fields.${qid}.probabilities.${level} is not a finite probability`);
      }
      weights[level] = supportProbability(value, probability);
    }
  }

  const total = levels.reduce((sum, level) => sum + (weights[level] ?? 0), 0);
  if (total <= 0) throw invalid("SCORE_NO_EVIDENCE", `fields.${qid} supports no level`);

  const probabilities: Record<string, number> = {};
  levels.forEach((level) => {
    probabilities[level] = (weights[level] ?? 0) / total;
  });
  const expected = levels.reduce((sum, level, index) => sum + index * (probabilities[level] ?? 0), 0);
  const entropy = -levels.reduce((sum, level) => {
    const p = probabilities[level] ?? 0;
    return sum + (p > 0 ? p * Math.log(p) : 0);
  }, 0);
  const maxEntropy = levels.length > 1 ? Math.log(levels.length) : 0;
  const confidence = maxEntropy > 0 ? 1 - entropy / maxEntropy : 0;
  const legend: Record<string, string> = {};
  levels.forEach((level, index) => {
    legend[String(index)] = level;
  });
  return { type: "score", score: expected, legend, probabilities, confidence };
}

/**
 * Resolve the per-question field container. The native field map surfaces each
 * question directly (`fields.<qid>`); the older draft wrapper surfaces them one
 * level down (`fields.result.<qid>`). A question literally named `result` makes
 * the two shapes collide, so the wrapper is recognised only by shape: `fields`
 * has the single key `result`, that value is not itself a field
 * (`value`/`probability`) and it carries at least one requested question id.
 */
function batchFieldsContainer(fields: Record<string, unknown>, questions: Array<[string, Question]>): Record<string, unknown> {
  const keys = Object.keys(fields);
  if (keys.length !== 1 || keys[0] !== "result") return fields;
  const wrapped = asRecord(fields.result);
  if (wrapped === null || "value" in wrapped || "probability" in wrapped) return fields;
  return questions.some(([qid]) => qid in wrapped) ? wrapped : fields;
}

/** Parse the batched reply, one answer per requested question, all-or-nothing. */
export function parseBatchResponse(raw: unknown, questions: Array<[string, Question]>): Record<string, Qwen4bAnswer> {
  const root = asRecord(raw);
  if (!root) throw invalid("MALFORMED_RESPONSE", "response was not a JSON object");
  const results = root.results;
  if (!Array.isArray(results) || results.length === 0) throw invalid("EMPTY_RESULT", "response carries no results");
  const first = asRecord(results[0]);
  if (!first) throw invalid("MALFORMED_RESPONSE", "results[0] must be an object");
  const fields = asRecord(first.fields);
  if (!fields) throw invalid("SCHEMA_VIOLATION", "results[0].fields is missing");
  const container = batchFieldsContainer(fields, questions);

  const answers: Record<string, Qwen4bAnswer> = {};
  for (const [qid, question] of questions) {
    const field = asRecord(container[qid]);
    if (!field) throw invalid("SCHEMA_VIOLATION", `results[0].fields.${qid} is missing`);
    if (question.type === "choice") answers[qid] = parseChoiceField(qid, field, choiceKeys(question));
    else if (question.type === "score") answers[qid] = parseScoreField(qid, field, [...question.criteria]);
    else throw invalid("UNSUPPORTED_TYPE", `question ${qid} is not batched`);
  }
  return answers;
}
