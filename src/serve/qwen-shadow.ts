/**
 * The Qwen4b Parallel Decision shadow orchestrator.
 *
 * Qwen4b is a local GPU llama.cpp Parallel Decision server (`POST /v1/decision`).
 * This module runs the pure adapter in `qwen-adapter.ts` against an injectable
 * transport and returns the `qwen4b` member of the Jev tap's joined record. It
 * is a *third* shadow: it has no authority, and its failure is recorded, never
 * propagated.
 *
 * Every call is bounded by `timeoutMs`; transport failure, a non-200 status, an
 * unparseable body, or a malformed/incomplete/contradictory result becomes an
 * error record and never an answer. Multiple questions run concurrently and the
 * result is all-or-nothing. `observeQwen4b` is the tap's isolation boundary and
 * never rejects.
 */
import type { Question } from "../types.js";
import {
  Qwen4bError,
  buildBatchRequest,
  buildNoulRequest,
  modelFromResponse,
  parseBatchResponse,
  parseNoulResponse,
  type Qwen4bAnswer,
  type Qwen4bExchange,
  type Qwen4bShadowResult,
} from "./qwen-adapter.js";
import type { Qwen4bForwardResult, Qwen4bTransport } from "./qwen-forward.js";

export {
  Qwen4bError,
  buildBatchRequest,
  buildNoulRequest,
  choiceKeys,
  parseBatchResponse,
  parseNoulResponse,
  renderBatchContext,
  renderNoulInstructions,
  renderQuestionText,
} from "./qwen-adapter.js";
export type { Qwen4bAnswer, Qwen4bChoiceAnswer, Qwen4bExchange, Qwen4bNoulAnswer, Qwen4bScoreAnswer, Qwen4bShadowResult } from "./qwen-adapter.js";

/** The minimal shadow surface the Jev tap depends on. */
export interface Qwen4bShadow {
  readonly enabled: true;
  readonly url: string;
  run(state: unknown, questions: Record<string, Question>): Promise<Qwen4bShadowResult>;
}

export interface Qwen4bShadowOptions {
  url: string;
  transport: Qwen4bTransport;
  timeoutMs: number;
  /** monotonic milliseconds; injectable for deterministic latency in tests */
  now?: () => number;
}

interface CallOutcome {
  exchange: Qwen4bExchange;
  answers: Record<string, Qwen4bAnswer>;
  error: Qwen4bError | null;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export function createQwen4bShadow(options: Qwen4bShadowOptions): Qwen4bShadow {
  const now = options.now ?? (() => performance.now());

  const call = async (request: unknown): Promise<Qwen4bForwardResult> => {
    const started = now();
    try {
      return await options.transport({ url: options.url, body: JSON.stringify(request), timeoutMs: options.timeoutMs });
    } catch {
      return { status: null, body: "", latencyMs: Math.max(0, now() - started), error: "QWEN4B_TRANSPORT_ERROR" };
    }
  };

  const runOne = async (questionIds: string[], request: unknown, normalize: (raw: unknown) => Record<string, Qwen4bAnswer>): Promise<CallOutcome> => {
    const result = await call(request);
    const parsed = parseJson(result.body);
    const exchange: Qwen4bExchange = {
      question_ids: questionIds,
      request,
      status: result.status,
      response: parsed === undefined ? (result.body.length > 0 ? result.body : null) : parsed,
      error: result.error,
      latency_ms: Math.max(0, Math.round(result.latencyMs)),
    };
    const fail = (error: Qwen4bError): CallOutcome => ({ exchange, answers: {}, error });
    if (result.error !== null) return fail(new Qwen4bError(result.error, `Qwen4b transport failed: ${result.error}`));
    if (result.status !== 200) return fail(new Qwen4bError(`QWEN4B_HTTP_${result.status ?? 0}`, `Qwen4b returned HTTP ${result.status}`));
    if (parsed === undefined) return fail(new Qwen4bError("QWEN4B_MALFORMED_RESPONSE", "response was not valid JSON"));
    try {
      return { exchange, answers: normalize(parsed), error: null };
    } catch (error) {
      return fail(error instanceof Qwen4bError ? error : new Qwen4bError("QWEN4B_MALFORMED_RESPONSE", "response could not be normalized"));
    }
  };

  const buildFailure = (questionIds: string[], error: unknown): CallOutcome => {
    const failure = error instanceof Qwen4bError ? error : new Qwen4bError("QWEN4B_MALFORMED_RESPONSE", "request could not be built");
    return {
      exchange: { question_ids: questionIds, request: null, status: null, response: null, error: failure.code, latency_ms: 0 },
      answers: {},
      error: failure,
    };
  };

  const run = async (state: unknown, questions: Record<string, Question>): Promise<Qwen4bShadowResult> => {
    const started = now();
    const entries = Object.entries(questions);
    const noul = entries.filter((entry): entry is [string, Extract<Question, { type: "noul" }>] => entry[1].type === "noul");
    const batched = entries.filter(([, question]) => question.type !== "noul");

    const tasks: Array<Promise<CallOutcome>> = [];
    for (const [qid, question] of noul) {
      // A bad question is isolated so it cannot reject the whole shadow.
      try {
        tasks.push(runOne([qid], buildNoulRequest(state, question), (raw) => ({ [qid]: parseNoulResponse(raw) })));
      } catch (error) {
        tasks.push(Promise.resolve(buildFailure([qid], error)));
      }
    }
    if (batched.length > 0) {
      const questionIds = batched.map(([qid]) => qid);
      try {
        const request = buildBatchRequest(state, batched);
        if (request !== null) tasks.push(runOne(questionIds, request, (raw) => parseBatchResponse(raw, batched)));
      } catch (error) {
        tasks.push(Promise.resolve(buildFailure(questionIds, error)));
      }
    }

    const outcomes = await Promise.all(tasks);
    const answers: Record<string, Qwen4bAnswer> = {};
    for (const outcome of outcomes) Object.assign(answers, outcome.answers);
    const failed = outcomes.find((outcome) => outcome.error !== null)?.error ?? null;
    const model = outcomes.map((outcome) => modelFromResponse(outcome.exchange.response)).find((value) => value !== null) ?? null;
    return {
      status: failed === null ? "ok" : "error",
      latency_ms: Math.max(0, Math.round(now() - started)),
      model,
      answers: failed === null ? answers : null,
      raw: outcomes.map((outcome) => outcome.exchange),
      error: failed === null ? null : failed.code,
    };
  };

  return { enabled: true, url: options.url, run };
}

/**
 * Run the third shadow and never reject: a disabled shadow yields `null`, and an
 * unexpected throw becomes an error view. This is the tap's whole isolation
 * boundary, so a slow/down/misbehaving Qwen4b cannot affect Jev or Laya.
 */
export async function observeQwen4b(
  shadow: Qwen4bShadow | null | undefined,
  state: unknown,
  questions: Record<string, Question>,
): Promise<Qwen4bShadowResult | null> {
  if (!shadow) return null;
  try {
    return await shadow.run(state, questions);
  } catch {
    return { status: "error", latency_ms: 0, model: null, answers: null, raw: [], error: "QWEN4B_ERROR" };
  }
}
