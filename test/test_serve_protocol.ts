import { test } from "node:test";
import assert from "node:assert/strict";
import { buildDecisionResponse, canonicalJson, computeRequestHash, validateDecisionRequest, DEFAULT_LIMITS } from "../src/serve/protocol.js";
import type { EngineIdentity } from "../src/serve/types.js";
import type { Answer } from "../src/types.js";

const identity: EngineIdentity = {
  engine: "laya",
  engineVersion: "0.1.2",
  model: "laya-test",
  modelDir: "/opt/auraforge/models/laya/base-fp32",
  modelSha256: "a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e",
  modelDataSha256: null,
  calibrationStatus: "UNQUALIFIED",
  executionProviders: ["cpu"],
};

const validBody = {
  model: "typesafe/jev-1.13",
  state: { evidence: "the oracle timed out twice" },
  questions: {
    result: { type: "noul", instructions: "Is this transient?", criteria: { true: "yes", false: "no" } },
    event_class: { type: "choice", instructions: "Which class?", criteria: { transient: "retryable", broken_oracle: "defective" } },
    severity: { type: "score", instructions: "How severe?", criteria: ["cosmetic", "minor", "major"] },
  },
};

test("validateDecisionRequest accepts Jev's typed-question shape and normalizes it", () => {
  const r = validateDecisionRequest(validBody);
  assert.ok(r.ok);
  assert.deepEqual(Object.keys(r.questions), ["result", "event_class", "severity"]);
  assert.equal(r.requestedModel, "typesafe/jev-1.13");
  assert.deepEqual(r.questions.event_class, {
    type: "choice",
    instructions: "Which class?",
    criteria: { transient: "retryable", broken_oracle: "defective" },
  });
  assert.deepEqual(r.questions.severity, { type: "score", instructions: "How severe?", criteria: ["cosmetic", "minor", "major"] });
  assert.deepEqual(r.questions.result, { type: "noul", instructions: "Is this transient?", criteria: { true: "yes", false: "no" } });
});

test("validateDecisionRequest accepts choice criteria as a plain list", () => {
  const r = validateDecisionRequest({ state: {}, questions: { q: { type: "choice", instructions: "pick", criteria: ["a", "b"] } } });
  assert.ok(r.ok);
  assert.deepEqual(r.questions.q, { type: "choice", instructions: "pick", criteria: ["a", "b"] });
});

test("validateDecisionRequest rejects malformed bodies with stable codes", () => {
  const cases: Array<[unknown, string]> = [
    ["not an object", "INVALID_BODY"],
    [null, "INVALID_BODY"],
    [{ questions: {} }, "INVALID_STATE"],
    [{ state: {} }, "MISSING_QUESTIONS"],
    [{ state: {}, questions: {} }, "MISSING_QUESTIONS"],
    [{ state: {}, questions: { q: { type: "magic", instructions: "x" } } }, "UNSUPPORTED_TYPE"],
    [{ state: {}, questions: { q: { type: "choice", instructions: "x" } } }, "INVALID_QUESTION"],
    [{ state: {}, questions: { q: { type: "choice", instructions: "x", criteria: { only: "one" } } } }, "INVALID_QUESTION"],
    [{ state: {}, questions: { q: { type: "score", instructions: "x", criteria: ["only"] } } }, "INVALID_QUESTION"],
    [{ state: {}, questions: { q: { type: "noul", instructions: 7 } } }, "INVALID_QUESTION"],
    [{ state: {}, questions: { q: { type: "noul", instructions: "x", criteria: { true: 1 } } } }, "INVALID_QUESTION"],
  ];
  for (const [body, code] of cases) {
    const r = validateDecisionRequest(body);
    assert.ok(!r.ok, `expected failure for ${JSON.stringify(body)}`);
    assert.equal(r.code, code);
    assert.equal(r.status, 400);
  }
});

test("validateDecisionRequest enforces the configured limits", () => {
  const tooMany = Object.fromEntries(Array.from({ length: DEFAULT_LIMITS.maxQuestions + 1 }, (_, i) => [`q${i}`, { type: "noul", instructions: "x" }]));
  const r = validateDecisionRequest({ state: {}, questions: tooMany });
  assert.ok(!r.ok);
  assert.equal(r.code, "TOO_MANY_QUESTIONS");

  const long = validateDecisionRequest({ state: {}, questions: { q: { type: "noul", instructions: "x".repeat(DEFAULT_LIMITS.maxInstructionsChars + 1) } } });
  assert.ok(!long.ok);
  assert.equal(long.code, "INVALID_QUESTION");

  const longState = validateDecisionRequest({ state: "x".repeat(DEFAULT_LIMITS.maxStateChars + 1), questions: { q: { type: "noul", instructions: "x" } } });
  assert.ok(!longState.ok);
  assert.equal(longState.code, "INVALID_STATE");
});

test("canonicalJson sorts object keys recursively", () => {
  assert.equal(canonicalJson({ b: 1, a: { d: [2, { z: 1, y: 0 }], c: "x" } }), '{"a":{"c":"x","d":[2,{"y":0,"z":1}]},"b":1}');
  assert.equal(canonicalJson(undefined), "null");
  assert.equal(canonicalJson(Number.NaN), "null");
});

test("computeRequestHash is deterministic, key-order independent and never the raw text", () => {
  const a = computeRequestHash(identity, { evidence: "raw-secret" }, { q: { type: "noul", instructions: "raw-secret?" } });
  const b = computeRequestHash(identity, { evidence: "raw-secret" }, { q: { instructions: "raw-secret?", type: "noul" } });
  assert.equal(a, b);
  assert.match(a, /^[0-9a-f]{64}$/);
  assert.ok(!a.includes("raw-secret"));

  const other = computeRequestHash(
    { ...identity, modelSha256: "00".repeat(32) },
    { evidence: "raw-secret" },
    { q: { type: "noul", instructions: "raw-secret?" } },
  );
  assert.notEqual(a, other);
});

test("buildDecisionResponse mirrors Jev's envelope plus Laya identity", () => {
  const answers: Record<string, Answer> = { result: { type: "noul", noul: 0.25, rl_agent: { act_probability: 1 } } };
  const body = buildDecisionResponse({
    identity,
    requestId: "11111111-1111-4111-8111-111111111111",
    requestHash: "ab".repeat(32),
    latencyMs: 12.34,
    answers,
    usage: { input_tokens: 10, output_tokens: 0 },
  });
  assert.deepEqual(body.answers, answers);
  assert.deepEqual(body.usage, { input_tokens: 10, output_tokens: 0 });
  assert.equal(body.engine, "laya");
  assert.equal(body.engine_version, "0.1.2");
  assert.equal(body.model, "laya-test");
  assert.equal(body.model_sha256, identity.modelSha256);
  assert.equal(body.model_data_sha256, null);
  assert.equal(body.calibration_status, "UNQUALIFIED");
  assert.equal(body.provider, "local-laya");
  assert.equal(body.request_hash, "ab".repeat(32));
  assert.equal(body.latency_ms, 12.34);
  assert.ok(!JSON.stringify(body).includes("raw-secret"));
});
