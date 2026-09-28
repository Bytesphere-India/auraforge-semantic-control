import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { extractShadowAnswer, ShadowLog } from "../src/serve/shadow.js";
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

const fixedNow = () => new Date("2026-09-28T12:00:00.000Z");

async function parseLines(file: string): Promise<Array<Record<string, unknown>>> {
  const text = await readFile(file, "utf8");
  return text
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

test("extractShadowAnswer projects each answer type onto training fields", () => {
  assert.deepEqual(extractShadowAnswer({ type: "choice", choice: "b", probabilities: { a: 0.4, b: 0.6 }, confidence: 0.2, rl_agent: { act_probability: 1 } }), {
    answer: "b",
    probabilities: { a: 0.4, b: 0.6 },
    confidence: 0.2,
  });
  assert.deepEqual(
    extractShadowAnswer({
      type: "score",
      score: 1.5,
      legend: { "0": "lo", "1": "hi" },
      probabilities: { "0": 0.5, "1": 0.5 },
      confidence: 0,
      rl_agent: { act_probability: 1 },
    }),
    {
      answer: 1.5,
      probabilities: { "0": 0.5, "1": 0.5 },
      confidence: 0,
    },
  );
  assert.deepEqual(extractShadowAnswer({ type: "noul", noul: 0.25, rl_agent: { act_probability: 1 } }), {
    answer: 0.25,
    probabilities: { false: 0.75, true: 0.25 },
    confidence: null,
  });
});

test("ShadowLog writes one JSONL line per question with no raw input text", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-shadow-"));
  const file = path.join(dir, "nested", "laya-decisions.jsonl");
  const shadow = new ShadowLog(file, fixedNow);
  try {
    const stats = shadow.open();
    assert.equal(stats.writable, true);

    const noul: Answer = { type: "noul", noul: 0.25, rl_agent: { act_probability: 1 } };
    const choice: Answer = {
      type: "choice",
      choice: "broken_oracle",
      probabilities: { transient: 0.1, broken_oracle: 0.9 },
      confidence: 0.8,
      rl_agent: { act_probability: 1 },
    };
    const score: Answer = {
      type: "score",
      score: 2.5,
      legend: { "0": "lo", "2": "hi" },
      probabilities: { "0": 0.25, "2": 0.75 },
      confidence: 0.5,
      rl_agent: { act_probability: 1 },
    };

    const base = { requestId: "req-1", requestHash: "cd".repeat(32), latencyMs: 151.5, inputTokens: 42, status: "ok" as const, error: null };
    shadow.record(identity, { ...base, questionId: "result", questionType: "noul", answer: noul });
    shadow.record(identity, { ...base, questionId: "event_class", questionType: "choice", answer: choice });
    shadow.record(identity, { ...base, questionId: "severity", questionType: "score", answer: score });
    shadow.record(identity, { ...base, questionId: "failed_q", questionType: "noul", answer: null, status: "error", error: "ENGINE_ERROR" });

    const lines = await parseLines(file);
    assert.equal(lines.length, 4);
    for (const line of lines) {
      assert.equal(line.schema, 1);
      assert.equal(line.ts, "2026-09-28T12:00:00.000Z");
      assert.equal(line.engine, "laya");
      assert.equal(line.model, "laya-test");
      assert.equal(line.model_sha256, identity.modelSha256);
      assert.equal(line.calibration_status, "UNQUALIFIED");
      assert.equal(line.request_id, "req-1");
      assert.equal(line.request_hash, "cd".repeat(32));
      assert.ok(typeof line.question_id === "string");
      assert.ok(typeof line.question_type === "string");
      assert.ok(typeof line.latency_ms === "number");
      assert.ok(typeof line.input_tokens === "number");
    }
    assert.deepEqual(lines[0]?.probabilities, { false: 0.75, true: 0.25 });
    assert.equal(lines[0]?.answer, 0.25);
    assert.equal(lines[0]?.latency_ms, 152);
    assert.equal(lines[1]?.answer, "broken_oracle");
    assert.equal(lines[1]?.confidence, 0.8);
    assert.equal(lines[2]?.answer, 2.5);
    assert.equal(lines[3]?.answer, null);
    assert.equal(lines[3]?.status, "error");
    assert.equal(lines[3]?.error, "ENGINE_ERROR");

    const stats2 = shadow.stats();
    assert.equal(stats2.lines, 4);
    assert.equal(stats2.errors, 0);
    assert.equal(stats2.writable, true);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("ShadowLog tolerates a null path (disabled) and an unwritable path", async () => {
  const disabled = new ShadowLog(null, fixedNow);
  assert.equal(disabled.open().enabled, false);
  disabled.record(identity, {
    requestId: "r",
    requestHash: "0".repeat(64),
    questionId: "q",
    questionType: "noul",
    answer: null,
    latencyMs: 1,
    inputTokens: 0,
    status: "ok",
    error: null,
  });
  assert.equal(disabled.stats().lines, 0);

  // A regular file where the parent directory should be: mkdir fails fast and
  // open() must report it without throwing.
  const dir = await mkdtemp(path.join(tmpdir(), "laya-shadow-blocked-"));
  try {
    const blocker = path.join(dir, "blocker");
    await writeFile(blocker, "not a directory");
    const unwritable = new ShadowLog(path.join(blocker, "laya.jsonl"), fixedNow);
    const stats = unwritable.open();
    assert.equal(stats.writable, false);
    assert.ok(stats.last_error !== null);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
