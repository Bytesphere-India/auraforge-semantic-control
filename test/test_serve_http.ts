import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import { DEFAULT_LIMITS } from "../src/serve/protocol.js";
import { createServer, type ServeConfig } from "../src/serve/server.js";
import { ShadowLog } from "../src/serve/shadow.js";
import type { DecisionEngine, EngineIdentity } from "../src/serve/types.js";
import type { Answer, Question, SystemOneResult } from "../src/types.js";

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

const stubAnswers = (questions: Record<string, Question>): Record<string, Answer> => {
  const answers: Record<string, Answer> = {};
  for (const [qid, q] of Object.entries(questions)) {
    if (q.type === "choice") {
      answers[qid] = {
        type: "choice",
        choice: "broken_oracle",
        probabilities: { transient: 0.1, broken_oracle: 0.9 },
        confidence: 0.8,
        rl_agent: { act_probability: 1 },
      };
    } else if (q.type === "score") {
      answers[qid] = {
        type: "score",
        score: 2.5,
        legend: { "0": "lo", "1": "hi" },
        probabilities: { "0": 0.25, "1": 0.75 },
        confidence: 0.5,
        rl_agent: { act_probability: 1 },
      };
    } else {
      answers[qid] = { type: "noul", noul: 0.25, rl_agent: { act_probability: 1 } };
    }
  }
  return answers;
};

const stubEngine: DecisionEngine = {
  async systemOne(_state, questions): Promise<SystemOneResult<Record<string, Question>>> {
    return { model: "laya", answers: stubAnswers(questions), usage: { input_tokens: 42, output_tokens: 0 } };
  },
};

const throwingEngine: DecisionEngine = {
  async systemOne(): Promise<SystemOneResult<Record<string, Question>>> {
    throw new Error("session exploded with raw input text");
  },
};

const RAW = "RAW_SECRET_EVIDENCE_9f3a";

function decisionBody() {
  return {
    model: "typesafe/jev-1.13",
    state: { evidence: `${RAW} oracle timed out` },
    questions: {
      result: { type: "noul", instructions: `Is ${RAW} transient?`, criteria: { true: "yes", false: "no" } },
      event_class: { type: "choice", instructions: `Classify ${RAW}`, criteria: { transient: "retryable", broken_oracle: "defective" } },
      severity: { type: "score", instructions: `Severity of ${RAW}`, criteria: ["cosmetic", "minor", "major"] },
    },
  };
}

interface Harness {
  url: string;
  shadow: ShadowLog;
  close: () => Promise<void>;
}

async function start(engine: DecisionEngine = stubEngine, shadowPath: string | null = null, overrides: Partial<ServeConfig> = {}): Promise<Harness> {
  const shadow = new ShadowLog(shadowPath, () => new Date());
  shadow.open();
  const config: ServeConfig = { host: "127.0.0.1", port: 0, maxBodyBytes: 1024 * 1024, limits: DEFAULT_LIMITS, startedAtMs: Date.now(), ...overrides };
  const server = createServer({ engine, identity, shadow, config, now: () => performance.now() });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    shadow,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

async function json(res: Response): Promise<Record<string, unknown>> {
  return (await res.json()) as Record<string, unknown>;
}

test("GET /health reports readiness, identity and the shadow log", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-health-"));
  const file = path.join(dir, "laya-decisions.jsonl");
  const h = await start(stubEngine, file);
  try {
    const res = await fetch(`${h.url}/health`);
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.status, "ok");
    assert.equal(body.engine, "laya");
    assert.equal(body.engine_version, "0.1.2");
    assert.equal(body.model, "laya-test");
    assert.equal(body.model_sha256, identity.modelSha256);
    assert.equal(body.calibration_status, "UNQUALIFIED");
    assert.deepEqual(body.execution_providers, ["cpu"]);
    assert.equal(typeof body.rss_mb, "number");
    const shadow = body.shadow_log as Record<string, unknown>;
    assert.equal(shadow.enabled, true);
    assert.equal(shadow.path, file);
    assert.equal(shadow.writable, true);
  } finally {
    await h.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("POST /api/alpha/decisions answers noul, choice and score in Jev's envelope", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-http-"));
  const file = path.join(dir, "laya-decisions.jsonl");
  const h = await start(stubEngine, file);
  try {
    const res = await fetch(`${h.url}/api/alpha/decisions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(decisionBody()),
    });
    assert.equal(res.status, 200);
    const text = await res.clone().text();
    assert.ok(!text.includes(RAW), "response must never echo raw input text");
    const body = JSON.parse(text) as Record<string, unknown>;

    assert.equal(body.engine, "laya");
    assert.equal(body.model, "laya-test");
    assert.equal(body.model_sha256, identity.modelSha256);
    assert.equal(body.calibration_status, "UNQUALIFIED");
    assert.equal(body.provider, "local-laya");
    assert.match(String(body.request_id), /^[0-9a-f-]{36}$/);
    assert.match(String(body.request_hash), /^[0-9a-f]{64}$/);
    assert.equal(typeof body.latency_ms, "number");
    assert.deepEqual(body.usage, { input_tokens: 42, output_tokens: 0 });

    const answers = body.answers as Record<string, Answer>;
    assert.deepEqual(
      Object.keys(answers).sort((a, b) => a.localeCompare(b)),
      ["event_class", "result", "severity"],
    );
    assert.equal(answers.result?.type, "noul");
    assert.equal(answers.event_class?.type, "choice");
    assert.equal(answers.severity?.type, "score");

    // the same request hashes identically
    const again = await fetch(`${h.url}/decisions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(decisionBody()),
    });
    assert.equal(again.status, 200);
    assert.equal((await json(again)).request_hash, body.request_hash);

    // one shadow line per question, and none of them carries the raw text
    const lines = (await readFile(file, "utf8")).trim().split("\n");
    assert.equal(lines.length, 6); // 3 questions x 2 requests
    for (const line of lines) {
      assert.ok(!line.includes(RAW));
      const record = JSON.parse(line) as Record<string, unknown>;
      assert.equal(record.status, "ok");
      assert.equal(record.request_hash, body.request_hash);
      assert.ok(typeof record.question_type === "string");
      assert.ok(record.probabilities !== null);
    }
    assert.equal(h.shadow.stats().lines, 6);
  } finally {
    await h.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("request errors are rejected with stable codes", async () => {
  const h = await start();
  try {
    const post = (body: string, headers: Record<string, string> = { "content-type": "application/json" }) =>
      fetch(`${h.url}/api/alpha/decisions`, { method: "POST", headers, body });

    const badJson = await post("{not json");
    assert.equal(badJson.status, 400);
    assert.equal((await json(badJson)).code, "INVALID_JSON");

    const missingState = await post(JSON.stringify({ questions: { q: { type: "noul", instructions: "x" } } }));
    assert.equal(missingState.status, 400);
    assert.equal((await json(missingState)).code, "INVALID_STATE");

    const badType = await post(JSON.stringify({ state: {}, questions: { q: { type: "magic", instructions: "x" } } }));
    assert.equal(badType.status, 400);
    assert.equal((await json(badType)).code, "UNSUPPORTED_TYPE");

    const method = await fetch(`${h.url}/api/alpha/decisions`);
    assert.equal(method.status, 405);
    assert.equal(method.headers.get("allow"), "POST");

    const missing = await fetch(`${h.url}/nope`);
    assert.equal(missing.status, 404);
    assert.equal((await json(missing)).code, "NOT_FOUND");
  } finally {
    await h.close();
  }
});

test("an oversized body is rejected with 413", async () => {
  const h = await start(stubEngine, null, { maxBodyBytes: 128 });
  try {
    const res = await fetch(`${h.url}/api/alpha/decisions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ state: { evidence: "x".repeat(1000) }, questions: { q: { type: "noul", instructions: "x" } } }),
    });
    assert.equal(res.status, 413);
    assert.equal((await json(res)).code, "BODY_TOO_LARGE");
  } finally {
    await h.close();
  }
});

test("an engine failure becomes 422 and an error shadow line, without leaking text", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-engine-"));
  const file = path.join(dir, "laya-decisions.jsonl");
  const h = await start(throwingEngine, file);
  try {
    const res = await fetch(`${h.url}/api/alpha/decisions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(decisionBody()),
    });
    assert.equal(res.status, 422);
    const body = await json(res);
    assert.equal(body.code, "ENGINE_ERROR");
    assert.ok(!JSON.stringify(body).includes("raw input text"));

    const lines = (await readFile(file, "utf8")).trim().split("\n");
    assert.equal(lines.length, 3);
    for (const line of lines) {
      assert.ok(!line.includes(RAW));
      const record = JSON.parse(line) as Record<string, unknown>;
      assert.equal(record.status, "error");
      assert.equal(record.answer, null);
      assert.equal(record.error, "ENGINE_ERROR");
    }
  } finally {
    await h.close();
    await rm(dir, { recursive: true, force: true });
  }
});
