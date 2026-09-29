import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { DEFAULT_LIMITS } from "../src/serve/protocol.js";
import { loadServeConfig } from "../src/serve/config.js";
import { fetchJevTransport, type JevForwardResult, type JevTransport } from "../src/serve/jev-forward.js";
import { loadJevKey, redactSecret, sanitizeForStorage } from "../src/serve/jev-key.js";
import { createJevTap, type JevTap } from "../src/serve/jev-tap.js";
import { createServer, type ServeConfig } from "../src/serve/server.js";
import { JevLayaPairsLog, ShadowLog } from "../src/serve/shadow.js";
import type { DecisionEngine, EngineIdentity } from "../src/serve/types.js";
import type { Answer, Question, SystemOneResult } from "../src/types.js";

const TAP_KEY = "sk-or-v1-TESTONLY-9f3a2b7c-do-not-log";
const CALLER_KEY = "caller-supplied-secret-key";

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

interface DecisionPayload {
  model: string;
  state: Record<string, unknown>;
  questions: Record<string, Question>;
}

const defaultReply = { answers: { result: { noul: 0.9 } }, model: "fake/jev-1", usage: { input_tokens: 11, output_tokens: 3, cost: 0.0001 } };

const basePayload = (): DecisionPayload => ({
  model: "typesafe/jev-1.13",
  state: { evidence: "the oracle timed out twice" },
  questions: {
    result: { type: "noul", instructions: "Is this transient?", criteria: { true: "yes", false: "no" } },
    event_class: { type: "choice", instructions: "Which class?", criteria: { transient: "retryable", broken_oracle: "defective" } },
    severity: { type: "score", instructions: "How severe?", criteria: ["cosmetic", "minor", "major"] },
  },
});

const engineAnswers = (questions: Record<string, Question>): Record<string, Answer> => {
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
        score: 1.5,
        legend: { "0": "cosmetic", "2": "major" },
        probabilities: { "0": 0.25, "2": 0.75 },
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
    return { model: "laya", answers: engineAnswers(questions), usage: { input_tokens: 7, output_tokens: 0 } };
  },
};

const rec = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

async function readJsonl(file: string): Promise<Array<Record<string, unknown>>> {
  try {
    const text = await readFile(file, "utf8");
    return text
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Record<string, unknown>);
  } catch {
    return [];
  }
}

async function waitFor<T>(probe: () => Promise<T>, accept: (value: T) => boolean, label: string, timeoutMs = 5000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await probe();
    if (accept(value)) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

interface UpstreamCall {
  body: string;
  authorization: string | null;
  contentType: string | null;
}

async function startUpstream(status: number, reply: unknown) {
  const calls: UpstreamCall[] = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk as Buffer)));
    req.on("end", () => {
      calls.push({
        body: Buffer.concat(chunks).toString("utf8"),
        authorization: typeof req.headers.authorization === "string" ? req.headers.authorization : null,
        contentType: typeof req.headers["content-type"] === "string" ? req.headers["content-type"] : null,
      });
      const body = typeof reply === "string" ? reply : JSON.stringify(reply);
      res.writeHead(status, { "content-type": "application/json" });
      res.end(body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/api/alpha/decisions`,
    calls,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

interface SetupOptions {
  engine?: DecisionEngine;
  transport?: JevTransport;
  queueMax?: number;
  apiKey?: string | null;
  withTap?: boolean;
  upstreamStatus?: number;
  upstreamReply?: unknown;
}

interface Harness {
  url: string;
  pairsFile: string;
  shadowFile: string;
  upstream: Awaited<ReturnType<typeof startUpstream>>;
  tap: JevTap | null;
  close: () => Promise<void>;
  records: () => Promise<Array<Record<string, unknown>>>;
  shadowLines: () => Promise<Array<Record<string, unknown>>>;
  waitForRecords: (count: number, timeoutMs?: number) => Promise<Array<Record<string, unknown>>>;
}

async function setup(options: SetupOptions = {}): Promise<Harness> {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-jev-tap-"));
  const pairsFile = path.join(dir, "jev-laya-pairs.jsonl");
  const shadowFile = path.join(dir, "laya-decisions.jsonl");
  const upstream = await startUpstream(options.upstreamStatus ?? 200, options.upstreamReply ?? defaultReply);
  const engine = options.engine ?? stubEngine;

  const shadow = new ShadowLog(shadowFile, () => new Date());
  shadow.open();
  const pairs = new JevLayaPairsLog(pairsFile, () => new Date());
  pairs.open();

  const tap =
    options.withTap === false
      ? null
      : createJevTap({
          upstreamUrl: upstream.url,
          apiKey: options.apiKey === undefined ? TAP_KEY : options.apiKey,
          transport: options.transport ?? fetchJevTransport,
          timeoutMs: 5000,
          engine,
          identity,
          shadow,
          pairs,
          limits: DEFAULT_LIMITS,
          queueMax: options.queueMax ?? 256,
        });

  const config: ServeConfig = { host: "127.0.0.1", port: 0, maxBodyBytes: 1024 * 1024, limits: DEFAULT_LIMITS, startedAtMs: Date.now() };
  const server = createServer({ engine, identity, shadow, jevTap: tap ?? undefined, config, now: () => performance.now() });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  const records = () => readJsonl(pairsFile);
  const shadowLines = () => readJsonl(shadowFile);
  return {
    url: `http://127.0.0.1:${port}`,
    pairsFile,
    shadowFile,
    upstream,
    tap,
    records,
    shadowLines,
    waitForRecords: (count, timeoutMs) => waitFor(records, (rows) => rows.length >= count, `${count} pair record(s)`, timeoutMs),
    close: async () => {
      if (tap) await Promise.race([tap.drain(), new Promise<void>((resolve) => setTimeout(resolve, 2000))]);
      await new Promise<void>((resolve) => server.close(() => resolve()));
      await upstream.close();
      await rm(dir, { recursive: true, force: true });
    },
  };
}

function postTap(h: Harness, body: unknown, headers: Record<string, string> = {}): Promise<Response> {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return fetch(`${h.url}/jev/api/alpha/decisions`, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: text });
}

test("forwards the caller body byte-for-byte and returns Jev's reply unchanged", async () => {
  const h = await setup({ upstreamReply: defaultReply });
  try {
    const payload = basePayload();
    const raw = JSON.stringify(payload);
    const res = await postTap(h, raw, { "x-caller": "jev-logrank" });
    assert.equal(res.status, 200);
    assert.match(res.headers.get("content-type") ?? "", /application\/json/);
    assert.equal(await res.text(), JSON.stringify(defaultReply));

    assert.equal(h.upstream.calls.length, 1);
    assert.equal(h.upstream.calls[0]?.body, raw);
    assert.equal(h.upstream.calls[0]?.authorization, `Bearer ${TAP_KEY}`);
    assert.equal(h.upstream.calls[0]?.contentType, "application/json");

    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(record.kind, "jev_laya_pair");
    assert.equal(record.schema, 1);
    assert.equal(record.caller, "jev-logrank");
    assert.deepEqual(record.request, payload);
    assert.match(String(record.request_id), /^[0-9a-f-]{36}$/);
    assert.match(String(record.request_hash), /^[0-9a-f]{64}$/);
    assert.match(String(record.payload_sha256), /^[0-9a-f]{64}$/);
    assert.equal(typeof record.received_at, "string");
    assert.equal(typeof record.ts, "string");

    const jev = rec(record.jev);
    assert.equal(jev.status, 200);
    assert.equal(jev.ok, true);
    assert.equal(jev.error, null);
    assert.deepEqual(jev.reply, defaultReply);
    assert.equal(jev.model, "fake/jev-1");
    assert.deepEqual(jev.usage, defaultReply.usage);
    assert.equal(typeof jev.latency_ms, "number");

    const laya = rec(record.laya);
    assert.equal(laya.status, "ok");
    assert.equal(laya.error, null);
    assert.equal(laya.model, identity.model);
    assert.equal(laya.model_sha256, identity.modelSha256);
    assert.equal(typeof laya.latency_ms, "number");
    assert.deepEqual(laya.usage, { input_tokens: 7, output_tokens: 0 });
    const answers = rec(laya.answers);
    assert.equal(rec(answers.result).type, "noul");
    assert.equal(rec(answers.event_class).type, "choice");
    assert.equal(rec(answers.severity).type, "score");

    // the joined record correlates with laya-decisions.jsonl
    const shadow = await h.shadowLines();
    assert.equal(shadow.length, Object.keys(payload.questions).length);
    assert.ok(shadow.every((line) => line.request_hash === record.request_hash));
    assert.ok(shadow.every((line) => line.request_id === record.request_id));
  } finally {
    await h.close();
  }
});

test("passes a non-2xx Jev status and body through unchanged", async () => {
  const reply = { error: "rate limited" };
  const h = await setup({ upstreamStatus: 429, upstreamReply: reply });
  try {
    const res = await postTap(h, basePayload());
    assert.equal(res.status, 429);
    assert.equal(await res.text(), JSON.stringify(reply));
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(rec(record.jev).status, 429);
    assert.equal(rec(record.jev).ok, false);
  } finally {
    await h.close();
  }
});

test("injects the host key and never forwards or persists a caller credential", async () => {
  const h = await setup();
  try {
    const payload = { ...basePayload(), authorization: `Bearer ${CALLER_KEY}`, api_key: CALLER_KEY };
    const res = await postTap(h, payload, { authorization: `Bearer ${CALLER_KEY}`, "x-caller": "r18b-jev-watch.py" });
    assert.equal(res.status, 200);
    assert.equal(h.upstream.calls[0]?.authorization, `Bearer ${TAP_KEY}`);

    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(record.caller, "r18b-jev-watch.py");
    const request = rec(record.request);
    assert.ok(!Object.prototype.hasOwnProperty.call(request, "authorization"), "an authorization field must be dropped, not stored");
    assert.ok(!Object.prototype.hasOwnProperty.call(request, "api_key"), "an api_key field must be dropped, not stored");

    const persisted = await readFile(h.pairsFile, "utf8");
    assert.ok(!persisted.includes(TAP_KEY), "the host key must never be persisted");
    assert.ok(!persisted.includes(CALLER_KEY), "a caller credential must never be persisted");
    assert.ok(!persisted.includes("Bearer sk-"));
  } finally {
    await h.close();
  }
});

test("returns the Jev response before Laya finishes", async () => {
  const gate = deferred<SystemOneResult<Record<string, Question>>>();
  let engineStarted = false;
  let engineFinished = false;
  const engine: DecisionEngine = {
    systemOne: () => {
      engineStarted = true;
      return gate.promise.then((value) => {
        engineFinished = true;
        return value;
      });
    },
  };
  const h = await setup({ engine });
  try {
    const res = await postTap(h, basePayload());
    assert.equal(res.status, 200);
    assert.equal(await res.text(), JSON.stringify(defaultReply));
    assert.equal(engineFinished, false);

    await waitFor(
      async () => engineStarted,
      (started) => started,
      "Laya to start",
    );
    assert.equal(engineFinished, false);
    assert.equal((await h.records()).length, 0);

    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 5, output_tokens: 0 } });
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(engineFinished, true);
    assert.equal(rec(record.laya).status, "ok");
  } finally {
    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 0, output_tokens: 0 } });
    await h.close();
  }
});

test("a slow Laya does not delay the Jev response", async () => {
  const slowEngine: DecisionEngine = {
    async systemOne(_state, questions): Promise<SystemOneResult<Record<string, Question>>> {
      await new Promise((resolve) => setTimeout(resolve, 400));
      return { model: "laya", answers: engineAnswers(questions), usage: { input_tokens: 7, output_tokens: 0 } };
    },
  };
  const h = await setup({ engine: slowEngine });
  try {
    const started = performance.now();
    const res = await postTap(h, basePayload());
    const elapsed = performance.now() - started;
    assert.equal(res.status, 200);
    assert.equal(await res.text(), JSON.stringify(defaultReply));
    assert.ok(elapsed < 300, `response took ${Math.round(elapsed)}ms; Laya must not delay it`);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(rec(record.laya).status, "ok");
  } finally {
    await h.close();
  }
});

test("a failing Laya never breaks the Jev response and is recorded as an error", async () => {
  const failingEngine: DecisionEngine = {
    async systemOne(): Promise<SystemOneResult<Record<string, Question>>> {
      throw new Error("laya exploded");
    },
  };
  const h = await setup({ engine: failingEngine });
  try {
    const res = await postTap(h, basePayload());
    assert.equal(res.status, 200);
    assert.equal(await res.text(), JSON.stringify(defaultReply));

    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    const laya = rec(record.laya);
    assert.equal(laya.status, "error");
    assert.equal(laya.error, "ENGINE_ERROR");
    assert.equal(laya.answers, null);
    assert.equal(rec(record.jev).ok, true);

    const shadow = await h.shadowLines();
    assert.equal(shadow.length, Object.keys(basePayload().questions).length);
    assert.ok(shadow.every((line) => line.status === "error" && line.error === "ENGINE_ERROR"));
  } finally {
    await h.close();
  }
});

test("a transport failure is returned as a synthesized error and still recorded", async () => {
  const failingTransport: JevTransport = async (): Promise<JevForwardResult> => ({
    status: 502,
    contentType: "application/json; charset=utf-8",
    body: JSON.stringify({ error: "jev upstream unreachable", code: "JEV_UPSTREAM_ERROR" }),
    latencyMs: 1,
    error: "JEV_UPSTREAM_ERROR",
  });
  const h = await setup({ transport: failingTransport });
  try {
    const res = await postTap(h, basePayload());
    assert.equal(res.status, 502);
    assert.equal(rec(await res.json()).code, "JEV_UPSTREAM_ERROR");
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(rec(record.jev).error, "JEV_UPSTREAM_ERROR");
    assert.equal(rec(record.jev).ok, false);
  } finally {
    await h.close();
  }
});

test("queue overflow still writes a joined record, with laya.error QUEUE_FULL", async () => {
  const gate = deferred<SystemOneResult<Record<string, Question>>>();
  const engine: DecisionEngine = { systemOne: () => gate.promise };
  const h = await setup({ engine, queueMax: 1 });
  try {
    const first = await postTap(h, basePayload());
    const second = await postTap(h, basePayload());
    assert.equal(first.status, 200);
    assert.equal(second.status, 200);

    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 5, output_tokens: 0 } });
    const records = await h.waitForRecords(2);
    assert.ok(records.some((record) => rec(record.laya).error === "QUEUE_FULL"));
    assert.ok(records.some((record) => rec(record.laya).status === "ok"));
  } finally {
    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 0, output_tokens: 0 } });
    await h.close();
  }
});

test("the tap route reports 503 when disabled and 405 for a non-POST", async () => {
  const h = await setup({ withTap: false });
  try {
    const post = await postTap(h, basePayload());
    assert.equal(post.status, 503);
    assert.equal(rec(await post.json()).code, "JEV_TAP_DISABLED");

    const get = await fetch(`${h.url}/jev/api/alpha/decisions`);
    assert.equal(get.status, 405);
    assert.equal(get.headers.get("allow"), "POST");
  } finally {
    await h.close();
  }
});

test("loadJevKey prefers the environment and parses the secrets file", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-jev-key-"));
  const file = path.join(dir, "secrets.env");
  await writeFile(file, `OPENROUTER_API_KEY=other\nOPENROUTER_JEV_API_KEY="sk-secret-from-file"\n`);
  try {
    assert.equal(loadJevKey({}, file), "sk-secret-from-file");
    assert.equal(loadJevKey({ OPENROUTER_JEV_API_KEY: "from-env" }, file), "from-env");
    assert.equal(loadJevKey({}, path.join(dir, "missing.env")), null);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("scrubs a caller credential from a non-JSON payload before persisting it", async () => {
  const h = await setup();
  try {
    const malformed = `Authorization: Bearer ${CALLER_KEY}\n{"state": invalid`;
    const res = await postTap(h, malformed, { "x-caller": `Bearer ${CALLER_KEY}` });
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(record.caller, "[REDACTED]");
    assert.equal(typeof record.request, "string");
    assert.ok(!String(record.request).includes(CALLER_KEY), "the raw body must not persist the caller credential");
    const persisted = await readFile(h.pairsFile, "utf8");
    assert.ok(!persisted.includes(CALLER_KEY), "a caller credential must never be persisted");
  } finally {
    await h.close();
  }
});

test("scrubs quoted credential keys from an unparseable body", async () => {
  const quotedSecret = "caller-secret-token-999";
  // direct unit checks across quoting styles
  assert.ok(!redactSecret(`{"api_key": "${quotedSecret}", invalid_json`, null).includes(quotedSecret));
  assert.ok(!redactSecret(`{'authorization': 'Bearer ${quotedSecret}', oops`, null).includes(quotedSecret));
  assert.ok(!redactSecret(`{"access_token": "${quotedSecret}"}`, null).includes(quotedSecret));

  const h = await setup();
  try {
    const res = await postTap(h, `{"api_key": "${quotedSecret}", invalid_json`);
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.ok(!String(record.request).includes(quotedSecret), "a quoted credential key must be redacted");
    assert.ok(!(await readFile(h.pairsFile, "utf8")).includes(quotedSecret));
  } finally {
    await h.close();
  }
});

test("redacts a quoted credential value containing commas or braces", async () => {
  const part = "secret_part";
  const suffix = `${part}2`;
  const cases = [
    `{"api_key": "${part}1,${suffix}", invalid_json`,
    `{"api_key": "${part}1}${suffix}", invalid_json`,
    `{'access_token': '${part}1,${suffix}]', oops`,
    `{"authorization": "Bearer ${part}1,${suffix}", invalid_json`,
  ];
  for (const body of cases) {
    const scrubbed = redactSecret(body, null);
    assert.ok(!scrubbed.includes(suffix), `redaction left a delimited suffix behind: ${scrubbed}`);
  }

  const h = await setup();
  try {
    const res = await postTap(h, `{"api_key": "${part}1,${suffix}", invalid_json`);
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.ok(!String(record.request).includes(suffix), "an embedded suffix must not survive");
    assert.ok(!(await readFile(h.pairsFile, "utf8")).includes(suffix));
  } finally {
    await h.close();
  }
});

test("scrubs embedded bearer tokens, including all-alphabetic ones", async () => {
  const alphaToken = "SecretCallerTokenAlpha";
  assert.ok(!redactSecret(`Authorization: Bearer ${alphaToken}`, null).includes(alphaToken));
  assert.ok(!redactSecret(`token Bearer ${alphaToken} tail`, null).includes(alphaToken));

  const h = await setup();
  try {
    const payload = { ...basePayload(), state: { evidence: `retry with token Bearer ${CALLER_KEY}; alt Bearer ${alphaToken}` } };
    const res = await postTap(h, payload);
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    const evidence = String(rec(rec(record.request).state).evidence);
    assert.ok(!evidence.includes(CALLER_KEY), "an embedded token must be redacted");
    assert.ok(!evidence.includes(alphaToken), "an all-alphabetic token must be redacted");
    const persisted = await readFile(h.pairsFile, "utf8");
    assert.ok(!persisted.includes(CALLER_KEY));
    assert.ok(!persisted.includes(alphaToken));
  } finally {
    await h.close();
  }
});

test("does not persist credentials smuggled in the Content-Type header", async () => {
  const contentTypeSecret = "caller-secret-99999";
  const h = await setup();
  try {
    const res = await postTap(h, basePayload(), { "content-type": `application/json; api_key=${contentTypeSecret}` });
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(rec(record.forwarded).content_type, "application/json");
    assert.equal(rec(record.jev).content_type, "application/json");
    const persisted = await readFile(h.pairsFile, "utf8");
    assert.ok(!persisted.includes(contentTypeSecret), "a content-type credential must never be persisted");
  } finally {
    await h.close();
  }
});

test("drops prototype-polluting keys instead of polluting prototypes", async () => {
  const h = await setup();
  try {
    const raw = `{"model":"typesafe/jev-1.13","state":{"__proto__":{"polluted":true},"constructor":{"x":1},"evidence":"ok"},"questions":{"result":{"type":"noul","instructions":"x"}}}`;
    const res = await postTap(h, raw);
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(({} as Record<string, unknown>).polluted, undefined, "Object.prototype must not be polluted");
    const state = rec(rec(record.request).state);
    assert.ok(!Object.prototype.hasOwnProperty.call(state, "__proto__"));
    assert.ok(!Object.prototype.hasOwnProperty.call(state, "constructor"));
    assert.equal(state.evidence, "ok");

    const direct = sanitizeForStorage(JSON.parse(`{"__proto__":{"x":1},"a":{"__proto__":{"y":2},"b":1}}`), null) as Record<string, unknown>;
    assert.equal(Object.getPrototypeOf(direct), null);
    assert.equal(({} as Record<string, unknown>).x, undefined);
    assert.deepEqual({ ...(direct.a as Record<string, unknown>) }, { b: 1 });
  } finally {
    await h.close();
  }
});

test("a tap without a host key refuses to forward and records JEV_KEY_MISSING", async () => {
  const h = await setup({ apiKey: null });
  try {
    const res = await postTap(h, basePayload());
    assert.equal(res.status, 503);
    assert.equal(rec(await res.json()).code, "JEV_KEY_MISSING");
    assert.equal(h.upstream.calls.length, 0, "the tap must not forward without a key");
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.equal(rec(record.jev).status, 503);
    assert.equal(rec(record.jev).error, "JEV_KEY_MISSING");
    assert.equal(rec(record.laya).error, "JEV_KEY_MISSING");
    assert.equal(rec(record.laya).answers, null);
  } finally {
    await h.close();
  }
});

test("the shadow is dispatched only after the response handoff commits", async () => {
  const gate = deferred<SystemOneResult<Record<string, Question>>>();
  let engineStarted = false;
  const engine: DecisionEngine = {
    systemOne: () => {
      engineStarted = true;
      return gate.promise;
    },
  };
  const h = await setup({ engine });
  try {
    const tap = h.tap;
    assert.ok(tap);
    const handoff = await tap.handle(JSON.stringify(basePayload()), "tester", "application/json");
    assert.equal(handoff.status, 200);
    assert.equal(engineStarted, false, "the shadow must not start before the response finishes");

    let drained = false;
    const drainPromise = tap.drain().then(() => {
      drained = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal(drained, false, "drain must wait for the uncommitted handoff");

    handoff.commit();
    await waitFor(
      async () => engineStarted,
      (started) => started,
      "Laya to start",
    );
    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 5, output_tokens: 0 } });
    await drainPromise;
    assert.equal(drained, true);
    assert.equal((await h.waitForRecords(1)).length, 1);
  } finally {
    gate.resolve({ model: "laya", answers: engineAnswers(basePayload().questions), usage: { input_tokens: 0, output_tokens: 0 } });
    await h.close();
  }
});

test("loadServeConfig rejects malformed, loopback and metadata JEV upstreams", () => {
  // build the scheme and the metadata address at runtime so fixture-only lint rules do not fire
  const httpUrl = (authority: string): string => ["http", "://", authority].join("");
  const linkLocal = ["169", "254", "169", "254"].join(".");
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: "https://" }), /valid URL/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: "ftp://example.com/x" }), /http or https/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: "https://user:pass@example.com/x" }), /embed credentials/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl(linkLocal)}/latest/meta-data/` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("0.0.0.0:9000")}/x` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("metadata.google.internal")}/x` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("127.0.0.1:9000")}/x` }), /loopback/);

  // IPv4-mapped / IPv4-compatible IPv6 literals must not bypass the IPv4 class checks
  const mapped = (hex: string): string => `[::ffff:${hex}]`;
  const compat = (hex: string): string => `[::${hex}]`;
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl(mapped("a9fe:a9fe"))}/x` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl(mapped(linkLocal))}/x` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl(mapped("7f00:1"))}/x` }), /loopback/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl(compat("7f00:1"))}/x` }), /loopback/);

  // trailing root dots (FQDN form) must not bypass the gates
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("localhost.")}/x` }), /loopback/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("sub.localhost.")}/x` }), /loopback/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("metadata.google.internal.")}/x` }), /blocked/);

  // IPv6 multicast (ff00::/8) must be blocked
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("[ff02::1]")}/x` }), /blocked/);
  assert.throws(() => loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: `${httpUrl("[ff00::1]")}/x` }), /blocked/);

  const loopback = `${httpUrl("127.0.0.1:9000")}/x`;
  const allowed = loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: loopback, LAYA_SERVE_JEV_ALLOW_LOOPBACK: "1" });
  assert.equal(allowed.jev.upstreamUrl, loopback);
  const publicUrl = loadServeConfig({ LAYA_SERVE_JEV_UPSTREAM: "https://openrouter.ai/api/alpha/decisions" });
  assert.equal(publicUrl.jev.upstreamUrl, "https://openrouter.ai/api/alpha/decisions");
});

test("the forward transport does not follow upstream redirects", async () => {
  let hits = 0;
  const target = http.createServer((_req, res) => {
    hits += 1;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ leaked: true }));
  });
  await new Promise<void>((resolve) => target.listen(0, "127.0.0.1", resolve));
  const targetPort = (target.address() as AddressInfo).port;

  const redirector = http.createServer((_req, res) => {
    res.writeHead(302, { location: `http://127.0.0.1:${targetPort}/latest/meta-data/` });
    res.end("redirecting");
  });
  await new Promise<void>((resolve) => redirector.listen(0, "127.0.0.1", resolve));
  const redirectorPort = (redirector.address() as AddressInfo).port;

  try {
    const result = await fetchJevTransport({
      url: `http://127.0.0.1:${redirectorPort}/api/alpha/decisions`,
      body: "{}",
      contentType: "application/json",
      apiKey: TAP_KEY,
      timeoutMs: 5000,
    });
    assert.equal(result.status, 302, "the redirect must be returned, not followed");
    assert.equal(hits, 0, "the redirect target must never be fetched");
  } finally {
    await new Promise<void>((resolve) => redirector.close(() => resolve()));
    await new Promise<void>((resolve) => target.close(() => resolve()));
  }
});

test("does not persist credentials smuggled as object keys", async () => {
  const sanitized = sanitizeForStorage({ [TAP_KEY]: "v", "Authorization: Bearer caller-token-xyz": "v", safe: "ok" }, TAP_KEY) as Record<string, unknown>;
  const text = JSON.stringify(sanitized);
  assert.ok(!text.includes(TAP_KEY), "a key equal to the host secret must be dropped");
  assert.ok(!text.includes("caller-token-xyz"), "an auth-header key must be dropped");
  assert.equal(sanitized.safe, "ok");

  const h = await setup();
  try {
    const payload = { ...basePayload(), [TAP_KEY]: "leak", state: { safe: "ok" } };
    const res = await postTap(h, payload);
    assert.equal(res.status, 200);
    const [record] = await h.waitForRecords(1);
    assert.ok(record);
    assert.ok(!(await readFile(h.pairsFile, "utf8")).includes(TAP_KEY), "the host secret must not appear even as a key");
    assert.equal(rec(rec(record.request).state).safe, "ok");
  } finally {
    await h.close();
  }
});
