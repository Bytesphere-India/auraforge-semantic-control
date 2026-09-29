/**
 * Fail-closed answer-parity tests (review findings: a variant that returns `{}`,
 * drops an answer key, or emits `NaN`/`Infinity` must never be reported as a
 * zero-difference pass). Pure: no model, no onnxruntime.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { compareMetricKeys, metrics, nonFiniteMetricKeys, planVariantProviders } from "../scripts/parity.js";
import type { Question, SystemOneResult } from "../src/types.js";

test("planVariantProviders strips cpu from a GPU stack so parity cannot pass on CPU", () => {
  const gpu = planVariantProviders(["tensorrt", "cuda", "cpu"]);
  assert.equal(gpu.gpuRequested, true);
  assert.deepEqual(gpu.loadProviders, ["tensorrt", "cuda"]);

  const cudaOnly = planVariantProviders(["cuda", "cpu"]);
  assert.equal(cudaOnly.gpuRequested, true);
  assert.deepEqual(cudaOnly.loadProviders, ["cuda"]);

  const cpu = planVariantProviders(["cpu"]);
  assert.equal(cpu.gpuRequested, false);
  assert.deepEqual(cpu.loadProviders, ["cpu"]);
});

test("compareMetricKeys accepts identical non-empty key sets", () => {
  const diff = compareMetricKeys({ "a.noul": 0.1, "a.confidence": 0.2 }, { "a.noul": 0.9, "a.confidence": 0.8 });
  assert.equal(diff.ok, true);
  assert.equal(diff.reason, undefined);
});

test("compareMetricKeys fails closed on an empty variant answer set", () => {
  const diff = compareMetricKeys({ "a.noul": 0.1 }, {});
  assert.equal(diff.ok, false);
  assert.equal(diff.reason, "VARIANT_EMPTY");
});

test("compareMetricKeys fails closed on an empty baseline answer set", () => {
  const diff = compareMetricKeys({}, { "a.noul": 0.1 });
  assert.equal(diff.ok, false);
  assert.equal(diff.reason, "BASELINE_EMPTY");
});

test("compareMetricKeys fails closed on a dropped or extra key", () => {
  const dropped = compareMetricKeys({ "a.noul": 0.1, "b.score": 1.5 }, { "a.noul": 0.1 });
  assert.equal(dropped.ok, false);
  assert.equal(dropped.reason, "ANSWER_KEY_MISMATCH");
  assert.deepEqual(dropped.missing_in_variant, ["b.score"]);
  assert.deepEqual(dropped.missing_in_baseline, []);

  const extra = compareMetricKeys({ "a.noul": 0.1 }, { "a.noul": 0.1, "b.noul": 0.2 });
  assert.equal(extra.ok, false);
  assert.deepEqual(extra.missing_in_baseline, ["b.noul"]);
});

test("compareMetricKeys fails closed on NaN or infinite variant metrics", () => {
  const nan = compareMetricKeys({ "a.noul": 0.1 }, { "a.noul": Number.NaN });
  assert.equal(nan.ok, false);
  assert.equal(nan.reason, "NON_FINITE_METRIC");
  assert.deepEqual(nan.non_finite_keys, ["a.noul"]);

  const infinite = compareMetricKeys({ "a.noul": 0.1, "a.confidence": 0.2 }, { "a.noul": 0.1, "a.confidence": Number.POSITIVE_INFINITY });
  assert.equal(infinite.ok, false);
  assert.deepEqual(infinite.non_finite_keys, ["a.confidence"]);
});

test("compareMetricKeys fails closed on a non-finite baseline metric", () => {
  const diff = compareMetricKeys({ "a.noul": Number.NaN }, { "a.noul": 0.1 });
  assert.equal(diff.ok, false);
  assert.equal(diff.reason, "NON_FINITE_METRIC");
});

test("nonFiniteMetricKeys reports every offending key once, sorted", () => {
  const keys = nonFiniteMetricKeys({ "b.noul": 0.1, "a.noul": Number.NaN }, { "b.noul": Number.NEGATIVE_INFINITY, "a.noul": 0.2 });
  assert.deepEqual(keys, ["a.noul", "b.noul"]);
  assert.deepEqual(nonFiniteMetricKeys({ "a.noul": 0.1 }, { "a.noul": 0.2 }), []);
});

test("metrics preserves a NaN so the comparison can reject it", () => {
  const result: SystemOneResult<Record<string, Question>> = {
    model: "laya",
    usage: { input_tokens: 1, output_tokens: 0 },
    answers: { n: { type: "noul", noul: Number.NaN, rl_agent: { act_probability: 1 } } },
  };
  assert.ok(Number.isNaN(metrics(result)["n.noul"]));
});

test("metrics flattens noul, choice and score answers", () => {
  const result: SystemOneResult<Record<string, Question>> = {
    model: "laya",
    usage: { input_tokens: 1, output_tokens: 0 },
    answers: {
      n: { type: "noul", noul: 0.25, rl_agent: { act_probability: 1 } },
      c: { type: "choice", choice: "a", probabilities: { a: 0.6, b: 0.4 }, confidence: 0.2, rl_agent: { act_probability: 1 } },
      s: {
        type: "score",
        score: 1.5,
        legend: { "0": "lo", "1": "hi" },
        probabilities: { "0": 0.5, "1": 0.5 },
        confidence: 0.1,
        rl_agent: { act_probability: 1 },
      },
    },
  };
  const flat = metrics(result);
  assert.equal(flat["n.noul"], 0.25);
  assert.equal(flat["c.p.a"], 0.6);
  assert.equal(flat["c.p.b"], 0.4);
  assert.equal(flat["c.confidence"], 0.2);
  assert.equal(flat["s.score"], 1.5);
  assert.equal(flat["s.p.0"], 0.5);
  assert.equal(flat["s.confidence"], 0.1);
});
