/**
 * Pure answer-parity helpers for `scripts/check-parity.ts`.
 *
 * Kept free of onnxruntime imports so the fail-closed comparison can be
 * unit-tested offline. `metrics` flattens an answer set into
 * `<question>.<field>` numbers; `compareMetricKeys` refuses to compare anything
 * unless the baseline and the variant expose exactly the same fields **and every
 * value is finite** (NaN/Infinity must never look like a zero-difference pass).
 */
import type { Answer, Question, SystemOneResult } from "../src/types.js";

/** flatten an answer set into comparable numeric metrics */
export function metrics(result: SystemOneResult<Record<string, Question>>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [qid, answer] of Object.entries(result.answers as Record<string, Answer>)) {
    if (answer.type === "noul") {
      out[`${qid}.noul`] = answer.noul;
    } else if (answer.type === "choice") {
      for (const [key, value] of Object.entries(answer.probabilities)) out[`${qid}.p.${key}`] = value;
      out[`${qid}.confidence`] = answer.confidence;
    } else {
      for (const [key, value] of Object.entries(answer.probabilities)) out[`${qid}.p.${key}`] = value;
      out[`${qid}.score`] = answer.score;
      out[`${qid}.confidence`] = answer.confidence;
    }
  }
  return out;
}

export type ParityKeyFailure = "BASELINE_EMPTY" | "VARIANT_EMPTY" | "ANSWER_KEY_MISMATCH" | "NON_FINITE_METRIC";

export interface ParityKeyDiff {
  ok: boolean;
  reason?: ParityKeyFailure;
  /** keys the baseline produced that the variant did not */
  missing_in_variant?: string[];
  /** keys the variant produced that the baseline did not */
  missing_in_baseline?: string[];
  /** keys whose baseline or variant value is NaN/±Infinity */
  non_finite_keys?: string[];
}

/** Metric keys whose value is not finite in either answer set (deduped, sorted). */
export function nonFiniteMetricKeys(baseline: Record<string, number>, variant: Record<string, number>): string[] {
  const keys = new Set<string>();
  for (const [key, value] of Object.entries(baseline)) if (!Number.isFinite(value)) keys.add(key);
  for (const [key, value] of Object.entries(variant)) if (!Number.isFinite(value)) keys.add(key);
  return [...keys].sort((a, b) => a.localeCompare(b));
}

/**
 * Fail closed: the two answer sets must have exactly the same non-empty key set
 * and every mapped value must be finite. A variant that returns `{}`, silently
 * drops a question, or emits `NaN`/`Infinity` must never look like a
 * zero-difference pass.
 */
export function compareMetricKeys(baseline: Record<string, number>, variant: Record<string, number>): ParityKeyDiff {
  if (Object.keys(baseline).length === 0) return { ok: false, reason: "BASELINE_EMPTY" };
  if (Object.keys(variant).length === 0) return { ok: false, reason: "VARIANT_EMPTY" };
  const missingInVariant = Object.keys(baseline).filter((key) => !(key in variant));
  const missingInBaseline = Object.keys(variant).filter((key) => !(key in baseline));
  if (missingInVariant.length > 0 || missingInBaseline.length > 0) {
    missingInVariant.sort((a, b) => a.localeCompare(b));
    missingInBaseline.sort((a, b) => a.localeCompare(b));
    return { ok: false, reason: "ANSWER_KEY_MISMATCH", missing_in_variant: missingInVariant, missing_in_baseline: missingInBaseline };
  }
  const nonFinite = nonFiniteMetricKeys(baseline, variant);
  if (nonFinite.length > 0) return { ok: false, reason: "NON_FINITE_METRIC", non_finite_keys: nonFinite };
  return { ok: true };
}

export interface VariantProviderPlan {
  /** true when at least one requested provider is a GPU provider */
  gpuRequested: boolean;
  /** providers to open the variant session with; `"cpu"` is dropped for GPU loads */
  loadProviders: string[];
}

/**
 * The variant session must load on the providers the variant actually needs.
 *
 * A GPU variant (fp8/nvfp4 on TensorRT, fp16 on CUDA) must never be validated
 * through a session that silently fell back to `"cpu"`: ONNX Runtime accepts a
 * mixed list like `["tensorrt", "cuda", "cpu"]` and can satisfy it on CPU when
 * the GPU provider fails to initialize, which would mark the variant `"ok"` and
 * stop the fp16 fallback. So for a GPU load the `"cpu"` entry is removed, making
 * a GPU init failure throw (exit 2) instead of passing parity.
 */
export function planVariantProviders(providers: string[]): VariantProviderPlan {
  const gpuRequested = providers.some((provider) => provider.toLowerCase() !== "cpu");
  const loadProviders = gpuRequested ? providers.filter((provider) => provider.toLowerCase() !== "cpu") : ["cpu"];
  return { gpuRequested, loadProviders };
}
