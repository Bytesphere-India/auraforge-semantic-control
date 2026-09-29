/**
 * Pure answer-parity helpers for `scripts/check-parity.ts`.
 *
 * Kept free of onnxruntime imports so the fail-closed key comparison can be
 * unit-tested offline. `metrics` flattens an answer set into
 * `<question>.<field>` numbers; `compareMetricKeys` refuses to compare anything
 * unless the baseline and the variant expose exactly the same fields.
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

export type ParityKeyFailure = "BASELINE_EMPTY" | "VARIANT_EMPTY" | "ANSWER_KEY_MISMATCH";

export interface ParityKeyDiff {
  ok: boolean;
  reason?: ParityKeyFailure;
  /** keys the baseline produced that the variant did not */
  missing_in_variant?: string[];
  /** keys the variant produced that the baseline did not */
  missing_in_baseline?: string[];
}

/**
 * Fail closed: the two answer sets must have exactly the same non-empty key set.
 * A variant that returns `{}` or silently drops a question must never look like a
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
  return { ok: true };
}
