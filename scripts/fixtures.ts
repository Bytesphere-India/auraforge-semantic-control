/**
 * Shared test fixtures for the Laya speed-lane tooling.
 *
 * These are exactly the request payloads the repo tests/docs already use
 * (test/test_model.ts and the docs/laya-serve.md sample). Quantization
 * calibration and answer-parity checks must only ever run on these, never on
 * shadow-log data.
 */
import type { Question } from "../src/types.js";

export interface Fixture {
  name: string;
  state: unknown;
  questions: Record<string, Question>;
}

export const noul = (instructions: string): Question => ({
  type: "noul",
  instructions,
  criteria: { true: "yes", false: "no" },
});
export const choice = (instructions: string, criteria: Record<string, string>): Question => ({ type: "choice", instructions, criteria });
export const score = (instructions: string, criteria: string[]): Question => ({ type: "score", instructions, criteria });

/** A deterministic state whose JSON serialization is `targetBytes` long. */
export function evidenceState(targetBytes: number): { evidence: string } {
  const overhead = '{"evidence": ""}'.length;
  const filler = "the oracle timed out after three retries and the executor returned a schema error; ";
  let text = "incident evidence: ";
  while (text.length < targetBytes - overhead) text += filler;
  return { evidence: text.slice(0, Math.max(0, targetBytes - overhead)) };
}

export const FIXTURES: Fixture[] = [
  {
    name: "test_model_refund",
    state: {
      subject: "Refund not received",
      body: "I cancelled my subscription two weeks ago and I still have not received my refund. This is the third time I am writing. If this is not resolved I will dispute the charge with my bank.",
    },
    questions: {
      department: choice("Which team should handle this ticket?", {
        billing: "payments, refunds, invoices",
        support: "product help and bugs",
        sales: "new purchases and upgrades",
      }),
      urgency: score("How urgent is this ticket?", ["not urgent", "somewhat urgent", "urgent", "critical"]),
      churn_risk: noul("Is the customer likely to cancel or dispute?"),
    },
  },
  {
    name: "docs_small",
    state: { evidence: "oracle timed out twice" },
    questions: {
      transient_recoverable: noul("Is the failure transient and recoverable by retrying the same call?"),
      event_class: choice("Classify the incident as a transient retryable failure or a defective oracle.", {
        transient: "retryable",
        broken_oracle: "defective oracle",
      }),
      severity: score("How severe is this incident?", ["cosmetic", "minor", "moderate", "major", "critical"]),
    },
  },
  {
    name: "docs_3kb",
    state: evidenceState(3072),
    questions: {
      transient_recoverable: noul("Is the failure transient and recoverable by retrying the same call?"),
      event_class: choice("Classify the incident as a transient retryable failure or a defective oracle.", {
        transient: "retryable",
        broken_oracle: "defective oracle",
      }),
      severity: score("How severe is this incident?", ["cosmetic", "minor", "moderate", "major", "critical"]),
    },
  },
];
