#!/usr/bin/env node
/**
 * Answer-parity check for the Laya speed lane (Raja 2026-09-29, item 6).
 *
 * Loads the fp32 baseline and one candidate variant (nvfp4/fp8/fp16) and runs
 * both on the repo test fixtures, then reports the maximum absolute difference
 * of `noul`, of every probability, and of every score. Exit code 2 when the
 * variant cannot be loaded with the requested providers (for example the
 * TensorRT EP without libnvinfer), so a caller can fall back.
 *
 *   ./node_modules/.bin/tsx scripts/check-parity.ts \
 *     --baseline /opt/auraforge/models/laya/base-fp32 \
 *     --variant ~/models/laya/base-fp8 --providers tensorrt,cuda,cpu
 */
import { Laya } from "../src/laya.js";
import type { Answer, Question, SystemOneResult } from "../src/types.js";
import { FIXTURES } from "./fixtures.js";

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

/** flatten an answer set into comparable numeric metrics */
function metrics(result: SystemOneResult<Record<string, Question>>): Record<string, number> {
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

async function main(): Promise<void> {
  const baselineDir = arg("--baseline") ?? "/opt/auraforge/models/laya/base-fp32";
  const variantDir = arg("--variant");
  if (!variantDir) throw new Error("--variant is required");
  const providers = (arg("--providers") ?? "tensorrt,cuda,cpu")
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);

  const baseline = await Laya.load({ modelDir: baselineDir, executionProviders: ["cpu"] });
  let variant: Laya;
  try {
    variant = await Laya.load({ modelDir: variantDir, executionProviders: providers });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify({ ok: false, variant: variantDir, providers, reason }, null, 2)}\n`);
    await baseline.close();
    process.exit(2);
    return;
  }

  const diffs: Record<string, number> = {};
  const perCase: Array<{ fixture: string; max_abs_diff: number; max_key: string | null }> = [];
  let maxNoul = 0;
  let maxProbability = 0;
  try {
    for (const fixture of FIXTURES) {
      const a = metrics(await baseline.systemOne(fixture.state, fixture.questions));
      const b = metrics(await variant.systemOne(fixture.state, fixture.questions));
      let caseMax = 0;
      let caseKey: string | null = null;
      for (const key of Object.keys(a)) {
        if (!(key in b)) continue;
        const delta = Math.abs((a[key] ?? 0) - (b[key] ?? 0));
        diffs[key] = delta;
        if (key.endsWith(".noul")) maxNoul = Math.max(maxNoul, delta);
        else if (key.includes(".p.")) maxProbability = Math.max(maxProbability, delta);
        if (delta > caseMax) {
          caseMax = delta;
          caseKey = key;
        }
      }
      perCase.push({ fixture: fixture.name, max_abs_diff: caseMax, max_key: caseKey });
    }
  } finally {
    await baseline.close();
    await variant.close();
  }

  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        baseline: baselineDir,
        variant: variantDir,
        variant_providers: variant.providers,
        baseline_providers: baseline.providers,
        max_abs_diff_noul: maxNoul,
        max_abs_diff_probability: maxProbability,
        per_case: perCase,
        per_metric: diffs,
      },
      null,
      2,
    )}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(`check-parity: fatal: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
