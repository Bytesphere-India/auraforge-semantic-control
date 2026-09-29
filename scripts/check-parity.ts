#!/usr/bin/env node
/**
 * Answer-parity check for the Laya speed lane (Raja 2026-09-29, item 6).
 *
 * Loads the fp32 baseline and one candidate variant (nvfp4/fp8/fp16) and runs
 * both on the repo test fixtures, then reports the maximum absolute difference
 * of `noul`, of every probability, and of every score.
 *
 * Exit codes are fail-closed:
 *   0  parity measured, answer key sets identical
 *   2  the variant could not be loaded with the requested providers
 *   3  the variant answers are empty or their key set differs from the baseline
 *      (the check must never pass a non-functional/corrupt model as 0-diff)
 *
 *   ./node_modules/.bin/tsx scripts/check-parity.ts \
 *     --baseline /opt/auraforge/models/laya/base-fp32 \
 *     --variant ~/models/laya/base-fp8 --providers tensorrt,cuda,cpu
 */
import { Laya } from "../src/laya.js";
import { FIXTURES } from "./fixtures.js";
import { compareMetricKeys, metrics } from "./parity.js";

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

/** Terminal parity failure: the caller must not read a diff out of this run. */
class ParityError extends Error {
  constructor(readonly payload: Record<string, unknown>) {
    super(String(payload.reason));
  }
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
      const baselineMetrics = metrics(await baseline.systemOne(fixture.state, fixture.questions));
      const variantMetrics = metrics(await variant.systemOne(fixture.state, fixture.questions));
      const keys = compareMetricKeys(baselineMetrics, variantMetrics);
      if (!keys.ok) {
        throw new ParityError({
          ok: false,
          variant: variantDir,
          providers,
          fixture: fixture.name,
          reason: keys.reason,
          missing_in_variant: keys.missing_in_variant,
          missing_in_baseline: keys.missing_in_baseline,
          non_finite_keys: keys.non_finite_keys,
        });
      }
      let caseMax = 0;
      let caseKey: string | null = null;
      for (const key of Object.keys(baselineMetrics)) {
        const delta = Math.abs((baselineMetrics[key] ?? 0) - (variantMetrics[key] ?? 0));
        if (!Number.isFinite(delta)) {
          // Belt and braces: compareMetricKeys already rejects non-finite values,
          // but a NaN delta must never be silently folded into a 0-max case.
          throw new ParityError({ ok: false, variant: variantDir, providers, fixture: fixture.name, reason: "NON_FINITE_METRIC", non_finite_keys: [key] });
        }
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
  } catch (error) {
    if (error instanceof ParityError) {
      process.stdout.write(`${JSON.stringify(error.payload, null, 2)}\n`);
      process.exitCode = 3;
      return;
    }
    throw error;
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
