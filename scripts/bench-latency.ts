#!/usr/bin/env node
/**
 * Laya decision-latency bench (Raja 2026-09-29 speed lane).
 *
 * Measures the wall-clock latency of one batched `systemOne` call (the docs'
 * three-question sample) for a small state and a 3 KB state, per device, and
 * reports p50/p95 over N runs. GPU is optional: a device whose execution
 * provider cannot initialize is reported as SKIP with the reason, never a crash.
 *
 *   yarn bench:latency
 *   LAYA_SERVE_MODEL_DIR=~/models/laya/base-fp16 LAYA_BENCH_DEVICES=cuda yarn bench:latency
 *
 * VRAM is sampled with nvidia-smi before and after each device's runs. Under
 * WSL2 per-process accounting is not exposed, so the number is the device total
 * delta, which is the honest ceiling check.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { loadDeviceConfig, loadEngine, probeCuda, warmupEngine, type ServeDevice } from "../src/serve/device.js";
import type { Question } from "../src/types.js";
import { evidenceState, FIXTURES } from "./fixtures.js";

const MODEL_DIR = process.env.LAYA_SERVE_MODEL_DIR ?? "/opt/auraforge/models/laya/base-fp32";
const RUNS = Math.max(1, Number(process.env.LAYA_BENCH_RUNS ?? 5));
const DEVICES = (process.env.LAYA_BENCH_DEVICES ?? "cuda,cpu")
  .split(",")
  .map((d) => d.trim().toLowerCase())
  .filter((d): d is ServeDevice => d === "cuda" || d === "cpu");

/** The three-question sample from scripts/fixtures.ts (docs/laya-serve.md). */
const SAMPLE = FIXTURES.find((fixture) => fixture.name === "docs_small");
if (!SAMPLE) throw new Error("bench-latency: docs_small fixture is missing");
const QUESTIONS: Record<string, Question> = SAMPLE.questions;

const SMALL = evidenceState(64);
const LARGE = evidenceState(3072);

function percentile(sorted: number[], p: number): number {
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length - 1, Math.max(0, rank - 1))] ?? 0;
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** Absolute candidates only: never resolve nvidia-smi through $PATH. */
const NVIDIA_SMI_CANDIDATES = ["/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/usr/local/cuda/bin/nvidia-smi"];

function gpuMemUsedMb(): number | null {
  const smi = NVIDIA_SMI_CANDIDATES.find((candidate) => existsSync(candidate));
  if (!smi) return null;
  const r = spawnSync(smi, ["--query-gpu=memory.used", "--format=csv,noheader,nounits"], { encoding: "utf8", timeout: 10_000 });
  if (r.status !== 0 || typeof r.stdout !== "string") return null;
  const value = Number(r.stdout.trim().split("\n")[0]);
  return Number.isFinite(value) ? value : null;
}

interface CaseResult {
  case: "small" | "3kb";
  state_bytes: number;
  runs: number;
  p50_ms: number;
  p95_ms: number;
  min_ms: number;
  max_ms: number;
  input_tokens: number;
}

interface DeviceResult {
  device: ServeDevice;
  requested: ServeDevice;
  providers: string[];
  fallback: string | null;
  status: "ok" | "skip";
  reason?: string;
  warmed: boolean;
  vram_used_before_mb: number | null;
  vram_used_after_mb: number | null;
  vram_delta_mb: number | null;
  cases: CaseResult[];
}

async function runDevice(requested: ServeDevice): Promise<DeviceResult> {
  const base = loadDeviceConfig({ ...process.env, LAYA_SERVE_DEVICE: requested });

  if (requested === "cuda") {
    const probe = await probeCuda(base);
    if (!probe.available) {
      return {
        device: "cuda",
        requested,
        providers: [],
        fallback: null,
        status: "skip",
        reason: probe.reason ?? "CUDA unavailable",
        warmed: false,
        vram_used_before_mb: gpuMemUsedMb(),
        vram_used_after_mb: gpuMemUsedMb(),
        vram_delta_mb: null,
        cases: [],
      };
    }
  }

  const before = gpuMemUsedMb();
  const outcome = await loadEngine({
    modelDir: MODEL_DIR,
    cfg: base,
    log: (m) => process.stderr.write(`${m}\n`),
  });
  const laya = outcome.laya;
  const warmed = await warmupEngine(laya);
  const cases: CaseResult[] = [];
  const toRun: Array<{ label: "small" | "3kb"; state: { evidence: string }; bytes: number }> = [
    { label: "small", state: SMALL, bytes: 64 },
    { label: "3kb", state: LARGE, bytes: 3072 },
  ];
  try {
    for (const { label, state, bytes } of toRun) {
      const times: number[] = [];
      let inputTokens = 0;
      for (let i = 0; i < RUNS; i++) {
        const t0 = performance.now();
        const result = await laya.systemOne(state, QUESTIONS);
        times.push(performance.now() - t0);
        inputTokens = result.usage.input_tokens;
      }
      const sorted = [...times].sort((a, b) => a - b);
      cases.push({
        case: label,
        state_bytes: bytes,
        runs: RUNS,
        p50_ms: round2(percentile(sorted, 50)),
        p95_ms: round2(percentile(sorted, 95)),
        min_ms: round2(sorted[0] ?? 0),
        max_ms: round2(sorted[sorted.length - 1] ?? 0),
        input_tokens: inputTokens,
      });
    }
  } finally {
    await laya.close();
  }
  const after = gpuMemUsedMb();
  return {
    device: outcome.device,
    requested,
    providers: outcome.providers,
    fallback: outcome.fallback ? outcome.fallback.reason : null,
    status: "ok",
    warmed,
    vram_used_before_mb: before,
    vram_used_after_mb: after,
    vram_delta_mb: before !== null && after !== null ? after - before : null,
    cases,
  };
}

async function main(): Promise<void> {
  const results: DeviceResult[] = [];
  for (const device of DEVICES) results.push(await runDevice(device));

  for (const r of results) {
    if (r.status === "skip") {
      process.stdout.write(`\n${r.requested}: SKIP (${r.reason})\n`);
      continue;
    }
    const fallbackNote = r.fallback ? ` (fallback: ${r.fallback})` : "";
    process.stdout.write(`\n${r.device}: providers=${r.providers.join(",")}${fallbackNote} warmed=${r.warmed} model=${MODEL_DIR}\n`);
    for (const c of r.cases) {
      process.stdout.write(
        `  ${c.case.padEnd(6)} ${String(c.state_bytes).padStart(5)} B  ` +
          `p50=${c.p50_ms.toFixed(1)}ms p95=${c.p95_ms.toFixed(1)}ms ` +
          `min=${c.min_ms.toFixed(1)}ms max=${c.max_ms.toFixed(1)}ms tokens=${c.input_tokens}\n`,
      );
    }
    if (r.vram_delta_mb !== null) {
      process.stdout.write(`  VRAM used ${r.vram_used_before_mb} -> ${r.vram_used_after_mb} MiB (delta ${r.vram_delta_mb} MiB)\n`);
    } else {
      process.stdout.write("  VRAM: nvidia-smi unavailable\n");
    }
  }

  process.stdout.write(`\n${JSON.stringify({ model_dir: MODEL_DIR, runs: RUNS, results }, null, 2)}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`bench-latency: fatal: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
