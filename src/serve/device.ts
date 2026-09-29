/**
 * Device / execution-provider policy for `laya-serve`.
 *
 * Raja's 2026-09-29 amendment allows Laya on the RTX PRO 4000 next to NInfer and
 * Qwen llama-server, under a HARD 2048 MiB VRAM ceiling and with a mandatory
 * CPU fallback so a broken/missing GPU can never take the service down:
 *
 *   - `LAYA_SERVE_DEVICE=cuda|cpu` (default `cuda` when available)
 *   - `LAYA_SERVE_PROVIDERS` explicit EP list (e.g. `tensorrt,cuda,cpu` for
 *     FP8/NVFP4). `LAYA_SERVE_DEVICE=cpu` always wins over it.
 *   - `LAYA_SERVE_GPU_MEM_MB` (default 2048) is the ceiling we advertise to the
 *     CUDA EP. onnxruntime-node 1.22 only forwards `deviceId` for the CUDA EP,
 *     so `gpu_mem_limit` / `arena_extend_strategy` are recorded but not honoured
 *     by the binding; the hard ceiling is enforced by refusing fp32/oversized
 *     bundles on the GPU (see `gpuCeilingViolation`) and by measuring the total
 *     GPU delta (see scripts/bench-latency.ts).
 *   - `LAYA_SERVE_GPU_DEVICE_ID` (default 0)
 *   - CPU knobs: `LAYA_SERVE_INTRA_OP_THREADS`, `LAYA_SERVE_INTER_OP_THREADS`
 *
 * No I/O happens at import time. The tiny-fixture probe is used to prove a GPU
 * provider stack can actually initialize before any identity claims `cuda`, and
 * by the (skipping) GPU test and the latency bench.
 */
import { existsSync, statSync } from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { Laya, type LayaOptions } from "../laya.js";

export type ServeDevice = "cuda" | "cpu";

export interface DeviceConfig {
  /** requested device; `cuda` is the default and falls back to cpu */
  requested: ServeDevice;
  /**
   * explicit execution-provider list (LAYA_SERVE_PROVIDERS), or null to derive
   * it from `requested`. Needed for FP8/NVFP4, which only the TensorRT EP can
   * execute: `tensorrt,cuda,cpu` (CPU still last as the per-node/init fallback).
   */
  providers: string[] | null;
  /** hard VRAM ceiling advertised to the CUDA EP, MiB (LAYA_SERVE_GPU_MEM_MB) */
  gpuMemMb: number;
  /** CUDA ordinal (LAYA_SERVE_GPU_DEVICE_ID) */
  gpuDeviceId: number;
  /** onnxruntime intra-op thread count for the CPU path, or null for the default */
  intraOpThreads: number | null;
  /** onnxruntime inter-op thread count for the CPU path, or null for the default */
  interOpThreads: number | null;
}

/** Provider names onnxruntime-node can actually bridge to a shared library. */
const KNOWN_PROVIDERS = new Set(["cpu", "cuda", "tensorrt"]);

/** Providers that mean "this is a GPU attempt". */
const GPU_PROVIDERS = new Set(["cuda", "tensorrt"]);

/** The largest ceiling we will accept, so a typo cannot claim the whole card. */
const MAX_GPU_MEM_MB = 1_000_000;

/**
 * Weight sizes above this cannot meet the 2048 MiB VRAM ceiling once the CUDA
 * context/arenas are added (fp32 weights are ~1.68 GB; quantized ≤ ~0.9 GB), so
 * such a bundle is forced onto CPU even if its directory name looks quantized.
 */
export const MAX_GPU_WEIGHTS_BYTES = 1024 * 1024 * 1024;

function integer(env: NodeJS.ProcessEnv, name: string, fallback: number, min: number, max: number): number {
  const raw = env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer in ${min}..${max}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function optionalInteger(env: NodeJS.ProcessEnv, name: string, min: number, max: number): number | null {
  const raw = env[name];
  if (raw === undefined || raw === "") return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer in ${min}..${max}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

/** Parse LAYA_SERVE_PROVIDERS (comma-separated, known names only), or null. */
function parseProviders(raw: string | undefined): string[] | null {
  if (raw === undefined || raw.trim() === "") return null;
  const list = raw
    .split(",")
    .map((name) => name.trim().toLowerCase())
    .filter((name) => name !== "");
  if (list.length === 0) return null;
  for (const name of list) {
    if (!KNOWN_PROVIDERS.has(name)) {
      throw new Error(`LAYA_SERVE_PROVIDERS entries must be one of ${[...KNOWN_PROVIDERS].join(", ")}, got ${JSON.stringify(raw)}`);
    }
  }
  return list;
}

/** Parse the process environment into the device policy. Fails closed on bad values. */
export function loadDeviceConfig(env: NodeJS.ProcessEnv = process.env): DeviceConfig {
  const rawDevice = (env.LAYA_SERVE_DEVICE ?? "cuda").trim().toLowerCase();
  if (rawDevice !== "cuda" && rawDevice !== "cpu") {
    throw new Error(`LAYA_SERVE_DEVICE must be "cuda" or "cpu", got ${JSON.stringify(env.LAYA_SERVE_DEVICE)}`);
  }
  const providers = parseProviders(env.LAYA_SERVE_PROVIDERS);
  if (rawDevice === "cpu" && providers?.some((name) => GPU_PROVIDERS.has(name))) {
    // `cpu` is the explicit opt-out; a GPU provider list would silently undo it.
    throw new Error("LAYA_SERVE_DEVICE=cpu conflicts with GPU entries in LAYA_SERVE_PROVIDERS; remove one of them");
  }
  return {
    requested: rawDevice,
    providers,
    gpuMemMb: integer(env, "LAYA_SERVE_GPU_MEM_MB", 2048, 1, MAX_GPU_MEM_MB),
    gpuDeviceId: integer(env, "LAYA_SERVE_GPU_DEVICE_ID", 0, 0, 1024),
    intraOpThreads: optionalInteger(env, "LAYA_SERVE_INTRA_OP_THREADS", 1, 4096),
    interOpThreads: optionalInteger(env, "LAYA_SERVE_INTER_OP_THREADS", 1, 4096),
  };
}

/** The provider name for a string or `{ name }` execution-provider config. */
export function providerName(provider: ort.InferenceSession.ExecutionProviderConfig): string {
  return typeof provider === "string" ? provider : provider.name;
}

/** True when the provider list names a GPU execution provider. */
export function isGpuAttempt(providers: ort.InferenceSession.ExecutionProviderConfig[]): boolean {
  return providers.some((provider) => GPU_PROVIDERS.has(providerName(provider)));
}

/**
 * Execution providers to hand to onnxruntime-node.
 *
 * `LAYA_SERVE_DEVICE=cpu` wins over everything. Otherwise `LAYA_SERVE_PROVIDERS`
 * is used when set (for FP8/NVFP4, which need `tensorrt,cuda,cpu`); otherwise the
 * CUDA entry carries `gpu_mem_limit` and `arena_extend_strategy` for a future
 * binding that honours them, with `cpu` last so ONNX Runtime can place nodes the
 * GPU EP does not implement instead of failing the whole session.
 */
export function buildExecutionProviders(cfg: DeviceConfig): ort.InferenceSession.ExecutionProviderConfig[] {
  if (cfg.requested === "cpu") return ["cpu"];
  if (cfg.providers) return cfg.providers.map((name) => (name === "cuda" ? cudaProvider(cfg) : name));
  return [cudaProvider(cfg), "cpu"];
}

function cudaProvider(cfg: DeviceConfig): ort.InferenceSession.ExecutionProviderConfig {
  return {
    name: "cuda",
    deviceId: cfg.gpuDeviceId,
    gpu_mem_limit: cfg.gpuMemMb * 1024 * 1024,
    arena_extend_strategy: "kSameAsRequested",
  } as unknown as ort.InferenceSession.ExecutionProviderConfig;
}

/** Session options shared by every device: full graph optimization + optional CPU threads. */
export function buildSessionOptions(cfg: DeviceConfig): ort.InferenceSession.SessionOptions {
  const options: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: "all" };
  if (cfg.intraOpThreads !== null) options.intraOpNumThreads = cfg.intraOpThreads;
  if (cfg.interOpThreads !== null) options.interOpNumThreads = cfg.interOpThreads;
  return options;
}

/** True when the bundle directory names itself as the unquantized fp32 export. */
export function isFp32ModelDir(modelDir: string): boolean {
  return path.basename(path.resolve(modelDir)).toLowerCase().includes("fp32");
}

/**
 * Total weight bytes for a bundle: `laya.onnx` plus `laya.onnx.data` when
 * present, or null when neither file can be read.
 *
 * The ONNX export stores weights externally (`laya.onnx.data`), but a single-file
 * model embeds them in `laya.onnx`; measuring only the `.data` file would report
 * 3.8 MB for a full 1.6 GB fp32 graph and let it onto the GPU.
 */
export function weightsByteLength(modelDir: string): number | null {
  let total = 0;
  let found = false;
  for (const file of ["laya.onnx", "laya.onnx.data"]) {
    try {
      total += statSync(path.join(modelDir, file)).size;
      found = true;
    } catch {
      // absent (or unreadable): the other file may still be measurable
    }
  }
  return found ? total : null;
}

/**
 * The reason a bundle may not run on the GPU under the 2048 MiB ceiling, or null
 * when it may.
 *
 * Fails closed: fp32 is always refused (≈2337 MiB measured before inference), an
 * oversized weight total is refused regardless of its directory name, and a
 * bundle whose weight size cannot be verified at all is refused too (an
 * unverifiable model must never be assumed small).
 */
export function gpuCeilingViolation(modelDir: string, weightsBytes: number | null): string | null {
  if (isFp32ModelDir(modelDir)) return `model directory ${path.basename(path.resolve(modelDir))} is the unquantized fp32 bundle`;
  if (weightsBytes === null) return `weight size could not be verified under ${path.resolve(modelDir)}`;
  if (weightsBytes > MAX_GPU_WEIGHTS_BYTES) {
    return `weights are ${weightsBytes} bytes, over the ${MAX_GPU_WEIGHTS_BYTES}-byte GPU budget`;
  }
  return null;
}

export interface EngineFallback {
  from: ServeDevice;
  to: ServeDevice;
  reason: string;
}

export interface EngineCeiling {
  from: ServeDevice;
  reason: string;
}

export interface EngineLoadOutcome {
  laya: Laya;
  /** device the session actually runs on */
  device: ServeDevice;
  /** provider names the session was created with (verified by the GPU probe) */
  providers: string[];
  /** set when a GPU attempt failed and CPU took over */
  fallback: EngineFallback | null;
  /** set when the 2048 MiB ceiling forced the bundle off the GPU before any attempt */
  ceiling: EngineCeiling | null;
}

export interface ProbeResult {
  available: boolean;
  reason: string | null;
}

export interface LoadEngineDeps {
  modelDir: string;
  cfg: DeviceConfig;
  /** injectable session loader (tests); defaults to `Laya.load` */
  load?: (options: LayaOptions) => Promise<Laya>;
  /** where fallback diagnostics go; defaults to stderr */
  log?: (message: string) => void;
  /** weight-size probe (tests); defaults to stat(laya.onnx.data) */
  weightsBytes?: () => number | null;
  /** GPU provider probe (tests); defaults to the tiny-fixture probe */
  probe?: (providers: ort.InferenceSession.ExecutionProviderConfig[]) => Promise<ProbeResult>;
}

const firstLine = (text: string): string => text.split("\n", 1)[0] ?? text;

/** Force a GPU request to CPU when the bundle cannot meet the 2048 MiB ceiling. */
function applyCeiling(
  cfg: DeviceConfig,
  modelDir: string,
  weightsBytes: number | null,
  log: (message: string) => void,
): { cfg: DeviceConfig; ceiling: EngineCeiling | null } {
  if (!isGpuAttempt(buildExecutionProviders(cfg))) return { cfg, ceiling: null };
  const reason = gpuCeilingViolation(modelDir, weightsBytes);
  if (reason === null) return { cfg, ceiling: null };
  log(`laya-serve: 2048 MiB VRAM ceiling: ${reason}; staying on the CPU execution provider`);
  return { cfg: { ...cfg, requested: "cpu", providers: null }, ceiling: { from: "cuda", reason } };
}

/**
 * Load the model on the requested providers, falling back to CPU when the bundle
 * cannot meet the VRAM ceiling or a GPU provider cannot initialize. Never throws
 * for a GPU failure: the CPU attempt is the failure path. A CPU failure still
 * throws, because no engine means no service.
 */
export async function loadEngine(deps: LoadEngineDeps): Promise<EngineLoadOutcome> {
  const load = deps.load ?? ((options: LayaOptions) => Laya.load(options));
  const log = deps.log ?? ((message: string) => process.stderr.write(`${message}\n`));
  const weights = deps.weightsBytes ? deps.weightsBytes() : weightsByteLength(deps.modelDir);
  const capped = applyCeiling(deps.cfg, deps.modelDir, weights, log);
  const sessionOptions = buildSessionOptions(capped.cfg);
  const requested = buildExecutionProviders(capped.cfg);

  if (!isGpuAttempt(requested)) {
    const laya = await load({ modelDir: deps.modelDir, executionProviders: ["cpu"], sessionOptions });
    return { laya, device: "cpu", providers: ["cpu"], fallback: null, ceiling: capped.ceiling };
  }

  const probe = deps.probe ?? probeProviders;
  const verified = await probe(requested);
  if (!verified.available) {
    const reason = verified.reason ?? "GPU provider probe failed";
    log(`laya-serve: GPU execution provider unavailable (${firstLine(reason)}); falling back to the CPU execution provider`);
    const laya = await load({ modelDir: deps.modelDir, executionProviders: ["cpu"], sessionOptions });
    return { laya, device: "cpu", providers: ["cpu"], fallback: { from: "cuda", to: "cpu", reason }, ceiling: capped.ceiling };
  }

  try {
    const laya = await load({ modelDir: deps.modelDir, executionProviders: requested, sessionOptions });
    return { laya, device: "cuda", providers: [...laya.providers], fallback: null, ceiling: capped.ceiling };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    log(`laya-serve: GPU execution provider unavailable (${firstLine(reason)}); falling back to the CPU execution provider`);
    const laya = await load({ modelDir: deps.modelDir, executionProviders: ["cpu"], sessionOptions });
    return { laya, device: "cpu", providers: ["cpu"], fallback: { from: "cuda", to: "cpu", reason }, ceiling: capped.ceiling };
  }
}

/** One dummy inference so the first real request does not pay the warm-up cost. */
export async function warmupEngine(laya: Laya, log: (message: string) => void = (m) => process.stderr.write(`${m}\n`)): Promise<boolean> {
  try {
    await laya.systemOne("laya-serve warmup", {
      warmup: { type: "noul", instructions: "warmup", criteria: { true: "yes", false: "no" } },
    });
    return true;
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    log(`laya-serve: warm-up inference failed (${firstLine(reason)}); continuing`);
    return false;
  }
}

export interface CudaProbe {
  available: boolean;
  /** first line of the CUDA failure, or null when available */
  reason: string | null;
  providers: string[];
}

/** Repo-relative tiny ONNX fixture used to probe the CUDA EP without the 1.6 GB bundle. */
export function tinyModelPath(): string {
  return path.resolve(import.meta.dirname, "../../test/fixtures/tiny-relu.onnx");
}

/**
 * The providers a probe must use: GPU entries only.
 *
 * The real session keeps `"cpu"` last for per-node fallback, but including it in
 * the probe would let ONNX Runtime satisfy the probe entirely on CPU after a GPU
 * provider failed to initialize — masking the failure and producing a false
 * `cuda` identity. When the list is CPU-only (already decided CPU), it is kept.
 */
export function gpuOnlyProviders(providers: ort.InferenceSession.ExecutionProviderConfig[]): ort.InferenceSession.ExecutionProviderConfig[] {
  const gpu = providers.filter((provider) => providerName(provider) !== "cpu");
  return gpu.length > 0 ? gpu : providers;
}

/**
 * Try to open a session on a 95-byte Relu fixture with the exact GPU provider
 * stack that will be used for the real model (the `"cpu"` fallback entry is
 * dropped, see `gpuOnlyProviders`). Returns `{ available: false, reason }`
 * instead of throwing when the provider library is missing or the device cannot
 * be initialized, so identity never claims `cuda` for a stack that cannot run.
 */
export async function probeProviders(providers: ort.InferenceSession.ExecutionProviderConfig[]): Promise<ProbeResult> {
  const fixture = tinyModelPath();
  if (!existsSync(fixture)) return { available: false, reason: `probe fixture missing: ${fixture}` };
  const probed = gpuOnlyProviders(providers);
  let session: ort.InferenceSession | null = null;
  try {
    session = await ort.InferenceSession.create(fixture, { executionProviders: probed, graphOptimizationLevel: "all" });
    await session.run({ input: new ort.Tensor("float32", Float32Array.from([-1, 2, -3, 4]), [1, 4]) });
    return { available: true, reason: null };
  } catch (error) {
    return { available: false, reason: firstLine(error instanceof Error ? error.message : String(error)) };
  } finally {
    if (session) await session.release().catch(() => undefined);
  }
}

/**
 * Probe the CUDA provider stack. Used by the skipping GPU test and the bench;
 * `loadEngine` probes the same way before it opens the real session.
 */
export async function probeCuda(cfg: DeviceConfig = loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" })): Promise<CudaProbe> {
  const providers = buildExecutionProviders({ ...cfg, requested: "cuda" });
  const result = await probeProviders(providers);
  return { available: result.available, reason: result.reason, providers: result.available ? providers.map(providerName) : [] };
}
