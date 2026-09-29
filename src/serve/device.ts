/**
 * Device / execution-provider policy for `laya-serve`.
 *
 * Raja's 2026-09-29 amendment allows Laya on the RTX PRO 4000 next to NInfer and
 * Qwen llama-server, under a HARD 2048 MiB VRAM ceiling and with a mandatory
 * CPU fallback so a broken/missing GPU can never take the service down:
 *
 *   - `LAYA_SERVE_DEVICE=cuda|cpu` (default `cuda` when available)
 *   - `LAYA_SERVE_GPU_MEM_MB` (default 2048) is the ceiling we advertise to the
 *     CUDA EP. onnxruntime-node 1.22 only forwards `deviceId` for the CUDA EP,
 *     so `gpu_mem_limit` / `arena_extend_strategy` are recorded but not honoured
 *     by the binding; the hard ceiling is enforced by choosing a small
 *     (quantized) model and by measuring the total GPU delta (see
 *     scripts/bench-latency.ts).
 *   - `LAYA_SERVE_GPU_DEVICE_ID` (default 0)
 *   - CPU knobs: `LAYA_SERVE_INTRA_OP_THREADS`, `LAYA_SERVE_INTER_OP_THREADS`
 *
 * No I/O happens at import time; the CUDA probe is only used by the bench and
 * the (skipping) GPU test.
 */
import { existsSync } from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { Laya, type LayaOptions } from "../laya.js";

export type ServeDevice = "cuda" | "cpu";

export interface DeviceConfig {
  /** requested device; `cuda` is the default and silently falls back to cpu */
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
  return {
    requested: rawDevice,
    providers: parseProviders(env.LAYA_SERVE_PROVIDERS),
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
 * `LAYA_SERVE_PROVIDERS` wins when set (for FP8/NVFP4, which need
 * `tensorrt,cuda,cpu`). Otherwise the CUDA entry carries `gpu_mem_limit` and
 * `arena_extend_strategy` for a future binding that honours them;
 * onnxruntime-node 1.22 ignores everything but `deviceId`. `cpu` is listed last
 * so ONNX Runtime can place nodes the GPU EP does not implement instead of
 * failing the whole session.
 */
export function buildExecutionProviders(cfg: DeviceConfig): ort.InferenceSession.ExecutionProviderConfig[] {
  if (cfg.providers) {
    return cfg.providers.map((name) => (name === "cuda" ? cudaProvider(cfg) : name));
  }
  if (cfg.requested === "cpu") return ["cpu"];
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

export interface EngineFallback {
  from: ServeDevice;
  to: ServeDevice;
  reason: string;
}

export interface EngineLoadOutcome {
  laya: Laya;
  /** device the session actually runs on */
  device: ServeDevice;
  /** provider names the session was created with */
  providers: string[];
  /** set when a CUDA attempt failed and CPU took over */
  fallback: EngineFallback | null;
}

export interface LoadEngineDeps {
  modelDir: string;
  cfg: DeviceConfig;
  /** injectable session loader (tests); defaults to `Laya.load` */
  load?: (options: LayaOptions) => Promise<Laya>;
  /** where fallback diagnostics go; defaults to stderr */
  log?: (message: string) => void;
}

const firstLine = (text: string): string => text.split("\n", 1)[0] ?? text;

/**
 * Load the model on the requested providers, falling back to CPU when a GPU
 * provider cannot initialize. Never throws for a GPU failure: the CPU attempt is
 * the failure path. A CPU failure still throws, because no engine means no
 * service.
 */
export async function loadEngine(deps: LoadEngineDeps): Promise<EngineLoadOutcome> {
  const load = deps.load ?? ((options: LayaOptions) => Laya.load(options));
  const log = deps.log ?? ((message: string) => process.stderr.write(`${message}\n`));
  const sessionOptions = buildSessionOptions(deps.cfg);
  const requested = buildExecutionProviders(deps.cfg);

  if (!isGpuAttempt(requested)) {
    const laya = await load({ modelDir: deps.modelDir, executionProviders: ["cpu"], sessionOptions });
    return { laya, device: "cpu", providers: ["cpu"], fallback: null };
  }

  try {
    const laya = await load({ modelDir: deps.modelDir, executionProviders: requested, sessionOptions });
    return { laya, device: "cuda", providers: [...laya.providers], fallback: null };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    log(`laya-serve: GPU execution provider unavailable (${firstLine(reason)}); falling back to the CPU execution provider`);
    const laya = await load({ modelDir: deps.modelDir, executionProviders: ["cpu"], sessionOptions });
    return { laya, device: "cpu", providers: ["cpu"], fallback: { from: "cuda", to: "cpu", reason } };
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
 * Try to open a CUDA session on a 95-byte Relu fixture. Returns
 * `{ available: false, reason }` instead of throwing when there is no GPU, no
 * CUDA runtime, or the provider library is missing. Used by the skipping GPU
 * test and by the latency bench.
 */
export async function probeCuda(cfg: DeviceConfig = loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" })): Promise<CudaProbe> {
  const fixture = tinyModelPath();
  if (!existsSync(fixture)) return { available: false, reason: `probe fixture missing: ${fixture}`, providers: [] };
  let session: ort.InferenceSession | null = null;
  try {
    session = await ort.InferenceSession.create(fixture, {
      executionProviders: buildExecutionProviders({ ...cfg, requested: "cuda" }),
      graphOptimizationLevel: "all",
    });
    await session.run({ input: new ort.Tensor("float32", Float32Array.from([-1, 2, -3, 4]), [1, 4]) });
    return { available: true, reason: null, providers: ["cuda", "cpu"] };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return { available: false, reason: firstLine(reason), providers: [] };
  } finally {
    if (session) await session.release().catch(() => undefined);
  }
}
