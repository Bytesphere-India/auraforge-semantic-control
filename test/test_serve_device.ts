/**
 * Device-policy tests: env parsing, execution-provider construction, the CUDA→CPU
 * fallback and the GPU probe. Everything here runs offline; the one test that
 * needs a real GPU skips cleanly when CUDA is absent.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import type { Laya, LayaOptions } from "../src/laya.js";
import {
  buildExecutionProviders,
  buildSessionOptions,
  isGpuAttempt,
  loadDeviceConfig,
  loadEngine,
  probeCuda,
  providerName,
  warmupEngine,
  type DeviceConfig,
} from "../src/serve/device.js";

const fakeLaya = (providers: string[]): Laya =>
  ({
    providers,
    systemOne: async () => ({ model: "laya", answers: {}, usage: { input_tokens: 1, output_tokens: 0 } }),
  }) as unknown as Laya;

const names = (options: LayaOptions): string[] => (options.executionProviders ?? []).map(providerName);

test("loadDeviceConfig defaults to cuda with a 2048 MiB ceiling and device 0", () => {
  const cfg = loadDeviceConfig({});
  assert.equal(cfg.requested, "cuda");
  assert.equal(cfg.gpuMemMb, 2048);
  assert.equal(cfg.gpuDeviceId, 0);
  assert.equal(cfg.intraOpThreads, null);
  assert.equal(cfg.interOpThreads, null);
});

test("loadDeviceConfig honors cpu, the VRAM ceiling, the device id and thread counts", () => {
  const cfg = loadDeviceConfig({
    LAYA_SERVE_DEVICE: "CPU",
    LAYA_SERVE_GPU_MEM_MB: "1024",
    LAYA_SERVE_GPU_DEVICE_ID: "2",
    LAYA_SERVE_INTRA_OP_THREADS: "6",
    LAYA_SERVE_INTER_OP_THREADS: "1",
  });
  assert.equal(cfg.requested, "cpu");
  assert.equal(cfg.gpuMemMb, 1024);
  assert.equal(cfg.gpuDeviceId, 2);
  assert.equal(cfg.intraOpThreads, 6);
  assert.equal(cfg.interOpThreads, 1);
});

test("loadDeviceConfig rejects malformed values", () => {
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_DEVICE: "tpu" }), /LAYA_SERVE_DEVICE/);
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_GPU_MEM_MB: "0" }), /LAYA_SERVE_GPU_MEM_MB/);
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_GPU_MEM_MB: "many" }), /LAYA_SERVE_GPU_MEM_MB/);
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_GPU_DEVICE_ID: "-1" }), /LAYA_SERVE_GPU_DEVICE_ID/);
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_INTRA_OP_THREADS: "0" }), /LAYA_SERVE_INTRA_OP_THREADS/);
});

test("buildExecutionProviders advertises the VRAM ceiling and caps CUDA with cpu fallback", () => {
  const cuda = buildExecutionProviders(loadDeviceConfig({ LAYA_SERVE_GPU_MEM_MB: "2048", LAYA_SERVE_GPU_DEVICE_ID: "1" }));
  assert.equal(cuda.length, 2);
  const primary = cuda[0] as { name: string; deviceId: number; gpu_mem_limit: number; arena_extend_strategy: string };
  assert.equal(primary.name, "cuda");
  assert.equal(primary.deviceId, 1);
  assert.equal(primary.gpu_mem_limit, 2048 * 1024 * 1024);
  assert.equal(primary.arena_extend_strategy, "kSameAsRequested");
  const fallback = cuda[1];
  assert.ok(fallback);
  assert.equal(providerName(fallback), "cpu");

  const cpu = buildExecutionProviders(loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu" }));
  assert.deepEqual(cpu, ["cpu"]);
});

test("LAYA_SERVE_PROVIDERS overrides the device-derived list (TensorRT for FP8/NVFP4)", () => {
  const cfg = loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "tensorrt, cuda, cpu", LAYA_SERVE_GPU_DEVICE_ID: "1" });
  assert.deepEqual(cfg.providers, ["tensorrt", "cuda", "cpu"]);
  const providers = buildExecutionProviders(cfg);
  assert.deepEqual(providers.map(providerName), ["tensorrt", "cuda", "cpu"]);
  assert.ok(isGpuAttempt(providers));
  assert.equal(isGpuAttempt(["cpu"]), false);

  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "cuda,openvino" }), /LAYA_SERVE_PROVIDERS/);
  assert.equal(loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "" }).providers, null);
});

test("loadEngine falls back to cpu from an explicit TensorRT provider list", async () => {
  const calls: string[][] = [];
  const load = async (options: LayaOptions): Promise<Laya> => {
    const requested = names(options);
    calls.push(requested);
    if (requested.includes("tensorrt")) throw new Error("libnvinfer.so.10: cannot open shared object file");
    return fakeLaya(requested);
  };
  const outcome = await loadEngine({
    modelDir: "/does/not/exist",
    cfg: loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "tensorrt,cuda,cpu" }),
    load,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["tensorrt", "cuda", "cpu"], ["cpu"]]);
  assert.equal(outcome.fallback?.from, "cuda");
});

test("buildSessionOptions always optimizes the graph and passes CPU threads only when set", () => {
  const plain = buildSessionOptions(loadDeviceConfig({}));
  assert.equal(plain.graphOptimizationLevel, "all");
  assert.equal(plain.intraOpNumThreads, undefined);
  assert.equal(plain.interOpNumThreads, undefined);

  const threaded = buildSessionOptions(loadDeviceConfig({ LAYA_SERVE_INTRA_OP_THREADS: "4", LAYA_SERVE_INTER_OP_THREADS: "2" }));
  assert.equal(threaded.intraOpNumThreads, 4);
  assert.equal(threaded.interOpNumThreads, 2);
});

test("loadEngine falls back from cuda to cpu and logs the reason", async () => {
  const calls: string[][] = [];
  const logs: string[] = [];
  const load = async (options: LayaOptions): Promise<Laya> => {
    const requested = names(options);
    calls.push(requested);
    if (requested.includes("cuda")) throw new Error("CUDA failure: no device\nsecond line");
    return fakeLaya(requested);
  };
  const outcome = await loadEngine({
    modelDir: "/does/not/exist",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load,
    log: (message) => logs.push(message),
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(outcome.providers, ["cpu"]);
  assert.equal(outcome.fallback?.from, "cuda");
  assert.match(outcome.fallback?.reason ?? "", /no device/);
  assert.deepEqual(calls, [["cuda", "cpu"], ["cpu"]]);
  assert.ok(logs.some((line) => line.includes("falling back to the CPU execution provider")));
});

test("loadEngine uses cuda directly when the session opens", async () => {
  const calls: string[][] = [];
  const load = async (options: LayaOptions): Promise<Laya> => {
    calls.push(names(options));
    return fakeLaya(names(options));
  };
  const outcome = await loadEngine({
    modelDir: "/does/not/exist",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cuda");
  assert.deepEqual(outcome.providers, ["cuda", "cpu"]);
  assert.equal(outcome.fallback, null);
  assert.deepEqual(calls, [["cuda", "cpu"]]);
});

test("loadEngine with LAYA_SERVE_DEVICE=cpu never attempts cuda", async () => {
  const calls: string[][] = [];
  const load = async (options: LayaOptions): Promise<Laya> => {
    calls.push(names(options));
    return fakeLaya(names(options));
  };
  const outcome = await loadEngine({
    modelDir: "/does/not/exist",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu" }),
    load,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["cpu"]]);
});

test("warmupEngine runs one dummy inference and never throws", async () => {
  let calls = 0;
  const laya = {
    systemOne: async () => {
      calls += 1;
      return { model: "laya", answers: {}, usage: { input_tokens: 1, output_tokens: 0 } };
    },
  } as unknown as Laya;
  assert.equal(await warmupEngine(laya, () => undefined), true);
  assert.equal(calls, 1);

  const broken = {
    systemOne: async () => {
      throw new Error("session exploded");
    },
  } as unknown as Laya;
  const logs: string[] = [];
  assert.equal(await warmupEngine(broken, (message) => logs.push(message)), false);
  assert.ok(logs.some((line) => line.includes("warm-up inference failed")));
});

test("probeCuda opens a GPU session when CUDA is present, else skips", async (t) => {
  const probe = await probeCuda(loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }));
  if (!probe.available) {
    t.skip(`CUDA unavailable in this environment: ${probe.reason}`);
    return;
  }
  assert.deepEqual(probe.providers, ["cuda", "cpu"]);
  assert.equal(probe.reason, null);
});

test("DeviceConfig stays structural for callers", () => {
  const cfg: DeviceConfig = loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu" });
  assert.equal(cfg.requested, "cpu");
});
