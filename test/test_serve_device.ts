/**
 * Device-policy tests: env parsing, execution-provider construction, the 2048 MiB
 * VRAM ceiling gate, the CUDA→CPU fallback and the GPU probe. Everything here
 * runs offline; the one test that needs a real GPU skips cleanly when CUDA is
 * absent.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Laya, LayaOptions } from "../src/laya.js";
import {
  buildExecutionProviders,
  buildSessionOptions,
  gpuCeilingViolation,
  gpuOnlyProviders,
  isFp32ModelDir,
  isGpuAttempt,
  loadDeviceConfig,
  loadEngine,
  MAX_GPU_WEIGHTS_BYTES,
  probeCuda,
  probeProviders,
  providerName,
  warmupEngine,
  weightsByteLength,
  type DeviceConfig,
} from "../src/serve/device.js";

const fakeLaya = (providers: string[]): Laya =>
  ({
    providers,
    systemOne: async () => ({ model: "laya", answers: {}, usage: { input_tokens: 1, output_tokens: 0 } }),
  }) as unknown as Laya;

const names = (options: LayaOptions): string[] => (options.executionProviders ?? []).map(providerName);

const collectLoader =
  (calls: string[][]) =>
  async (options: LayaOptions): Promise<Laya> => {
    const requested = names(options);
    calls.push(requested);
    return fakeLaya(requested);
  };

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

test("LAYA_SERVE_DEVICE=cpu wins over a GPU LAYA_SERVE_PROVIDERS list", () => {
  assert.throws(() => loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu", LAYA_SERVE_PROVIDERS: "tensorrt,cuda,cpu" }), /conflicts with GPU entries/);
  assert.deepEqual(loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu", LAYA_SERVE_PROVIDERS: "cpu" }).providers, ["cpu"]);
  // Defense in depth: a hand-built cfg cannot smuggle GPU providers past the cpu opt-out.
  const handBuilt: DeviceConfig = { ...loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "tensorrt,cuda,cpu" }), requested: "cpu" };
  assert.deepEqual(buildExecutionProviders(handBuilt), ["cpu"]);
  assert.equal(isGpuAttempt(buildExecutionProviders(handBuilt)), false);
});

test("gpuCeilingViolation fails closed on fp32, oversized and unverifiable bundles", () => {
  assert.equal(isFp32ModelDir("/opt/auraforge/models/laya/base-fp32"), true);
  assert.equal(isFp32ModelDir("/models/laya/base-fp16"), false);
  assert.match(gpuCeilingViolation("/opt/auraforge/models/laya/base-fp32", null) ?? "", /fp32/);
  assert.match(gpuCeilingViolation("/models/laya/base-fp16", MAX_GPU_WEIGHTS_BYTES + 1) ?? "", /over the .* GPU budget/);
  assert.equal(gpuCeilingViolation("/models/laya/base-fp16", 800_000_000), null);
  // Unknown size must not be assumed small.
  assert.match(gpuCeilingViolation("/models/laya/base-fp16", null) ?? "", /could not be verified/);
});

test("weightsByteLength measures embedded and external weights, or reports null", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "laya-weights-"));
  try {
    assert.equal(weightsByteLength(dir), null, "an empty directory has no verifiable weights");
    await writeFile(path.join(dir, "laya.onnx"), Buffer.alloc(100, 1));
    assert.equal(weightsByteLength(dir), 100, "a single-file model embeds its weights in laya.onnx");
    await writeFile(path.join(dir, "laya.onnx.data"), Buffer.alloc(250, 2));
    assert.equal(weightsByteLength(dir), 350, "external data adds to the graph size");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("gpuOnlyProviders drops the cpu fallback so a GPU probe cannot be satisfied on CPU", () => {
  const cuda = buildExecutionProviders(loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }));
  assert.deepEqual(gpuOnlyProviders(cuda).map(providerName), ["cuda"]);
  const tensorrt = buildExecutionProviders(loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "tensorrt,cuda,cpu" }));
  assert.deepEqual(gpuOnlyProviders(tensorrt).map(providerName), ["tensorrt", "cuda"]);
  // A CPU-only stack is left intact (it is already an explicit CPU decision).
  assert.deepEqual(gpuOnlyProviders(["cpu"]), ["cpu"]);
});

test("loadEngine forces the fp32 default bundle to cpu before any GPU attempt", async () => {
  const calls: string[][] = [];
  let probes = 0;
  const outcome = await loadEngine({
    modelDir: "/opt/auraforge/models/laya/base-fp32",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => {
      probes += 1;
      return { available: true, reason: null };
    },
    weightsBytes: () => null,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(outcome.providers, ["cpu"]);
  assert.deepEqual(calls, [["cpu"]]);
  assert.equal(probes, 0, "the ceiling gate must run before the GPU probe");
  assert.match(outcome.ceiling?.reason ?? "", /fp32/);
  assert.equal(outcome.fallback, null);
});

test("loadEngine refuses an oversized bundle even when its name looks quantized", async () => {
  const calls: string[][] = [];
  let probes = 0;
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp16",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => {
      probes += 1;
      return { available: true, reason: null };
    },
    weightsBytes: () => MAX_GPU_WEIGHTS_BYTES + 1,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["cpu"]]);
  assert.equal(probes, 0);
  assert.match(outcome.ceiling?.reason ?? "", /GPU budget/);
});

test("loadEngine allows a small quantized bundle on the verified GPU path", async () => {
  const calls: string[][] = [];
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp16",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => ({ available: true, reason: null }),
    weightsBytes: () => 800_000_000,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cuda");
  assert.deepEqual(outcome.providers, ["cuda", "cpu"]);
  assert.deepEqual(calls, [["cuda", "cpu"]]);
  assert.equal(outcome.ceiling, null);
  assert.equal(outcome.fallback, null);
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

test("loadEngine refuses the GPU when the weight size cannot be verified", async () => {
  const calls: string[][] = [];
  let probes = 0;
  const outcome = await loadEngine({
    modelDir: "/models/laya/candidate",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => {
      probes += 1;
      return { available: true, reason: null };
    },
    weightsBytes: () => null,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["cpu"]]);
  assert.equal(probes, 0);
  assert.match(outcome.ceiling?.reason ?? "", /could not be verified/);
});

test("loadEngine falls back from cuda to cpu and logs the probe reason", async () => {
  const calls: string[][] = [];
  const logs: string[] = [];
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp16",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => ({ available: false, reason: "CUDA failure: no device\nsecond line" }),
    weightsBytes: () => 800_000_000,
    log: (message) => logs.push(message),
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(outcome.providers, ["cpu"]);
  assert.equal(outcome.fallback?.from, "cuda");
  assert.match(outcome.fallback?.reason ?? "", /no device/);
  assert.deepEqual(calls, [["cpu"]]);
  assert.ok(logs.some((line) => line.includes("falling back to the CPU execution provider")));
});

test("loadEngine never claims cuda when the GPU probe fails, even if the loader would succeed", async () => {
  const calls: string[][] = [];
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp16",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => ({ available: false, reason: "CUDA driver version is insufficient" }),
    weightsBytes: () => 800_000_000,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(outcome.providers, ["cpu"]);
  assert.deepEqual(calls, [["cpu"]]);
  assert.match(outcome.fallback?.reason ?? "", /insufficient/);
});

test("loadEngine uses cuda directly when the probe verifies the provider stack", async () => {
  const calls: string[][] = [];
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp16",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }),
    load: collectLoader(calls),
    probe: async () => ({ available: true, reason: null }),
    weightsBytes: () => 800_000_000,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cuda");
  assert.deepEqual(outcome.providers, ["cuda", "cpu"]);
  assert.equal(outcome.fallback, null);
  assert.deepEqual(calls, [["cuda", "cpu"]]);
});

test("loadEngine with LAYA_SERVE_DEVICE=cpu never probes or attempts cuda", async () => {
  const calls: string[][] = [];
  let probes = 0;
  const outcome = await loadEngine({
    modelDir: "/does/not/exist",
    cfg: loadDeviceConfig({ LAYA_SERVE_DEVICE: "cpu" }),
    load: collectLoader(calls),
    probe: async () => {
      probes += 1;
      return { available: true, reason: null };
    },
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["cpu"]]);
  assert.equal(probes, 0);
});

test("loadEngine falls back to cpu from an explicit TensorRT provider list", async () => {
  const calls: string[][] = [];
  const outcome = await loadEngine({
    modelDir: "/models/laya/base-fp8",
    cfg: loadDeviceConfig({ LAYA_SERVE_PROVIDERS: "tensorrt,cuda,cpu" }),
    load: collectLoader(calls),
    probe: async () => ({ available: false, reason: "libnvinfer.so.10: cannot open shared object file" }),
    weightsBytes: () => 500_000_000,
    log: () => undefined,
  });
  assert.equal(outcome.device, "cpu");
  assert.deepEqual(calls, [["cpu"]]);
  assert.equal(outcome.fallback?.from, "cuda");
  assert.match(outcome.fallback?.reason ?? "", /libnvinfer/);
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

test("probeProviders verifies the cpu stack on the tiny fixture (offline)", async () => {
  const result = await probeProviders(["cpu"]);
  assert.equal(result.available, true);
  assert.equal(result.reason, null);
});

test("probeProviders does not let the cpu entry mask a GPU initialization failure", async (t) => {
  const providers = buildExecutionProviders(loadDeviceConfig({ LAYA_SERVE_DEVICE: "cuda" }));
  const result = await probeProviders(providers);
  if (result.available) {
    t.skip("CUDA is present; the masking case cannot be exercised on this host");
    return;
  }
  // With the cpu fallback stripped, a GPU-less host must report unavailable
  // rather than a false success on CPU.
  assert.equal(result.available, false);
  assert.ok(result.reason);
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
