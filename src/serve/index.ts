/**
 * `laya-serve` entrypoint.
 *
 * Loads the ONNX bundle once, opens the shadow log, warms the session, then
 * serves Jev-shaped typed questions on 127.0.0.1 only. The execution provider is
 * chosen by `LAYA_SERVE_DEVICE` (cuda by default, cpu fallback); the CUDA EP is
 * capped at 2048 MiB VRAM and a GPU init failure is logged and falls back to CPU.
 *
 *   LAYA_SERVE_MODEL_DIR=/opt/auraforge/models/laya/base-fp32 \
 *   LAYA_SERVE_SHADOW_LOG=~/.auraforge-work/shadow/laya-decisions.jsonl \
 *   ./node_modules/.bin/tsx src/serve/index.ts
 */
import { createHash } from "node:crypto";
import { createReadStream, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { loadServeConfig, deriveModelId } from "./config.js";
import { loadEngine, warmupEngine } from "./device.js";
import { fetchJevTransport } from "./jev-forward.js";
import { JEV_KEY_ENV, loadJevKey } from "./jev-key.js";
import { createJevTap } from "./jev-tap.js";
import { fetchQwen4bTransport } from "./qwen-forward.js";
import { createQwen4bShadow } from "./qwen-shadow.js";
import { createServer } from "./server.js";
import { JevLayaPairsLog, ShadowLog } from "./shadow.js";
import type { EngineIdentity, DecisionEngine } from "./types.js";

const require = createRequire(import.meta.url);

function engineVersion(): string {
  try {
    const pkg = require("../../package.json") as { version?: unknown };
    return typeof pkg.version === "string" ? pkg.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

async function sha256File(file: string): Promise<string | null> {
  try {
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
    return hash.digest("hex");
  } catch {
    return null;
  }
}

async function main(): Promise<void> {
  const config = loadServeConfig();

  // Fail closed before loading anything: a tap without a host key would forward
  // unauthenticated Jev calls (and disable key redaction), so refuse to start.
  const jevKey = config.jev.enabled ? loadJevKey(process.env, config.jev.secretsPath) : null;
  if (config.jev.enabled && jevKey === null) {
    throw new Error(
      `LAYA_SERVE_JEV_ENABLED=1 but ${JEV_KEY_ENV} is not set (in the environment or ${config.jev.secretsPath}); ` +
        "refusing to start so the tap cannot forward unauthenticated calls. Set the key or LAYA_SERVE_JEV_ENABLED=0.",
    );
  }

  // Identity before the (slow) session load, so a bad bundle fails fast.
  const graphPath = path.join(config.modelDir, "laya.onnx");
  statSync(graphPath);
  const modelSha256 = await sha256File(graphPath);
  if (modelSha256 === null) throw new Error(`could not hash model graph ${graphPath}`);
  const weightsPath = path.join(config.modelDir, "laya.onnx.data");
  const modelDataSha256 = config.hashWeights ? await sha256File(weightsPath) : null;

  const identity: EngineIdentity = {
    engine: "laya",
    engineVersion: engineVersion(),
    model: config.modelId ?? deriveModelId(config.modelDir),
    modelDir: path.resolve(config.modelDir),
    modelSha256,
    modelDataSha256,
    calibrationStatus: "UNQUALIFIED",
    // Replaced below with the providers the session actually opened with.
    executionProviders: [config.device.requested],
  };

  // Load the model exactly once, honoring the CUDA/CPU policy. `loadEngine`
  // refuses a bundle that cannot meet the 2048 MiB VRAM ceiling and falls back to
  // CPU (logging) when the GPU probe or the real session cannot initialize, so
  // the service never crashes because of a missing/broken GPU and never claims
  // `cuda` for a provider stack that cannot run.
  const startedAtMs = Date.now();
  const outcome = await loadEngine({ modelDir: config.modelDir, cfg: config.device });
  const laya = outcome.laya;
  identity.executionProviders = [...outcome.providers];
  const device = outcome.device;
  const warmed = await warmupEngine(laya);
  const engine: DecisionEngine = {
    systemOne: (state, questions) => laya.systemOne(state, questions),
  };

  const shadow = new ShadowLog(config.shadowLogPath, () => new Date());
  const shadowStats = shadow.open();
  if (config.requireShadow && !shadowStats.writable) {
    throw new Error(`LAYA_SERVE_REQUIRE_SHADOW=1 but the shadow log is not writable: ${shadowStats.path}`);
  }

  // The Jev shadow tap: host-side key, joined pairs log, transparent forwarder.
  const pairs = new JevLayaPairsLog(config.jev.pairsLogPath, () => new Date());
  const pairsStats = config.jev.enabled ? pairs.open() : pairs.stats();
  if (config.requireShadow && config.jev.enabled && !pairsStats.writable) {
    throw new Error(`LAYA_SERVE_REQUIRE_SHADOW=1 but the Jev-Laya pairs log is not writable: ${pairsStats.path}`);
  }
  const jevTap = config.jev.enabled
    ? createJevTap({
        upstreamUrl: config.jev.upstreamUrl,
        apiKey: jevKey,
        transport: fetchJevTransport,
        timeoutMs: config.jev.timeoutMs,
        engine,
        identity,
        shadow,
        pairs,
        limits: config.limits,
        queueMax: config.jev.queueMax,
        // The third shadow is an external local GPU service; laya-serve itself
        // stays CPU-only and never blocks on it.
        qwen4b:
          config.jev.qwen4bUrl === null
            ? null
            : createQwen4bShadow({ url: config.jev.qwen4bUrl, transport: fetchQwen4bTransport, timeoutMs: config.jev.qwen4bTimeoutMs }),
      })
    : null;

  const server = createServer({
    engine,
    identity,
    shadow,
    jevTap: jevTap ?? undefined,
    config: {
      host: config.host,
      port: config.port,
      maxBodyBytes: config.maxBodyBytes,
      limits: config.limits,
      startedAtMs,
    },
  });

  let shuttingDown = false;
  const shutdown = async (signal: string): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    process.stderr.write(`laya-serve: ${signal}; shutting down\n`);
    await new Promise<void>((resolve) => server.close(() => resolve()));
    if (jevTap) {
      // Flush queued Laya shadows, but never hang shutdown on a stuck run.
      await Promise.race([jevTap.drain(), new Promise<void>((resolve) => setTimeout(resolve, 20_000))]);
    }
    try {
      await laya.close();
    } catch {
      // release is best-effort on the way out
    }
    process.exit(0);
  };
  process.on("SIGINT", () => void shutdown("SIGINT"));
  process.on("SIGTERM", () => void shutdown("SIGTERM"));

  server.on("error", (error) => {
    process.stderr.write(`laya-serve: server error: ${error.message}\n`);
    process.exitCode = 1;
  });

  server.listen(config.port, config.host, () => {
    const tap = jevTap?.stats() ?? null;
    process.stderr.write(
      `laya-serve: listening on http://${config.host}:${config.port} ` +
        `engine=${identity.engine}@${identity.engineVersion} model=${identity.model} ` +
        `model_sha256=${identity.modelSha256} weights_sha256=${identity.modelDataSha256 ?? "unhashed"} ` +
        `dir=${identity.modelDir} calibration=${identity.calibrationStatus} ` +
        `device=${device} providers=${identity.executionProviders.join(",")} ` +
        `vram_ceiling=${outcome.ceiling ? "forced-cpu" : "ok"} gpu_mem_limit_mb=${config.device.gpuMemMb} ` +
        `warmed=${warmed ? "yes" : "no"} ` +
        `shadow=${shadowStats.path ?? "disabled"}${shadowStats.writable ? "" : " (NOT WRITABLE)"} ` +
        `jev_upstream=${tap?.upstream ?? "disabled"} jev_key=${tap?.key_present ? "present" : "absent"} ` +
        `jev_pairs=${pairsStats.path ?? "disabled"}${pairsStats.writable ? "" : " (NOT WRITABLE)"} ` +
        `qwen4b=${config.jev.qwen4bUrl ?? "disabled"} ` +
        `load_ms=${Date.now() - startedAtMs}\n`,
    );
  });
}

main().catch((error: unknown) => {
  process.stderr.write(`laya-serve: fatal: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
