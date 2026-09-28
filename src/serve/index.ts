/**
 * `laya-serve` entrypoint.
 *
 * Loads the ONNX bundle once, opens the shadow log, then serves Jev-shaped typed
 * questions on 127.0.0.1 only. CPU execution provider only; no outbound network,
 * no credentials, no GPU.
 *
 *   LAYA_SERVE_MODEL_DIR=/opt/auraforge/models/laya/base-fp32 \
 *   LAYA_SERVE_SHADOW_LOG=~/.auraforge-work/shadow/laya-decisions.jsonl \
 *   ./node_modules/.bin/tsx src/serve/index.ts
 */
import { createHash } from "node:crypto";
import { createReadStream, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { Laya } from "../laya.js";
import { loadServeConfig, deriveModelId } from "./config.js";
import { createServer } from "./server.js";
import { ShadowLog } from "./shadow.js";
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
    executionProviders: [...config.executionProviders],
  };

  // Load the model exactly once, pinned to the CPU execution provider.
  const startedAtMs = Date.now();
  const laya = await Laya.load({ modelDir: config.modelDir, executionProviders: ["cpu"] });
  const engine: DecisionEngine = {
    systemOne: (state, questions) => laya.systemOne(state, questions),
  };

  const shadow = new ShadowLog(config.shadowLogPath, () => new Date());
  const shadowStats = shadow.open();
  if (config.requireShadow && !shadowStats.writable) {
    throw new Error(`LAYA_SERVE_REQUIRE_SHADOW=1 but the shadow log is not writable: ${shadowStats.path}`);
  }

  const server = createServer({
    engine,
    identity,
    shadow,
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
    process.stderr.write(
      `laya-serve: listening on http://${config.host}:${config.port} ` +
        `engine=${identity.engine}@${identity.engineVersion} model=${identity.model} ` +
        `model_sha256=${identity.modelSha256} weights_sha256=${identity.modelDataSha256 ?? "unhashed"} ` +
        `dir=${identity.modelDir} calibration=${identity.calibrationStatus} ` +
        `providers=${identity.executionProviders.join(",")} ` +
        `shadow=${shadowStats.path ?? "disabled"}${shadowStats.writable ? "" : " (NOT WRITABLE)"} ` +
        `load_ms=${Date.now() - startedAtMs}\n`,
    );
  });
}

main().catch((error: unknown) => {
  process.stderr.write(`laya-serve: fatal: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
