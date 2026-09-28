/**
 * Environment-driven configuration for `laya-serve`.
 *
 * Fails closed on anything that would violate the service's contract: a
 * non-loopback bind address, a bad port, or a missing model directory.
 */
import os from "node:os";
import path from "node:path";
import { DEFAULT_LIMITS, type ServeLimits } from "./protocol.js";

export const DEFAULT_MODEL_DIR = "/opt/auraforge/models/laya/base-fp32";
export const DEFAULT_SHADOW_LOG = path.join(os.homedir(), ".auraforge-work", "shadow", "laya-decisions.jsonl");
export const DEFAULT_HOST = "127.0.0.1";
export const DEFAULT_PORT = 8790;

export interface ServeConfig {
  host: string;
  port: number;
  modelDir: string;
  /** explicit model id; null means derive from the directory name */
  modelId: string | null;
  /** null disables the shadow log (not recommended) */
  shadowLogPath: string | null;
  /** hash the multi-GB weights file at startup (pinned identity) */
  hashWeights: boolean;
  /** refuse to start when the shadow log is not writable */
  requireShadow: boolean;
  maxBodyBytes: number;
  limits: ServeLimits;
  /** CPU only by policy; the GPU belongs to NInfer */
  executionProviders: string[];
}

function expandHome(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith("~/")) return path.join(os.homedir(), value.slice(2));
  return value;
}

function positiveInt(env: NodeJS.ProcessEnv, name: string, fallback: number, max: number): number {
  const raw = env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > max) {
    throw new Error(`${name} must be an integer in 1..${max}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function boolean(env: NodeJS.ProcessEnv, name: string, fallback: boolean): boolean {
  const raw = env[name];
  if (raw === undefined || raw === "") return fallback;
  if (["1", "true", "yes", "on"].includes(raw.toLowerCase())) return true;
  if (["0", "false", "no", "off"].includes(raw.toLowerCase())) return false;
  throw new Error(`${name} must be a boolean, got ${JSON.stringify(raw)}`);
}

const LOOPBACK = new Set(["127.0.0.1", "::1", "localhost"]);

/** Derive a stable model id from a bundle directory, e.g. base-fp32 -> laya-base-fp32. */
export function deriveModelId(modelDir: string): string {
  const base = path.basename(path.resolve(modelDir));
  return base.startsWith("laya") ? base : `laya-${base}`;
}

export function loadServeConfig(env: NodeJS.ProcessEnv = process.env): ServeConfig {
  const host = env.LAYA_SERVE_HOST || DEFAULT_HOST;
  if (!LOOPBACK.has(host)) {
    throw new Error(`LAYA_SERVE_HOST must be a loopback address (127.0.0.1/::1/localhost); refusing to bind ${JSON.stringify(host)}`);
  }

  const modelDir = path.resolve(expandHome(env.LAYA_SERVE_MODEL_DIR || DEFAULT_MODEL_DIR));
  const shadowRaw = env.LAYA_SERVE_SHADOW_LOG === undefined ? DEFAULT_SHADOW_LOG : env.LAYA_SERVE_SHADOW_LOG;

  return {
    host,
    port: positiveInt(env, "LAYA_SERVE_PORT", DEFAULT_PORT, 65535),
    modelDir,
    modelId: env.LAYA_SERVE_MODEL_ID || null,
    shadowLogPath: shadowRaw === "" ? null : path.resolve(expandHome(shadowRaw)),
    hashWeights: boolean(env, "LAYA_SERVE_HASH_WEIGHTS", true),
    requireShadow: boolean(env, "LAYA_SERVE_REQUIRE_SHADOW", false),
    maxBodyBytes: positiveInt(env, "LAYA_SERVE_MAX_BODY_BYTES", 1024 * 1024, 64 * 1024 * 1024),
    limits: {
      ...DEFAULT_LIMITS,
      maxQuestions: positiveInt(env, "LAYA_SERVE_MAX_QUESTIONS", DEFAULT_LIMITS.maxQuestions, 4096),
    },
    executionProviders: ["cpu"],
  };
}
