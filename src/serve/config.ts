/**
 * Environment-driven configuration for `laya-serve`.
 *
 * Fails closed on anything that would violate the service's contract: a
 * non-loopback bind address, a bad port, or a missing model directory.
 */
import os from "node:os";
import path from "node:path";
import { parseJevUpstream } from "./jev-forward.js";
import { DEFAULT_LIMITS, type ServeLimits } from "./protocol.js";

export const DEFAULT_MODEL_DIR = "/opt/auraforge/models/laya/base-fp32";
export const DEFAULT_SHADOW_LOG = path.join(os.homedir(), ".auraforge-work", "shadow", "laya-decisions.jsonl");
export const DEFAULT_JEV_UPSTREAM = "https://openrouter.ai/api/alpha/decisions";
export const DEFAULT_JEV_PAIRS_LOG = path.join(os.homedir(), ".auraforge-work", "shadow", "jev-laya-pairs.jsonl");
export const DEFAULT_JEV_SECRETS = path.join(os.homedir(), ".config", "auraforge", "secrets.env");
export const DEFAULT_HOST = "127.0.0.1";
export const DEFAULT_PORT = 8790;

/** The Jev shadow tap's host-side configuration. */
export interface JevTapConfig {
  enabled: boolean;
  /** upstream OpenRouter decisions endpoint (overridable for tests/self-host) */
  upstreamUrl: string;
  /** joined `jev-laya-pairs.jsonl` path; null disables the joined record */
  pairsLogPath: string | null;
  /** `secrets.env`-style file searched for OPENROUTER_JEV_API_KEY */
  secretsPath: string;
  /** upstream timeout; the caller's Jev response is bounded by this */
  timeoutMs: number;
  /** queued Laya shadows before overflow (overflow still writes a pair) */
  queueMax: number;
  /** permit a loopback upstream (self-hosted Jev); off by default */
  allowLoopback: boolean;
}

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
  jev: JevTapConfig;
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

const JEV_BLOCKED_HOSTNAMES = new Set(["metadata", "metadata.google.internal", "metadata.goog"]);

function unbracket(hostname: string): string {
  // strip the IPv6 brackets and any trailing root dots (`localhost.` -> `localhost`)
  return hostname
    .replace(/^\[|\]$/g, "")
    .replace(/\.+$/, "")
    .toLowerCase();
}

/**
 * Decode an IPv4 address embedded in IPv6 (`::ffff:a.b.c.d`, `::a.b.c.d`, their
 * normalized hex forms, and the RFC 6052 NAT64 well-known prefix
 * `64:ff9b::a.b.c.d`) so mapped addresses cannot bypass the IPv4 class checks.
 * Returns a dotted quad, or null when the host is not an embedded-IPv4 form.
 */
function ipv4FromEmbedded(host: string): string | null {
  const match = /^(?:::ffff:|::|64:ff9b::|64:ff9b:1::)([0-9a-f:.]+)$/i.exec(host);
  if (!match) return null;
  const rest = match[1] ?? "";
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(rest)) return rest;
  const groups = rest.split(":");
  if (groups.length === 2 && groups.every((group) => /^[0-9a-f]{1,4}$/i.test(group))) {
    const hi = Number.parseInt(groups[0] ?? "0", 16);
    const lo = Number.parseInt(groups[1] ?? "0", 16);
    return [hi >> 8, hi & 0xff, lo >> 8, lo & 0xff].join(".");
  }
  return null;
}

/** standard loopback hostnames, including common Linux `/etc/hosts` aliases. */
const LOOPBACK_HOSTNAMES = new Set(["localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"]);

/** loopback: 127.0.0.0/8, ::1, IPv4-mapped/NAT64 loopback and loopback hostnames. */
function isLoopbackHost(hostname: string): boolean {
  const host = unbracket(hostname);
  const mapped = ipv4FromEmbedded(host);
  if (mapped !== null) return /^127\./.test(mapped);
  if (LOOPBACK_HOSTNAMES.has(host) || host.endsWith(".localhost") || host.endsWith(".localhost.localdomain")) return true;
  if (host === "::1") return true;
  return /^127\./.test(host);
}

/**
 * Address classes the tap must never forward to, so an operator mistake cannot
 * turn the tap into an SSRF primitive against cloud metadata or reserved space.
 * IPv4-mapped IPv6 literals are decoded and checked as their IPv4 address.
 */
function isBlockedJevHost(hostname: string): boolean {
  const host = unbracket(hostname);
  if (JEV_BLOCKED_HOSTNAMES.has(host)) return true;

  const candidate = ipv4FromEmbedded(host) ?? host;
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(candidate);
  if (v4) {
    const a = Number(v4[1]);
    const b = Number(v4[2]);
    const c = Number(v4[3]);
    if (a === 0) return true; // this-network / unspecified
    if (a === 10) return true; // RFC1918 10.0.0.0/8
    if (a === 169 && b === 254) return true; // link-local (cloud metadata)
    if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT
    if (a === 172 && b >= 16 && b <= 31) return true; // RFC1918 172.16.0.0/12
    if (a === 192 && b === 168) return true; // RFC1918 192.168.0.0/16
    if (a === 192 && b === 0 && c === 0) return true; // IETF protocol assignments
    if (a === 198 && (b === 18 || b === 19)) return true; // benchmarking
    if (a >= 224) return true; // multicast / reserved
    return false;
  }

  if (host === "::") return true; // IPv6 unspecified
  if (/^fe[89ab]/.test(host)) return true; // fe80::/10 link-local
  if (host.startsWith("fc") || host.startsWith("fd")) return true; // fc00::/7 unique-local
  if (host.startsWith("ff")) return true; // ff00::/8 multicast
  return false;
}

/** Parse the upstream, enforce the address-class policy, and return it normalized. */
export function resolveJevUpstream(raw: string, allowLoopback: boolean): string {
  const url = parseJevUpstream(raw);
  const host = url.hostname;
  if (isLoopbackHost(host)) {
    if (!allowLoopback) {
      throw new Error(`LAYA_SERVE_JEV_UPSTREAM must not be loopback (${host}); set LAYA_SERVE_JEV_ALLOW_LOOPBACK=1 to allow a self-hosted Jev`);
    }
  } else if (isBlockedJevHost(host)) {
    throw new Error(`LAYA_SERVE_JEV_UPSTREAM host ${host} is in a blocked link-local/metadata/reserved range`);
  }
  return url.toString();
}

export function loadServeConfig(env: NodeJS.ProcessEnv = process.env): ServeConfig {
  const host = env.LAYA_SERVE_HOST || DEFAULT_HOST;
  if (!LOOPBACK.has(host)) {
    throw new Error(`LAYA_SERVE_HOST must be a loopback address (127.0.0.1/::1/localhost); refusing to bind ${JSON.stringify(host)}`);
  }

  const modelDir = path.resolve(expandHome(env.LAYA_SERVE_MODEL_DIR || DEFAULT_MODEL_DIR));
  const shadowRaw = env.LAYA_SERVE_SHADOW_LOG === undefined ? DEFAULT_SHADOW_LOG : env.LAYA_SERVE_SHADOW_LOG;
  const pairsRaw = env.LAYA_SERVE_JEV_PAIRS_LOG === undefined ? DEFAULT_JEV_PAIRS_LOG : env.LAYA_SERVE_JEV_PAIRS_LOG;
  const allowLoopback = boolean(env, "LAYA_SERVE_JEV_ALLOW_LOOPBACK", false);
  const upstreamUrl = resolveJevUpstream(env.LAYA_SERVE_JEV_UPSTREAM || DEFAULT_JEV_UPSTREAM, allowLoopback);

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
    jev: {
      enabled: boolean(env, "LAYA_SERVE_JEV_ENABLED", true),
      upstreamUrl,
      pairsLogPath: pairsRaw === "" ? null : path.resolve(expandHome(pairsRaw)),
      secretsPath: path.resolve(expandHome(env.LAYA_SERVE_JEV_SECRETS || DEFAULT_JEV_SECRETS)),
      timeoutMs: positiveInt(env, "LAYA_SERVE_JEV_TIMEOUT_MS", 60_000, 300_000),
      queueMax: positiveInt(env, "LAYA_SERVE_JEV_QUEUE_MAX", 256, 1_000_000),
      allowLoopback,
    },
  };
}
