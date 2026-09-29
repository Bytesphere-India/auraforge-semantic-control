/**
 * Host-side Jev credential handling for the shadow tap.
 *
 * The key is read from the service environment first, then from the dedicated
 * AuraForge secrets file. It is only ever placed in the outbound `Authorization`
 * header. It is never logged, printed, returned to a caller, or written to a
 * shadow record: `sanitizeForStorage` / `redactSecret` are applied to everything
 * that is persisted.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

export const JEV_KEY_ENV = "OPENROUTER_JEV_API_KEY";
export const DEFAULT_SECRETS_PATH = path.join(os.homedir(), ".config", "auraforge", "secrets.env");

const ASSIGNMENT = /^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$/;

/**
 * Resolve only `OPENROUTER_JEV_API_KEY`. The environment wins; otherwise the
 * dedicated secrets file is read. No other file is ever searched.
 */
export function loadJevKey(env: NodeJS.ProcessEnv = process.env, secretsPath: string = DEFAULT_SECRETS_PATH): string | null {
  const fromEnv = env[JEV_KEY_ENV];
  if (typeof fromEnv === "string" && fromEnv.trim().length > 0) return fromEnv.trim();

  let text: string;
  try {
    text = readFileSync(secretsPath, "utf8");
  } catch {
    return null;
  }
  for (const line of text.split(/\r?\n/)) {
    const match = ASSIGNMENT.exec(line);
    if (!match || match[1] !== JEV_KEY_ENV) continue;
    let raw = match[2] ?? "";
    if (raw.length >= 2 && (raw[0] === '"' || raw[0] === "'") && raw[raw.length - 1] === raw[0]) raw = raw.slice(1, -1);
    return raw.length > 0 ? raw : null;
  }
  return null;
}

/** Non-reversible correlation fingerprint; safe to log or report in `/health`. */
export function secretFingerprint(secret: string): string {
  return createHash("sha256").update(secret).digest("hex").slice(0, 12);
}

// Credential-name matching is word-based: camelCase and snake_case both split
// into words, so `secretKey`/`userPassword` match while `input_tokens` (a token
// *count*), `session_id` and `credential_type` (legitimate domain fields) do not.
// `SENSITIVE_SUFFIX` additionally keeps the older, broader gate for a single
// alphanumeric prefix (`Xtoken`, `AToken`, `SPassword`).
const SENSITIVE_KEY = /(authorization|api[_-]?key)/i;
const SENSITIVE_AMBIGUOUS = /^(?:session|credential|credentials)$/i;
const SENSITIVE_SUFFIX = /[a-z0-9](?:password|passwd|passphrase|secret|cookie|token)(?![a-z0-9])/i;
const SENSITIVE_WORDS = new Set(["password", "passwd", "passphrase", "secret", "cookie", "token"]);
const keyWords = (key: string): string[] =>
  key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .split(/[^A-Za-z0-9]+/)
    .filter((word) => word.length > 0)
    .map((word) => word.toLowerCase());
const isSensitiveKey = (key: string): boolean =>
  SENSITIVE_KEY.test(key) || SENSITIVE_AMBIGUOUS.test(key) || SENSITIVE_SUFFIX.test(key) || keyWords(key).some((word) => SENSITIVE_WORDS.has(word));

/** Redact a `key[:=]value` assignment when the key names a credential. */
const redactAssignment = (match: string, quote: string, key: string): string => (isSensitiveKey(key) ? `${quote}${key}${quote}=[REDACTED]` : match);

/**
 * Scrub a string for persistence: the host key is replaced, and generic
 * `Bearer`/`Basic` credentials plus `authorization:`/`api_key=` assignments are
 * redacted even when no host key is known, so a caller-supplied credential can
 * never reach a shadow record. Any scheme-prefixed token of length >= 8 is
 * redacted, including all-alphabetic tokens; this errs toward over-redaction.
 */
export function redactSecret(text: string, secret: string | null): string {
  let out = text;
  if (secret !== null && secret.length > 0) out = out.split(secret).join("[REDACTED]");
  // a value that *is* an auth header is a credential regardless of token shape
  if (/^\s*(?:bearer|basic)\s+\S+/i.test(out)) return "[REDACTED]";
  return (
    out
      .replace(/\b(?:bearer|basic)\s+([-a-z0-9._~+/=]+)/gi, (match, token: string) => (token.length >= 8 ? "[REDACTED]" : match))
      // header- and assignment-style credentials, including quoted JSON keys in
      // an unparseable/non-JSON body (e.g. `{"api_key": "..."`). Quoted values
      // are consumed whole so an embedded `,`/`}` cannot leave a suffix behind.
      .replace(/(["']?)([a-z_][\w-]*)\1\s*[:=]\s*"(?:[^"\\]|\\.)*"/gi, redactAssignment)
      .replace(/(["']?)([a-z_][\w-]*)\1\s*[:=]\s*'(?:[^'\\]|\\.)*'/gi, redactAssignment)
      .replace(/(["']?)([a-z_][\w-]*)\1\s*[:=]\s*[^\r\n,}]*/gi, redactAssignment)
  );
}

const UNSAFE_KEY = /^(?:__proto__|prototype|constructor)$/;

/**
 * Deep-copy a JSON value for persistence. A credential-named field is kept but
 * its value is replaced with `[REDACTED]`, and the field path is appended to
 * `redacted` so the record carries a visible marker instead of silently losing
 * part of the "full request payload". Prototype-polluting keys, and keys that
 * themselves contain the host secret, are dropped (the latter recorded without
 * echoing the key); string values are scrubbed. Uses a null-prototype object so
 * a `__proto__` key can never mutate the prototype of the stored object.
 */
export function sanitizeForStorage(value: unknown, secret: string | null, redacted: string[] = [], path = ""): unknown {
  return sanitizeValue(value, secret, redacted, path, 0);
}

function sanitizeValue(value: unknown, secret: string | null, redacted: string[], path: string, depth: number): unknown {
  if (depth > 32) return "[depth-limit]";
  if (typeof value === "string") return redactSecret(value, secret);
  if (Array.isArray(value)) return value.map((item, index) => sanitizeValue(item, secret, redacted, `${path}[${index}]`, depth + 1));
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      const childPath = path === "" ? key : `${path}.${key}`;
      if (UNSAFE_KEY.test(key)) continue;
      // a credential can also be smuggled as the key name itself; never echo it
      if (secret !== null && secret.length > 0 && key.includes(secret)) {
        redacted.push("[key withheld: contained the host secret]");
        continue;
      }
      const safeKey = redactSecret(key, secret);
      if (isSensitiveKey(key)) {
        out[safeKey] = "[REDACTED]";
        redacted.push(redactSecret(childPath, secret));
        continue;
      }
      out[safeKey] = sanitizeValue(item, secret, redacted, childPath, depth + 1);
    }
    return out;
  }
  return value;
}
