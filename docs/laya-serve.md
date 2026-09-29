# laya-serve — local always-on shadow decision service

`laya-serve` runs the Laya ONNX decision model as a small, always-on HTTP service
on **127.0.0.1:8790**, next to Jev. It speaks the same typed-question
request/response shape as Jev's OpenRouter `/api/alpha/decisions` API, so a
caller can switch engine by URL alone. It also hosts the **Jev shadow tap**
(`POST /jev/api/alpha/decisions`): a transparent forwarder to Jev that shadows
every Jev call through Laya and, when configured, through a local GPU
**Qwen4b Parallel Decision** server, writing one joined record per call. The
service only **serves and logs**; it never learns online.

`laya-serve` itself stays **CPU-only** (fp32 is forced to CPU by the device
policy); Qwen4b is a separate, external local GPU service reached over loopback
and is never allowed to delay or fail the Jev reply or the Laya shadow.

## Invariants

- **Loopback only.** The bind host is validated at startup; a non-loopback
  address fails closed. There is no auth and no TLS because nothing is exposed.
- **GPU allowed under a hard ceiling (Raja 2026-09-29).** Laya may share the
  RTX PRO 4000 with NInfer Bonsai2 and Qwen4b llama-server, but only under a
  **hard 2048 MiB total VRAM ceiling** (context included) and never at NInfer's
  expense. The execution provider is chosen by `LAYA_SERVE_DEVICE` (`cuda` by
  default, `cpu` to opt out). The CPU fallback is mandatory: a missing GPU, an
  insufficient driver, or a CUDA init failure is logged and the session opens on
  CPU instead of crashing. fp32 (~1.6 GiB of weights) cannot meet the ceiling, so
  the fp32 default bundle and any oversized bundle are **forced to CPU before any
  GPU attempt** (`gpuCeilingViolation`); only a small quantized (nvfp4/fp8/fp16)
  variant may run on CUDA, and only after its measured total GPU delta is
  ≤ 2048 MiB.
- **No unrequested egress.** The decision route loads the model from a local
  directory (`Laya.load({ modelDir })`), never the network. The only outbound
  calls in the process are the tap forwarding a caller's request to the
  operator-configured Jev upstream and, when enabled, the third shadow posting
  the same bounded question to the operator-configured **loopback** Qwen4b
  endpoint (a non-loopback URL is refused at startup).
- **No leaked secrets.** The decision route reads no credentials. The tap holds
  the host-side Jev key only to inject it into the upstream `Authorization`
  header; the key is never logged, returned, or written to a shadow record, and
  caller-supplied credentials are stripped before storage.
- **No online learning.** The shadow logs are training data for a later,
  separate, qualified offline job. They are appended to, never read or optimized
  against.
- **Model identity is pinned.** `/health` and every response carry the sha256 of
  `laya.onnx` (and `laya.onnx.data`). For
  `/opt/auraforge/models/laya/base-fp32` these are the AF-LAYA-BASELINE-001
  hashes: `a874eb25…dba1e` (graph) and `4877463…42aba` (weights).
- **`calibration_status` is always `"UNQUALIFIED"`.** The bundle passed
  AF-LAYA-REPEATABILITY-001, which is repeatability, not decision calibration.

## Device policy (GPU, Raja 2026-09-29)

Laya may run on the RTX PRO 4000 shared with NInfer Bonsai2 and Qwen4b
llama-server, subject to two non-negotiables:

1. **Hard 2048 MiB total VRAM ceiling**, context included. onnxruntime-node
   1.22's CUDA bridge only forwards `deviceId`; `gpu_mem_limit` and
   `arena_extend_strategy: kSameAsRequested` are recorded in the provider config
   (`src/serve/device.ts`) but are **not honoured by the binding**, so the
   ceiling is enforced in code and verified by measuring the total GPU delta.
   fp32 on CUDA adds ≈ 2337 MiB before any inference, so a bundle is refused on
   the GPU before any session/probe when its directory name says `fp32`, when the
   total size of **every file in the bundle** (embedded or external weights,
   whatever their names — ONNX external-data `location` is arbitrary and may be
   renamed or nested) exceeds the budget **derived from `LAYA_SERVE_GPU_MEM_MB`**
   (`gpuWeightsBudgetBytes` = ceiling − 1024 MiB context reserve, i.e. 1 GiB at
   the default 2048 MiB ceiling), **or when the bundle cannot be measured at all**
   (`gpuCeilingViolation` fails closed — an unverifiable bundle is never assumed
   small); the request is downgraded to CPU and logged. Lowering
   `LAYA_SERVE_GPU_MEM_MB` tightens the enforced gate (e.g. 1536 MiB admits only
   512 MiB of weights), and a value above the hard 2048 MiB ceiling is clamped —
   for both the weight budget and the `gpu_mem_limit` advertised to the CUDA EP.
   Only a small quantized variant (nvfp4 / fp8 / fp16, ≈ 0.8 GiB) may be promoted
   to CUDA, and only after its measured total delta is ≤ 2048 MiB.
2. **Never crowd out NInfer.** The provider list is `["cuda", "cpu"]`, so
   unsupported nodes fall back per-node rather than failing the session. Before
   the real session opens, the exact provider stack is exercised on a 95-byte
   Relu fixture (`probeProviders`); the `"cpu"` fallback entry is stripped for
   the probe (`gpuOnlyProviders`) so a failed GPU init cannot be satisfied on CPU
   and reported as success. If the probe cannot initialize, the service logs and
   starts on CPU, so `/health` never claims `cuda` for a stack that cannot run.
   Startup then warms the session with one dummy inference so steady-state
   latency is not paid by the first caller.

`LAYA_SERVE_DEVICE=cuda` (default) attempts CUDA and falls back to CPU, logging
the first line of the GPU error, when the driver, device, or provider library is
unavailable. `LAYA_SERVE_DEVICE=cpu` skips CUDA entirely and **wins over**
`LAYA_SERVE_PROVIDERS`; a GPU entry in the provider list together with
`LAYA_SERVE_DEVICE=cpu` is rejected at startup rather than silently ignored. For
a quantized FP8/NVFP4 bundle, which the CUDA EP cannot execute, set
`LAYA_SERVE_PROVIDERS=tensorrt,cuda,cpu` (the TensorRT EP needs the matching
`libnvinfer.so.*` on `LD_LIBRARY_PATH`); `cpu` stays last so a failed TensorRT/CUDA
init still lands on CPU. `[N/A]` per-process VRAM accounting under WSL2 is
expected; use the device-total delta from `scripts/bench-latency.ts`.

## Endpoints

| Method | Path                       | Purpose                                                   |
| ------ | -------------------------- | --------------------------------------------------------- |
| `GET`  | `/health`                  | readiness, identity, hashes, RSS, shadow-log/tap counters |
| `POST` | `/api/alpha/decisions`     | typed decisions (aliases: `/decisions`, `/v1/decisions`)  |
| `POST` | `/jev/api/alpha/decisions` | Jev shadow tap (alias: `/jev/decisions`); forwards to Jev |

### Request

Jev's body verbatim, with optional `model` (accepted and recorded, not used to
select the model — the URL does that):

```json
{
  "model": "typesafe/jev-1.13",
  "state": { "evidence": "…" },
  "questions": {
    "result": { "type": "noul", "instructions": "…", "criteria": { "true": "…", "false": "…" } },
    "event": { "type": "choice", "instructions": "…", "criteria": { "transient": "…", "broken_oracle": "…" } },
    "sev": { "type": "score", "instructions": "…", "criteria": ["cosmetic", "minor", "moderate", "major", "critical"] }
  }
}
```

All three question types are supported. Limits (overridable via env, see below):
≤32 questions, ≤20 000 instruction characters, 2–64 options/levels, ≤400 000
state characters, ≤1 MiB body.

`state` is passed to Laya unchanged and serialized by the checkpoint's own
`json.dumps` path (a plain string is used verbatim). A Jev caller's
`state: { "evidence": "…" }` therefore reaches Laya as `{"evidence": "…"}`,
carrying the same information Jev sees.

### Response

Jev's envelope (`answers`, `model`, `provider`, `usage`) plus Laya identity and
latency:

```json
{
  "engine": "laya",
  "engine_version": "0.1.2",
  "model": "laya-base-fp32",
  "model_sha256": "a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e",
  "model_data_sha256": "487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba",
  "model_dir": "/opt/auraforge/models/laya/base-fp32",
  "calibration_status": "UNQUALIFIED",
  "provider": "local-laya",
  "request_id": "0725aa3c-…",
  "request_hash": "211286bc…",
  "latency_ms": 3104.76,
  "answers": {
    "transient_recoverable": { "type": "noul", "noul": 0.5103, "rl_agent": { "act_probability": 1 } }
  },
  "usage": { "input_tokens": 923, "output_tokens": 0 }
}
```

Errors are `{ "error": "…", "code": "…" }` with `400` (validation),
`413` (body too large), `422` (engine failure) or `405`/`404`.

### `/health`

```json
{
  "status": "ok",
  "engine": "laya",
  "engine_version": "0.1.2",
  "model": "laya-base-fp32",
  "model_sha256": "a874eb25…",
  "model_data_sha256": "4877463…",
  "model_dir": "/opt/auraforge/models/laya/base-fp32",
  "calibration_status": "UNQUALIFIED",
  "execution_providers": ["cpu"],
  "pid": 1234,
  "uptime_s": 47.5,
  "rss_mb": 1829.8,
  "requests": 6,
  "errors": 0,
  "shadow_log": { "enabled": true, "path": "…/laya-decisions.jsonl", "writable": true, "lines": 10, "errors": 0, "last_error": null }
}
```

`execution_providers` is reported truthfully: `["cuda", "cpu"]` when the CUDA
session opened (CPU is ONNX Runtime's per-node fallback), or `["cpu"]` when the
CUDA attempt failed and the process fell back at startup. There is no way to
claim CUDA when the session is on CPU.

## Shadow log

Every answered question appends one JSONL line to
`~/.auraforge-work/shadow/laya-decisions.jsonl`:

```json
{
  "schema": 1,
  "ts": "2026-09-28T21:28:13.229Z",
  "engine": "laya",
  "engine_version": "0.1.2",
  "model": "laya-base-fp32",
  "model_sha256": "a874eb25…",
  "calibration_status": "UNQUALIFIED",
  "request_id": "0725aa3c-…",
  "request_hash": "211286bc…",
  "question_id": "severity",
  "question_type": "score",
  "answer": 1.4974,
  "probabilities": { "0": 0.0941, "1": 0.4774, "2": 0.2842, "3": 0.1255, "4": 0.0187 },
  "confidence": 0.2122,
  "latency_ms": 496,
  "input_tokens": 154,
  "status": "ok",
  "error": null
}
```

- The `request_hash` is a canonical sha256 over `{engine, model, model_sha256,
state, questions}`; the raw state/instructions are **never** written.
- `answer` is the chosen label (choice), the expected level (score) or P(true)
  (noul); `probabilities` is the full distribution. `rl_agent` is not logged.
- A failed engine call still records one line per question with
  `status:"error"`, `answer:null` and a short machine code (never a stack trace).
- If the log cannot be written the service keeps serving and reports it under
  `shadow_log` in `/health`. Set `LAYA_SERVE_REQUIRE_SHADOW=1` to fail startup
  instead.

## Jev shadow tap

`POST /jev/api/alpha/decisions` (alias `/jev/decisions`) lets a caller move from
Jev to Laya without changing anything but the URL:

1. **Forward, unchanged.** The tap forwards the caller's body byte-for-byte to
   the configured Jev upstream and returns the upstream status, response headers
   and body unchanged and immediately. Caller headers (trace ids, idempotency
   keys, ...) are forwarded too, minus hop-by-hop and transport-controlled ones
   (`host`, `content-length`, `accept-encoding`, ...). It keeps the caller's
   `Content-Type` and `Accept` (defaulting `Accept` to `application/json` only
   when the caller sent none); the one header it injects is the host-side
   `Authorization`, and a caller-supplied `Authorization` is never forwarded.
   Callers send no key. The caller id comes from the `X-Caller` request header,
   which the tap consumes for the record and does **not** forward upstream.
   Redirects are **not followed** (`redirect: "manual"`), so a 3xx `Location`
   cannot point the tap at loopback or cloud metadata.
2. **Shadow after the reply.** Dispatch is a handoff: the server commits it only
   from the response `finish`/`close` event, so the shadow is scheduled strictly
   after the response is on the wire (and `drain()` can see the in-flight handoff
   during shutdown). Everything that exists only to prepare the shadow — JSON
   decode, SHA-256, request validation and request-hash computation — also runs
   after the response, on the queue, not on the forward path, so prep can never
   add latency to or fail the Jev response (a prep failure is recorded as
   `SHADOW_PREP_ERROR`). The payload then runs through Laya on a serial queue (no
   short timeout; overflow is reported, not dropped) and, when configured,
   through the Qwen4b shadow at the same time. A slow or failing Laya **or**
   Qwen4b can never delay or fail the Jev response.
3. **One joined record per call** (`jev-laya-pairs.jsonl`): the full request
   payload, Jev's full reply (status, answer, probabilities, usage/cost,
   latency), Laya's full reply (answers, probabilities, latency, model sha),
   Qwen4b's shadow reply (answers, raw exchanges, latency, returned model), the
   caller, request hash and timestamps. When a credential-named field's value is
   withheld, the record carries a `redacted_fields` list of the affected paths so
   a truncated payload is never silently presented as full. If Laya fails, the
   record is still written with `laya.error` (`INVALID_JSON`, `ENGINE_ERROR`,
   `QUEUE_FULL`, ...). `laya-decisions.jsonl` keeps receiving the per-question
   lines as before.
4. **Key handling.** `OPENROUTER_JEV_API_KEY` is read from the service
   environment or `~/.config/auraforge/secrets.env` inside the process and is
   only ever placed in the outbound `Authorization` header. It is never logged,
   printed, returned, or persisted. **The service fails closed**: with the tap
   enabled and no key it refuses to start, and a tap constructed without a key
   returns `503 JEV_KEY_MISSING` rather than forwarding unauthenticated calls.
   `/health` exposes only a non-reversible fingerprint.
5. **Credential scrubbing with a visible marker.** A credential-named field is
   _kept_ but its value is replaced with `[REDACTED]`, and the field path is
   listed in the record's `redacted_fields`, so the "full request payload" is
   never silently truncated. String values are scrubbed of `Bearer`/`Basic`
   tokens (any scheme-prefixed token of 8+ characters, including all-alphabetic
   ones) and `authorization:`/`api_key=` assignments, including in a non-JSON
   body and when the key is quoted (`{"api_key": "..."}`), consuming a whole
   quoted value so embedded commas/braces/escapes cannot leave a suffix; the
   `X-Caller` value is scrubbed the same way. A key that itself contains the host
   secret is dropped (recorded without echoing the key). Matching recognises both
   `snake_case` and `camelCase` (`apiToken`, `userPassword`, `sessionCookie`) for
   `authorization`, `api_key`, `password`/`passwd`/`passphrase`, `secret`,
   `cookie` and `token`, while `session`/`credential` match only exactly — so
   legitimate domain fields such as `session_id`, `sessionId`, `credential_type`,
   `credentialType` and token _counts_ (`input_tokens`, `inputTokens`, ...) are
   preserved. Only the media type of a `Content-Type` is persisted
   (`application/json`, never its parameters). Prototype-polluting keys
   (`__proto__`, `constructor`, `prototype`) are ignored and storage uses
   null-prototype objects.

The tap is inactive when `LAYA_SERVE_JEV_ENABLED=0` (the route then reports
`503 JEV_TAP_DISABLED`).

### Qwen4b third shadow

When `LAYA_SERVE_QWEN4B_URL` is non-empty (the default is the local GPU
`http://127.0.0.1:8082/v1/decision`), the same request is also put to the
llama.cpp **Parallel Decision** server as a third, strictly advisory shadow. The
endpoint must be loopback (`http`/`https`, no embedded credentials, no fragment);
anything else fails startup, so a shadow can never become an SSRF primitive.
`laya-serve` remains CPU-only — Qwen4b is an external local GPU service.

**Semantic parity (`noul`).** Every `noul` question is one native Parallel
Decision call, rendered deterministically (no LLM transforms the question pack):

```json
{
  "instructions": "<question>\n\nTRUE:\n<true criterion>\n\nFALSE:\n<false criterion>\n\nClassify only from the supplied evidence.",
  "schema": { "result": { "type": "boolean", "description": "Does the evidence support the predicate?" } },
  "contexts": ["<canonical JSON of state>"],
  "mode": "tree",
  "cache_prompt": true
}
```

A question or criterion that embeds a `TRUE:`/`FALSE:` section header or the
classifier terminator is rejected (`QWEN4B_STRUCTURAL_HEADER`) rather than asked
with a spoofed structure. The header gate anchors to any line start (`\n`,
`\r\n`, a bare `\r`, and the Unicode line/paragraph separators `U+2028`/`U+2029`)
and tolerates any Unicode horizontal whitespace (a no-break space included)
before the keyword; the classifier terminator is matched case-insensitively with
optional terminal punctuation. A `noul` question without criteria renders the
question and the classifier line only.

**Probability normalization.** The endpoint reports the probability of the
*selected* value, not P(TRUE): `value = true` yields `p_true = p`,
`p_false = 1 - p`; `value = false` yields `p_true = 1 - p`, `p_false = p`. A
probability that is not finite or is outside `[0, 1]`, a missing/non-boolean
result, a disagreement between `results[0].decision.result` and
`results[0].fields.result.value`, or an unparseable body becomes an error record
and **never** an answer.

**`choice` / `score` (older draft mapping).** All non-`noul` questions go in one
batched call with the brief's exact draft schema
(`{"type":"object","properties":{<qid>: <per type>},"required":[all qids]}`):
`choice` -> `{"type":"string","enum":[<criteria keys>]}`; `score` ->
`{"type":"object","properties":{<level>:{"type":"boolean"}}}`. The batched
context is the canonical `state` followed by each question's id, type,
instructions and criteria. The reply is read from
`results[0].fields.<qid> = {value, probability}`; the one-level wrapper shape (a
lone `result` object holding the question map) is also recognized, so a question
literally named `result` cannot be mistaken for the wrapper. `choice` records the
selected option and its reported probability (the enum returns only the
selection, so no full distribution is claimed). For `score`, each level's support
weight is the reported probability when the selected value is `true` and
`1 - probability` when it is `false` (a hard `1`/`0` when the endpoint omits a
probability — a high probability for `false` is *low* support, never high). The
weights are normalized into a level distribution and the record carries the
expected level, the level legend, the distribution and the `1 - normalized
entropy` confidence; a field that supports no level fails closed
(`QWEN4B_SCORE_NO_EVIDENCE`).

**Record and failure isolation.** The joined record gains a `qwen4b` member:

```json
{
  "qwen4b": {
    "status": "ok",
    "latency_ms": 323,
    "model": "Qwen3.5-4B-IQ4_XS.gguf",
    "answers": { "result": { "type": "noul", "noul": 0.9879, "p_true": 0.9879, "p_false": 0.0121 } },
    "raw": [{ "question_ids": ["result"], "request": { "…": "…" }, "status": 200, "response": { "…": "…" }, "error": null, "latency_ms": 320 }],
    "error": null
  }
}
```

The member is `null` when the shadow is disabled or was never attempted (an
invalid request). Qwen4b answers are all-or-nothing: any failed call records
`status:"error"`, `answers:null` and a short machine code (`QWEN4B_UNREACHABLE`,
`QWEN4B_TIMEOUT`, `QWEN4B_HTTP_<status>`, `QWEN4B_MALFORMED_RESPONSE`,
`QWEN4B_PROBABILITY_OUT_OF_RANGE`, ...), while the raw exchanges are still kept
for audit. Qwen4b starts together with Laya and its promise never rejects, and
the third-shadow join plus the record append run **outside** Laya's serial queue
worker: Laya's slot is released as soon as inference finishes, so a down, slow or
hung Qwen4b can never delay the Jev reply, delay the Laya shadow, hold the queue
or overflow later Laya shadows with `QUEUE_FULL`. `drain()` still waits for every
detached record write, so no pair is lost at shutdown. The `qwen4b` view
(including its raw exchanges) is scrubbed exactly like the Jev payload, so a
credential cannot reach the record through the third shadow's audit trail.
`/health` reports `qwen4b_enabled`, `qwen4b_url`, `qwen4b_ok` and `qwen4b_error`.

## Configuration

| Env                             | Default                                         | Meaning                                                                           |
| ------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------- |
| `LAYA_SERVE_HOST`               | `127.0.0.1`                                     | loopback only; others are rejected                                                |
| `LAYA_SERVE_PORT`               | `8790`                                          | listen port                                                                       |
| `LAYA_SERVE_MODEL_DIR`          | `/opt/auraforge/models/laya/base-fp32`          | ONNX bundle                                                                       |
| `LAYA_SERVE_MODEL_ID`           | derived (`laya-base-fp32`)                      | identity string                                                                   |
| `LAYA_SERVE_DEVICE`             | `cuda`                                          | `cuda` (CPU fallback) or `cpu`                                                    |
| `LAYA_SERVE_PROVIDERS`          | unset                                           | explicit EP list, e.g. `tensorrt,cuda,cpu`; rejected with `LAYA_SERVE_DEVICE=cpu` |
| `LAYA_SERVE_GPU_MEM_MB`         | `2048`                                          | VRAM ceiling (MiB); enforced weight budget = min(value, 2048) − 1024 MiB reserve  |
| `LAYA_SERVE_GPU_DEVICE_ID`      | `0`                                             | CUDA device ordinal                                                               |
| `LAYA_SERVE_INTRA_OP_THREADS`   | unset (ORT default)                             | CPU intra-op threads                                                              |
| `LAYA_SERVE_INTER_OP_THREADS`   | unset (ORT default)                             | CPU inter-op threads                                                              |
| `LAYA_SERVE_SHADOW_LOG`         | `~/.auraforge-work/shadow/laya-decisions.jsonl` | JSONL path; empty disables                                                        |
| `LAYA_SERVE_HASH_WEIGHTS`       | `1`                                             | hash `laya.onnx.data` at startup                                                  |
| `LAYA_SERVE_REQUIRE_SHADOW`     | `0`                                             | fail startup when the log is unwritable                                           |
| `LAYA_SERVE_MAX_BODY_BYTES`     | `1048576`                                       | request body cap                                                                  |
| `LAYA_SERVE_MAX_QUESTIONS`      | `32`                                            | questions per request                                                             |
| `LAYA_SERVE_JEV_ENABLED`        | `1`                                             | enable the Jev shadow tap                                                         |
| `LAYA_SERVE_JEV_UPSTREAM`       | `https://openrouter.ai/api/alpha/decisions`     | tap target; parsed, no embedded creds                                             |
| `LAYA_SERVE_JEV_ALLOW_LOOPBACK` | `0`                                             | allow a loopback upstream (self-host)                                             |
| `LAYA_SERVE_JEV_PAIRS_LOG`      | `~/.auraforge-work/shadow/jev-laya-pairs.jsonl` | joined record; empty disables                                                     |
| `LAYA_SERVE_JEV_SECRETS`        | `~/.config/auraforge/secrets.env`               | file searched for the key                                                         |
| `LAYA_SERVE_JEV_TIMEOUT_MS`     | `60000`                                         | upstream forward timeout                                                          |
| `LAYA_SERVE_JEV_QUEUE_MAX`      | `256`                                           | queued Laya shadows before `QUEUE_FULL`                                           |
| `LAYA_SERVE_QWEN4B_URL`         | `http://127.0.0.1:8082/v1/decision`             | third shadow; loopback only; empty disables                                       |
| `LAYA_SERVE_QWEN4B_TIMEOUT_MS`  | `30000`                                         | Qwen4b per-call timeout (never blocks the tap)                                    |

The upstream must be a valid `http(s)` URL without embedded credentials. Loopback
is rejected unless `LAYA_SERVE_JEV_ALLOW_LOOPBACK=1`, and link-local/metadata/
reserved ranges (`169.254.0.0/16`, RFC1918 `10.0.0.0/8`, `172.16.0.0/12`,
`192.168.0.0/16`, `100.64.0.0/10`, `fe80::/10`, `ff00::/8`, `fd00:ec2::254`,
`metadata.google.internal`, ...) are always rejected at startup. Hostnames are
normalized (trailing root dots and case are stripped) before matching, so
`localhost.`/`metadata.google.internal.` cannot bypass the gates. IPv4-mapped and
IPv4-compatible IPv6 literals (`::ffff:169.254.169.254`, `::ffff:127.0.0.1`,
`::7f00:1`, and their normalized hex forms) are decoded and classified as the
IPv4 address they represent, so they cannot bypass these checks. The checks are
string/address-class based: the hostname is resolved by the transport at connect
time, so a DNS answer that changes after startup (rebinding) is not re-checked.

## Run

Development (tsx):

```sh
LAYA_SERVE_MODEL_DIR=/opt/auraforge/models/laya/base-fp32 \
LAYA_SERVE_SHADOW_LOG="$PWD/scratch/laya-decisions.jsonl" \
./node_modules/.bin/tsx src/serve/index.ts
```

Production entrypoint (this is what the systemd unit runs and what produced the
sample numbers below; `dist/` is built by `yarn build` or the install script):

```sh
/usr/bin/node ~/projects/auraforge-semantic-control/dist/serve/index.js
```

Model load is once at startup (session load ≈ 2.2 s warm; plus ≈ 3.4 s to hash
the 1.6 GB weights when `LAYA_SERVE_HASH_WEIGHTS=1`), followed by one warm-up
inference so the first caller does not pay the first-run cost. Set
`LAYA_SERVE_DEVICE=cuda` (default) with `LAYA_SERVE_MODEL_DIR` pointing at a
quantized bundle that fits the 2048 MiB ceiling; leave it unset for fp32, which
must stay on CPU. The startup line reports `device=… providers=… warmed=…`.

### Latency bench

```sh
yarn bench:latency                                  # cuda then cpu, 5 runs each
LAYA_SERVE_MODEL_DIR=~/models/laya/base-fp16 LAYA_BENCH_DEVICES=cuda yarn bench:latency
```

The bench runs the docs' three-question batch against a small (~64 B) and a 3 KB
state, reports p50/p95/min/max per device, samples `nvidia-smi` before/after for
the total VRAM delta, and reports a device as `SKIP` with the CUDA reason when no
GPU is reachable. It never fails the run for a missing GPU.

### Quantization (NVFP4 → FP8 → fp16)

`scripts/quantize-laya.py` writes quantized bundles under `~/models/laya/<variant>/`
without ever touching `/opt/auraforge/models/laya/base-fp32`. It creates the
dedicated venv `~/.venv-laya-quant` (`nvidia-modelopt[onnx]`, `tensorrt`, `onnx`)
and calibrates on the repo test fixtures only:

```sh
python3 scripts/quantize-laya.py --variant nvfp4   # falls back fp8, then fp16
```

onnxruntime's CUDA EP cannot execute FP8/NVFP4: those need the **TensorRT EP**,
which requires `libnvinfer.so.*` on `LD_LIBRARY_PATH` (from the TensorRT pip
wheel). The variants form a **fallback chain**: a variant is `"ok"` only when its
quantization _and_ answer parity both pass **on the provider it requires**
(fp8/nvfp4 → `tensorrt,cuda`, fp16 → `cuda`; `"cpu"` is never in the required
stack), and the script stops at the first `"ok"` variant (so fp16 is built only
when fp8 did not clear both gates; skipped variants are recorded as `"skipped"`
with the reason, and `fallback_stopped_after` names the winner). A variant that
quantizes but cannot load on its required GPU EP, or fails parity, is recorded as
`"failed"` and is not counted as produced — it can never pass on a silent CPU
fallback. fp32 remains the default until answer parity against it is measured
with `yarn bench:latency` plus the parity check below.

## Install (systemd user unit)

The unit is non-activating; the install script builds the project, writes
`~/.config/systemd/user/laya-serve.service` and reloads the user manager, but
does **not** enable or start anything.

```sh
scripts/install-laya-serve.sh --print        # inspect the rendered unit
scripts/install-laya-serve.sh                # build + install + daemon-reload
systemctl --user enable --now laya-serve.service
systemctl --user status laya-serve.service
curl -s http://127.0.0.1:8790/health
```

To keep it running without an active login: `loginctl enable-linger "$USER"`.
The unit sets `Restart=on-failure`, `RestartSec=3`, `MemoryMax=3G`, `Nice=10`,
the model-dir environment, and the 2048 MiB VRAM ceiling. It no longer blanks
`CUDA_VISIBLE_DEVICES` (the old CPU-only guard) and instead leaves the device
policy to `LAYA_SERVE_DEVICE`; point `LAYA_SERVE_MODEL_DIR` at a quantized
bundle before enabling CUDA.

## Tests

```sh
./node_modules/.bin/tsx --test test/test_serve_protocol.ts test/test_serve_device.ts test/test_serve_shadow.ts test/test_serve_http.ts test/test_serve_jev_tap.ts test/test_serve_qwen4b.ts test/test_serve_parity.ts
# or: yarn test:serve
```

Covers request validation and limits, response shape/identity,
`canonicalJson`/request hashing, the shadow-line format and its no-raw-text
property, HTTP status codes, the engine-failure path, the Jev-tap forward/shadow
path, the Qwen4b adapter (native `noul` rendering, selected-value probability
normalization for `noul` and per-level `score` support, the choice/score field-map
mapping with the lone-`result` wrapper disambiguation, injection and terminator
fail-closed parsing, down/400/timeout error records, three-engine record joining
and the hung-Qwen4b queue-isolation guarantee), the device policy (env
parsing, provider construction, the 2048 MiB ceiling gate, GPU-probe-gated
CUDA→CPU fallback, the `cpu`-wins rule) and the fail-closed parity comparison.
The CUDA probe test skips cleanly when no GPU/CUDA driver is reachable; the
`probeProviders(["cpu"])` test always runs offline.

### Answer parity

`scripts/quantize-laya.py` also runs each produced variant and the fp32 baseline
on the repo test fixtures and reports the maximum absolute difference of `noul`
(and of each score level/choice probability). The comparison **fails closed**:
`scripts/check-parity.ts` exits 3 with `ok:false` when the variant returns an
empty answer set, a key set that differs from the baseline, **or any `NaN`/infinite
metric** (a non-finite delta must never be folded into a zero-difference pass), so
a corrupt model can never be reported as passing parity. Exit 2 means the variant
— or its **required GPU provider** — could not be loaded: `check-parity.ts`
removes `"cpu"` from a GPU variant's provider stack and probes the GPU stack
first (`planVariantProviders` + `probeProviders`), so a TensorRT/CUDA init
failure is a hard failure, never a CPU-backed parity pass that would stop the
fp16 fallback. A variant is only eligible for `LAYA_SERVE_DEVICE=cuda` once the
parity number is recorded; fp32 stays the default until then.

## Sample run (2026-09-28, base-fp32)

- Batched 5-question request: **3113.5 ms** wall (server `latency_ms` 3104.76),
  923 input tokens, RSS 1829.8 MB.
- Per-question (one question per request): event_class_oracle_timeout 537 ms,
  event_class_executor_schema 604 ms, transient_recoverable 457 ms,
  requires_escalation 393 ms, severity 501 ms.
- 10 shadow lines for 10 answered questions; 0 errors; no raw state substrings
  present in the log or responses.

Laya is a shadow engine: the numbers above are the raw, uncalibrated model.
Jev remains the decision engine; Laya's output is evidence and training data.
