# laya-serve — local always-on shadow decision service

`laya-serve` runs the Laya ONNX decision model as a small, always-on HTTP service
on **127.0.0.1:8790**, next to Jev. It speaks the same typed-question
request/response shape as Jev's OpenRouter `/api/alpha/decisions` API, so a
caller can switch engine by URL alone. It only **serves and logs**; it never
learns online.

## Invariants

- **Loopback only.** The bind host is validated at startup; a non-loopback
  address fails closed. There is no auth and no TLS because nothing is exposed.
- **CPU only.** The session is pinned to `executionProviders: ["cpu"]`. The GPU
  belongs to NInfer and is never advertised to ONNX Runtime.
- **No network egress.** The model is loaded from a local directory
  (`Laya.load({ modelDir })`), so the Hugging Face download path is never taken.
- **No secrets.** The service reads no credentials and sends no headers to
  anyone.
- **No online learning.** The shadow log is training data for a later, separate,
  qualified offline job. It is appended to, never read or optimized against.
- **Model identity is pinned.** `/health` and every response carry the sha256 of
  `laya.onnx` (and `laya.onnx.data`). For
  `/opt/auraforge/models/laya/base-fp32` these are the AF-LAYA-BASELINE-001
  hashes: `a874eb25…dba1e` (graph) and `4877463…42aba` (weights).
- **`calibration_status` is always `"UNQUALIFIED"`.** The bundle passed
  AF-LAYA-REPEATABILITY-001, which is repeatability, not decision calibration.

## Endpoints

| Method | Path                   | Purpose                                                  |
| ------ | ---------------------- | -------------------------------------------------------- |
| `GET`  | `/health`              | readiness, identity, hashes, RSS, shadow-log counters    |
| `POST` | `/api/alpha/decisions` | typed decisions (aliases: `/decisions`, `/v1/decisions`) |

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

## Configuration

| Env                         | Default                                         | Meaning                                 |
| --------------------------- | ----------------------------------------------- | --------------------------------------- |
| `LAYA_SERVE_HOST`           | `127.0.0.1`                                     | loopback only; others are rejected      |
| `LAYA_SERVE_PORT`           | `8790`                                          | listen port                             |
| `LAYA_SERVE_MODEL_DIR`      | `/opt/auraforge/models/laya/base-fp32`          | ONNX bundle                             |
| `LAYA_SERVE_MODEL_ID`       | derived (`laya-base-fp32`)                      | identity string                         |
| `LAYA_SERVE_SHADOW_LOG`     | `~/.auraforge-work/shadow/laya-decisions.jsonl` | JSONL path; empty disables              |
| `LAYA_SERVE_HASH_WEIGHTS`   | `1`                                             | hash `laya.onnx.data` at startup        |
| `LAYA_SERVE_REQUIRE_SHADOW` | `0`                                             | fail startup when the log is unwritable |
| `LAYA_SERVE_MAX_BODY_BYTES` | `1048576`                                       | request body cap                        |
| `LAYA_SERVE_MAX_QUESTIONS`  | `32`                                            | questions per request                   |

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
the 1.6 GB weights when `LAYA_SERVE_HASH_WEIGHTS=1`).

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
the model-dir environment, and blank GPU-visibility variables.

## Tests

Offline (no model, no network):

```sh
./node_modules/.bin/tsx --test test/test_serve_protocol.ts test/test_serve_shadow.ts test/test_serve_http.ts
# or: yarn test:serve
```

Covers request validation and limits, response shape/identity,
`canonicalJson`/request hashing, the shadow-line format and its no-raw-text
property, HTTP status codes and the engine-failure path.

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
