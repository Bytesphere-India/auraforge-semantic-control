> **CONTROLLED MIRROR — PLATFORM LEGISLATION**  
> Canonical source: `Bytesphere-India/AuraForge/docs/governance/AF-ARCH-DECISION-FABRIC-001.md`.  
> This repository copy is normative for local implementation but MUST NOT independently redefine, weaken, widen, or supersede the canonical architecture. Material changes originate in the canonical platform document and are then synchronized here.

# AuraForge Decision Fabric — Canonical Architecture and Implementation Blueprint

**Document ID:** AF-ARCH-DECISION-FABRIC-001  
**Status:** PLATFORM LEGISLATION — ADOPTED ARCHITECTURE BASELINE  
**Scope:** AuraForge AI Platform — AFIE / AFSE / AFEE / LOOP Automation / Decision-capable platform fabrics  
**Canonical repository:** `Bytesphere-India/AuraForge`  
**Canonical path:** `docs/governance/AF-ARCH-DECISION-FABRIC-001.md`  
**Change rule:** This document is platform legislation. A repository mirror MUST NOT independently redefine, weaken, widen, or supersede it. Material change requires a new approved platform architecture version and synchronized dependent profiles.

---

## 1. Executive Summary

The **AuraForge Decision Fabric** is the governed semantic-decision layer of the AuraForge platform.

Its purpose is to handle decisions that are:

- too semantic for ordinary deterministic rules;
- too frequent, narrow, or latency-sensitive to justify a frontier LLM invocation;
- expressible as bounded typed judgments;
- measurable against later outcomes; and
- safe only when enclosed by deterministic authority, fallback, evidence, and replay.

Examples include:

- whether a test modification appears to weaken the test oracle;
- which tests are most likely to expose a regression;
- whether an agent command appears semantically hazardous;
- whether a work brief is ambiguous;
- where reviewer and human attention should be concentrated;
- whether an agent is repeating a previously failed approach;
- whether a failed candidate should be repaired, restarted, or escalated;
- which memories, tools, skills, or knowledge fragments are relevant;
- whether two concurrent work items may interfere semantically;
- which already-authorized intervention is most likely to help a stuck agent.

The Fabric is **not an authority layer**.

It does not:

- transition the authoritative FSM;
- approve or accept work;
- broaden LOOP or Generation scope;
- grant tools, permissions, budgets, or leases;
- create runtime authority;
- bypass the Authority Pack;
- replace deterministic safety controls;
- replace human cryptographic acceptance;
- independently decide architecture policy.

The foundational rule is:

> **System 1 advises. Deterministic software disposes.**

A Decision Site produces typed semantic judgments with probabilities, calibration metadata, provenance, and abstention. Deterministic policy decides what those judgments mean operationally.

This preserves AuraForge's authority chain:

`AFIE Definition -> AFSE Runtime Binding -> AFEE Execution`.

---

## 2. Why AuraForge Needs a Decision Fabric

AuraForge already distinguishes two major classes of computation.

### 2.1 Deterministic control

Examples:

- FSM transitions;
- leases and write masks;
- authority and scope;
- cryptographic acceptance;
- budgets;
- signature and schema verification;
- Git ancestry;
- CI/test facts;
- evidence binding;
- execution ownership.

### 2.2 System 2 intelligence

Examples:

- architecture;
- research;
- open-ended coding and debugging;
- complex review;
- requirement interpretation;
- novel failure analysis;
- human judgment.

A substantial middle layer remains:

- "Does this test change look like weakening?"
- "Which three tests should run first?"
- "Is this command surprising relative to the goal?"
- "Does this brief contain ambiguous acceptance criteria?"
- "Is this Run repeating a failed approach?"
- "Which hunk deserves deeper review attention?"
- "Does this retrieved memory contain an actual reusable fix?"
- "Are these separate changes semantically coupled?"

These are **semantic decisions**: not reliably reducible to ordinary Boolean rules, but not worthy of full frontier deliberation on every occurrence.

The Decision Fabric industrializes this class of judgment.

---

## 3. Architectural Position

AuraForge SHALL NOT integrate "Laya" or "Jev" as isolated features.

AuraForge SHALL implement a **Decision Fabric** in which Laya, Jev, specialized local decision models, logit-based deciders, and future semantic engines are replaceable runtimes behind stable governed Decision Sites.

The durable architecture is:

- typed Decision Sites;
- deterministic state projection;
- bounded candidate spaces;
- calibrated typed outputs;
- abstention;
- deterministic disposition;
- evidence and immutable ledgering;
- outcome-label joining;
- replay;
- customer specialization;
- champion/candidate lifecycle;
- automatic demotion;
- judgment-to-guard conversion.

---

## 4. Definition of a Decision Site

A **Decision Site** is a versioned, independently governed semantic judgment point.

```text
Deterministic System State
          |
          v
    State Projector
          |
          v
 Canonical Decision State
          |
          v
   Typed Question Pack
          |
          v
    Decision Engine
          |
          v
 Typed Judgment + Probabilities
          |
          v
 Calibration / Abstention
          |
          v
 Deterministic Disposition Policy
          |
          v
 FSM / Executor / Scheduler / Reviewer Action
```

The learned model does not directly perform the action.

The disposition policy does.

---

## 5. Foundational Principles

### 5.1 Authority remains deterministic

No decision model may grant authority.

A model may report:

```text
semantic_risk = HIGH
```

but it may not authoritatively report:

```text
permission = GRANTED
```

Authority derives only from:

- signed AFIE Definition;
- signed AFSE Runtime Binding;
- Authority Pack;
- deterministic policy;
- authoritative FSM state;
- leases and write masks;
- resource constraints;
- accepted human authority.

### 5.2 Decision and disposition are separate

A Decision Site may return:

```text
is_destructive = 0.91
reversibility = LOW
matches_current_goal = 0.32
```

Deterministic disposition code may then apply:

```text
IF not_authorized:
    BLOCK
ELSE IF semantic_risk >= threshold:
    ESCALATE
ELSE:
    EXECUTE
```

The model supplies semantic evidence. Code owns consequence.

### 5.3 Abstention is a valid successful result

`INCONCLUSIVE`, `ABSTAIN`, or `ESCALATE` are first-class outputs, not inference failures.

### 5.4 Deterministic baselines come first

If a decision is reliably computable in ordinary code, AuraForge MUST use ordinary code.

Semantic models complement deterministic mechanisms; they do not replace precise mechanisms for fashion or convenience.

---

## 6. Relationship to AFIE, AFSE, and AFEE

### 6.1 AFIE — permission origin and architectural governance

AFIE authors and signs the LOOP Definition and governs architecture.

AFIE may specify:

- required semantic checks;
- Decision Site categories;
- safety invariants;
- risk posture;
- acceptance criteria;
- mandatory evidence;
- semantic policies that are architectural rather than runtime-specific.

AFIE MUST NOT bind runtime-specific details such as:

- a particular Laya checkpoint;
- Jev endpoint;
- GPU/CPU;
- executor instance;
- harness version;
- machine or worktree;
- runtime retry count;
- runtime inference slot.

### 6.2 AFSE — deterministic supervision and runtime binding

AFSE selects and signs runtime bindings from qualified capabilities immediately before execution.

Example:

```yaml
decision_site: test-integrity/v1
engine: laya/customer-a/test-integrity:v7
fallback:
  - jev/test-integrity:v2
  - system2-review
threshold_profile: enterprise-strict-v3
calibration: integrity-cal-2026-09
```

AFSE may rebind runtime engines without returning to AFIE provided it does not widen AFIE authority.

AFSE owns:

- disposition policy;
- thresholds and runtime policy;
- fallback legality;
- decision timeout behavior;
- deterministic scheduling consequences;
- lifecycle status;
- runtime engine selection from the approved registry.

### 6.3 AFEE — bounded execution

AFEE verifies:

1. AFIE Definition signature;
2. AFSE Binding signature;
3. Definition/Binding hash linkage;
4. Binding within Definition;
5. execution authority.

AFEE invokes Decision Sites where bound, receives typed evidence, submits that evidence to deterministic disposition, and executes only the resulting authorized action.

Decision Fabric output is execution evidence, not execution authority.

---

## 7. Three-Rate Control Model

AuraForge SHALL describe control as three conceptual rates.

### Layer 1 — deterministic reflex

Runs on every relevant event.

Examples:

- permissions;
- FSM validation;
- leases;
- write masks;
- signatures;
- schema checks;
- hard budgets;
- path rules;
- immutable evidence checks.

### Layer 2 — System 1 semantic judgment

Runs at bounded checkpoints.

Examples:

- semantic risk;
- relevance;
- test likelihood;
- ambiguity;
- semantic consistency;
- recurrence;
- attention prioritization.

### Layer 3 — System 2 deliberation

Invoked for:

- architecture;
- complex debugging;
- novel failure;
- deep semantic reconciliation;
- ambiguous requirement interpretation;
- complex review;
- human decision.

Desired shape:

```text
DETERMINISTIC RULE
    |
    +-- conclusive ----------------------> ACT
    |
    +-- semantic judgment needed
                   |
                   v
                SYSTEM 1
                   |
             +-----+-----+
             |           |
         confident     abstain
             |           |
             v           v
            ACT       SYSTEM 2
```

---

## 8. Decision Cascade

A Decision Site MAY bind an escalation cascade such as:

```text
Deterministic rule
      |
Customer-specialized Laya
      |
Generic local System-1 engine
      |
Hosted Jev
      |
Local reasoning model
      |
Frontier System-2 model
      |
Human
```

Not every site requires every rung.

**Escalation may increase intelligence. It MUST NOT increase authority.**

---

## 9. Canonical Decision-Site Registration

Every production Decision Site MUST register, at minimum:

```yaml
decision_site_id:
decision_family:
schema_version:

state_projection:
  projector_id:
  projector_version:
  max_tokens:
  deterministic_features:
  excluded_fields:

questions:
  question_pack_id:
  question_pack_version:
  types:

policy:
  policy_version:
  abstention_policy:
  fallback_chain:
  timeout_policy:
  disposition_policy:

model:
  champion:
  candidate:
  teacher:
  calibration_version:

learning:
  outcome_label_function:
  sampling_policy:
  counterfactual_policy:
  drift_policy:

governance:
  status:
  owner:
  risk_class:
  authority_effect:
  promotion_requirements:

observability:
  required_metrics:
  evidence_retention:
```

---

## 10. Supported Decision Types

### 10.1 NOUL / YES-NO

Binary probability-bearing judgment.

### 10.2 CHOICE

Selection from a bounded set of deterministic candidate IDs.

Large choices MUST be partitioned, hierarchically resolved, or converted to one yes/no question per candidate.

### 10.3 SCORE

Bounded ordinal judgment with explicitly anchored score semantics.

### 10.4 RANK

Rank a deterministic candidate set.

Ranking is often preferable to pretending that an absolute probability threshold is stable across domains.

### 10.5 ACT / ESCALATE

Explicit abstention/escalation decision.

---

## 11. State Projection Architecture

State construction is a first-class subsystem.

```text
Full System State
      |
      v
Deterministic Extraction
      |
      v
Normalization
      |
      v
Feature Reduction
      |
      v
Untrusted-Prose Removal
      |
      v
Canonical Serialization
      |
      v
Token-Budget Validation
      |
      v
Decision State
```

### 11.1 Facts before narratives

Prefer:

```text
changed_assertions: -2
new_skip_markers: 1
changed_files: 3
test_exit_code: 0
```

over agent claims such as "this is harmless".

### 11.2 Self-advocacy is untrusted

Agent-generated comments, explanations, commit prose, safety claims, and completion claims MUST NOT become trusted safety features merely because the agent authored them.

### 11.3 Bounded state

A Decision Site requiring huge context is probably System 2 work.

### 11.4 Immutable evidence references

Prefer IDs and hashes:

- SHA;
- hunk ID;
- test ID;
- finding ID;
- tool ID;
- artifact ID;
- path ID;
- capability ID.

The model chooses among identifiers created by deterministic software.

---

## 12. Decision Ledger

Every production Decision Site invocation MUST create an immutable ledger event.

Minimum logical fields:

```json
{
  "decision_id": "...",
  "decision_site_id": "test-integrity/v1",
  "timestamp": "...",

  "loop_id": "...",
  "generation_id": "...",
  "run_id": "...",

  "state_projection_hash": "...",
  "question_pack_hash": "...",

  "engine": "...",
  "model_hash": "...",
  "calibration_version": "...",

  "answers": {},
  "raw_probabilities": {},
  "calibrated_probabilities": {},
  "abstained": false,

  "disposition_policy_version": "...",
  "disposition": "...",

  "fsm_state_before": "...",
  "fsm_state_after": "...",

  "evidence_refs": [],
  "eventual_outcome": null
}
```

The outcome may be joined later.

---

## 13. Outcome Learning

AuraForge SHOULD train and qualify Decision Sites primarily from what actually happened.

Examples:

```text
Decision: test X unlikely to fail
Later: full CI -> test X failed
Label: FALSE NEGATIVE
```

```text
Decision: memory M useful
Later: agent used M and scoped test passed
Label: POSITIVE
```

```text
Decision: approach repeats failed strategy
Later: architect confirms recurrence
Label: POSITIVE
```

### 13.1 Label hierarchy

Prefer evidence in this order:

1. deterministic ground truth;
2. accepted human judgment;
3. independent cross-model adjudication;
4. teacher-model agreement.

Teacher agreement is cold-start supervision, not ground truth.

---

## 14. Calibration

Raw probabilities MUST NOT automatically be treated as calibrated.

Track calibration per:

- Decision Site;
- question family;
- model/checkpoint;
- state-projector version;
- customer/domain;
- calibration transform;
- calibration dataset.

Metrics SHOULD include:

- expected calibration error;
- Brier score where applicable;
- precision/recall;
- coverage;
- selective risk;
- abstention quality.

A model upgrade without qualification is a behavior change.

---

## 15. Champion / Candidate Lifecycle

Each production site SHOULD maintain:

```text
CHAMPION -------- production
    |
CANDIDATE ------- replay / shadow / canary
```

Qualification stages:

1. offline replay;
2. shadow;
3. bounded canary;
4. production champion;
5. continuous drift monitoring.

A trained candidate MUST NOT replace a champion merely because training completed.

---

## 16. Automatic Demotion

Deterministic lifecycle policy SHOULD automatically demote a model on conditions such as:

- false-negative rate exceeds the site SLO;
- calibration exceeds allowed error;
- abstention rate collapses unexpectedly;
- outcome accuracy falls;
- input distribution drifts materially;
- projector schema becomes incompatible;
- model identity does not match binding;
- required calibration is absent.

Rollback itself is deterministic.

---

## 17. Decision-Site Risk Classes

### D0 — advisory

Wrong answer changes presentation/order only.

### D1 — efficiency

Wrong answer wastes compute/time.

### D2 — workflow

Wrong answer may create extra retries, holds, or escalation.

### D3 — safety-sensitive

Wrong answer may fail to identify dangerous behavior.

Every D3 site MUST have an independent deterministic backstop.

No learned model may be the sole protection.

---

# Part II — Canonical Decision Sites

## 18. Decision Site: Test Integrity / Oracle-Tamper Sentinel

**Priority:** Wave 1 / D3.

Purpose:

> Detect whether a coding agent weakened the evidence used to prove its own change correct.

Questions may include:

- `test_weakened?`
- `mock_bypasses_unit_under_test?`
- `golden_rewritten_to_match_output?`
- `verification_authenticity?`
- `test_change_intent?`

Preferred state features:

- assertion-count delta;
- skip/xfail/only marker delta;
- mock delta;
- tolerance changes;
- snapshot/golden churn;
- test hunk before/after;
- declared behavior change.

Coder comments and justification SHOULD be stripped.

Deterministic controls may include:

- deletion of protected tests blocked;
- old-oracle execution;
- protected assertion rules;
- mutation checks where configured.

Outcome labels:

- old-oracle result;
- mutation result;
- adversarial review;
- accepted human override;
- later defect.

This site protects the validity of all downstream evidence.

---

## 19. Decision Site: Adaptive Verification Planner

**Priority:** Wave 1 / D1-D2.

Merge:

- predictive test selection;
- semantic test targeting;
- fail-fast test ordering;
- Semantic Test Frontier.

Flow:

```text
Diff
 |
 +-- deterministic impact analysis
 +-- protected tests
 +-- System-1 relevance/failure likelihood
             |
             v
       ordered Run test plan
             |
             v
        fail-fast execution
             |
             v
      Generation-close FULL CI
```

Questions per candidate test/cluster:

- `likely_relevant?`
- `p_failure?`
- `priority?`
- `marginal_value?`

Hard invariants:

- Authority-Pack-mandated tests never skipped;
- protected tests never skipped;
- incident-linked tests never skipped;
- explicit acceptance tests never skipped;
- Generation-close full CI remains mandatory.

A deterministic/random sample of skipped candidates SHOULD run periodically to measure false negatives.

Full CI provides high-quality outcome labels.

---

## 20. Decision Site: Semantic Contract / Brief Linter

**Priority:** Wave 1 / initially D0-D1.

Inputs:

- LOOP/Generation brief;
- acceptance criteria;
- interfaces;
- output contract;
- scope;
- relevant Authority Pack projection;
- accepted glossary.

Questions:

- `ambiguity?`
- `contradiction?`
- `undefined_terms?`
- `acceptance_oracle_defined?`
- `scope_boundary_clear?`
- `requires_ungranted_authority?`

Initial deployment SHOULD be shadow/advisory.

System 1 MUST NOT silently rewrite requirements.

Labels:

- later clarification;
- scope amendment;
- review dispute;
- acceptance rejection;
- Generation churn.

---

## 21. Decision Site: Pre-Execution Side-Effect Hazard Sentinel

**Priority:** Wave 1 / D3.

Applies immediately before meaningful side effects, including:

- destructive Git operations;
- file deletion/overwrite;
- package installation;
- database mutation;
- SQL mutation;
- cloud mutations;
- deployment;
- process termination;
- credential/security changes.

Questions:

- `surprising_relative_to_goal?`
- `destructive?`
- `reversible?`
- `scope_aligned?`
- `semantic_blast_radius?`

Core separation:

```text
Authority Pack:
"Is this class of operation permitted?"

Decision Fabric:
"Does this particular permitted operation make semantic sense here?"
```

A Decision Site MAY convert:

```text
ALLOW -> HOLD/ESCALATE
```

It MUST NEVER convert:

```text
DENY -> ALLOW
```

---

## 22. Decision Site: Judgment-to-Guard Ratchet

**Priority:** Wave 2 / strategic platform capability.

Purpose:

> Convert repeated probabilistic judgment into deterministic protection.

```text
System-2 reviewer discovers reusable rule
             |
             v
Decision Fabric recognizes recurrence/generality
             |
             v
          mechanizable?
        /              \
      no                yes
      |                  |
   memory          create normal LOOP
                         |
                         v
                deterministic guard
                         |
                         v
                 fail-before/pass-after
                         |
                         v
                review + human accept
                         |
                         v
               future deterministic rule
```

Possible mechanisms:

- unit test;
- integration test;
- lint rule;
- Roslyn/static analyzer;
- AST rule;
- grep/pattern rule;
- schema rule;
- policy rule;
- `not_mechanizable`.

A new guard enters production only through ordinary governed engineering and acceptance.

Strategic objective:

> Mature customers should require less probabilistic judgment per unit of engineering over time.

---

## 23. Decision Site: Review and Acceptance Attention Map

**Priority:** Wave 2 / D0-D1.

Purpose:

> Direct scarce reviewer/human attention without hiding evidence.

Inputs may include:

- hunk/path;
- subsystem criticality;
- behavior change;
- test evidence;
- novelty;
- reviewer disagreement;
- defect history;
- blast radius;
- Authority Pack sensitivity.

Questions:

- `attention_score?`
- `behavior_change?`
- `security_surface?`
- `claim_matches_evidence?`
- `review_reason?`

Hard rule:

**No hunk disappears.**

The map changes ordering/emphasis only.

Random low-ranked hunk promotion SHOULD be used to measure automation bias and miss rate.

---

## 24. Decision Site: Recovery Policy Engine

**Priority:** Wave 2 / D2.

Combine:

- agent health;
- anti-repeat fingerprint;
- signed steer library;
- candidate salvage;
- failure triage;
- repair-vs-restart.

Questions:

- `repeating_failed_approach?`
- `salvageable?`
- `approach_novelty?`
- `repair_complexity?`
- `likely_intervention?`

Allowed intervention IDs may include:

```text
CONTINUE
REREAD_ERROR
RUN_SCOPED_TEST
NARROW_TO_FAILING_TEST
REVERT_LAST_EDIT
LOAD_RESOURCE
PATCH_FORWARD
RESET_CANDIDATE
SPLIT_TASK
REQUEST_ARCHITECT
ABORT_RUN
```

Any steer injected into an agent MUST be:

- pre-approved;
- signed/versioned;
- bounded;
- parameterized only with deterministic values;
- incapable of widening authority or tool access.

Environment changes MUST invalidate stale anti-repeat equivalence where appropriate.

---

## 25. Decision Site: Resource and Memory Utility Filter

**Priority:** Wave 2 / D1.

This consolidates:

- resource shortlist;
- skill/tool/MCP relevance;
- knowledge relevance;
- episodic-memory utility.

Do NOT expose hundreds of tools/skills in one Choice.

Use:

```text
deterministic partition
       |
lexical/embedding recall
       |
System-1 rerank
       |
small candidate set
```

Memory questions SHOULD distinguish:

- relevant vs merely similar;
- verified reusable fix vs historical discussion;
- stale vs current;
- actionable vs narrative;
- contradicts current state vs compatible.

---

## 26. Decision Site: Candidate Portfolio Controller

**Priority:** Wave 3 / D1-D2.

When local capacity permits best-of-N candidate generation:

```text
N candidate Runs
      |
cheap trajectory scoring
      |
retain diverse survivors
      |
test top-K
      |
review winner(s)
```

Questions:

- `on_track?`
- `likely_to_pass?`
- `candidate_quality?`
- `continue_or_prune?`

Safeguards:

- never prune final candidate;
- minimum candidate age;
- preserve model-family diversity;
- journal pruned work;
- no candidate accepted without normal verification;
- deterministic reprieve/exploration cohort to measure pruning regret.

---

## 27. Decision Site: Semantic Interference Predictor

**Priority:** Wave 3 / D1-D2.

Purpose:

> Detect semantic concurrency/integration hazards that mechanical file overlap may miss.

Questions:

- `safe_to_parallelize?`
- `semantic_interaction?`
- `likely_collision_kind?`
- `integration_order?`

Inputs:

- write masks;
- dependency graph;
- symbols/contracts;
- task briefs;
- migrations;
- tests;
- branch ancestry.

Any deterministic conflict MUST serialize regardless of the model.

System 1 may make scheduling more conservative; never less safe than deterministic controls.

---

## 28. Decision Site: Speculative Next-State Prewarmer

**Priority:** Wave 3 / D1.

Questions:

- `likely_next_role?`
- `likely_next_model?`
- `likely_context_pack?`
- `probability_tests_pass?`
- `likely_reviewer_path?`

Allowed speculative actions:

- load model;
- warm immutable prefix/KV;
- materialize read-only context;
- resolve MCP metadata;
- stage inactive worktree.

All speculative allocations MUST be:

`VOLATILE`, `PREEMPTIBLE`, and `NON-AUTHORITATIVE`.

They MUST NOT:

- transition FSM;
- widen authority;
- execute mutation;
- block authorized work;
- incur unapproved spend.

---

## 29. Specialized Decision Packs

The Fabric SHALL support domain/application packs rather than forcing every site into the core.

Examples:

### Review Finding Disposition

Classify finding:

`blocking | suggestion | question | nit | praise`

and identify whether an already-authorized mechanical remediation exists.

### Dependency Upgrade Risk

Use changelog/API/lock graph/history evidence, with CI as ground truth.

### Capability-Gap Advisor

Classify recurring failure as:

`unknown_api | missing_tool | missing_permission | missing_domain_context | model_too_weak | bad_brief`

Recommendations never self-install capabilities.

### E2E Locator Healing

Use prior/current DOM, failing selector, assertions, and application diff to propose a new target while deterministic browser guards and unchanged assertions remain authoritative.

---

# Part III — Revised Status of Original Ideas

## 30. Fine-Grained `route_task` Is Demoted

AuraForge SHALL NOT depend on a highly granular learned task-difficulty router as a foundational control primitive.

Use only coarse starting-rung selection:

```text
LOCAL
MID
FRONTIER
```

and rely on observed deterministic evidence to escalate.

Reasons:

- counterfactual labels are missing without exploration;
- task difficulty may not be visible in the brief;
- a bad route may waste a whole Run/review cycle;
- model capabilities and cost change;
- tests and review outcomes provide stronger evidence.

Correct pattern:

```text
coarse semantic starting-rung recommendation
          +
deterministic failure-driven escalation
```

---

## 31. State-vs-Reality Reconciliation

Retain the capability but split responsibilities.

Deterministic code establishes:

- SHA;
- branch;
- ancestry;
- CI status;
- tests;
- PR state;
- approvals;
- files;
- leases.

System 1 may:

- identify likely claim/evidence mismatch;
- shortlist relevant evidence;
- identify discrepancies requiring System 2.

Deep proof such as "the implementation fully satisfies architectural intent" is System 2 work.

---

## 32. Verdict Extraction

Free-text verdict extraction is a compatibility adapter, not a first-party architectural goal.

First-party reviewers SHOULD emit typed structured findings.

Malformed first-party review output SHOULD produce:

`MALFORMED_REVIEW_OUTPUT`

followed by deterministic retry or escalation.

Do not permanently train models to compensate for an avoidable interface defect.

---

## 33. DAG Decomposition

System 1 may assist DAG construction but MUST NOT independently prove task independence.

Correct order:

```text
static dependency analysis
write masks
symbol graph
declared contracts
        |
System-1 semantic coupling check
        |
System-2 architect if ambiguous
        |
deterministic DAG
```

---

# Part IV — Fabric Runtime Architecture

## 34. Logical Components

```text
+------------------------------------------------+
| AuraForge Decision Fabric                      |
|                                                |
|  Decision Gateway                              |
|      |                                         |
|  Decision Site Registry                        |
|      |                                         |
|  State Projector Runtime                       |
|      |                                         |
|  Question-Pack Runtime                         |
|      |                                         |
|  Engine Router                                 |
|      +-- Laya                                  |
|      +-- Jev                                   |
|      +-- local/logit deciders                  |
|      +-- future engines                        |
|      |                                         |
|  Calibration + Abstention                      |
|      |                                         |
|  Deterministic Disposition Engine              |
|      |                                         |
|  Decision Ledger                               |
|      |                                         |
|  Outcome Joiner                                |
|      |                                         |
|  Replay / Qualification Lab                    |
|      |                                         |
|  Model Lifecycle Controller                    |
|      |                                         |
|  Telemetry                                     |
+------------------------------------------------+
```

---

## 35. Decision Gateway

Expose a stable internal API independent of model engine.

Conceptual request:

```http
POST /v1/decisions/{decision_site_id}
```

```json
{
  "context_ref": "...",
  "state": {},
  "candidate_ids": [],
  "authority_context_ref": "...",
  "correlation": {
    "loop_id": "...",
    "generation_id": "...",
    "run_id": "..."
  }
}
```

Conceptual response:

```json
{
  "decision_id": "...",
  "site": "...",
  "answers": {},
  "confidence": {},
  "abstained": false,
  "engine": "...",
  "model_version": "...",
  "calibration_version": "...",
  "evidence_ref": "..."
}
```

No endpoint returns authoritative permission.

---

## 36. Engine Abstraction

Conceptual interface:

```text
DecisionRequest
      |
IDecisionEngine
      |
DecisionResponse
```

Possible implementations:

- `LayaDecisionEngine`
- `JevDecisionEngine`
- `OpenAICompatibleLogitEngine`
- `LocalDecisionEngine`
- `ReplayDecisionEngine`
- `MockDecisionEngine`

---

## 37. Decision Site Registry

The registry MUST answer:

- which Decision Sites exist;
- site/schema versions;
- projector version;
- qualified engines;
- champion/candidate;
- fallback chain;
- calibration;
- threshold profile;
- risk class;
- disposition policy;
- SLO;
- customer/domain scope.

---

## 38. Deterministic Disposition Engine

The Disposition Engine consumes:

```text
deterministic facts
+
Decision Site output
+
Authority Pack
+
FSM state
+
risk profile
```

and produces only a finite legal disposition such as:

```text
CONTINUE
WARN
HOLD
ESCALATE
RETRY
SERIALIZE
RUN_EXTRA_TEST
REQUEST_ARCHITECT
REQUEST_HUMAN
```

The Disposition Engine MUST NOT use generative AI.

It SHOULD be exhaustively unit-testable.

---

## 39. Replayability

Every site MUST maintain replay fixtures including:

- state projection;
- question pack;
- engine identity;
- model hash/version;
- calibration;
- disposition policy;
- expected output envelope.

Support:

### Exact replay

Where deterministic inference allows it.

### Behavioral contract replay

Where external engines are not bitwise stable.

Examples:

- must abstain on unsafe/underspecified fixture;
- must never classify protected destructive fixture as safe;
- risk score must remain within an accepted envelope.

Model upgrade that breaks qualification MUST NOT silently deploy.

---

## 40. Determinism vs Statistical Semantics

Record whether a site provides:

### Bitwise repeatability

Same input -> same output.

or

### Statistical qualification

Meets an accepted envelope for calibration, risk, coverage, abstention, and outcome quality.

Do not conflate the two.

---

## 41. Deadlines and Timeout

Every Decision Site has a deadline.

Late result normally becomes `INCONCLUSIVE` and enters deterministic fallback.

For efficiency sites:

```text
timeout -> do nothing / baseline
```

For safety sites:

```text
timeout -> hold / escalate
```

Never wait indefinitely for a semantic model.

---

## 42. Site-Specific Thresholds

There is no platform-global confidence threshold.

Threshold depends on:

- customer;
- site;
- question;
- risk class;
- FSM state;
- model;
- calibration;
- cost of false positive;
- cost of false negative.

---

## 43. Exploration and Counterfactual Learning

Some choices hide the outcome of rejected alternatives.

Use bounded deterministic exploration where safe:

- skipped-test sampling;
- candidate prune reprieves;
- random low-ranked hunk promotion;
- steer withholding cohort;
- occasional resource-filter bypass.

Exploration MUST be explicitly governed by site risk and policy.

---

## 44. Customer Specialization

Prefer:

```text
shared encoder
+
independently versioned customer/site-specific heads
```

rather than one giant customer model.

Example:

```text
customer-a/
  test-integrity/v4
  brief-lint/v2
  failure-triage/v8
  memory-utility/v3
```

This isolates regression and calibration by decision family.

---

## 45. Versioning

Version all behaviorally material parts:

- Decision Site;
- state projector;
- feature schema;
- question pack;
- model/checkpoint;
- calibration;
- threshold profile;
- fallback chain;
- disposition policy;
- outcome-label function.

Decision version != model version.

---

## 46. Multi-Tenant Isolation

By default, customer-specific:

- history;
- labels;
- calibration;
- fine-tunes;
- thresholds;
- memory;
- vocabulary;
- outcome metrics

remain customer-isolated.

Cross-customer learning requires an explicit separately governed design.

---

## 47. Security Requirements

1. Treat repository/model/external text as untrusted input.
2. Prefer deterministic features for D3 sites.
3. Decision output SHOULD name registered IDs rather than manufacture executable instructions.
4. Question packs, steer templates, disposition policies, and models MUST have immutable identity.
5. A semantic recommendation cannot widen tools, permissions, scope, budget, or authority.
6. Failure/timeout cannot silently become approval.
7. Missing model/calibration/registry evidence MUST fail according to the site's deterministic fallback.

---

## 48. Observability

Each site SHOULD expose:

### Traffic

- decisions/hour;
- p50/p95/p99 latency;
- engine utilization;
- fallback rate.

### Quality

- precision;
- recall;
- false-negative/false-positive rate;
- coverage;
- abstention;
- ECE;
- Brier where applicable.

### Operational

- timeout;
- schema failure;
- model failure;
- projector failure;
- fallback invocation.

### Learning

- label count;
- label delay;
- teacher disagreement;
- champion/candidate disagreement;
- drift.

### Business/engineering outcome

- LLM calls avoided;
- tokens saved;
- wall time saved;
- extra Runs caused;
- defects caught;
- unnecessary escalations.

---

## 49. Enterprise Decision Audit

The ledger SHOULD answer:

- Why did this Run receive extra review?
- Why was this tool call held?
- Why did test A run before test B?
- Which model/checkpoint made the judgment?
- What probability/confidence was returned?
- Which deterministic policy consumed it?
- What authority applied?
- Was the judgment later correct?
- Has quality drifted since deployment?

---

## 50. Judgment-Reduction Flywheel

AuraForge SHALL optimize for useful work per expensive deliberation, not for number of AI decisions.

Desired learning progression:

```text
System 2 discovers
      |
System 1 learns recurrence
      |
Deterministic guard captures stable rule
```

Flywheel:

```text
More governed work
  -> more outcomes
  -> better Decision Sites
  -> fewer expensive judgments
  -> more deterministic guards
  -> higher autonomous throughput
  -> more governed work
```

---

## 51. LOOP Integration

Representative Generation:

```text
AFIE Definition
      |
AFSE Runtime Binding
      |
Semantic Contract Lint
      |
Generation admitted
      |
Coder Run
      +-- resource/memory utility
      +-- agent-health / recovery
      +-- signed steer decisions
      +-- pre-tool hazard sentinel
      |
Candidate patch
      |
Test Integrity Sentinel
      |
Adaptive Verification Planner
      |
Impact-scoped tests
      |
Failure triage / recovery
      |
Reviewer
      +-- attention map
      +-- typed finding disposition
      +-- judgment-to-guard candidate
      |
Generation close
      |
FULL CI
      |
Adversarial review
      |
Human acceptance
      |
Outcome join
      |
Decision Ledger labels
      |
Qualification / learning
```

---

## 52. What MUST Never Become Probabilistic

These remain deterministic:

- authority;
- scope;
- signature verification;
- valid FSM transitions;
- leases;
- write masks;
- hard budget ceilings;
- mandatory tests;
- Generation-close full-CI requirement;
- human acceptance requirement;
- model promotion floors;
- fallback legality;
- registry integrity;
- evidence-to-SHA linkage;
- signed Definition/Binding verification.

System 1 may add caution. It may not weaken these controls.

---

## 53. Failure Philosophy

The Fabric assumes semantic decisions will sometimes be wrong.

Prefer sites where:

```text
wrong judgment -> wasted milliseconds/minutes
```

rather than:

```text
wrong judgment -> unauthorized action
```

This is why ordering, prioritization, retrieval, and speculative preparation are attractive; authority granting is prohibited.

---

# Part V — Delivery Plan

## 54. Phase 0 — Decision Fabric Kernel

Build:

1. Decision Gateway;
2. Decision Site Registry;
3. typed question schemas;
4. state-projector SDK/runtime;
5. engine abstraction;
6. local Laya adapter;
7. Jev adapter;
8. calibration subsystem;
9. deterministic Disposition Engine;
10. Decision Ledger;
11. replay harness;
12. outcome joiner;
13. champion/candidate lifecycle;
14. telemetry.

### Exit condition

A trivial Decision Site can run end-to-end in shadow mode, be replayed, produce evidence, receive an eventual outcome label, and qualify a candidate model without gaining any LOOP authority.

---

## 55. Phase 1 — High-Value / High-Label-Quality Sites

Implement first:

1. **Test Integrity Sentinel**
2. **Adaptive Verification Planner**
3. **Semantic Contract Linter**
4. **Pre-Execution Side-Effect Hazard Sentinel**

These provide immediate safety/throughput value and strong labels.

---

## 56. Phase 2 — Learning and Human Efficiency

Implement:

1. Judgment-to-Guard Ratchet;
2. Review/Acceptance Attention Map;
3. Recovery Policy Engine;
4. Resource/Memory Utility Filter;
5. richer agent-health semantic sites.

---

## 57. Phase 3 — Parallelism and Compute Optimization

Implement when scheduler scale requires it:

1. Candidate Portfolio Controller;
2. Semantic Interference Predictor;
3. Integration Order Advisor;
4. Speculative Next-State Prewarmer;
5. GPU/KV scheduling hints.

These SHALL NOT block Phase 0/1.

---

## 58. Phase 4 — Decision Packs

Add reusable packs such as:

- Playwright locator healing;
- dependency maintenance;
- SQL safety;
- deployment risk;
- MIE domain decisions;
- incident triage;
- knowledge retrieval;
- capability-gap advisor.

---

## 59. Suggested Implementation Shape

Conceptual module/repository layout:

```text
decision-fabric/
|
+-- gateway/
+-- registry/
+-- projectors/
+-- questions/
+-- engines/
|   +-- laya/
|   +-- jev/
|   +-- replay/
+-- calibration/
+-- dispositions/
+-- ledger/
+-- outcomes/
+-- qualification/
+-- telemetry/
+-- sites/
|   +-- test-integrity/
|   +-- adaptive-verification/
|   +-- brief-lint/
|   +-- side-effect-hazard/
|   +-- attention-map/
|   +-- recovery-policy/
+-- schemas/
+-- fixtures/
+-- policies/
+-- tests/
```

Repository topology may evolve. The architectural boundaries above are normative.

---

## 60. Required Test Strategy

The Fabric itself MUST receive unusually strong qualification.

### Unit

- projector;
- schema;
- disposition;
- threshold;
- timeout;
- fallback;
- precedence.

### Replay

Frozen historical decision fixtures.

### Calibration

Per site/question/model.

### Adversarial

Inputs attempting to manipulate semantic classification.

### Drift

Champion vs candidate.

### Failure injection

Simulate:

- Laya unavailable;
- Jev unavailable;
- missing calibration;
- malformed response;
- timeout;
- oversized state;
- impossible choice;
- ledger failure according to risk class.

### Fallback qualification

Every semantic component MUST be removable while leaving a safe deterministic system.

---

## 61. Normative Invariants

**DF-I01** No Decision Site can grant authority.  
**DF-I02** No Decision Site directly performs an authoritative FSM transition.  
**DF-I03** Every production Decision Site has a deterministic fallback.  
**DF-I04** Every production decision is ledgered.  
**DF-I05** Every production model/checkpoint is pinned.  
**DF-I06** Every state projector is versioned.  
**DF-I07** Every question pack is versioned.  
**DF-I08** Every disposition policy is versioned.  
**DF-I09** `INCONCLUSIVE` is valid.  
**DF-I10** Timeout cannot silently become approval.  
**DF-I11** Learned judgment cannot bypass a mandatory deterministic check.  
**DF-I12** Candidate promotion requires deterministic qualification criteria.  
**DF-I13** Authority remains traceable to AFIE Definition and AFSE Runtime Binding.  
**DF-I14** Human cryptographic acceptance remains where required.  
**DF-I15** Decision output is evidence, not permission.  
**DF-I16** Engine substitution cannot widen authority.  
**DF-I17** State projection and model output must be provenance-addressable.  
**DF-I18** A D3 site cannot be the sole safety protection.  
**DF-I19** Repository/customer-specific mirrors cannot independently amend this legislation.  
**DF-I20** Stable recurring judgment SHOULD preferentially migrate toward deterministic protection.

---

## 62. Non-Goals

The Decision Fabric is not:

- another agent framework;
- another LLM orchestrator;
- a substitute for MAF;
- a substitute for AFSE;
- a substitute for AFEE;
- an autonomous policy authority;
- a giant general-purpose classifier;
- a replacement for deterministic verification;
- a requirement that every decision use ML.

If code can decide reliably, use code.

---

## 63. Product Differentiation

The long-term differentiator is not merely a semantic router.

It is the lifecycle:

```text
typed decision
+
deterministic authority
+
outcome evidence
+
immutable ledger
+
customer specialization
+
calibration
+
shadow deployment
+
canary promotion
+
automatic demotion
+
judgment-to-guard conversion
```

This converts semantic decision models from utilities into governed infrastructure.

---

## 64. Architectural North Star

AuraForge should make large numbers of bounded semantic judgments while reserving expensive reasoning for the minority that genuinely require deliberation.

> **DETERMINISTIC WHEN POSSIBLE**  
> **SYSTEM 1 WHEN SEMANTIC BUT BOUNDED**  
> **SYSTEM 2 WHEN DELIBERATION IS REQUIRED**  
> **HUMAN WHEN AUTHORITY OR JUDGMENT SHOULD REMAIN HUMAN**

---

## 65. Canonical Backlog

### Fabric Kernel

- Decision Gateway
- Site Registry
- State Projector Runtime
- Question Runtime
- Engine Abstraction
- Calibration
- Abstention
- Deterministic Disposition Engine
- Decision Ledger
- Outcome Joiner
- Replay/Qualification Lab
- Champion/Candidate Lifecycle
- Telemetry

### Wave 1

- Test Integrity Sentinel
- Adaptive Verification Planner
- Semantic Contract Linter
- Side-Effect Hazard Sentinel

### Wave 2

- Judgment-to-Guard Ratchet
- Review/Acceptance Attention Map
- Recovery Policy Engine
- Resource/Memory Utility Filter

### Wave 3

- Candidate Portfolio Controller
- Semantic Interference Predictor
- Integration Order Advisor
- Speculative Next-State Prewarmer

### Specialized Packs

- Review Finding Disposition
- Dependency Upgrade Risk
- Capability-Gap Advisor
- Locator Healing
- product/domain-specific Decision Sites

---

## 66. Repository Responsibilities

### AuraForge platform repository

Owns this canonical legislation and platform-wide architectural meaning.

### AFIE repository

Mirrors this legislation and consumes it for Definition authoring, architectural governance, Decision Site requirements, evidence policy, and qualification principles. AFIE does not select runtime-specific engines.

### AFSE / LOOP platform

Implements deterministic Decision Fabric supervision: Runtime Bindings, registry qualification, disposition policy, lifecycle, outcome joining, replay/qualification integration, and orchestration boundaries.

### AFEE / execution paths

Consume authorized Decision Site bindings at executor boundaries, especially pre-tool hazard checks, test-integrity evidence, and bounded advisory sites. Execution cannot widen authority from a semantic result.

### Semantic Control

Implements or hosts qualified System-1 engine adapters/runtimes. It is an engine/runtime implementation, not the owner of Decision Fabric authority or disposition.

### Communication Fabric

Carries Decision Fabric messages/evidence only where required by a bound architecture. Communication transport cannot reinterpret, authorize, or widen Decision Fabric judgments.

### Memory capabilities

Memory supplies candidate context and receives accepted Decision Ledger-derived knowledge according to memory governance. Memory is not decision authority. Resource/memory utility scoring remains advisory and must respect provenance/promotion rules.

---

## 67. Final Architecture Decision

**ADOPT:** AuraForge SHALL implement a first-class Decision Fabric for bounded semantic judgment.

**AUTHORITY:** The Fabric owns no platform authority.

**CONTROL:** Operational consequence is always deterministic disposition under AFIE-defined authority and AFSE runtime binding.

**MODELS:** Jev, Laya, customer-specialized heads, local semantic models, and future engines are replaceable.

**LEARNING:** Real production outcomes feed a governed Decision Ledger and customer-specific qualification/training corpus.

**DEPLOYMENT:** Learned sites follow replay -> shadow -> canary -> champion with deterministic demotion/rollback.

**FIRST IMPLEMENTATIONS:** Test Integrity, Adaptive Verification, Contract Lint, Side-Effect Hazard.

**NORTH STAR:** System 2 teaches System 1; real outcomes teach System 1 better than teachers; stable System-1 judgments become deterministic guards.

This document is the canonical architectural foundation for the AuraForge Decision Fabric.
