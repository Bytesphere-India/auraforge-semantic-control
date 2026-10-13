# HYDRATE (coordinator control head) - PRIMARY hydration source for auraforge-semantic-control

**ROADMAP RULE (Raja 2026-10-13, Medhika 36edfc45, binding): planned work exists only as Milestone > Requirement > LOOP > GEN in `.medhika/ROADMAP.json` (af-roadmap/v1) on auraforge-loop-platform medhika/control (portfolio root). The tracker (Bytesphere-India/auraforge-tracker) is loaded from it by tracker-sync.py on the tick; never hand-edit board structure: ask Medhika (ADD GEN to LOOP-X / UPDATE LOOP-X/Gn / ADD LOOP / ADD REQ, each with DoD+UAT). Every round targets an open GEN in ROADMAP.json (gen-gate.py); a closed GEN is never redone. Execution state (status, RUNs, jobs) stays here and on the issues, never in ROADMAP.json.**

Planes:
- coord/control (this branch, ~/projects/auraforge-semantic-control-coord-control): EXECUTION. Lanes, jobs, SIDs, briefs, transitions, pointers, runbooks, RUNS, NOW, AUTOPILOT. Owner Raja + Claude. All execution state is written HERE.
- medhika/control (~/projects/auraforge-semantic-control-medhika-control): ARCHITECTURE REFERENCE ONLY. Decisions, DoD/UAT plans, reviews, contracts. Never write execution state there.
- qa/control (optional, ~/projects/auraforge-semantic-control-qa-control): sandbox gate plane; copy from auraforge-loop-platform qa/control when gates are needed.
Order after compaction: HYDRATE -> STATE -> TODO -> NOW -> lanes/<running>.md -> newest transitions/. Then scan the medhika/control commits the hook prints.
Medhika workflow: new commits there -> read -> translate into RUNBOOK.md / TODO.md here. Asks go only via ~/.auraforge-work/bin/medhika-send-queue.sh.
Verify live: pueue status; ~/.auraforge-work/bin/job-health check.
Rules: build lanes need NO Raja signing/Bindings/leases. git pull -q --rebase before push. No force-push. $5/day per paid API. Secrets only in ~/.config/auraforge/secrets.env. Shared lane tooling: ~/.auraforge-work/bin (mirrored in auraforge-loop-platform coord/control tools/coordinator).
