# LLM Coding Agent V4.1 ICS — Architecture

## 1. Design principle

Rules that matter for correctness are kernel mechanisms, not only prompt instructions.
The LLM proposes actions; the kernel owns mutation permission, evidence, ceilings,
verification, persistence and closure.

## 2. Incident lifecycle

```text
INTAKE -> OBJECTIVES -> PLANNING -> APPROVED -> EXECUTING
   -> VERIFYING -> SAFETY_REVIEW -> CLOSED | BLOCKED
```

`IncidentManager` persists the lifecycle under `AGENT_STATE_DIR/incidents`.

## 3. Planning / mutation gate

For normal and architectural incidents:

1. Planner creates objectives and plan.
2. Kernel writes `IAP.md`.
3. Approval stores SHA-256 in `IAP-APPROVED.json`.
4. Every source mutation calls `mutation_gate()`.
5. Any IAP content change invalidates the approval automatically.

Trivial incidents bypass the full plan gate by design.

## 4. Operational periods and context

One operational period is a bounded LLM working context. `record_llm_round()` tracks
rounds. When the period limit is reached, the kernel writes `PERIOD-XX-HANDOFF.md`,
compacts LLM messages and continues from persistent incident artifacts.

No-progress and safety-stop counters survive handoff. A fresh context therefore cannot
reset mechanical attempt ceilings.

## 5. Convergence controls

The old `max_steps=60` remains only a compatibility hard ceiling. V4 additionally has:

- `max_total_llm_rounds`;
- `max_period_rounds` / `max_periods`;
- `max_read_only_rounds`;
- `max_no_progress_rounds`;
- plan-revision ceiling;
- safety-stop ceiling;
- duplicate-call protection;
- read-before-each-mutation;
- automatic post-mutation verification.

A repeated reconnaissance loop is blocked before it can consume the global hard cap.

## 6. Verification

For run/test tasks the kernel runs deterministic baseline quality gates. After every
successful mutation it reruns gates automatically without waiting for an LLM `final`.

`FailureAnalyzer` turns pytest-like output into a compact digest:

- failed tests;
- relevant Python files;
- exception classes;
- stable failure signature.

Evidence from an older code revision cannot prove the current revision is correct.

## 7. Safety Officer

Safety is a separate model role. It receives:

- project contract;
- incident state/IAP;
- actual unified diff built from captured pre-change originals;
- execution/tool evidence.

It is instructed to treat Executor self-report only as a claim. `STOP` prevents closure.
If all safety-provider fallbacks are unavailable, the incident is blocked as an
infrastructure failure.

## 8. Persistent forms

- `201-BRIEF.md` — intake.
- `202-OBJECTIVES.md` — objectives / acceptance criteria.
- `IAP.md` — action plan.
- `IAP-APPROVED.json` — actor/time/SHA seal.
- `214-LOG.jsonl` — append-only chronological audit log.
- `SAFETY.md` — current independent verdict.
- `FOLLOWUPS.md` — adjacent findings outside current scope.
- `PERIOD-XX-HANDOFF.md` — context-window transfer.
- `AAR.md` — after-action review on closure.

## 9. Workspace and execution

Workspace defaults to `../AGENT_WORKSPACE` when the env variable is absent. Traversal,
absolute escape, symlink escape and protected secrets are rejected.

Execution uses `shell=False`, constrained cwd, timeout/output limits and a scrubbed child
environment. Project `.venv` is preferred. It is not an OS-level sandbox.

## 10. Context priority

1. kernel/security rules;
2. technical specification / project contract;
3. incident/IAP state;
4. TaskState / acceptance criteria;
5. project file index and relevant failure digest;
6. project facts/decisions;
7. rolling summary;
8. recent raw chat/tool results.

## 11. Protocol and providers

TOOL PROTOCOL V3 remains the data/control interface: small JSON envelopes, raw payloads,
batch calls, protocol repair and model fallback. Planner, Executor, Safety and Reviewer
can use different providers. Persistent incident files allow transfer of command without
binding an incident to one model session.

## 12. Task sizing

`IncidentKind` selects `trivial`, `normal` or `architectural` ceremony. This avoids full
bureaucracy for small edits. V4 does not yet automatically split a multi-root-cause task
into multiple incidents; such findings are focused by `FailureAnalyzer` and can be
registered as follow-ups.
