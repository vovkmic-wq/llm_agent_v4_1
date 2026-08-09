# Changelog

## 0.4.1 — 2026-08-09

- `/spec show` now refreshes the specification from the active project before displaying it.
- `/spec run` now refreshes the specification before starting work, preventing stale contracts.
- `/project show` now refreshes project specification metadata before displaying project state.
- Conventional specification files created or changed after project selection are detected automatically.
- Explicit `/spec load <path>` workspace specifications are re-synchronized when the source file changes.
- Removed stale cached auto-discovered specifications when their conventional source file disappears.
- Added regression tests for late-created, changed, deleted and explicitly imported specifications.

## 0.4.0 — 2026-08-08

- Added ICS-inspired persistent incident lifecycle and task classification.
- Added 201 brief, 202 objectives, IAP, SHA-256 approval seal, 214 append-only journal,
  Safety report, period handoffs, follow-ups and AAR artifacts.
- Added mechanical no-source-mutation-before-approved-IAP gate for non-trivial tasks.
- Added operational periods as bounded LLM contexts with persistent transfer-of-command.
- Added read-only/no-progress, plan-revision and safety-stop mechanical ceilings.
- Added automatic quality-gate execution immediately after every successful mutation.
- Added deterministic failure digest for focused repair work.
- Added independent Safety Officer role using real diff and evidence.
- Safety-provider unavailability now blocks closure instead of being treated as a code bug.
- Added default `../AGENT_WORKSPACE`; `AGENT_WORKSPACE_DIR` is optional.
- Preserved V3 protocol raw payload, batch calls, repair/fallback and V3.1 hardening.
- Added regression test proving read-only loops stop far below the legacy 60-step ceiling.
