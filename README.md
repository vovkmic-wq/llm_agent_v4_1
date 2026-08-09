# LLM Coding Agent V4.1 ICS

Production-oriented multi-LLM coding agent for autonomous inspection, editing,
execution, testing and repair of Python projects inside a constrained workspace.

V4 adds a mechanically enforced incident workflow inspired by the Incident Command
System / Development Command System approach described in the Habr case study that
motivated this release. The important change is not extra prompt text: critical rules
are enforced by kernel gates and persistent files.

## Why V4 exists

V3.1 solved malformed/truncated tool JSON, but a real project could still enter a long
read/analyse loop and hit `max_steps=60`. V4 replaces the single anonymous step counter
with incident state, operational periods, no-progress ceilings and automatic verification
immediately after mutations.

## Core workflow

```text
USER TASK
  -> Incident intake / classification
  -> objectives
  -> IAP (Incident Action Plan)
  -> SHA-256 approval gate
  -> execution
  -> automatic post-mutation verification
  -> independent Safety Officer
  -> Reviewer
  -> AAR / close
```

For trivial changes the ceremony is intentionally reduced: the full IAP/Safety cycle is
not required.

## Mechanical controls

Each non-trivial task gets an incident directory under `AGENT_STATE_DIR/incidents/`:

```text
<incident>/
  201-BRIEF.md
  202-OBJECTIVES.md
  IAP.md
  IAP-APPROVED.json
  214-LOG.jsonl
  SAFETY.md
  FOLLOWUPS.md
  PERIOD-XX-HANDOFF.md
  AAR.md
  STATE.json
  originals/
```

Important invariants:

- Source mutation is denied until `IAP.md` has an approved matching SHA-256.
- Editing `IAP.md` after approval mechanically closes the mutation gate.
- Activity journal `214-LOG.jsonl` is append-only.
- Read-only/no-progress loops are mechanically stopped well before the global hard cap.
- Operational-period handoff writes state to disk and starts a fresh LLM context.
- Handoff does not reset no-progress / safety-stop ceilings.
- After every successful mutation the kernel automatically reruns quality gates.
- Final success requires real evidence from the current code revision.
- Safety Officer inspects actual diff/evidence and can block closure.
- If the independent safety provider is unavailable after fallback, the incident is
  blocked rather than falsely declared safe.

## Technical specification as project contract

Put the specification in the active project using one of these names:

```text
TECHNICAL_SPECIFICATION.md
SPECIFICATION.md
SPEC.md
REQUIREMENTS.md
SRS.md
ТЕХНИЧЕСКОЕ_ЗАДАНИЕ.md
ТЗ.md
```

It is stored separately from chat history and receives higher prompt priority.

CLI alternative:

```text
/project set ozon_market_analytics
/spec load ozon_market_analytics/TECHNICAL_SPECIFICATION.md
/spec show
```

For a large pasted specification:

```text
/spec begin
... paste fragments ...
/spec end
```

## Workspace without an explicit .env path

`AGENT_WORKSPACE_DIR` is optional. If it is absent, the agent uses and creates:

```text
../AGENT_WORKSPACE
```

relative to the process working directory. A custom location can still be configured:

```dotenv
AGENT_WORKSPACE_DIR=D:/projects/AGENT_WORKSPACE
```

Runtime state defaults to:

```text
~/.llm_agent_v4
```

and can be overridden with `AGENT_STATE_DIR`.

## Execution

Supported execution tools:

```text
execution.run_python
execution.run_compileall
execution.run_pytest
execution.run_ruff
execution.run_mypy
terminal.execute        # compatibility allow-list, not a shell
```

Properties:

- `shell=False`;
- project cwd is constrained to workspace;
- timeout and stdout/stderr limits;
- child environment is scrubbed of API keys/secrets;
- project `.venv` Python is preferred when present;
- arbitrary PowerShell/cmd/bash/pip/curl pipelines are not allowed.

Enable autonomous execution:

```dotenv
AGENT_ALLOW_CODE_EXECUTION=true
```

Executed project code still has the OS permissions of the user account. This is an
execution guard, not a VM/container sandbox.

## Tool protocol

V4 retains TOOL PROTOCOL V3 from the previous release:

- small JSON control envelope;
- raw `PAYLOAD` for large source patches;
- truncated/invalid/schema protocol classification;
- deterministic repair profile;
- protocol fallback between models;
- batch tool calls.

Independent reads should be batched to reduce LLM round trips.

## Model roles

Roles are configured in `config/agent.yaml`:

```text
planner
executor
safety_officer
reviewer
```

Command decisions can therefore move to another provider/model without losing incident
state. Persistent incident files are the transfer-of-command interface.

## Commands

```text
/project set <path>
/project show
/spec begin | /spec end | /spec load <path> | /spec show | /spec clear | /spec run
/context | /context add <fact>
/decision add <text>
/execution on|off
/task <text>
/state
/incident
/incident approve
/resume [note]
/doctor
/reload
/exit
```

`/incident approve` is needed only when `incident.approval_mode: human` is configured.
The default is kernel approval so autonomous tasks do not stop for routine confirmations.

## Recommended project workflow

```text
/project set ozon_market_analytics
/spec show
/task Протестируй проект на соответствие ТЗ. Исправляй подтверждённые ошибки и повторяй проверки до успешного результата либо конкретного blocker.
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
python main.py
```

Or install the wheel:

```powershell
python -m pip install llm_agent_v4-0.4.1-py3-none-any.whl
llm-agent
```

## Scope limitation

V4 implements the core mechanical incident workflow. It does **not** automatically infer
that several failing tests have several independent root causes and spawn one incident per
root cause. `FailureAnalyzer` focuses the current failure set and `FOLLOWUPS.md` can hold
additional work, but automatic root-cause incident splitting is deliberately not claimed.


## Specification refresh in V4.1

`/spec show`, `/spec run`, and `/project show` refresh the active project contract before
using it. A specification created or edited after `/project set` is therefore detected
without re-selecting the project. Conventional file names include
`TECHNICAL_SPECIFICATION.md`, `SPECIFICATION.md`, `SPEC.md`, `REQUIREMENTS.md`, `SRS.md`,
`ТЕХНИЧЕСКОЕ_ЗАДАНИЕ.md`, and `ТЗ.md`. Explicit `/spec load <path>` sources are also
re-synchronized when their workspace file changes.
