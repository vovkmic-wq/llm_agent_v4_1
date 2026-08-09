# LLM Coding Agent V4.1 ICS — Test Report

Date: 2026-08-09
Version: 0.4.1

## Change under test

V4.1 fixes technical-specification refresh semantics. `/spec show`, `/spec run`, and
`/project show` now synchronize the active project contract before displaying or using it.
The project context also refreshes changed source files by SHA-256 and removes stale cached
auto-discovered contracts when the conventional source file disappears.

## Test results

Default deterministic suite:

```text
97 passed, 5 deselected
```

Real-subprocess suite:

```text
5 passed, 97 deselected
```

Total collected scenarios: 102.

## V4.1-specific regression coverage

- Specification created after `/project set` is discovered by the next refresh.
- Changed conventional specification is reloaded and gets a new SHA-256.
- Deleted conventional auto-discovered specification is not served as stale cache.
- Explicit `/spec load <path>` workspace source is re-synchronized after source changes.
- `/spec show` helper refreshes before printing source, SHA-256 and text.
- Missing contract is reported explicitly.
- `/project show` refreshes specification metadata before printing project state.
- Existing V4 incident/IAP/Safety/convergence behavior remains covered by regression tests.

## Inherited critical coverage

- IAP SHA-256 mutation gate and append-only 214 journal;
- operational-period handoff and no-progress ceilings;
- automatic post-mutation verification;
- read-only loop convergence guard;
- raw PAYLOAD protocol and truncated-output repair;
- protocol/provider fallback and checkpoint/resume;
- duplicate call-id and read-before-mutation controls;
- workspace traversal/protected-file/symlink controls;
- pytest/ruff/mypy/compileall execution routing;
- project `.venv` preference;
- context budget, rolling summary and tool-result clipping;
- provider/research/GigaChat hardening.

## Packaging / smoke checks

```text
compileall                                          PASS
AST parse                                           PASS (61 Python files)
Python/test lines > 100 chars                       0
Wheel build 0.4.1                                   PASS
Wheel package data                                  PASS
Wheel target-install import outside source tree     PASS
CLI /project set -> /spec show smoke                PASS
```

The clean-venv dependency download could not be completed because the sandbox package
registry returned no distribution even for PyYAML. Therefore the wheel smoke test was
performed by installing the wheel into an isolated target directory and running it outside
the source tree with the already verified runtime dependencies of the current environment.

## Ruff / mypy

The sandbox does not provide `ruff` or `mypy` (`No module named ruff/mypy`). They remain
declared dependencies and their execution routing is covered by tests. They are not falsely
reported as having been run against V4.1 source code.

## Exact production ZIP verification

The production ZIP was extracted into a separate directory and the exact archived source
was re-tested:

```text
97 passed, 5 deselected
5 passed, 97 deselected   # real-subprocess marker
compileall PASS
AST parse PASS (61 Python files)
Python/test lines > 100 chars: 0
```
