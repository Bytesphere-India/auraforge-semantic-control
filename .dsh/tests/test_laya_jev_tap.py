"""Polyglot conformance runner for the Jev shadow tap.

The authoritative service tests are TypeScript (`tsx --test`, wired into
`package.json` `"test"` per the brief).  This module lets a pytest-based CI run
the same impact-scoped serve/tap suite and asserts a positive pass count with
zero failures, so the result is visible to whichever harness collected it.  It
is impact-scoped: it runs the four `test_serve_*.ts` modules, not the full
`yarn test` (which also covers model/download tests).  It does not reimplement,
replace or weaken the TypeScript tests.  This file lives inside the project
worktree (`<repo>/.dsh/tests/`), not in a shared or external location.

Fail-closed on every infrastructure problem: a missing `tsx`, a timeout, a
non-zero exit, or a summary with no tests / no passes / any failure all fail the
test rather than reporting a false positive.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# <repo>/.dsh/tests/test_laya_jev_tap.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
TSX = REPO_ROOT / "node_modules" / ".bin" / "tsx"

# The impact-scoped TypeScript modules that verify the `laya-serve` surface;
# `test_serve_jev_tap.ts` is the tap suite.
SERVE_TEST_MODULES = (
    "test/test_serve_protocol.ts",
    "test/test_serve_shadow.ts",
    "test/test_serve_http.ts",
    "test/test_serve_jev_tap.ts",
)
TIMEOUT_SECONDS = 900

_INT = r"(\d+)"
_TESTS = re.compile(rf"^# tests {_INT}$", re.MULTILINE)
_PASS = re.compile(rf"^# pass {_INT}$", re.MULTILINE)
_FAIL = re.compile(rf"^# fail {_INT}$", re.MULTILINE)


def test_serve_tap_suite_passes() -> None:
    """The authoritative TypeScript serve/tap suite must pass."""
    assert TSX.is_file(), f"missing TypeScript test runner: {TSX}"
    try:
        result = subprocess.run(
            [str(TSX), "--test", *SERVE_TEST_MODULES],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(f"tsx --test timed out after {TIMEOUT_SECONDS}s") from error

    output = result.stdout + result.stderr
    assert result.returncode == 0, f"tsx --test exited {result.returncode}:\n{output}"

    tests = _TESTS.search(output)
    passed = _PASS.search(output)
    failed = _FAIL.search(output)
    assert tests is not None and int(tests.group(1)) > 0, f"no tests executed:\n{output}"
    assert passed is not None and int(passed.group(1)) > 0, f"no positive pass count:\n{output}"
    assert failed is not None and int(failed.group(1)) == 0, f"failures reported:\n{output}"
