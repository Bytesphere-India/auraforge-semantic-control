"""Lane test bridge for the Jev shadow tap (Raja 2026-09-29).

`lane.sh` records test evidence by running changed Python test modules with
`pytest`.  This lane's service tests are TypeScript (`tsx --test`), so without a
bridge the coordinator's evidence file only ever says
``no targeted python tests (warn)``.  This module runs the real, impact-scoped
TypeScript serve/tap suite and asserts a positive pass count with zero failures,
so the lane's evidence records the actual result of the tests that matter.

It never modifies the repo, uses only a local fake upstream (no network), and is
safe to run repeatedly.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# <repo>/.dsh/tests/test_laya_jev_tap.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
TSX = REPO_ROOT / "node_modules" / ".bin" / "tsx"

# Every TypeScript module this lane's `laya-serve` surface is verified by.  The
# tap suite is the brief's subject; the others cover the shared server/shadow
# code paths it builds on.
SERVE_TEST_MODULES = (
    "test/test_serve_protocol.ts",
    "test/test_serve_shadow.ts",
    "test/test_serve_http.ts",
    "test/test_serve_jev_tap.ts",
)

_PASS = re.compile(r"^# pass (\d+)$", re.MULTILINE)
_FAIL = re.compile(r"^# fail (\d+)$", re.MULTILINE)
_TESTS = re.compile(r"^# tests (\d+)$", re.MULTILINE)


def test_serve_tap_suite_passes() -> None:
    """The TypeScript serve/tap suite must pass with a positive test count."""
    result = subprocess.run(
        [str(TSX), "--test", *SERVE_TEST_MODULES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"tsx --test exited {result.returncode}:\n{output}"

    tests = _TESTS.search(output)
    passed = _PASS.search(output)
    failed = _FAIL.search(output)
    assert tests is not None and int(tests.group(1)) > 0, f"no tests executed:\n{output}"
    assert passed is not None and int(passed.group(1)) > 0, f"no positive pass count:\n{output}"
    assert failed is not None and int(failed.group(1)) == 0, f"failures reported:\n{output}"
