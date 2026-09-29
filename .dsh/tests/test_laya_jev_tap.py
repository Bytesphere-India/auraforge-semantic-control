"""Polyglot conformance runner for the Jev shadow tap (lane evidence entrypoint).

The authoritative service tests are TypeScript (`tsx --test`, wired into
`package.json` `"test"` per the brief).  This module lets a pytest-based CI run
the same impact-scoped serve/tap suite and asserts a positive pass count with
zero failures, so the result is visible to whichever harness collected it.  It
is impact-scoped: it runs the `test_serve_*.ts` modules (protocol, device
policy, shadow, HTTP, Jev tap, parity), not the full `yarn test` (which also
covers model/download tests), plus a focused check of the
`scripts/quantize-laya.py` parity/fallback gating.  It does not reimplement,
replace or weaken the TypeScript tests.  This file lives inside the project
worktree (`<repo>/.dsh/tests/`), not in a shared or external location.

Fail-closed on every infrastructure problem: a missing `tsx`, a timeout, a
non-zero exit, or a summary with no tests / no passes / any failure all fail the
test rather than reporting a false positive.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path
from types import ModuleType

# <repo>/.dsh/tests/test_laya_jev_tap.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
TSX = REPO_ROOT / "node_modules" / ".bin" / "tsx"

# The impact-scoped TypeScript modules that verify the `laya-serve` surface;
# `test_serve_jev_tap.ts` is the tap suite, `test_serve_qwen4b.ts` the Qwen4b
# third-shadow suite, `test_serve_device.ts` is the GPU policy/ceiling suite and
# `test_serve_parity.ts` the fail-closed parity suite.
SERVE_TEST_MODULES = (
    "test/test_serve_protocol.ts",
    "test/test_serve_device.ts",
    "test/test_serve_shadow.ts",
    "test/test_serve_http.ts",
    "test/test_serve_jev_tap.ts",
    "test/test_serve_qwen4b.ts",
    "test/test_serve_parity.ts",
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


def _load_quantize_module() -> ModuleType:
    """Load scripts/quantize-laya.py (the hyphen blocks a normal import)."""
    spec = importlib.util.spec_from_file_location("quantize_laya", REPO_ROOT / "scripts" / "quantize-laya.py")
    assert spec is not None and spec.loader is not None, "could not load scripts/quantize-laya.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantize_parity_gates_status_and_fallback(tmp_path: Path) -> None:
    """A variant is only "ok" when its required GPU parity passes, and fp16 is a fallback."""
    quantize = _load_quantize_module()

    # Fallback chain: fp16 is only attempted when fp8/parity did not succeed.
    assert quantize.fallback_order("auto", False) == ["base-fp8", "base-fp16"]
    assert quantize.fallback_order("auto", True) == []
    assert quantize.fallback_order("base-fp8", False) == ["base-fp8"]
    assert quantize.fallback_order("base-nvfp4", False) == []

    # Required providers never include cpu: a GPU variant must not pass parity on
    # a silent CPU fallback (that would stop the fp16 fallback).
    assert quantize.required_providers("base-fp8") == "tensorrt,cuda"
    assert quantize.required_providers("base-nvfp4") == "tensorrt,cuda"
    assert quantize.required_providers("base-fp16") == "cuda"
    assert "cpu" not in quantize.required_providers("base-fp8")
    assert "cpu" not in quantize.required_providers("base-fp16")

    # build_variant: quantization success + parity failure must be "failed", and
    # parity must be asked for the GPU stack (no cpu).
    src = tmp_path / "src"
    (src / "tokenizer").mkdir(parents=True)
    (src / "laya_config.json").write_text("{}", encoding="utf8")
    (src / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf8")
    args = argparse.Namespace(src=src, no_parity=False)

    parity_calls: list[str] = []

    def fake_parity(_baseline: Path, _variant: Path, providers: str) -> dict[str, object]:
        parity_calls.append(providers)
        return {"ok": False, "reason": "GPU_PROVIDER_UNAVAILABLE"}

    quantize.try_fp8 = lambda *a, **k: (True, "")
    quantize.run_parity = fake_parity
    failed = quantize.build_variant(args, "base-fp8", tmp_path / "out-failed", {})
    assert failed["status"] == "failed", failed
    assert "parity failed" in str(failed["reason"]), failed

    def fake_parity_ok(_baseline: Path, _variant: Path, providers: str) -> dict[str, object]:
        parity_calls.append(providers)
        return {"ok": True, "max_abs_diff_noul": 0}

    quantize.run_parity = fake_parity_ok
    passed = quantize.build_variant(args, "base-fp8", tmp_path / "out-ok", {})
    assert passed["status"] == "ok", passed
    assert parity_calls == ["tensorrt,cuda", "tensorrt,cuda"], parity_calls
