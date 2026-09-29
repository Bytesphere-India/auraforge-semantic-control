#!/usr/bin/env python3
"""Quantize the Laya ONNX bundle for the GPU lane (Raja 2026-09-29).

Produces ``~/models/laya/<variant>/`` bundles and never writes into
``/opt/auraforge/models/laya/base-fp32`` (or anywhere under it). Everything is
installed only into the dedicated venv ``~/.venv-laya-quant`` — never
system-wide and never into ``~/.dsh``.

Order of preference for the Blackwell RTX PRO 4000:

1. **NVFP4** — ModelOpt has *no* NVFP4 mode in ONNX post-training quantization
   (``quantize_mode="nvfp4"`` raises ``RuntimeError`` on 0.47.0; the pinned
   ``onnx`` opset normalizer only feeds the export path). NVFP4-as-ONNX is an
   *export format*: quantize the PyTorch checkpoint (``mtq`` nvfp4 recipe) and
   export with ``NVFP4QuantExporter`` / ``torch_quant_to_onnx.py --qformat=nvfp4``,
   which needs Blackwell + TensorRT >= 10.11 and has not been validated on
   ModernBERT. This script attempts the ONNX PTQ call anyway (so a future
   ModelOpt that gains the mode just works) and records the exact failure.
2. **FP8** — ``modelopt.onnx.quantization.quantize(quantize_mode="fp8")`` with
   CPU calibration; QDQ that TensorRT (or a CUDA EP that understands the QDQ)
   can consume.
3. **fp16** — last resort, converted with ``onnxconverter_common`` so the graph
   IO stays fp32 and ``src/laya.ts`` keeps seeing float32 outputs.

Calibration uses only the repo test fixtures (``scripts/fixtures.ts`` via
``scripts/calibration-fixtures.ts``), never shadow-log data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path("/opt/auraforge/models/laya/base-fp32")
DEFAULT_OUT_ROOT = Path.home() / "models" / "laya"
DEFAULT_VENV = Path.home() / ".venv-laya-quant"
# tensorrt (the bare metapackage) is CUDA-13/TRT-11 today, whose
# libnvinfer.so.11 the ORT 1.22/1.24 TensorRT EP (built against TRT 10.9) cannot
# load. tensorrt-cu12 10.9.x provides the libnvinfer.so.10 the EP asks for.
DEFAULT_TRT_PACKAGE = "tensorrt-cu12>=10.9,<11"
VARIANTS = ("base-nvfp4", "base-fp8", "base-fp16")

NVFP4_NOTE = (
    "ModelOpt ONNX PTQ has no nvfp4 mode (modelopt 0.47.0 raises "
    "'Invalid quantization mode choice: nvfp4'). NVFP4-as-ONNX requires the "
    "PyTorch -> ONNX export path (mtq nvfp4 recipe + NVFP4QuantExporter / "
    "torch_quant_to_onnx.py --qformat=nvfp4), Blackwell SM100+ and TensorRT "
    ">= 10.11; ModernBERT is not in the validated model list."
)


def log(message: str) -> None:
    sys.stderr.write(f"quantize-laya: {message}\n")


def validate_venv(venv: Path) -> None:
    """Only the dedicated venv may ever receive pip packages (brief containment)."""
    if venv.expanduser().resolve() != DEFAULT_VENV.expanduser().resolve():
        raise SystemExit(f"quantize-laya: refusing --venv {venv}; packages may only be installed into {DEFAULT_VENV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"fp32 bundle to read (default: {DEFAULT_SRC})")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help=f"output root (default: {DEFAULT_OUT_ROOT})")
    parser.add_argument("--variant", choices=("auto",) + VARIANTS, default="auto", help="force one variant instead of trying in order")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help=f"dedicated venv; must be {DEFAULT_VENV}")
    parser.add_argument("--skip-install", action="store_true", help="do not pip install into the venv")
    parser.add_argument("--trt-package", default=DEFAULT_TRT_PACKAGE, help=f"TensorRT wheel to install (default: {DEFAULT_TRT_PACKAGE})")
    parser.add_argument("--no-parity", action="store_true", help="skip the onnxruntime-node parity check")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what this host can do (GPU, TensorRT libs, venv writability) and exit without installing",
    )
    args = parser.parse_args()
    validate_venv(args.venv)
    return args


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _libnvinfer() -> str | None:
    from ctypes.util import find_library

    found = find_library("nvinfer") or find_library("nvinfer.so.10")
    return found


def _gpu_memory_mb() -> int | None:
    candidates = ["/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/usr/local/cuda/bin/nvidia-smi"]
    smi = next((path for path in candidates if Path(path).exists()), None)
    if smi is None:
        return None
    result = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, capture_output=True)  # noqa: S603
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def check_environment(args: argparse.Namespace) -> int:
    """Dry preflight: what is possible on this host, with no install and no GPU use."""
    target = args.out_root if args.out_root.exists() else args.out_root.parent
    report = {
        "python": sys.version.split()[0],
        "python_ok": (3, 10) <= sys.version_info[:2] < (3, 15),
        "venv": str(args.venv),
        "venv_exists": args.venv.exists(),
        "venv_parent_writable": bool(args.venv.parent.exists() and os.access(args.venv.parent, os.W_OK)),
        "out_root": str(args.out_root),
        "out_root_writable": bool(target.exists() and os.access(target, os.W_OK)),
        "source_present": (args.src / "laya.onnx").exists(),
        "modelopt_importable": _module_available("modelopt"),
        "tensorrt_importable": _module_available("tensorrt"),
        "torch_importable": _module_available("torch"),
        "numpy_importable": _module_available("numpy"),
        "libnvinfer": _libnvinfer(),
        "gpu_memory_used_mb": _gpu_memory_mb(),
        "nvfp4_note": NVFP4_NOTE,
        "planned_order": ["base-nvfp4 (expected to fail: no ONNX PTQ mode)", "base-fp8", "base-fp16"],
        "tensorrt_package": args.trt_package,
    }
    print(json.dumps(report, indent=2))
    return 0


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(command))
    return subprocess.run(command, check=True, text=True, **kwargs)  # type: ignore[arg-type]


def ensure_venv(args: argparse.Namespace) -> None:
    """Create the dedicated venv and install the quant stack into it only."""
    if not args.venv.exists():
        log(f"creating venv {args.venv}")
        run([sys.executable, "-m", "venv", str(args.venv)])
    if args.skip_install:
        return
    pip = args.venv / "bin" / "pip"
    run([str(pip), "install", "--upgrade", "pip"])
    packages = ["nvidia-modelopt[onnx]", args.trt_package, "onnx"]
    run([str(pip), "install", *packages])


def reexec_in_venv(args: argparse.Namespace) -> None:
    """Re-exec the script with the venv interpreter (the modules live there)."""
    if os.environ.get("_LAYA_QUANT_VENV") == "1":
        return
    python = args.venv / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"quantize-laya: venv interpreter missing: {python}")
    env = dict(os.environ, _LAYA_QUANT_VENV="1")
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def guard_paths(src: Path, out_root: Path) -> None:
    src = src.resolve()
    out_root = out_root.resolve()
    if not (src / "laya.onnx").exists():
        raise SystemExit(f"quantize-laya: no laya.onnx under {src}")
    protected = Path("/opt/auraforge/models/laya").resolve()
    if out_root == protected or protected in out_root.parents:
        raise SystemExit(f"quantize-laya: refusing to write under the protected model root {protected}")
    if out_root == src or src in out_root.parents:
        raise SystemExit("quantize-laya: output root must not be inside the source bundle")


def generate_calibration(model_dir: Path, out_json: Path) -> list[str]:
    tsx = REPO / "node_modules" / ".bin" / "tsx"
    if not tsx.exists():
        raise SystemExit(f"quantize-laya: tsx not found at {tsx}; run `yarn install`")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    run([str(tsx), "scripts/calibration-fixtures.ts", "--model-dir", str(model_dir), "--out", str(out_json)], cwd=str(REPO))
    raw = json.loads(out_json.read_text(encoding="utf8"))
    return list(raw.get("samples", []))


def calibration_dict(out_json: Path) -> dict[str, object]:
    import numpy as np  # installed with the quant stack

    raw = json.loads(out_json.read_text(encoding="utf8"))
    return {
        "input_ids": np.asarray(raw["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(raw["attention_mask"], dtype=np.int64),
        "marker_pos": np.asarray(raw["marker_pos"], dtype=np.int64),
        "marker_mask": np.asarray(raw["marker_mask"], dtype=bool),
        "qtype": np.asarray(raw["qtype"], dtype=np.int64),
    }


def copy_bundle_metadata(src: Path, variant_dir: Path) -> None:
    shutil.copy2(src / "laya_config.json", variant_dir / "laya_config.json")
    target = variant_dir / "tokenizer"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src / "tokenizer", target)


def produced_files(variant_dir: Path) -> list[str]:
    return sorted(p.name for p in variant_dir.iterdir() if p.is_file() and p.name != "quantization.json")


def write_manifest(variant_dir: Path, manifest: dict[str, object]) -> None:
    (variant_dir / "quantization.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


def try_nvfp4(src: Path, variant_dir: Path, calib: dict[str, object]) -> tuple[bool, str]:
    out = variant_dir / "laya.onnx"
    try:
        from modelopt.onnx.quantization import quantize  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"modelopt import failed: {type(error).__name__}: {error}"
    try:
        quantize(
            onnx_path=str(src / "laya.onnx"),
            quantize_mode="nvfp4",
            calibration_data=calib,
            calibration_method="max",
            calibration_eps=["cpu"],
            use_external_data_format=True,
            output_path=str(out),
        )
        return True, ""
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"{type(error).__name__}: {error}"


def try_fp8(src: Path, variant_dir: Path, calib: dict[str, object]) -> tuple[bool, str]:
    out = variant_dir / "laya.onnx"
    try:
        from modelopt.onnx.quantization import quantize  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"modelopt import failed: {type(error).__name__}: {error}"
    try:
        quantize(
            onnx_path=str(src / "laya.onnx"),
            quantize_mode="fp8",
            calibration_data=calib,
            calibration_method="max",
            calibration_eps=["cpu"],
            high_precision_dtype="fp16",
            use_external_data_format=True,
            output_path=str(out),
        )
        return True, ""
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"{type(error).__name__}: {error}"


def try_fp16(src: Path, variant_dir: Path) -> tuple[bool, str]:
    out = variant_dir / "laya.onnx"
    try:
        import onnx  # type: ignore[import-not-found]
        from onnxconverter_common import float16  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"onnx/onnxconverter_common import failed: {type(error).__name__}: {error}"
    try:
        model = onnx.load(str(src / "laya.onnx"))
        # keep_io_types keeps the (int64/bool) inputs and the float32 outputs, so
        # src/laya.ts still sees Float32Array logits/act_probs.
        converted = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
        onnx.save(
            converted,
            str(out),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="laya.onnx.data",
            size_threshold=1024,
        )
        return True, ""
    except Exception as error:  # pragma: no cover - depends on the venv
        return False, f"{type(error).__name__}: {error}"


# The provider each variant must actually run on. `"cpu"` is deliberately absent:
# a GPU variant validated through a silent CPU fallback would wrongly pass parity
# and stop the fp16 fallback. fp8/nvfp4 need the TensorRT EP; fp16 needs CUDA.
REQUIRED_PROVIDERS = {
    "base-nvfp4": "tensorrt,cuda",
    "base-fp8": "tensorrt,cuda",
    "base-fp16": "cuda",
}


def required_providers(mode: str) -> str:
    """The GPU provider stack a variant must load on (never `cpu`)."""
    return REQUIRED_PROVIDERS.get(mode, "cuda")


def run_parity(baseline: Path, variant_dir: Path, providers: str) -> dict[str, object]:
    """Answer parity via onnxruntime-node on the variant's required GPU stack.

    `providers` names GPU providers only; `check-parity.ts` strips any `cpu`
    entry and probes the GPU stack first, so an unavailable EP returns exit 2
    (failed) instead of passing parity on a CPU-backed session.
    """
    tsx = REPO / "node_modules" / ".bin" / "tsx"
    script = REPO / "scripts" / "check-parity.ts"
    command = [
        str(tsx),
        str(script),
        "--baseline",
        str(baseline),
        "--variant",
        str(variant_dir),
        "--providers",
        providers,
    ]
    result = subprocess.run(command, cwd=str(REPO), text=True, capture_output=True)  # noqa: S603
    if result.returncode in (0, 2, 3):
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {"ok": False, "providers": providers, "reason": result.stdout[-2000:] or result.stderr[-2000:]}
    parsed["attempted_providers"] = providers
    parsed["exit_code"] = result.returncode
    return parsed


def build_variant(args: argparse.Namespace, mode: str, variant_dir: Path, calib: dict[str, object]) -> dict[str, object]:
    src = args.src.resolve()
    variant_dir.mkdir(parents=True, exist_ok=True)
    copy_bundle_metadata(src, variant_dir)

    if mode == "base-nvfp4":
        ok, reason = try_nvfp4(src, variant_dir, calib)
    elif mode == "base-fp8":
        ok, reason = try_fp8(src, variant_dir, calib)
    else:
        ok, reason = try_fp16(src, variant_dir)

    manifest: dict[str, object] = {
        "schema": 1,
        "source": str(src),
        "variant": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if ok else "failed",
        "reason": reason or None,
        "files": produced_files(variant_dir) if ok else [],
        "nvfp4_note": NVFP4_NOTE,
    }
    if ok and not args.no_parity:
        providers = required_providers(mode)
        log(f"checking answer parity for {mode} on {providers}")
        parity = run_parity(src, variant_dir, providers)
        manifest["parity"] = parity
        if not parity.get("ok", False):
            # A variant that cannot load on its required GPU EP, or fails answer
            # parity, is not "ok" and must not stop the fallback chain.
            manifest["status"] = "failed"
            manifest["reason"] = f"parity failed: {parity.get('reason') or parity.get('attempted_providers') or 'unknown'}"
    write_manifest(variant_dir, manifest)
    return manifest


def fallback_order(variant: str, nvfp4_ok: bool) -> list[str]:
    """Which variants to attempt after NVFP4, in order.

    ``auto`` is a fallback chain: fp8 first, fp16 only if it does not clear both
    gates. An explicit ``--variant`` builds exactly that one (or, for nvfp4,
    nothing more).
    """
    if variant == "base-nvfp4":
        return []
    if variant != "auto":
        return [variant]
    if nvfp4_ok:
        return []
    return ["base-fp8", "base-fp16"]


def main() -> int:
    args = parse_args()
    if args.check:
        return check_environment(args)
    # Validate every write target before creating a venv or installing anything.
    guard_paths(args.src, args.out_root)
    ensure_venv(args)
    reexec_in_venv(args)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    calib_json = out_root / "calibration-fixtures.json"
    samples = generate_calibration(args.src.resolve(), calib_json)
    log(f"calibration fixtures: {len(samples)} samples")

    calib = calibration_dict(calib_json)
    results: dict[str, dict[str, object]] = {}

    # NVFP4 is attempted first (the preferred format) and its failure recorded,
    # but it cannot succeed through ONNX PTQ today.
    if args.variant in ("auto", "base-nvfp4"):
        results["base-nvfp4"] = build_variant(args, "base-nvfp4", out_root / "base-nvfp4", calib)

    nvfp4_ok = results.get("base-nvfp4", {}).get("status") == "ok"
    order = fallback_order(args.variant, nvfp4_ok)

    # Fallback chain: stop at the first variant whose quantization *and* parity
    # passed; only fall through to fp16 when fp8 did not clear both gates.
    stopped_after: str | None = None
    for mode in order:
        results[mode] = build_variant(args, mode, out_root / mode, calib)
        if results[mode].get("status") == "ok":
            stopped_after = mode
            break
    for mode in VARIANTS:
        if mode not in results:
            reason = f"not needed: {stopped_after} passed" if stopped_after else "not attempted"
            results[mode] = {"status": "skipped", "reason": reason}

    summary = {
        "source": str(args.src.resolve()),
        "out_root": str(out_root),
        "calibration_fixtures": samples,
        "variants": results,
        "preferred_order": list(VARIANTS),
        "fallback_stopped_after": stopped_after,
    }
    (out_root / "quantization-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))

    produced = [name for name, result in results.items() if result.get("status") == "ok"]
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
