"""Smoke test: every public module imports / parses cleanly.

Catches the cheap class of breakage:
  - Syntax errors
  - Module-level NameErrors and typos
  - Missing imports (e.g. `from X import Y` where Y was renamed/removed)
  - Circular imports

Function-body bugs are NOT caught here — see `test_zsic_roundtrip.py` for that.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


# Real-import these — they're the core library and we want full import-time checks.
CORE_IMPORTS = [
    "quant_layerwise",
    "quant_layerwise.pipeline",
    "quant_layerwise.methods.zsic",
    "quant_layerwise.methods.gptq",
    "quant_layerwise.qronos_stats",
    "quant_layerwise.rate_control",
    "quant_layerwise.bucket",
    "quant_layerwise.names",
    "quant_layerwise.precompute",
    "quant_layerwise.hessian_runtime",
    "quant_layerwise.partial_model",
    "quant_layerwise.eval",
    "quant_layerwise.finetune",
    "quant_layerwise.data",
    "quant_layerwise.storage",
    "quant_layerwise.storage.artifacts",
]


@pytest.mark.parametrize("modname", CORE_IMPORTS)
def test_core_import(modname):
    """Every quant_layerwise.* module imports without error."""
    __import__(modname)


def _collect_py(*dirs):
    out = []
    for d in dirs:
        p = REPO / d
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            out.append(f)
    return out


# Ast-parse-only these — heavier scripts / adapters that pull in CUDA / fairscale / HF
# downloads at import time. Syntax-check is enough to catch the typical breakage class.
PARSE_DIRS = ["scripts", "examples", "parallel", "llama", "llama2", "lm_bench"]


@pytest.mark.parametrize(
    "path",
    _collect_py(*PARSE_DIRS),
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_parseable(path):
    """Every script / adapter file parses without SyntaxError."""
    src = path.read_text()
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path}: {e}")
