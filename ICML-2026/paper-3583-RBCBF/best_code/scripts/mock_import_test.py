#!/usr/bin/env python3
"""Runtime import smoke test using mocked heavy dependencies.

Replaces `torch` and `transformers` (and friends) with `MagicMock` in sys.modules
*before* importing `rbcbf`
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


def install_mocks() -> None:
    """Insert MagicMock entries for heavy deps that aren't installed locally."""
    heavy_packages = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "transformers",
        "transformers.utils",
        "transformers.utils.hub",
        "transformers.modeling_outputs",
        "sentencepiece",
        "accelerate",
    ]
    # First create the parents
    for name in heavy_packages:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
    # Re-attach known classes so `from transformers import AutoModelForCausalLM` etc. works
    sys.modules["torch"].Tensor = MagicMock
    sys.modules["torch"].device = MagicMock
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].Module = type("MockModule", (), {})
    sys.modules["torch.nn"].Linear = MagicMock
    sys.modules["torch"].inference_mode = MagicMock
    sys.modules["torch"].no_grad = MagicMock
    sys.modules["torch"].cuda = MagicMock()
    sys.modules["torch"].cuda.is_available = lambda: False
    sys.modules["torch"].bfloat16 = "bfloat16"
    sys.modules["torch"].float16 = "float16"
    sys.modules["torch"].float32 = "float32"
    sys.modules["transformers"].AutoModelForCausalLM = MagicMock
    sys.modules["transformers"].AutoTokenizer = MagicMock
    sys.modules["transformers"].AutoModel = MagicMock
    sys.modules["transformers"].PreTrainedModel = type("MockPreTrainedModel", (), {})
    sys.modules["transformers"].PreTrainedTokenizerBase = type("MockTok", (), {})


def main() -> int:
    install_mocks()
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))

    print(f"[mock_import_test] importing rbcbf from {here}")
    try:
        import rbcbf  # noqa: F401
    except Exception as e:
        print(f"  ✗ import rbcbf FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"  rbcbf imported ok; version = {getattr(rbcbf, '__version__', '?')}")

    # Verify every name in __all__ is actually attached
    missing = [n for n in rbcbf.__all__ if not hasattr(rbcbf, n)]
    if missing:
        print(f"  ✗ Names missing from rbcbf namespace: {missing}")
        return 1
    print(f"  All {len(rbcbf.__all__)} __all__ exports present: {rbcbf.__all__}")

    # Also try importing the runner module and the scripts package files
    for sub in (
        "rbcbf.runner",
        "rbcbf.controllers",
        "rbcbf.scorers",
        "rbcbf.judges",
        "rbcbf.models",
        "rbcbf.detectors",
    ):
        try:
            __import__(sub)
            print(f"  ✓ {sub}")
        except Exception as e:
            print(f"  ✗ {sub}: {type(e).__name__}: {e}")
            return 1

    # Test the entry scripts (just AST + module-level execution, not main())
    for sp in (here / "scripts" / "demo.py", here / "scripts" / "run_rbcbf.py"):
        src = sp.read_text()
        # The scripts do `from rbcbf import RBCBFRunner`; with our mocks that works.
        try:
            compile(src, str(sp), "exec")
            print(f"  ✓ {sp.relative_to(here)} compiles")
        except SyntaxError as e:
            print(f"  ✗ {sp}: {e}")
            return 1

    print("\n✓ Mock import smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
