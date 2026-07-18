import ast
import py_compile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "diffusion_strings"


def test_all_python_files_compile():
    """All source files should be syntactically valid Python."""
    for path in SOURCE_ROOT.rglob("*.py"):
        py_compile.compile(path, doraise=True)


def test_all_functions_have_docstrings():
    """Every function and method in diffusion_strings should have a docstring."""
    missing = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}:{node.name}")

    assert missing == []
