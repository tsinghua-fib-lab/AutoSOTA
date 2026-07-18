#!/usr/bin/env python3

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "rbcbf"
assert PKG.is_dir(), f"Expected rbcbf/ under {ROOT}, but it does not exist."


# ----------------------------------------------------------------------
# Step 1: enumerate every .py and collect its imports + top-level defs.

def parse_module(p: Path) -> Tuple[List[ast.stmt], Set[str]]:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    defined: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    defined.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # also count re-imports as "defined" symbols
            if isinstance(node, ast.ImportFrom):
                for n in node.names:
                    defined.add(n.asname or n.name)
            else:
                for n in node.names:
                    defined.add((n.asname or n.name).split(".")[0])
    return tree.body, defined


def module_path_for(dotted: str) -> Path | None:
    """Given `rbcbf.scorers.qwen_margin_scorer`, return its file path or None."""
    parts = dotted.split(".")
    if parts[0] != "rbcbf":
        return None
    p = ROOT / Path(*parts).with_suffix(".py")
    if p.exists():
        return p
    pkg_init = ROOT / Path(*parts) / "__init__.py"
    if pkg_init.exists():
        return pkg_init
    return None


def relative_to_absolute(current: Path, level: int, module: str | None) -> str:
    """Resolve `from .x import ...` to absolute dotted module name.

    Python relative-import rules: `level=1` means the current *package*.
    For a non-__init__ file, the current package is its parent directory, so
    the file name must be stripped before applying additional level steps.
    """
    rel = current.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    else:
        # non-__init__ file: drop the file name to reach its package
        parts = parts[:-1]
    # walk up (level - 1) additional parents
    for _ in range(level - 1):
        if parts:
            parts.pop()
    if module:
        parts.append(module)
    return ".".join(parts)


# ----------------------------------------------------------------------
# Step 2: walk and collect

all_py: List[Path] = []
for r, _, fs in os.walk(PKG):
    for f in fs:
        if f.endswith(".py"):
            all_py.append(Path(r) / f)

print(f"Found {len(all_py)} Python files in {PKG}")
if len(all_py) == 0:
    print("ERROR: zero Python files found — package layout broken.")
    sys.exit(2)

module_defined: Dict[str, Set[str]] = {}  # dotted name -> set of defined names
issues: List[str] = []

for p in all_py:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        dotted = ".".join(parts[:-1])
    else:
        dotted = ".".join(parts)
    try:
        body, defined = parse_module(p)
    except SyntaxError as e:
        issues.append(f"SYNTAX in {p}: {e}")
        continue
    module_defined[dotted] = defined


# ----------------------------------------------------------------------
# Step 3: re-walk every import and verify

for p in all_py:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        current_dotted = ".".join(parts[:-1])
    else:
        current_dotted = ".".join(parts)

    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                abs_mod = relative_to_absolute(p, node.level, node.module)
            else:
                abs_mod = node.module or ""
            if not abs_mod.startswith("rbcbf"):
                continue  # skip external imports (torch, transformers, etc.)
            target_path = module_path_for(abs_mod)
            if target_path is None:
                issues.append(
                    f"MISSING MODULE: {p.relative_to(ROOT)}:{node.lineno}  "
                    f"`from {abs_mod} import ...` -> module file not found"
                )
                continue
            target_defined = module_defined.get(abs_mod, set())
            for n in node.names:
                if n.name == "*":
                    continue
                if n.name not in target_defined:
                    issues.append(
                        f"MISSING SYMBOL: {p.relative_to(ROOT)}:{node.lineno}  "
                        f"`from {abs_mod} import {n.name}` -> not defined in {abs_mod}"
                    )

        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("rbcbf"):
                    if module_path_for(n.name) is None:
                        issues.append(
                            f"MISSING MODULE: {p.relative_to(ROOT)}:{node.lineno}  "
                            f"`import {n.name}` -> not found"
                        )


# ----------------------------------------------------------------------
# Step 4: report

print("\n" + "=" * 70)
if not issues:
    print("✓ Static scan PASSED — no dangling imports detected in rbcbf/")
else:
    print(f"✗ Static scan FAILED — {len(issues)} issue(s):")
    for s in issues:
        print(f"  {s}")
print("=" * 70)
sys.exit(0 if not issues else 1)
