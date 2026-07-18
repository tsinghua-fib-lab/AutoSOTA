"""
Analysis Tools for SafeFlow - Atomic, Absolute-Path Operations (Refactored)

This module provides code analysis tools for Python projects.

Design principles:
- Atomic operations: each tool call performs one well-defined action and returns a complete result.
- Stateless: no session state is required. Optional caches are stored on disk under the project root.
- Absolute-path only: all entrypoint paths are validated to be absolute.

Provided tools:
1) analysis_tools__module_dependency_graph
   - Static import dependency graph via AST
   - Relative import resolution (best-effort)
   - Cycle detection (SCC) + partial topo order
   - Internal/external module classification (heuristic)

2) analysis_tools__semantic_search
   - Semantic code search using sentence-transformers (optional dependency)
   - Chunk extraction (function/class/method/file chunks)
   - Embedding caching under <root>/.agent_cache
   - Designed for SWE-style codebase navigation

Limitations:
- Only detects static imports (not dynamic imports via __import__/importlib).
- Import resolution is best-effort and cannot fully model runtime sys.path modifications.
- Chunk extraction depends on AST parsing; syntax errors reduce coverage.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function


# -----------------------------
# Helper Functions (absolute-path safe)
# -----------------------------

_DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    ".tox",
    "build",
    "dist",
    ".eggs",
    ".idea",
    ".vscode",
]


def _validate_absolute_dir(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        raise ValueError(f"Path must be absolute, got: {path_str}")
    p = p.resolve()
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")
    return p


def _read_text_best_effort(p: Path, max_bytes: int = 2_000_000) -> str:
    """Read file content with best effort, handling encoding errors and size limits."""
    try:
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _is_excluded(rel_parts: Tuple[str, ...], exclude_dirs: List[str]) -> bool:
    excl = set(exclude_dirs)
    return any(part in excl for part in rel_parts)


def _iter_py_files(
    root: Path,
    file_pattern: str = "**/*.py",
    exclude_dirs: Optional[List[str]] = None,
) -> List[Path]:
    """
    List Python files under root, honoring `file_pattern` (supports '**') and excluding directories.

    NOTE: Uses Path.glob(file_pattern) for consistent '**' handling.
    """
    exclude_dirs = (exclude_dirs or []) + _DEFAULT_EXCLUDE_DIRS
    root = Path(root).resolve()

    out: List[Path] = []
    for p in root.glob(file_pattern):
        try:
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if _is_excluded(rel.parts, exclude_dirs):
                continue
            out.append(p)
        except Exception:
            continue

    return out


def _module_name_from_path(root: Path, file_path: Path) -> str:
    """
    Convert file path to a module-like name (best-effort).

    Example:
        pkg/sub/mod.py         -> pkg.sub.mod
        pkg/sub/__init__.py    -> pkg.sub
    """
    root = Path(root).resolve()
    file_path = Path(file_path).resolve()

    try:
        rel = file_path.relative_to(root)
    except Exception:
        rel = Path(file_path.name)

    if rel.name == "__init__.py":
        parts = list(rel.parts[:-1])
    else:
        parts = list(rel.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

    parts = [p for p in parts if p]
    return ".".join(parts)


def _package_name_for_module(module_name: str) -> str:
    """
    Get the package context for a module (best-effort).

    - For 'a.b.c' (module file), package context is 'a.b'
    - For top-level 'a', package context is '' (empty)
    """
    parts = module_name.split(".") if module_name else []
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def _resolve_import_from(current_module: str, level: int, module: Optional[str]) -> str:
    """
    Resolve relative import to absolute module name (best-effort).

    Key fix vs naive approach:
    - Relative imports are resolved against the *package* containing the current module,
      not the module itself.

    Args:
        current_module: current module name (e.g., 'a.b.c')
        level: number of leading dots in relative import
        module: the module part after dots (can be None)

    Returns:
        Resolved module name, or a special marker like '<invalid_relative_import:...>'
    """
    # Relative import base should be current package, not the module itself.
    current_pkg = _package_name_for_module(current_module)
    base_parts = current_pkg.split(".") if current_pkg else []

    # Validate level
    # level=1: current package; level=2: parent package, etc.
    if level < 0:
        return f"<invalid_relative_import:level={level}:module={current_module}>"
    if level > len(base_parts) + 1 and base_parts:
        return f"<invalid_relative_import:level={level}:module={current_module}>"
    if level > 1 and not base_parts:
        # cannot go above root
        return f"<invalid_relative_import:level={level}:module={current_module}>"

    # Go up (level-1) packages
    up = max(0, level - 1)
    base_parts = base_parts[:-up] if up else base_parts

    mod_parts = module.split(".") if module else []
    resolved = ".".join([p for p in (base_parts + mod_parts) if p])
    return resolved if resolved else "<empty_module>"


def _is_probably_test_file(p: Path) -> bool:
    """
    Best-effort test file detection using common conventions.

    More conservative than substring matching:
    - file name starts with test_ or ends with _test.py
    - any directory component equals 'tests' or 'test'
    """
    name = p.name.lower()
    parts_lower = [x.lower() for x in p.parts]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts_lower
        or "test" in parts_lower
    )


def _topological_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Topological sort with cycle detection.

    Returns:
        (partial_topo_order, cycles)
        - If cycles exist, topo order may be partial (Kahn's algorithm result).
        - cycles are SCCs (size>1) detected by Tarjan.
    """
    g = defaultdict(set)
    indeg = defaultdict(int)

    for n in nodes:
        indeg[n] = 0

    for a, b in edges:
        if b not in g[a]:
            g[a].add(b)
            indeg[b] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    topo: List[str] = []

    while q:
        n = q.popleft()
        topo.append(n)
        for nb in g[n]:
            indeg[nb] -= 1
            if indeg[nb] == 0:
                q.append(nb)

    if len(topo) == len(nodes):
        return topo, []

    # Tarjan SCC
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    cycles: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in g[v]:
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            scc = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                cycles.append(list(reversed(scc)))

    for n in nodes:
        if n not in idx:
            strongconnect(n)

    return topo, cycles


# -----------------------------
# AST Visitors
# -----------------------------

class _ImportCollector(ast.NodeVisitor):
    """Collect all import statements from an AST."""

    def __init__(self):
        self.imports: List[Dict[str, Any]] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                "kind": "import",
                "module": alias.name,
                "name": alias.asname or alias.name,
                "lineno": getattr(node, "lineno", None),
            })

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module  # can be None for "from . import x"
        for alias in node.names:
            self.imports.append({
                "kind": "from",
                "module": mod,
                "level": node.level or 0,
                "name": alias.name,
                "asname": alias.asname,
                "lineno": getattr(node, "lineno", None),
            })


# -----------------------------
# Chunk extraction for semantic search
# -----------------------------

def _extract_docstring(node: ast.AST) -> Optional[str]:
    try:
        return ast.get_docstring(node)
    except Exception:
        return None


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    module_name: str
    chunk_type: str  # function, class, method, file
    name: Optional[str]
    start_line: int
    end_line: int
    code: str
    docstring: Optional[str]
    signature: Optional[str]


def _extract_code_chunks(root: Path, file_path: Path, max_chunk_lines: int = 100) -> List[CodeChunk]:
    chunks: List[CodeChunk] = []
    text = _read_text_best_effort(file_path)
    if not text.strip():
        return chunks

    try:
        tree = ast.parse(text, filename=str(file_path))
    except Exception:
        return chunks

    module_name = _module_name_from_path(root, file_path)
    lines = text.split("\n")

    class ChunkExtractor(ast.NodeVisitor):
        def __init__(self):
            self.current_class: Optional[str] = None

        def _get_code_snippet(self, node: ast.AST) -> str:
            start = max(0, getattr(node, "lineno", 1) - 1)
            end = getattr(node, "end_lineno", None)
            if end is None:
                end = start + 1
            end = min(len(lines), max(start + 1, end))
            return "\n".join(lines[start:end])

        def _get_signature(self, node: ast.AST) -> str:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ""
            args = [arg.arg for arg in node.args.args]
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            return f"{prefix}{node.name}({', '.join(args)})"

        def visit_ClassDef(self, node: ast.ClassDef):
            chunk_id = f"{module_name}::{node.name}"
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=str(file_path),
                module_name=module_name,
                chunk_type="class",
                name=node.name,
                start_line=getattr(node, "lineno", 1),
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                code=self._get_code_snippet(node),
                docstring=_extract_docstring(node),
                signature=f"class {node.name}",
            ))
            prev = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = prev

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if self.current_class:
                chunk_id = f"{module_name}::{self.current_class}.{node.name}"
                chunk_type = "method"
            else:
                chunk_id = f"{module_name}::{node.name}"
                chunk_type = "function"

            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=str(file_path),
                module_name=module_name,
                chunk_type=chunk_type,
                name=node.name,
                start_line=getattr(node, "lineno", 1),
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                code=self._get_code_snippet(node),
                docstring=_extract_docstring(node),
                signature=self._get_signature(node),
            ))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            # Mirror FunctionDef logic
            if self.current_class:
                chunk_id = f"{module_name}::{self.current_class}.{node.name}"
                chunk_type = "method"
            else:
                chunk_id = f"{module_name}::{node.name}"
                chunk_type = "function"

            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=str(file_path),
                module_name=module_name,
                chunk_type=chunk_type,
                name=node.name,
                start_line=getattr(node, "lineno", 1),
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                code=self._get_code_snippet(node),
                docstring=_extract_docstring(node),
                signature=self._get_signature(node),
            ))
            self.generic_visit(node)

    ChunkExtractor().visit(tree)

    # If file small and no chunks, add whole file
    if not chunks and len(lines) < max_chunk_lines:
        chunks.append(CodeChunk(
            chunk_id=f"{module_name}::__file__",
            file_path=str(file_path),
            module_name=module_name,
            chunk_type="file",
            name=None,
            start_line=1,
            end_line=len(lines),
            code=text,
            docstring=None,
            signature=None,
        ))

    return chunks


# -----------------------------
# Cache helpers (semantic search)
# -----------------------------

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _project_fingerprint(root: Path, files: List[Path]) -> str:
    """
    Best-effort fingerprint based on file list + mtimes + sizes.
    Used to validate cached chunk lists/embeddings.
    """
    h = hashlib.sha256()
    h.update(str(root).encode())
    for p in sorted(files, key=lambda x: str(x)):
        try:
            st = p.stat()
            h.update(str(p).encode())
            h.update(str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))).encode())
            h.update(str(st.st_size).encode())
        except Exception:
            continue
    return h.hexdigest()


# -----------------------------
# Main Tool Class
# -----------------------------

class AnalysisTools(Tool):
    """
    Code analysis tools for Python projects (refactored).

    Atomic + absolute-path operations.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "analysis_tools",
        description: str = "Code analysis tools for dependency graphs and semantic/impact analysis",
    ):
        super().__init__(name=name, description=description, category=ToolCategory.ANALYSIS)
        self.item_id = item_id

    @tool_function(
        description=(
            "Build a Python module dependency graph based on static AST import analysis.\n"
            "- Atomic: scans files and returns graph in one call.\n"
            "- Stateless: no session state required.\n"
            "- Absolute-path only: root_path must be absolute.\n"
            "NOTE: Only captures static imports; dynamic imports are not detected."
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to project root", required=True),
            ToolParameter("file_pattern", "string", "Glob pattern for files (default: '**/*.py')", required=False, default="**/*.py"),
            ToolParameter("include_tests", "boolean", "Include test files in analysis", required=False, default=False),
            ToolParameter("collapse", "string", "Node granularity: 'module' (file-level) or 'package' (top-level package)", required=False, default="module"),
            ToolParameter("include_external", "boolean", "Include external/stdlib modules as nodes", required=False, default=False),
            ToolParameter("exclude_dirs", "array", "Additional directory names to exclude", required=False),
        ],
        returns="Dependency graph with nodes, edges, cycles, partial topo order, and warnings.",
        category=ToolCategory.ANALYSIS,
    )
    def analysis_tools__module_dependency_graph(
        self,
        root_path: str,
        file_pattern: str = "**/*.py",
        include_tests: bool = False,
        collapse: str = "module",
        include_external: bool = False,
        exclude_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        start_t = time.time()
        try:
            root = _validate_absolute_dir(root_path)
            files = _iter_py_files(root, file_pattern=file_pattern, exclude_dirs=exclude_dirs)

            # Filter tests
            test_filtered_count = 0
            if not include_tests:
                original_count = len(files)
                files = [p for p in files if not _is_probably_test_file(p)]
                test_filtered_count = original_count - len(files)

            # Module-to-file mapping
            module_to_file: Dict[str, str] = {}
            duplicates: List[Dict[str, Any]] = []
            for f in files:
                mod_name = _module_name_from_path(root, f)
                if mod_name in module_to_file:
                    duplicates.append({"module": mod_name, "files": [module_to_file[mod_name], str(f)]})
                else:
                    module_to_file[mod_name] = str(f)

            # Collapse strategy
            if collapse == "package":
                def _collapse(m: str) -> str:
                    parts = m.split(".")
                    return parts[0] if parts else m
            else:
                def _collapse(m: str) -> str:
                    return m

            # Build internal-prefix set for fast membership checks
            internal_prefixes: Set[str] = set()
            for m in module_to_file.keys():
                parts = m.split(".")
                for i in range(1, len(parts) + 1):
                    internal_prefixes.add(".".join(parts[:i]))

            def is_internal_module(resolved: str) -> bool:
                # Exact or prefix membership (e.g., package)
                if resolved in internal_prefixes:
                    return True
                # Also handle "a.b.c.something" where "a.b.c" exists
                # by checking parent prefixes
                parts = resolved.split(".")
                for i in range(len(parts), 0, -1):
                    if ".".join(parts[:i]) in internal_prefixes:
                        return True
                return False

            nodes_set: Set[str] = set()
            edges_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
            skipped_files: List[Dict[str, Any]] = []
            invalid_imports: List[Dict[str, Any]] = []

            for f in files:
                mod = _module_name_from_path(root, f)
                text = _read_text_best_effort(f)
                if not text.strip():
                    skipped_files.append({"file": str(f), "reason": "empty_or_unreadable"})
                    continue

                try:
                    tree = ast.parse(text, filename=str(f))
                except SyntaxError as e:
                    skipped_files.append({
                        "file": str(f),
                        "reason": "syntax_error",
                        "error": f"Line {e.lineno}: {str(e.msg)[:200]}",
                    })
                    continue
                except Exception as e:
                    skipped_files.append({"file": str(f), "reason": "parse_error", "error": str(e)[:200]})
                    continue

                ic = _ImportCollector()
                ic.visit(tree)

                for it in ic.imports:
                    if it["kind"] == "import":
                        resolved = it["module"]
                    else:
                        resolved = _resolve_import_from(mod, it.get("level", 0), it.get("module"))
                        if resolved.startswith("<invalid_relative_import") or resolved.startswith("<empty_module"):
                            invalid_imports.append({
                                "file": str(f),
                                "line": it.get("lineno"),
                                "module": mod,
                                "import_statement": it,
                                "resolved": resolved,
                            })
                            continue

                    # external filtering decision uses non-collapsed resolved name
                    internal = is_internal_module(resolved)
                    if not include_external and not internal:
                        continue

                    src = _collapse(mod)
                    dst = _collapse(resolved)

                    nodes_set.add(src)
                    nodes_set.add(dst)

                    key = (src, dst)
                    meta = {
                        "kind": it["kind"],
                        "lineno": it.get("lineno"),
                        "raw_module": it.get("module"),
                        "level": it.get("level", 0),
                        "file": str(f),
                        "is_self_loop": (src == dst),
                        "is_internal": internal,
                        "resolved_module": resolved,
                    }

                    # De-duplicate edges while preserving evidence list
                    if key not in edges_map:
                        edges_map[key] = {"from": src, "to": dst, "evidence": [meta]}
                    else:
                        edges_map[key]["evidence"].append(meta)

            nodes = sorted(nodes_set)
            edge_pairs = [(a, b) for (a, b) in edges_map.keys() if a != b]
            topo, cycles = _topological_sort(nodes, edge_pairs)

            edges_out = list(edges_map.values())
            self_loop_count = sum(1 for (a, b) in edges_map.keys() if a == b)

            internal_edge_count = 0
            for e in edges_out:
                # internal if any evidence says internal
                if any(ev.get("is_internal") for ev in e.get("evidence", [])):
                    internal_edge_count += 1
            external_edge_count = len(edges_out) - internal_edge_count

            elapsed = round(time.time() - start_t, 4)

            return {
                "success": True,
                "result": {
                    "root_path": str(root),
                    "file_count": len(files),
                    "collapse": collapse,
                    "nodes": nodes,
                    "edges": edges_out,
                    "cycles": cycles,
                    "topo_order": topo,
                    "module_to_file": module_to_file,
                    "statistics": {
                        "total_nodes": len(nodes),
                        "total_edges": len(edges_out),
                        "internal_edges": internal_edge_count,
                        "external_edges": external_edge_count,
                        "self_loops": self_loop_count,
                        "cycles_detected": len(cycles),
                        "test_files_filtered": test_filtered_count,
                        "elapsed_seconds": elapsed,
                    },
                    "warnings": {
                        "skipped_files": skipped_files[:50],
                        "skipped_files_total": len(skipped_files),
                        "duplicate_modules": duplicates,
                        "invalid_imports": invalid_imports[:50],
                        "invalid_imports_total": len(invalid_imports),
                    },
                },
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    @tool_function(
        description=(
            "Semantic code search using sentence-transformers (optional dependency).\n"
            "- Atomic: computes (or loads) embeddings, runs query, returns ranked results.\n"
            "- Stateless: cache stored under <root_path>/.agent_cache.\n"
            "- Absolute-path only: root_path must be absolute.\n"
            "\n"
            "Security note: trust_remote_code is disabled by default."
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to project root", required=True),
            ToolParameter("query", "string", "Natural language search query", required=True),
            ToolParameter("top_k", "integer", "Number of results to return (default: 10)", required=False, default=10),
            ToolParameter("chunk_type", "string", "Filter by type: all, function, class, method, file", required=False, default="all"),
            ToolParameter("exclude_tests", "boolean", "Exclude test files from results", required=False, default=True),
            ToolParameter("exclude_dirs", "array", "Additional directory names to exclude", required=False),
            ToolParameter("min_similarity", "number", "Minimum cosine similarity (0-1, default: 0.3)", required=False, default=0.3),
            ToolParameter("use_cache", "boolean", "Use cached embeddings if available", required=False, default=True),
            ToolParameter("rebuild_cache", "boolean", "Force rebuild embeddings cache", required=False, default=False),
        ],
        returns="Ranked search results with similarity scores, code snippets, and metadata.",
        category=ToolCategory.ANALYSIS,
    )
    def analysis_tools__semantic_search(
        self,
        root_path: str,
        query: str,
        top_k: int = 10,
        chunk_type: str = "all",
        exclude_tests: bool = True,
        exclude_dirs: Optional[List[str]] = None,
        min_similarity: float = 0.3,
        use_cache: bool = True,
        rebuild_cache: bool = False,
    ) -> Dict[str, Any]:
        start_t = time.time()
        try:
            # Optional dependency
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
            except ImportError:
                return {
                    "success": False,
                    "error": "sentence-transformers not installed. Run: pip install sentence-transformers",
                }

            root = _validate_absolute_dir(root_path)

            cache_dir = root / ".agent_cache"
            cache_dir.mkdir(exist_ok=True)

            model_name = "nomic-ai/CodeRankEmbed"
            embeddings_cache_path = cache_dir / "code_embeddings.npz"
            chunks_cache_path = cache_dir / "code_chunks.json"
            meta_cache_path = cache_dir / "code_cache_meta.json"

            if rebuild_cache:
                for p in (embeddings_cache_path, chunks_cache_path, meta_cache_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

            # Collect files for fingerprint
            files = _iter_py_files(root, file_pattern="**/*.py", exclude_dirs=exclude_dirs)
            if exclude_tests:
                files = [f for f in files if not _is_probably_test_file(f)]

            if not files:
                return {"success": False, "error": "No Python files found in project"}

            fingerprint = _project_fingerprint(root, files)

            chunks: List[CodeChunk] = []
            embeddings = None
            cache_loaded = False

            def _load_cache() -> bool:
                nonlocal chunks, embeddings, cache_loaded
                if not (use_cache and embeddings_cache_path.exists() and chunks_cache_path.exists() and meta_cache_path.exists()):
                    return False
                try:
                    meta = json.loads(meta_cache_path.read_text(encoding="utf-8"))
                    if meta.get("fingerprint") != fingerprint:
                        return False
                    if meta.get("model_name") != model_name:
                        return False

                    chunks_data = json.loads(chunks_cache_path.read_text(encoding="utf-8"))
                    chunks = [CodeChunk(**c) for c in chunks_data]

                    data = np.load(embeddings_cache_path)
                    embeddings = data["embeddings"]

                    if embeddings is None or len(chunks) != embeddings.shape[0]:
                        return False

                    cache_loaded = True
                    return True
                except Exception:
                    return False

            def _save_cache():
                try:
                    chunks_data = [
                        {
                            "chunk_id": c.chunk_id,
                            "file_path": c.file_path,
                            "module_name": c.module_name,
                            "chunk_type": c.chunk_type,
                            "name": c.name,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "code": c.code,
                            "docstring": c.docstring,
                            "signature": c.signature,
                        }
                        for c in chunks
                    ]
                    chunks_cache_path.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")
                    meta_cache_path.write_text(json.dumps({
                        "fingerprint": fingerprint,
                        "model_name": model_name,
                        "chunk_count": len(chunks),
                        "created_at": time.time(),
                    }, indent=2), encoding="utf-8")
                    if embeddings is not None:
                        np.savez_compressed(embeddings_cache_path, embeddings=embeddings)
                except Exception:
                    # Cache is best-effort
                    pass

            # Load cache or build chunks
            loaded = _load_cache()
            if not loaded:
                chunks = []
                # Extract chunks (best-effort)
                for f in files:
                    chunks.extend(_extract_code_chunks(root, f))

                if not chunks:
                    return {"success": False, "error": "No code chunks extracted (files may be empty or have syntax errors)."}

            # Load model (security: trust_remote_code disabled by default)
            try:
                model = SentenceTransformer(model_name, trust_remote_code=False)
            except TypeError:
                # Compatibility with older sentence-transformers without trust_remote_code kwarg
                model = SentenceTransformer(model_name)
            except Exception as e:
                return {"success": False, "error": f"Failed to load model {model_name}: {e}"}

            # Compute embeddings if needed
            if embeddings is None:
                code_texts: List[str] = []
                for chunk in chunks:
                    parts: List[str] = []
                    if chunk.signature:
                        parts.append(chunk.signature)
                    if chunk.docstring:
                        parts.append(f'"""{chunk.docstring}"""')
                    code_preview = chunk.code[:1000] if chunk.code else ""
                    parts.append(code_preview)
                    code_texts.append("\n".join(parts))

                embeddings = model.encode(
                    code_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=32,
                )
                _save_cache()

            # Prepare query (recommended prefix)
            if query.startswith("Represent this query"):
                query_with_prefix = query
            else:
                query_with_prefix = f"Represent this query for searching relevant code: {query}"

            query_embedding = model.encode([query_with_prefix], convert_to_numpy=True)[0]

            # Cosine similarity
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
            similarities = (embeddings_norm @ query_norm)

            results: List[Dict[str, Any]] = []
            for idx, sim in enumerate(similarities):
                sim_f = float(sim)
                if sim_f < min_similarity:
                    continue
                ch = chunks[idx]
                if chunk_type != "all" and ch.chunk_type != chunk_type:
                    continue

                code_preview = ch.code
                if len(code_preview) > 500:
                    code_preview = code_preview[:500] + "\n... [truncated]"

                results.append({
                    "chunk_id": ch.chunk_id,
                    "similarity": sim_f,
                    "file_path": ch.file_path,
                    "module_name": ch.module_name,
                    "type": ch.chunk_type,
                    "name": ch.name,
                    "signature": ch.signature,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "docstring": ch.docstring,
                    "code_preview": code_preview,
                })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:max(1, int(top_k))]

            elapsed = round(time.time() - start_t, 4)

            return {
                "success": True,
                "result": {
                    "query": query,
                    "query_with_prefix": query_with_prefix,
                    "total_chunks_searched": len(chunks),
                    "matching_chunks": len(results),
                    "top_k": top_k,
                    "min_similarity": min_similarity,
                    "chunk_type_filter": chunk_type,
                    "model_used": model_name,
                    "cache_used": cache_loaded,
                    "elapsed_seconds": elapsed,
                    "results": results,
                },
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}