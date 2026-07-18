"""
File System Tool for SafeFlow - Minimal Utility Operations (Refactored)

Provides utility operations that complement windowed_editor:
- Bulk file discovery (before editing)
- Quick path checks (before operations)
- Lightweight code structure / symbol search utilities

Design principles:
- Atomic operations: each tool call performs one well-defined action.
- Stateless tools: the tool holds no session state; any caching is purely internal optimization.
- Absolute-path only: all filesystem entry points validate absolute paths.

NOTE:
- This tool is read-only with respect to the repository under analysis.
- For editing, use windowed_editor.
"""

import ast
import logging
import os
import re
import tokenize
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class FileSystemTool(Tool):
    """
    Enhanced File System Utilities for SafeFlow (refactored).

    Provides:
    - File discovery
    - Path validation
    - Text-based code element search
    - AST-based symbol search

    No overlap with windowed_editor or base_tools.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "file_system",
        description: str = "File discovery, code search, and path validation utilities",
        read_only: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.FILE_SYSTEM
        )

        self.item_id = item_id
        self.read_only = read_only

        # Internal performance cache: (path, mtime_ns, size) -> extracted elements
        # This is NOT session state in the SWE sense; it is a best-effort optimization.
        self._code_cache: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}

    # -------------------------
    # Core helpers (absolute path + file iteration)
    # -------------------------

    def _validate_absolute_path(self, path: str) -> Path:
        """Validate that the path is absolute and return resolved Path object."""
        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {path}")
        return p.resolve()

    def _iter_files(self, root: Path, pattern: str) -> List[Path]:
        """
        Return a list of Paths matching `pattern` rooted at `root`.

        We intentionally use Path.glob(pattern) because it supports '**/*.py' patterns
        in a consistent manner, unlike mixing Path.rglob() with '**/' patterns.
        """
        # Path.glob supports '**' recursion.
        # For plain patterns like '*.py', it works as expected too.
        return [p for p in root.glob(pattern)]

    def _safe_stat_key(self, file_path: Path) -> Optional[Tuple[str, int, int]]:
        """Cache key based on path + mtime + size. Returns None if stat fails."""
        try:
            st = file_path.stat()
            return (str(file_path), getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)), st.st_size)
        except OSError:
            return None

    def _read_text_file(self, file_path: Path) -> str:
        """
        Read file content robustly.

        - For .py files, prefer tokenize.open (PEP263 encoding detection).
        - For others, read as utf-8 with replacement to preserve offsets/line structure.
        """
        if file_path.suffix == ".py":
            try:
                with tokenize.open(file_path) as f:
                    return f.read()
            except Exception:
                # Fall back to replacement decoding.
                return file_path.read_text(encoding="utf-8", errors="replace")
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize into identifier-like words, suitable for code and docstrings.

        This is more stable than split() because it handles punctuation, underscores, etc.
        """
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)

    def _get_code_snippet(self, content: str, start_line: int, num_lines: int = 5) -> str:
        """Get a snippet of code around a specific line (1-indexed line input)."""
        lines = content.splitlines()
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), start_idx + num_lines)

        snippet_lines = []
        for i in range(start_idx, end_idx):
            snippet_lines.append(f"{i+1:4d}: {lines[i]}")
        return "\n".join(snippet_lines)

    # -------------------------
    # Code element extraction
    # -------------------------

    def _extract_code_elements(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract functions, classes, and meaningful blocks from a file."""
        cache_key = self._safe_stat_key(file_path)
        if cache_key is not None and cache_key in self._code_cache:
            return self._code_cache[cache_key]

        try:
            content = self._read_text_file(file_path)
            if file_path.suffix == ".py":
                elements = self._extract_python_elements(content, file_path)
            else:
                elements = self._extract_text_elements(content, file_path)

            if cache_key is not None:
                self._code_cache[cache_key] = elements
            return elements

        except Exception as e:
            logger.warning(f"Failed to extract elements from {file_path}: {e}")
            return []

    def _extract_python_elements(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract Python functions, classes, and file-level docstring using AST.

        Improvement vs old version:
        - Uses a visitor to provide qualname and method detection.
        - Avoids counting nested functions as top-level in ambiguous ways.
        """
        elements: List[Dict[str, Any]] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._extract_text_elements(content, file_path)

        # File-level docstring
        file_docstring = ast.get_docstring(tree)
        if file_docstring:
            elements.append({
                "type": "file_docstring",
                "name": file_path.name,
                "line": 1,
                "file_path": str(file_path),
                "signature": f"File: {file_path.name}",
                "docstring": file_docstring,
                "content_preview": file_docstring[:200] + "..." if len(file_docstring) > 200 else file_docstring
            })

        class Visitor(ast.NodeVisitor):
            def __init__(self, outer: "FileSystemTool"):
                self.outer = outer
                self.class_stack: List[str] = []
                self.func_stack: List[str] = []

            def visit_ClassDef(self, node: ast.ClassDef):
                self.class_stack.append(node.name)
                qualname = ".".join(self.class_stack)

                elements.append({
                    "type": "class",
                    "name": node.name,
                    "qualname": qualname,
                    "line": getattr(node, "lineno", 1),
                    "file_path": str(file_path),
                    "signature": f"class {node.name}",
                    "docstring": ast.get_docstring(node) or "",
                    "content_preview": self.outer._get_code_snippet(content, getattr(node, "lineno", 1), 5),
                })
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.func_stack.append(node.name)
                in_class = bool(self.class_stack)
                qualname = ".".join(self.class_stack + [node.name]) if in_class else ".".join(self.func_stack)

                elements.append({
                    "type": "method" if in_class else "function",
                    "name": node.name,
                    "qualname": qualname,
                    "line": getattr(node, "lineno", 1),
                    "file_path": str(file_path),
                    "signature": f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})",
                    "docstring": ast.get_docstring(node) or "",
                    "content_preview": self.outer._get_code_snippet(content, getattr(node, "lineno", 1), 5),
                })

                self.generic_visit(node)
                self.func_stack.pop()

            # Python 3.8+: also catch async defs
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                # Represent async functions similarly
                self.func_stack.append(node.name)
                in_class = bool(self.class_stack)
                qualname = ".".join(self.class_stack + [node.name]) if in_class else ".".join(self.func_stack)

                elements.append({
                    "type": "method" if in_class else "function",
                    "name": node.name,
                    "qualname": qualname,
                    "line": getattr(node, "lineno", 1),
                    "file_path": str(file_path),
                    "signature": f"async def {node.name}({', '.join(arg.arg for arg in node.args.args)})",
                    "docstring": ast.get_docstring(node) or "",
                    "content_preview": self.outer._get_code_snippet(content, getattr(node, "lineno", 1), 5),
                })

                self.generic_visit(node)
                self.func_stack.pop()

        Visitor(self).visit(tree)
        return elements

    def _extract_text_elements(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract meaningful text chunks from non-Python files."""
        elements: List[Dict[str, Any]] = []
        lines = content.splitlines()

        current_block: List[str] = []
        block_start = 1

        def flush_block():
            nonlocal current_block, block_start
            if current_block and len(current_block) >= 2:
                block_text = "\n".join(current_block)
                elements.append({
                    "type": "comment_block",
                    "name": f"Comment block (line {block_start})",
                    "line": block_start,
                    "file_path": str(file_path),
                    "signature": f"Comment in {file_path.name}",
                    "docstring": block_text,
                    "content_preview": block_text[:200] + "..." if len(block_text) > 200 else block_text
                })
            current_block = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
                or "TODO" in stripped
                or "FIXME" in stripped
                or "NOTE" in stripped
            ):
                if not current_block:
                    block_start = i
                current_block.append(line)
            else:
                flush_block()

        flush_block()
        return elements

    # -------------------------
    # Tools
    # -------------------------

    @tool_function(
        description=(
            "Search for files matching a glob pattern.\n"
            "- Atomic: scans filesystem under root_path once.\n"
            "- Stateless: no persistent state.\n"
            "- Absolute-path only: root_path must be absolute.\n"
            "Pattern supports '**' via Path.glob(), e.g. '**/*.py' or '**/quantity.py'."
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to search from", required=True),
            ToolParameter("pattern", "string", "Glob pattern (e.g., '*.py', '**/*.txt', '**/quantity.py')", required=True),
            ToolParameter("max_results", "integer", "Maximum results", required=False, default=100),
        ],
        returns="List of matching files for bulk discovery",
        category=ToolCategory.FILE_SYSTEM,
    )
    def file_system__find_files(
        self,
        root_path: str,
        pattern: str,
        max_results: int = 100
    ) -> Dict[str, Any]:
        """Search for files matching glob pattern. Use for bulk file discovery."""
        try:
            root = self._validate_absolute_path(root_path)

            if not root.exists():
                return {"success": False, "error": f"Root directory not found: {root}"}
            if not root.is_dir():
                return {"success": False, "error": f"Root path is not a directory: {root}"}

            matches: List[Dict[str, Any]] = []
            count = 0

            for match in self._iter_files(root, pattern):
                if count >= max_results:
                    break
                try:
                    stat = match.stat()
                    match_info = {
                        "name": match.name,
                        "absolute_path": str(match.resolve()),
                        "relative_path": str(match.relative_to(root)),
                        "type": "directory" if match.is_dir() else "file",
                        "size": stat.st_size if match.is_file() else None,
                    }
                    if match.is_file():
                        match_info["extension"] = match.suffix
                    matches.append(match_info)
                    count += 1
                except (OSError, PermissionError):
                    continue

            return {
                "success": True,
                "result": {
                    "matches": matches,
                    "pattern": pattern,
                    "root_path": str(root),
                }
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Search failed: {e}"}

    @tool_function(
        description=(
            "Quick check if a path exists.\n"
            "- Atomic: checks filesystem metadata once.\n"
            "- Stateless.\n"
            "- Absolute-path only."
        ),
        parameters=[
            ToolParameter("path", "string", "Absolute path to check", required=True)
        ],
        returns="Path existence and basic info",
        category=ToolCategory.FILE_SYSTEM,
    )
    def file_system__path_info(self, path: str) -> Dict[str, Any]:
        """Quick path existence check. Use before opening files in windowed_editor."""
        try:
            path_obj = self._validate_absolute_path(path)

            exists = path_obj.exists()
            result: Dict[str, Any] = {"exists": exists, "path": str(path_obj)}

            if exists:
                try:
                    stat = path_obj.stat()
                    result.update({
                        "type": "directory" if path_obj.is_dir() else "file",
                        "size": stat.st_size if path_obj.is_file() else None,
                    })
                    if path_obj.is_file():
                        result["extension"] = path_obj.suffix
                except (OSError, PermissionError) as e:
                    result["metadata_error"] = str(e)

            return {"success": True, "result": result}

        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Path check failed: {e}"}

    @tool_function(
        description=(
            "Text-based search for code elements (functions, classes, methods).\n"
            "- Atomic: scans matching files and returns best matches.\n"
            "- Stateless (internal cache may speed up repeated calls).\n"
            "- Absolute-path only.\n"
            "Returns heuristic match_score as 'similarity_score' for backward compatibility."
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to search from", required=True),
            ToolParameter("query", "string", "Search query with keywords", required=True),
            ToolParameter("file_pattern", "string", "Glob pattern for files (default: '**/*.py')", required=False, default="**/*.py"),
            ToolParameter("max_results", "integer", "Maximum number of results", required=False, default=10),
            ToolParameter("include_preview", "boolean", "Include code preview in results", required=False, default=True),
        ],
        returns="Text-matched code elements with relevance scores",
        category=ToolCategory.FILE_SYSTEM,
    )
    def file_system__keyword_search(
        self,
        root_path: str,
        query: str,
        file_pattern: str = "**/*.py",
        max_results: int = 10,
        include_preview: bool = True
    ) -> Dict[str, Any]:
        """Search for code elements using text-based matching."""
        try:
            root = self._validate_absolute_path(root_path)
            if not root.exists():
                return {"success": False, "error": f"Root directory not found: {root}"}
            return self._fallback_text_search(root, query, file_pattern, max_results, include_preview)
        except Exception as e:
            return {"success": False, "error": f"Keyword search failed: {e}"}

    def _fallback_text_search(
        self,
        root: Path,
        query: str,
        file_pattern: str,
        max_results: int,
        include_preview: bool
    ) -> Dict[str, Any]:
        """Fallback text-based search when semantic search is not available."""
        try:
            query_lower = query.lower()
            query_words = set(w.lower() for w in self._tokenize_words(query))

            all_elements: List[Dict[str, Any]] = []
            files_searched = 0

            for file_path in self._iter_files(root, file_pattern):
                if not file_path.is_file():
                    continue
                try:
                    if file_path.stat().st_size >= 1024 * 1024:
                        continue
                except OSError:
                    continue

                files_searched += 1
                elements = self._extract_code_elements(file_path)
                all_elements.extend(elements)

            scored_results: List[Tuple[Dict[str, Any], float]] = []

            for element in all_elements:
                search_text = f"{element.get('name','')} {element.get('signature','')} {element.get('docstring','')}".lower()
                element_words = set(w.lower() for w in self._tokenize_words(search_text))

                score = 0.0
                if query_lower and query_lower in search_text:
                    score += 0.8

                if query_words:
                    word_matches = len(query_words & element_words)
                    score += 0.6 * (word_matches / max(1, len(query_words)))

                if query_lower and query_lower in str(element.get("name", "")).lower():
                    score += 0.4

                # Cap to 1.0 to match "similarity" semantics better (backward-compatible field name).
                score = min(1.0, score)

                if score > 0.1:
                    scored_results.append((element, score))

            scored_results.sort(key=lambda x: x[1], reverse=True)

            matches: List[Dict[str, Any]] = []
            for element, score in scored_results[:max_results]:
                out = {
                    "type": element.get("type", ""),
                    "name": element.get("name", ""),
                    "file_path": element.get("file_path", ""),
                    "line": element.get("line", 1),
                    "signature": element.get("signature", ""),
                    # Keep field name for compatibility
                    "similarity_score": float(score),
                    "docstring": (element.get("docstring", "")[:200] + "...")
                    if len(element.get("docstring", "")) > 200 else element.get("docstring", ""),
                }
                if include_preview:
                    out["code_preview"] = element.get("content_preview", "")
                matches.append(out)

            return {
                "success": True,
                "result": {
                    "matches": matches,
                    "query": query,
                    # Keep old field, but clarify meaning and add files_searched
                    "total_searched": len(all_elements),  # elements searched (legacy)
                    "files_searched": files_searched,
                    "elements_searched": len(all_elements),
                    "search_method": "text_matching",
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Text search failed: {e}"}

    @tool_function(
        description=(
            "Analyze code structure and extract functions/classes/methods from files.\n"
            "- Atomic: scans matching files and returns extracted structure.\n"
            "- Stateless (internal cache may speed up repeated calls).\n"
            "- Absolute-path only."
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to analyze", required=True),
            ToolParameter("file_pattern", "string", "Glob pattern for files (default: '**/*.py')", required=False, default="**/*.py"),
            ToolParameter("group_by_file", "boolean", "Group results by file", required=False, default=True),
        ],
        returns="Structured analysis of code elements",
        category=ToolCategory.FILE_SYSTEM,
    )
    def file_system__analyze_code_structure(
        self,
        root_path: str,
        file_pattern: str = "**/*.py",
        group_by_file: bool = True
    ) -> Dict[str, Any]:
        """Analyze and extract code structure from files."""
        try:
            root = self._validate_absolute_path(root_path)
            if not root.exists():
                return {"success": False, "error": f"Root directory not found: {root}"}

            analysis_by_file: Dict[str, Any] = {}
            flat_elements: List[Dict[str, Any]] = []
            total_elements = 0
            files_searched = 0

            for file_path in self._iter_files(root, file_pattern):
                if not file_path.is_file():
                    continue
                try:
                    if file_path.stat().st_size >= 1024 * 1024:
                        continue
                except OSError:
                    continue

                files_searched += 1
                elements = self._extract_code_elements(file_path)
                if not elements:
                    continue

                rel = str(file_path.relative_to(root))
                total_elements += len(elements)

                if group_by_file:
                    analysis_by_file[rel] = {
                        "file_path": str(file_path),
                        "elements": elements,
                        "element_count": len(elements),
                    }
                else:
                    for el in elements:
                        el["relative_path"] = rel
                    flat_elements.extend(elements)

            if group_by_file:
                analysis_result: Any = analysis_by_file
                total_files = len(analysis_by_file)
            else:
                by_type: Dict[str, List[Dict[str, Any]]] = {}
                for el in flat_elements:
                    by_type.setdefault(el.get("type", "unknown"), []).append(el)

                analysis_result = {"by_type": by_type, "all_elements": flat_elements}
                total_files = len(set(el.get("relative_path") for el in flat_elements if el.get("relative_path")))

            return {
                "success": True,
                "result": {
                    "analysis": analysis_result,
                    "total_files": total_files,
                    "files_searched": files_searched,
                    "total_elements": total_elements,
                    "root_path": str(root),
                    "pattern": file_pattern,
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Code structure analysis failed: {e}"}

    @tool_function(
        description=(
            "Search for symbol definitions/references/usages using AST analysis.\n"
            "- Atomic: scans matching files and returns symbol occurrences.\n"
            "- Stateless (internal cache may speed up repeated calls).\n"
            "- Absolute-path only.\n"
            "\n"
            "symbol_type controls what entity you look for: function/class/variable/import/attribute/any.\n"
            "search_type controls which occurrences to return: definition/reference/usage/all.\n"
        ),
        parameters=[
            ToolParameter("root_path", "string", "Absolute path to search from", required=True),
            ToolParameter("symbol_name", "string", "Symbol name to search for (exact match)", required=True),
            ToolParameter("symbol_type", "string", "Type of symbol: 'function', 'class', 'variable', 'import', 'attribute', or 'any'", required=False, default="any"),
            ToolParameter("search_type", "string", "Search type: 'definition', 'reference', 'usage', or 'all'", required=False, default="all"),
            ToolParameter("file_pattern", "string", "Glob pattern for files to search", required=False, default="**/*.py"),
            ToolParameter("include_builtin", "boolean", "Include built-in and imported symbols", required=False, default=False),
            ToolParameter("max_results", "integer", "Maximum number of results", required=False, default=50),
        ],
        returns="Symbol definitions, references, and usage locations with context",
        category=ToolCategory.FILE_SYSTEM,
    )
    def file_system__symbol_search(
        self,
        root_path: str,
        symbol_name: str,
        symbol_type: str = "any",
        search_type: str = "all",
        file_pattern: str = "**/*.py",
        include_builtin: bool = False,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """Search for symbol definitions and references using AST analysis."""
        try:
            root = self._validate_absolute_path(root_path)
            if not root.exists():
                return {"success": False, "error": f"Root directory not found: {root}"}

            symbol_results = {"definitions": [], "references": [], "usages": [], "imports": []}
            total_files_searched = 0

            # We prioritize definitions first to avoid max_results being filled by usages
            want_def = search_type in ("definition", "all")
            want_ref = search_type in ("reference", "all")
            want_use = search_type in ("usage", "all")

            # Pass 1: collect definitions/imports (higher signal)
            for file_path in self._iter_files(root, file_pattern):
                if not file_path.is_file():
                    continue
                try:
                    if file_path.stat().st_size >= 2 * 1024 * 1024:
                        continue
                except OSError:
                    continue

                total_files_searched += 1
                matches = self._analyze_file_for_symbols(
                    file_path=file_path,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    search_type="definition" if want_def else "all",
                    include_builtin=include_builtin,
                )

                if want_def:
                    symbol_results["definitions"].extend(matches.get("definitions", []))
                # imports are always useful to include when asked for import/any
                symbol_results["imports"].extend(matches.get("imports", []))

                if sum(len(v) for v in symbol_results.values()) >= max_results:
                    break

            # Pass 2: collect references/usages if requested and still under max_results
            if (want_ref or want_use) and sum(len(v) for v in symbol_results.values()) < max_results:
                for file_path in self._iter_files(root, file_pattern):
                    if not file_path.is_file():
                        continue
                    try:
                        if file_path.stat().st_size >= 2 * 1024 * 1024:
                            continue
                    except OSError:
                        continue

                    matches = self._analyze_file_for_symbols(
                        file_path=file_path,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        search_type="all",
                        include_builtin=include_builtin,
                    )

                    if want_ref:
                        symbol_results["references"].extend(matches.get("references", []))
                    if want_use:
                        symbol_results["usages"].extend(matches.get("usages", []))

                    if sum(len(v) for v in symbol_results.values()) >= max_results:
                        break

            # Now construct final limited results prioritizing: definitions, imports, references, usages
            prioritized: List[Tuple[Dict[str, Any], str, float]] = []
            prioritized.extend([(m, "definition", 3.0) for m in symbol_results["definitions"]])
            prioritized.extend([(m, "import", 2.5) for m in symbol_results["imports"]])
            prioritized.extend([(m, "reference", 2.0) for m in symbol_results["references"]])
            prioritized.extend([(m, "usage", 1.0) for m in symbol_results["usages"]])

            prioritized.sort(key=lambda x: (x[2], x[0].get("confidence", 0)), reverse=True)
            limited = prioritized[:max_results]

            final_results = {"definitions": [], "references": [], "usages": [], "imports": []}
            for m, kind, _prio in limited:
                if kind == "definition":
                    final_results["definitions"].append(m)
                elif kind == "import":
                    final_results["imports"].append(m)
                elif kind == "reference":
                    final_results["references"].append(m)
                elif kind == "usage":
                    final_results["usages"].append(m)

            return {
                "success": True,
                "result": {
                    "symbol_name": symbol_name,
                    "symbol_type": symbol_type,
                    "search_type": search_type,
                    "matches": final_results,
                    "total_matches": len(limited),
                    "files_searched": total_files_searched,
                    "root_path": str(root),
                    "search_method": "ast_analysis"
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Symbol search failed: {e}"}

    # -------------------------
    # Symbol analysis helpers
    # -------------------------

    def _analyze_file_for_symbols(
        self,
        file_path: Path,
        symbol_name: str,
        symbol_type: str,
        search_type: str,
        include_builtin: bool
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze a single file for symbol occurrences using AST."""
        results = {"definitions": [], "references": [], "usages": [], "imports": []}
        try:
            content = self._read_text_file(file_path)

            if file_path.suffix == ".py":
                try:
                    tree = ast.parse(content)
                    return self._ast_symbol_search(
                        tree=tree,
                        content=content,
                        file_path=file_path,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        search_type=search_type,
                        include_builtin=include_builtin
                    )
                except SyntaxError:
                    return self._text_symbol_search(content, file_path, symbol_name)
            else:
                return self._text_symbol_search(content, file_path, symbol_name)

        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return results

    def _ast_symbol_search(
        self,
        tree: ast.AST,
        content: str,
        file_path: Path,
        symbol_name: str,
        symbol_type: str,
        search_type: str,
        include_builtin: bool
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform AST-based symbol search."""
        results = {"definitions": [], "references": [], "usages": [], "imports": []}

        want_def = search_type in ("definition", "all")
        want_ref = search_type in ("reference", "all")
        want_use = search_type in ("usage", "all")

        # Helper to add with common context
        def _ctx(line: int, n: int = 2) -> str:
            return self._get_code_snippet(content, line, n)

        for node in ast.walk(tree):
            # Definitions
            if want_def:
                if symbol_type in ("function", "any") and isinstance(node, ast.FunctionDef) and node.name == symbol_name:
                    results["definitions"].append({
                        "type": "function_definition",
                        "name": node.name,
                        "file_path": str(file_path),
                        "line": node.lineno,
                        "column": getattr(node, "col_offset", 0),
                        "context": _ctx(node.lineno, 3),
                        "signature": f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})",
                        "docstring": ast.get_docstring(node) or "",
                        "confidence": 1.0
                    })

                if symbol_type in ("class", "any") and isinstance(node, ast.ClassDef) and node.name == symbol_name:
                    results["definitions"].append({
                        "type": "class_definition",
                        "name": node.name,
                        "file_path": str(file_path),
                        "line": node.lineno,
                        "column": getattr(node, "col_offset", 0),
                        "context": _ctx(node.lineno, 3),
                        "signature": f"class {node.name}",
                        "docstring": ast.get_docstring(node) or "",
                        "confidence": 1.0
                    })

                if symbol_type in ("variable", "any") and isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == symbol_name:
                            results["definitions"].append({
                                "type": "variable_assignment",
                                "name": target.id,
                                "file_path": str(file_path),
                                "line": node.lineno,
                                "column": getattr(target, "col_offset", 0),
                                "context": _ctx(node.lineno, 2),
                                "signature": f"{target.id} = ...",
                                "confidence": 0.9
                            })

            # Imports (these are separate from "definition/reference/usage", but we keep them in imports)
            if symbol_type in ("import", "any") and isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        if imported_name == symbol_name or alias.name == symbol_name:
                            results["imports"].append({
                                "type": "import",
                                "name": imported_name,
                                "original_name": alias.name,
                                "file_path": str(file_path),
                                "line": node.lineno,
                                "column": getattr(node, "col_offset", 0),
                                "context": _ctx(node.lineno, 1),
                                "signature": f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                                "confidence": 1.0
                            })
                else:
                    module = node.module or ""
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        if imported_name == symbol_name or alias.name == symbol_name:
                            results["imports"].append({
                                "type": "from_import",
                                "name": imported_name,
                                "original_name": alias.name,
                                "module": module,
                                "file_path": str(file_path),
                                "line": node.lineno,
                                "column": getattr(node, "col_offset", 0),
                                "context": _ctx(node.lineno, 1),
                                "signature": f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                                "confidence": 1.0
                            })

            # Attribute references
            if want_ref and symbol_type in ("attribute", "any") and isinstance(node, ast.Attribute) and node.attr == symbol_name:
                results["references"].append({
                    "type": "attribute_access",
                    "name": node.attr,
                    "file_path": str(file_path),
                    "line": node.lineno,
                    "column": getattr(node, "col_offset", 0),
                    "context": _ctx(node.lineno, 1),
                    "confidence": 0.7
                })

            # Variable usages/references (Name nodes)
            # For external symbol_type "variable", Name occurrences are usages/references.
            if (want_ref or want_use) and symbol_type in ("variable", "any") and isinstance(node, ast.Name) and node.id == symbol_name:
                if isinstance(node.ctx, ast.Load) and want_use:
                    results["usages"].append({
                        "type": "usage",
                        "name": node.id,
                        "file_path": str(file_path),
                        "line": node.lineno,
                        "column": getattr(node, "col_offset", 0),
                        "context": _ctx(node.lineno, 1),
                        "confidence": 0.8
                    })
                elif isinstance(node.ctx, ast.Store) and want_ref:
                    results["references"].append({
                        "type": "assignment_target",
                        "name": node.id,
                        "file_path": str(file_path),
                        "line": node.lineno,
                        "column": getattr(node, "col_offset", 0),
                        "context": _ctx(node.lineno, 1),
                        "confidence": 0.75
                    })

        return results

    def _text_symbol_search(self, content: str, file_path: Path, symbol_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback text-based symbol search for non-Python files or parsing errors."""
        results = {"definitions": [], "references": [], "usages": [], "imports": []}

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            if symbol_name not in line:
                continue

            stripped = line.strip()

            # Simple heuristics
            if stripped.startswith(f"def {symbol_name}") or f"function {symbol_name}" in line:
                results["definitions"].append({
                    "type": "function_definition",
                    "name": symbol_name,
                    "file_path": str(file_path),
                    "line": line_num,
                    "context": self._get_code_snippet(content, line_num, 2),
                    "signature": stripped,
                    "confidence": 0.6
                })
            elif stripped.startswith(f"class {symbol_name}") or f"class {symbol_name}" in line:
                results["definitions"].append({
                    "type": "class_definition",
                    "name": symbol_name,
                    "file_path": str(file_path),
                    "line": line_num,
                    "context": self._get_code_snippet(content, line_num, 2),
                    "signature": stripped,
                    "confidence": 0.6
                })
            elif "import" in line:
                results["imports"].append({
                    "type": "import",
                    "name": symbol_name,
                    "file_path": str(file_path),
                    "line": line_num,
                    "context": stripped,
                    "signature": stripped,
                    "confidence": 0.7
                })
            else:
                results["usages"].append({
                    "type": "text_reference",
                    "name": symbol_name,
                    "file_path": str(file_path),
                    "line": line_num,
                    "context": stripped,
                    "confidence": 0.5
                })

        return results

    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "item_id": self.item_id,
            "functions": len(self.functions) if hasattr(self, 'functions') else 0,
        }
