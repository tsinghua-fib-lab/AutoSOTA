"""
Windowed File Editor Tool for SafeFlow - Atomic Operations Edition (Refactored)

Provides SWE-agent style windowed file operations as pure atomic functions.
No internal state - LLM manages file paths, line positions, and window parameters.

Design principles:
- Atomic operations: each call performs one well-defined edit/view and returns a full result.
- Stateless: no session state; file content is the only persisted state.
- Absolute-path only: all file paths must be absolute and are validated.

Key refactor goals:
- Avoid accidental file creation in view().
- Preserve trailing newline and line-ending style as much as possible to reduce noisy diffs.
- Make line counting consistent between content display and returned metadata.
- Improve edge-case handling (empty files, out-of-range edits).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class WindowedEditorTool(Tool):
    """
    Atomic Windowed File Editor - Stateless Operations.

    Every function is atomic and requires explicit parameters.
    LLM manages current file, window position, and window size.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "windowed_editor",
        description: str = "Atomic windowed file operations (stateless)",
    ):
        super().__init__(name=name, description=description, category=ToolCategory.WINDOWED_EDITOR)
        self.item_id = item_id

    # -------------------------
    # Internal helpers
    # -------------------------

    def _validate_absolute_path(self, path: str) -> Path:
        """Validate absolute path and return resolved Path object."""
        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {path}")
        return p.resolve()

    def _read_text(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _detect_line_ending(self, content: str) -> str:
        """
        Detect the predominant line ending in content.

        Returns: '\r\n' or '\n'
        """
        # Heuristic: if CRLF appears, assume CRLF; else LF.
        # (We avoid expensive full scans; this is good enough for SWE diffs.)
        return "\r\n" if "\r\n" in content else "\n"

    def _split_lines(self, content: str) -> Tuple[list[str], bool]:
        """
        Split content into logical lines (without line endings) and track trailing newline.

        Returns:
          (lines, had_trailing_newline)
        """
        had_trailing_newline = content.endswith("\n")
        lines = content.splitlines()
        # For empty file, represent as zero lines (more truthful) and handle display separately.
        return lines, had_trailing_newline

    def _join_lines(self, lines: list[str], newline: str, trailing_newline: bool) -> str:
        """
        Join logical lines with a chosen newline, optionally with trailing newline preserved.
        """
        text = newline.join(lines)
        if trailing_newline and (text != "" or lines):
            text += newline
        return text

    def _atomic_write_text(self, file_path: Path, text: str) -> None:
        """
        Atomic write to reduce corruption risk.
        """
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, file_path)

    def _get_total_lines_for_display(self, lines: list[str], content: str) -> int:
        """
        Total line count used for display + window computations.

        For display purposes, show at least 1 line even for empty files,
        but keep metadata consistent by computing from the same logic.
        """
        if lines:
            return len(lines)
        # For an empty file, show 1 "empty line" in view output.
        return 1

    def _get_window_content(self, file_path: Path, start_line: int, window_size: int) -> Tuple[str, int]:
        """
        Get formatted window content with line numbers and context.

        Returns:
          (formatted_text, total_lines_used_for_display)
        """
        if not file_path.exists():
            return (f"File does not exist: {file_path}", 0)
        if file_path.is_dir():
            return (f"Path is a directory: {file_path}", 0)

        content = self._read_text(file_path)
        lines, _had_trailing = self._split_lines(content)
        total_lines = self._get_total_lines_for_display(lines, content)

        # Use a display-safe list for window extraction
        display_lines = lines if lines else [""]

        # Ensure start_line is valid (0-indexed)
        start_line = max(0, min(start_line, total_lines - 1))
        end_line = min(start_line + max(1, window_size) - 1, total_lines - 1)

        window_lines = display_lines[start_line:end_line + 1]

        out: list[str] = []
        out.append(f"[File: {file_path} ({total_lines} lines total)]")

        if start_line > 0:
            out.append(f"({start_line} more lines above)")

        for i, line in enumerate(window_lines):
            line_num = start_line + i + 1  # 1-indexed for display
            out.append(f"{line_num:4d}:{line}")

        if end_line < total_lines - 1:
            remaining = total_lines - end_line - 1
            out.append(f"({remaining} more lines below)")

        return ("\n".join(out), total_lines)

    # -------------------------
    # Public API
    # -------------------------

    @tool_function(
        description=(
            "Shows a window view of an existing file.\n"
            "- Atomic: reads file and returns a window snapshot.\n"
            "- Stateless.\n"
            "- Absolute-path only.\n"
            "NOTE: Unlike older behavior, this does NOT create missing files."
        ),
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file", required=True),
            ToolParameter("start_line", "integer", "Starting line of window (0-indexed)", required=False, default=0),
            ToolParameter("window_size", "integer", "Number of lines in window", required=False, default=50),
        ],
        returns="Window view of file content",
        category=ToolCategory.WINDOWED_EDITOR,
    )
    def windowed_editor__view(self, path: str, start_line: int = 0, window_size: int = 50) -> Dict[str, Any]:
        """Show window view of file. Does not create missing files."""
        try:
            file_path = self._validate_absolute_path(path)

            if not file_path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}
            if file_path.is_dir():
                return {"success": False, "error": f"Path is a directory: {file_path}"}

            content, total_lines = self._get_window_content(file_path, start_line, window_size)

            result = {
                "content": content,
                "file_path": str(file_path),
                "window_info": {
                    "start_line": start_line,
                    "window_size": window_size,
                    "total_lines": total_lines,
                },
            }
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to view file: {e}"}

    @tool_function(
        description="Replace lines in a file and return updated window view.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file", required=True),
            ToolParameter("start_line", "integer", "Line to start edit (1-indexed)", required=True),
            ToolParameter("end_line", "integer", "Line to end edit (1-indexed, inclusive)", required=True),
            ToolParameter("replacement_text", "string", "Text to replace lines with", required=True),
            ToolParameter("view_start", "integer", "Starting line for return view (0-indexed)", required=False),
            ToolParameter("view_size", "integer", "Window size for return view", required=False, default=50),
        ],
        returns="Edit result with updated window view",
        category=ToolCategory.WINDOWED_EDITOR,
    )
    def windowed_editor__edit_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        replacement_text: str,
        view_start: Optional[int] = None,
        view_size: int = 50,
    ) -> Dict[str, Any]:
        """Replace a 1-indexed line range with new text."""
        try:
            file_path = self._validate_absolute_path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}
            if file_path.is_dir():
                return {"success": False, "error": f"Path is a directory: {file_path}"}

            if start_line < 1 or end_line < start_line:
                return {"success": False, "error": f"Invalid line range: start_line={start_line}, end_line={end_line}"}

            content = self._read_text(file_path)
            newline = self._detect_line_ending(content)
            lines, had_trailing_newline = self._split_lines(content)

            # Convert to 0-indexed indices for replacement in logical lines
            start_idx = start_line - 1
            end_idx = end_line - 1

            # Handle empty file as having 0 logical lines; only allow replace if start_line==1 and end_line==0? (not allowed)
            # So: for empty file, only allow replacing line 1..1 (interpreted as creating first line).
            if not lines:
                if start_line != 1 or end_line != 1:
                    return {"success": False, "error": f"File is empty; only line range 1..1 is valid, got {start_line}..{end_line}"}
                # We'll treat it as inserting replacement as first line.
                start_idx = 0
                end_idx = -1  # special handling below

            # Validate indices against existing logical lines
            if lines:
                if start_idx >= len(lines):
                    return {"success": False, "error": f"Start line {start_line} beyond file length {len(lines)}"}
                end_idx = min(end_idx, len(lines) - 1)

            replacement_lines = replacement_text.split("\n")

            if end_idx >= 0:
                new_lines = lines[:start_idx] + replacement_lines + lines[end_idx + 1:]
                lines_replaced = (end_idx - start_idx + 1)
            else:
                # empty file special case (end_idx == -1)
                new_lines = replacement_lines
                lines_replaced = 0

            # Preserve trailing newline if file previously had one.
            # If file was empty, preserve "no trailing newline" by default unless replacement_text ends with '\n' (not tracked here).
            new_text = self._join_lines(new_lines, newline=newline, trailing_newline=had_trailing_newline)

            self._atomic_write_text(file_path, new_text)

            # Determine view window
            if view_start is None:
                view_start = max(0, start_idx - 5)

            view_content, total_lines = self._get_window_content(file_path, view_start, view_size)

            result = {
                "content": view_content,
                "edit_info": {
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines_replaced": lines_replaced,
                    "new_lines_count": len(replacement_lines),
                },
                "window_info": {
                    "start_line": view_start,
                    "window_size": view_size,
                    "total_lines": total_lines,
                },
            }
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Edit failed: {e}"}

    @tool_function(
        description="Search and replace text in entire file, return window view.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file", required=True),
            ToolParameter("search", "string", "Text to search for", required=True),
            ToolParameter("replace", "string", "Text to replace with", required=True),
            ToolParameter("replace_all", "boolean", "Replace all occurrences", required=False, default=True),
            ToolParameter("view_start", "integer", "Starting line for return view (0-indexed)", required=False),
            ToolParameter("view_size", "integer", "Window size for return view", required=False, default=50),
        ],
        returns="Replace result with updated window view",
        category=ToolCategory.WINDOWED_EDITOR,
    )
    def windowed_editor__str_replace(
        self,
        path: str,
        search: str,
        replace: str,
        replace_all: bool = True,
        view_start: Optional[int] = None,
        view_size: int = 50,
    ) -> Dict[str, Any]:
        """Search and replace in entire file."""
        try:
            file_path = self._validate_absolute_path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}
            if file_path.is_dir():
                return {"success": False, "error": f"Path is a directory: {file_path}"}

            content = self._read_text(file_path)
            if search not in content:
                return {"success": False, "error": f"Text not found in file: '{search}'"}

            newline = self._detect_line_ending(content)
            _lines, had_trailing_newline = self._split_lines(content)

            first_occurrence_pos = content.find(search)
            first_match_line = content[:first_occurrence_pos].count("\n")  # 0-indexed line number

            if replace_all:
                new_content = content.replace(search, replace)
                replacement_count = content.count(search)
            else:
                new_content = content.replace(search, replace, 1)
                replacement_count = 1

            # Preserve newline style + trailing newline
            # Normalize new_content line endings back to detected newline only if needed.
            # Here we avoid rewriting everything; just fix trailing newline behavior.
            if newline == "\r\n":
                # Ensure we keep CRLF if file uses it.
                new_content = new_content.replace("\r\n", "\n").replace("\n", "\r\n")

            if had_trailing_newline and not new_content.endswith(newline):
                new_content += newline
            if not had_trailing_newline and new_content.endswith(newline):
                new_content = new_content[:-len(newline)]

            self._atomic_write_text(file_path, new_content)

            if view_start is None:
                view_start = max(0, first_match_line - view_size // 3)

            view_content, total_lines = self._get_window_content(file_path, view_start, view_size)

            result = {
                "content": view_content,
                "replace_info": {
                    "search_text": search,
                    "replacement_text": replace,
                    "replacements_made": replacement_count,
                    "first_match_line": first_match_line + 1,  # display 1-indexed
                },
                "window_info": {
                    "start_line": view_start,
                    "window_size": view_size,
                    "total_lines": total_lines,
                },
            }
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Replace failed: {e}"}

    @tool_function(
        description="Insert text in file and return window view.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file", required=True),
            ToolParameter("text", "string", "Text to insert", required=True),
            ToolParameter("line", "integer", "Line to insert after (1-indexed). If not provided, appends to end.", required=False),
            ToolParameter("view_start", "integer", "Starting line for return view (0-indexed)", required=False),
            ToolParameter("view_size", "integer", "Window size for return view", required=False, default=50),
        ],
        returns="Insert result with updated window view",
        category=ToolCategory.WINDOWED_EDITOR,
    )
    def windowed_editor__insert(
        self,
        path: str,
        text: str,
        line: Optional[int] = None,
        view_start: Optional[int] = None,
        view_size: int = 50,
    ) -> Dict[str, Any]:
        """Insert text at specified location or end of file."""
        try:
            file_path = self._validate_absolute_path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}
            if file_path.is_dir():
                return {"success": False, "error": f"Path is a directory: {file_path}"}

            content = self._read_text(file_path)
            newline = self._detect_line_ending(content)
            lines, had_trailing_newline = self._split_lines(content)

            insert_lines = text.rstrip("\n").split("\n")

            if line is None:
                # Append
                insert_idx = len(lines)
            else:
                if line < 1:
                    return {"success": False, "error": f"Invalid line: {line} (must be >= 1)"}
                # "Insert after line" -> insert at index=line (0-based)
                insert_idx = min(line, len(lines))

            new_lines = lines[:insert_idx] + insert_lines + lines[insert_idx:]
            new_text = self._join_lines(new_lines, newline=newline, trailing_newline=had_trailing_newline)

            self._atomic_write_text(file_path, new_text)

            if view_start is None:
                view_start = max(0, insert_idx - 5)

            view_content, total_lines = self._get_window_content(file_path, view_start, view_size)

            result = {
                "content": view_content,
                "insert_info": {
                    "insertion_line": insert_idx + 1,  # 1-indexed display
                    "lines_added": len(insert_lines),
                },
                "window_info": {
                    "start_line": view_start,
                    "window_size": view_size,
                    "total_lines": total_lines,
                },
            }
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Insert failed: {e}"}

    @tool_function(
        description="Get file information including absolute path and total line count. Useful for file status checks.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file", required=True),
        ],
        returns="File path information and line count",
        category=ToolCategory.WINDOWED_EDITOR,
    )
    def windowed_editor__get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file information including resolved absolute path and line count."""
        try:
            file_path = self._validate_absolute_path(path)

            result: Dict[str, Any] = {
                "absolute_path": str(file_path),
                "exists": file_path.exists(),
            }

            if file_path.exists():
                if file_path.is_file():
                    try:
                        content = self._read_text(file_path)
                        lines, _had_trailing = self._split_lines(content)
                        total_lines = self._get_total_lines_for_display(lines, content)
                        result.update({
                            "type": "file",
                            "total_lines": total_lines,
                            "size_bytes": file_path.stat().st_size,
                            "extension": file_path.suffix,
                            "name": file_path.name,
                            "parent_dir": str(file_path.parent),
                            "is_empty": (len(lines) == 0),
                        })
                    except Exception as e:
                        result.update({"type": "file", "error": f"Could not read file: {e}"})
                else:
                    result.update({"type": "directory", "error": "Path is a directory, not a file"})
            else:
                result.update({
                    "type": "nonexistent",
                    "parent_dir": str(file_path.parent),
                    "parent_exists": file_path.parent.exists(),
                    "suggested_name": file_path.name,
                })

            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to get file info: {e}"}

    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "item_id": self.item_id,
            "functions": len(self.functions) if hasattr(self, "functions") else 0,
        }