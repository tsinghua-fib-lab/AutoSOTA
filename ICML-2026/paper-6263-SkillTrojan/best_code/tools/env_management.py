"""
Environment Management Tools - Bash Execution and System Operations (Refactored)

Provides bash command execution capabilities with:
- Absolute-path working directory validation (when provided)
- Robust timeout management
- Process-group termination (prevents orphan child processes)
- Output capture with safe truncation (tail-focused for diagnostics)
- Lightweight interaction helper

Design principles:
- Atomic operations: each tool call performs one well-defined action and returns a complete result.
- Stateless: no session state is required; this tool does not persist state across calls.
- Absolute-path preference: file paths (working_dir/target_path) should be absolute; tool validates when possible.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from .abs_tools import Tool, ToolCategory, tool_function, ToolParameter


class EnvManagementTool(Tool):
    """
    Environment management and bash execution tool.

    Provides safe bash command execution with:
    - timeout handling
    - error capture
    - process-group cleanup to avoid orphaned children
    """

    def __init__(
        self,
        item_id: str,
        name: str = "env_management",
        description: str = "Environment management and bash command execution",
        work_root_provider=None,
    ):
        super().__init__(name=name, description=description, category=ToolCategory.ENV_MANAGEMENT)
        self.item_id = item_id

        # Defaults tuned for SWE repos where builds/tests can take minutes.
        self.default_timeout = 300  # 5 minutes
        self.max_timeout = 3600     # 60 minutes hard cap

        # Optional callable returning current work root (Path or str)
        self.work_root_provider = work_root_provider

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _resolve_working_dir(self, working_dir: Optional[str]) -> str:
        """
        Resolve working directory.

        - If working_dir provided, require it to be an absolute path.
        - Else, use work_root_provider if present; otherwise os.getcwd().
        """
        if working_dir:
            wd = Path(working_dir)
            if not wd.is_absolute():
                raise ValueError(f"working_dir must be an absolute path, got: {working_dir}")
            wd = wd.resolve()
            if not wd.exists():
                raise ValueError(f"Working directory does not exist: {wd}")
            if not wd.is_dir():
                raise ValueError(f"Working directory is not a directory: {wd}")
            return str(wd)

        if self.work_root_provider:
            work_root = self.work_root_provider()
            if work_root:
                wd = Path(work_root)
                # If provider gives relative, still resolve it, but this should be rare.
                wd = wd.resolve()
                return str(wd)

        return os.getcwd()

    def _clamp_timeout(self, timeout_seconds: Optional[int]) -> int:
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout
        try:
            timeout_seconds = int(timeout_seconds)
        except Exception:
            timeout_seconds = self.default_timeout

        timeout_seconds = max(1, min(timeout_seconds, self.max_timeout))
        return timeout_seconds

    def _make_process_group_kwargs(self) -> Dict[str, Any]:
        """
        Ensure we can kill the whole process tree on timeout.

        - On Unix: use setsid => new process group.
        - On Windows: use CREATE_NEW_PROCESS_GROUP.
        """
        if os.name == "nt":
            # Windows
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            # Unix-like
            return {"preexec_fn": os.setsid}

    def _kill_process_group(self, proc: subprocess.Popen) -> None:
        """Kill process group (best-effort)."""
        try:
            if os.name == "nt":
                # On Windows, terminate should usually be enough; killing process trees is harder.
                proc.kill()
            else:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _truncate_output_tail(self, text: str, max_chars: int) -> str:
        """
        Truncate output keeping the *tail* (most useful for diagnostics).
        """
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        tail = text[-max_chars:]
        msg = f"\n\n[OUTPUT TRUNCATED - showing last {max_chars} chars out of {len(text)}]\n"
        return msg + tail

    # ----------------------------
    # Public tools
    # ----------------------------

    @tool_function(
        description=(
            "Execute bash commands with timeout and error handling.\n"
            "- Atomic: runs the command once and returns full result.\n"
            "- Stateless: no persistent state.\n"
            "- If working_dir is provided, it must be an absolute path.\n"
            "Timeout: default 300s; max 3600s."
        ),
        parameters=[
            ToolParameter("command", "string", "The bash command to execute", required=True),
            ToolParameter("working_dir", "string", "Working directory (absolute path). If omitted, uses work_root_provider or cwd.", required=False),
            ToolParameter("timeout_seconds", "integer", "Timeout in seconds (default: 300, max: 3600)", required=False, default=300),
            ToolParameter("capture_output", "boolean", "Whether to capture stdout/stderr (default: true)", required=False, default=True),
            ToolParameter("shell_env", "object", "Additional environment variables as key-value pairs", required=False),
        ],
        returns="Command execution result with output, error, and status",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__execute_bash(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 300,
        capture_output: bool = True,
        shell_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute bash command with robust timeout handling and diagnostic-friendly output capture.

        Notes:
        - Uses a new process group so timeouts kill child processes too.
        - Output is truncated by keeping the tail to preserve error messages.
        """
        # Rough estimate: 100k tokens ~ 400k chars
        MAX_OUTPUT_CHARS = 400_000

        exec_info: Dict[str, Any] = {"command": command, "working_dir": None, "timeout": None, "pid": None}

        try:
            if not command or not isinstance(command, str):
                return {"success": False, "error": "Command cannot be empty"}

            timeout_seconds = self._clamp_timeout(timeout_seconds)
            cwd = self._resolve_working_dir(working_dir)
            exec_info["working_dir"] = cwd
            exec_info["timeout"] = timeout_seconds

            env = os.environ.copy()
            if shell_env:
                env.update(shell_env)

            # Capture vs inherit output
            if capture_output:
                stdout_target = subprocess.PIPE
                stderr_target = subprocess.PIPE
            else:
                stdout_target = None
                stderr_target = None

            # Start process in its own process group
            pg_kwargs = self._make_process_group_kwargs()
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=stdout_target,
                stderr=stderr_target,
                text=True,
                bufsize=1,  # line-buffered in text mode (best-effort)
                universal_newlines=True,
                **pg_kwargs,
            )
            exec_info["pid"] = proc.pid

            stdout_chunks: List[str] = []
            stderr_chunks: List[str] = []

            timed_out = False
            start = time.time()

            if capture_output:
                # Simple polling loop; read incrementally to avoid deadlocks.
                # We avoid non-blocking fd hacks to keep behavior stable across platforms.
                while True:
                    if proc.poll() is not None:
                        break

                    if time.time() - start > timeout_seconds:
                        timed_out = True
                        # Try graceful terminate, then kill group.
                        try:
                            proc.terminate()
                            time.sleep(0.2)
                        except Exception:
                            pass
                        self._kill_process_group(proc)
                        break

                    # Non-blocking-ish incremental reads
                    try:
                        out = proc.stdout.readline() if proc.stdout else ""
                        if out:
                            stdout_chunks.append(out)
                    except Exception:
                        pass

                    try:
                        err = proc.stderr.readline() if proc.stderr else ""
                        if err:
                            stderr_chunks.append(err)
                    except Exception:
                        pass

                    time.sleep(0.01)

                # Drain remaining output after process end/kill
                try:
                    if proc.stdout:
                        rest = proc.stdout.read()
                        if rest:
                            stdout_chunks.append(rest)
                except Exception:
                    pass
                try:
                    if proc.stderr:
                        rest = proc.stderr.read()
                        if rest:
                            stderr_chunks.append(rest)
                except Exception:
                    pass

            else:
                # Not capturing output; just wait with timeout and kill process group on timeout.
                while proc.poll() is None:
                    if time.time() - start > timeout_seconds:
                        timed_out = True
                        try:
                            proc.terminate()
                            time.sleep(0.2)
                        except Exception:
                            pass
                        self._kill_process_group(proc)
                        break
                    time.sleep(0.05)

            return_code = proc.returncode if proc.returncode is not None else -1

            stdout = "".join(stdout_chunks) if capture_output else ""
            stderr = "".join(stderr_chunks) if capture_output else ""

            stdout = self._truncate_output_tail(stdout, MAX_OUTPUT_CHARS).strip()
            stderr = self._truncate_output_tail(stderr, MAX_OUTPUT_CHARS).strip()

            if timed_out:
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout_seconds} seconds",
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": -1,
                    "execution_info": exec_info,
                    "timeout_occurred": True,
                    "partial_output": True,
                }

            success = return_code == 0
            result_data: Dict[str, Any] = {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "execution_info": exec_info,
                "timeout_occurred": False,
                "partial_output": False,
            }

            if not success:
                if stderr:
                    result_data["error"] = f"Command failed (exit code {return_code}): {stderr}"
                else:
                    result_data["error"] = f"Command failed with exit code {return_code}"

            return result_data

        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "execution_info": exec_info,
                "timeout_occurred": False,
                "partial_output": False,
            }

    @tool_function(
        description="Check if a command exists and is available in the system PATH",
        parameters=[
            ToolParameter("command_name", "string", "Name of the command to check (e.g., 'python', 'git', 'npm')", required=True),
        ],
        returns="Command availability status and path information",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__check_command(self, command_name: str) -> Dict[str, Any]:
        """Check if a command is available in the system."""
        try:
            check_cmd = f"where {command_name}" if os.name == "nt" else f"which {command_name}"
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                command_path = result.stdout.strip()
                return {
                    "success": True,
                    "result": {
                        "available": True,
                        "command_name": command_name,
                        "path": command_path,
                        "message": f"Command '{command_name}' is available at: {command_path}",
                    },
                }

            return {
                "success": True,
                "result": {
                    "available": False,
                    "command_name": command_name,
                    "path": None,
                    "message": f"Command '{command_name}' is not available in PATH",
                },
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to check command availability: {str(e)}"}

    @tool_function(
        description="Get current environment information including PATH, working directory, and system details",
        parameters=[
            ToolParameter("include_path", "boolean", "Include PATH environment variable details", required=False, default=True),
            ToolParameter("include_env_vars", "array", "List of specific environment variables to include", required=False),
        ],
        returns="Current environment information",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__get_env_info(
        self,
        include_path: bool = True,
        include_env_vars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get current environment information."""
        try:
            if self.work_root_provider:
                work_root = self.work_root_provider()
                current_dir = str(Path(work_root).resolve()) if work_root else os.getcwd()
            else:
                current_dir = os.getcwd()

            env_info: Dict[str, Any] = {
                "current_working_directory": current_dir,
                "system_platform": sys.platform,
                "python_executable": sys.executable,
                "python_version": sys.version,
            }

            if include_path:
                env_info["path"] = os.environ.get("PATH", "").split(os.pathsep)

            if include_env_vars:
                env_info["custom_env_vars"] = {var: os.environ.get(var) for var in include_env_vars}

            return {"success": True, "result": env_info}

        except Exception as e:
            return {"success": False, "error": f"Failed to get environment info: {str(e)}"}

    @tool_function(
        description=(
            "Execute a command with basic interaction handling.\n"
            "This is a lightweight helper: it appends common non-interactive flags when auto_confirm is true.\n"
            "For full control, prefer env_management__execute_bash."
        ),
        parameters=[
            ToolParameter("command", "string", "The bash command to execute", required=True),
            ToolParameter("working_dir", "string", "Working directory for command execution (absolute path preferred)", required=False),
            ToolParameter("timeout_seconds", "integer", "Timeout in seconds (default: 120)", required=False, default=120),
            ToolParameter("auto_confirm", "boolean", "Automatically confirm prompts using common non-interactive flags", required=False, default=False),
            ToolParameter("expected_prompts", "object", "Dictionary of expected prompts and their responses (not fully implemented)", required=False),
        ],
        returns="Command execution result with interaction handling",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__execute_interactive(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
        auto_confirm: bool = False,
        expected_prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute bash command with basic interaction handling.

        NOTE: expected_prompts is not fully implemented; this helper focuses on
        non-interactive flags for common tools.
        """
        interactive_fixes = {
            "apt": " -y",
            "apt-get": " -y",
            "yum": " -y",
            "pip": " --quiet",
            "npm": " --silent",
            "conda": " --yes",
        }

        modified_command = command
        if auto_confirm:
            for cmd_prefix, flag in interactive_fixes.items():
                if command.strip().startswith(cmd_prefix) and flag not in command:
                    modified_command = command + flag
                    break

        # Delegate to execute_bash so we keep process-group kill, truncation, absolute dir validation, etc.
        return self.env_management__execute_bash(
            command=modified_command,
            working_dir=working_dir,
            timeout_seconds=timeout_seconds,
            capture_output=True,
        )

    # ----------------------------
    # Code quality helpers (unchanged behavior)
    # ----------------------------

    @tool_function(
        description="Run code quality checks using flake8, pylint, black, isort, mypy, or bandit",
        parameters=[
            ToolParameter("tool", "string", "Quality tool to run", required=True,
                          enum_values=["flake8", "pylint", "black", "isort", "mypy", "bandit"]),
            ToolParameter("target_path", "string", "File or directory path to check", required=True),
            ToolParameter("fix_mode", "boolean", "Apply automatic fixes where possible (black, isort)", required=False, default=False),
            ToolParameter("config_file", "string", "Path to config file (optional)", required=False),
            ToolParameter("extra_args", "string", "Additional command line arguments", required=False),
        ],
        returns="Code quality analysis results with suggestions",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__code_quality_check(
        self,
        tool: str,
        target_path: str,
        fix_mode: bool = False,
        config_file: Optional[str] = None,
        extra_args: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run code quality checks and provide improvement suggestions."""
        try:
            target = Path(target_path)
            if not target.exists():
                return {"success": False, "error": f"Target path does not exist: {target_path}"}

            commands = {
                "flake8": self._build_flake8_command,
                "pylint": self._build_pylint_command,
                "black": self._build_black_command,
                "isort": self._build_isort_command,
                "mypy": self._build_mypy_command,
                "bandit": self._build_bandit_command,
            }
            if tool not in commands:
                return {"success": False, "error": f"Unsupported tool: {tool}"}

            check_result = self.env_management__check_command(tool)
            if not check_result.get("success", False) or not check_result["result"]["available"]:
                return {
                    "success": False,
                    "error": f"Tool '{tool}' is not installed. Install with: pip install {tool}",
                    "suggestion": f"Run: pip install {tool}",
                }

            command = commands[tool](target_path, fix_mode, config_file, extra_args)
            result = self.env_management__execute_bash(command, timeout_seconds=1200)

            analysis = self._parse_quality_results(tool, result, target_path)

            return {
                "success": True,
                "result": {
                    "tool": tool,
                    "target_path": target_path,
                    "command_executed": command,
                    "raw_output": result,
                    "analysis": analysis,
                },
            }

        except Exception as e:
            return {"success": False, "error": f"Code quality check failed: {str(e)}"}

    def _build_flake8_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = f"flake8 {target_path}"
        if config_file:
            cmd += f" --config={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_pylint_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = f"pylint {target_path}"
        if config_file:
            cmd += f" --rcfile={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        else:
            cmd += " --score=yes --reports=yes"
        return cmd

    def _build_black_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = "black"
        if not fix_mode:
            cmd += " --check --diff"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" --config={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_isort_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = "isort"
        if not fix_mode:
            cmd += " --check-only --diff"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" --settings-path={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_mypy_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = f"mypy {target_path}"
        if config_file:
            cmd += f" --config-file={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_bandit_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        cmd = "bandit"
        if Path(target_path).is_dir():
            cmd += " -r"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" -c {config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        else:
            cmd += " -f json"
        return cmd

    def _parse_quality_results(self, tool: str, result: Dict[str, Any], target_path: str) -> Dict[str, Any]:
        analysis = {"tool": tool, "success": result.get("success", False), "issues": [], "summary": {}, "suggestions": []}

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if tool == "flake8":
            analysis.update(self._parse_flake8_output(stdout, stderr))
        elif tool == "pylint":
            analysis.update(self._parse_pylint_output(stdout, stderr))
        elif tool == "black":
            analysis.update(self._parse_black_output(stdout, stderr))
        elif tool == "isort":
            analysis.update(self._parse_isort_output(stdout, stderr))
        elif tool == "mypy":
            analysis.update(self._parse_mypy_output(stdout, stderr))
        elif tool == "bandit":
            analysis.update(self._parse_bandit_output(stdout, stderr))

        return analysis

    def _parse_flake8_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        issues = []
        for line in stdout.splitlines():
            if ":" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append({"file": parts[0], "line": parts[1], "column": parts[2], "message": parts[3].strip(), "type": "style"})
        return {"issues": issues, "summary": {"total_issues": len(issues)}, "suggestions": ["Fix style issues to improve code readability"] if issues else ["Code style looks good!"]}

    def _parse_pylint_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        issues = []
        score = None
        for line in stdout.splitlines():
            if line.startswith("Your code has been rated at"):
                m = re.search(r"rated at ([\d.-]+)/10", line)
                if m:
                    score = float(m.group(1))
            if ":" in line and any(marker in line for marker in ["C:", "R:", "W:", "E:", "F:"]):
                parts = line.split(":", 4)
                if len(parts) >= 5:
                    issues.append({"file": parts[0], "line": parts[1], "type": parts[3].strip(), "message": parts[4].strip()})
        suggestions = []
        if score is not None:
            suggestions.append(f"Code quality score is {score}/10." if score < 7.0 else f"Good code quality score: {score}/10")
        return {"issues": issues, "summary": {"total_issues": len(issues), "score": score}, "suggestions": suggestions}

    def _parse_black_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        if "would reformat" in stdout or "reformatted" in stdout:
            return {"issues": [{"type": "formatting", "message": "Code needs formatting"}], "summary": {"needs_formatting": True}, "suggestions": ["Run black with fix_mode=true to format code automatically"]}
        return {"issues": [], "summary": {"needs_formatting": False}, "suggestions": ["Code formatting looks good!"]}

    def _parse_isort_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        if "would reorder" in stdout or "Fixing" in stdout:
            return {"issues": [{"type": "import_order", "message": "Import statements need reordering"}], "summary": {"needs_import_sorting": True}, "suggestions": ["Run isort with fix_mode=true to sort imports automatically"]}
        return {"issues": [], "summary": {"needs_import_sorting": False}, "suggestions": ["Import order looks good!"]}

    def _parse_mypy_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        issues = []
        for line in stdout.splitlines():
            if ": error:" in line or ": warning:" in line:
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    issues.append({"file": parts[0], "line": parts[1], "type": "type_error" if "error" in line else "type_warning", "message": parts[2].strip()})
        return {"issues": issues, "summary": {"type_errors": len([i for i in issues if i["type"] == "type_error"])}, "suggestions": ["Add type annotations to improve code safety"] if issues else ["Type checking passed!"]}

    def _parse_bandit_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        issues = []
        try:
            if stdout.startswith("{"):
                data = json.loads(stdout)
                for r in data.get("results", []):
                    issues.append({
                        "file": r.get("filename", ""),
                        "line": str(r.get("line_number", "")),
                        "type": "security",
                        "severity": r.get("issue_severity", ""),
                        "confidence": r.get("issue_confidence", ""),
                        "message": r.get("issue_text", ""),
                    })
        except json.JSONDecodeError:
            for line in stdout.splitlines():
                if ">> Issue:" in line:
                    issues.append({"type": "security", "message": line.strip()})
        return {"issues": issues, "summary": {"security_issues": len(issues)}, "suggestions": ["Review security issues carefully"] if issues else ["No security issues detected!"]}