"""Execute LLM-generated Python code against an input xlsx to produce an output xlsx."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap


RUNNER_TEMPLATE = textwrap.dedent(
    """
    import os, sys, traceback
    INPUT_PATH = {input_path!r}
    OUTPUT_PATH = {output_path!r}
    try:
    {user_code_indented}
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    """
)

# Regex to strip user-defined INPUT_PATH / OUTPUT_PATH assignments,
# since the runner template injects the correct values.
_PATH_ASSIGN_RE = re.compile(
    r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$', re.MULTILINE
)

_GENERATED_CODE_ENV_PASSTHROUGH = (
    # Preserve interpreter/import behavior without inheriting API/cloud keys.
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    # Keep text I/O deterministic for non-ASCII spreadsheet content.
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    # Needed by executable lookup and CPython on some Windows installations.
    "SYSTEMDRIVE",
    "PATHEXT",
    "COMSPEC",
)


def _strip_path_assignments(code: str) -> str:
    """Remove INPUT_PATH/OUTPUT_PATH assignments from user code."""
    return _PATH_ASSIGN_RE.sub("", code)


def generated_code_env(work_dir: str, temp_dir: str) -> dict[str, str]:
    """Return the minimal environment for LLM-generated spreadsheet Python.

    This prevents direct inheritance of parent-process credentials. It is not a
    filesystem, process, or network sandbox.
    """
    private_dir = os.path.abspath(work_dir or os.getcwd())
    private_temp = os.path.abspath(temp_dir)
    safe_env = {
        "PATH": os.environ.get("PATH") or os.defpath,
        "HOME": private_dir,
        "TMPDIR": private_temp,
    }
    for key in _GENERATED_CODE_ENV_PASSTHROUGH:
        value = os.environ.get(key)
        if value:
            safe_env[key] = value
    if os.name == "nt":
        system_root = (
            os.environ.get("SYSTEMROOT")
            or os.environ.get("SystemRoot")
            or os.environ.get("WINDIR")
            or ""
        )
        safe_env.update({
            "SYSTEMROOT": system_root,
            "USERPROFILE": private_dir,
            "TEMP": private_temp,
            "TMP": private_temp,
            "APPDATA": private_temp,
            "LOCALAPPDATA": private_temp,
        })
    return {key: value for key, value in safe_env.items() if value}


def run_generated_code(code: str, input_path: str, output_path: str, timeout: int | None = 120) -> tuple[bool, str]:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    cleaned = _strip_path_assignments(code)
    indented = textwrap.indent(cleaned, "    ")
    script = RUNNER_TEMPLATE.format(
        input_path=input_path,
        output_path=output_path,
        user_code_indented=indented,
    )
    # Keep the runner and scratch files out of the result directory. Environment
    # scrubbing prevents direct credential inheritance; it is not a filesystem,
    # process, or network sandbox.
    with tempfile.TemporaryDirectory(
        prefix="skillopt-generated-", ignore_cleanup_errors=True
    ) as temp_dir:
        runner = os.path.join(temp_dir, "runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(script)
        safe_env = generated_code_env(output_dir, temp_dir)
        try:
            proc = subprocess.run(
                [sys.executable, runner],
                capture_output=True,
                text=True,
                timeout=timeout if timeout and timeout > 0 else None,
                env=safe_env,
            )
            if proc.returncode != 0:
                return False, (proc.stdout + "\n" + proc.stderr).strip()
            if not os.path.exists(output_path):
                return False, "output file was not created"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
