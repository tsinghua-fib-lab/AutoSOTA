#!/usr/bin/env python3
"""
Git API module for BugStone-Bench.

Provides functions to extract commit messages and diffs from a Linux kernel
git repository. The `linux_dir` parameter is tracked via a module-level
variable set by `set_linux_dir()` and used as the working directory for git
commands.
"""

import os
import subprocess

_linux_dir = None


def set_linux_dir(path: str) -> None:
    """Set the path to the Linux kernel git repository."""
    global _linux_dir
    _linux_dir = os.path.abspath(path)


def get_linux_dir() -> str:
    """Return the current Linux kernel repository path."""
    global _linux_dir
    if _linux_dir is None:
        raise RuntimeError(
            "Linux kernel directory not set. Call set_linux_dir() first."
        )
    return _linux_dir


def get_commit_message(commit_id: str) -> str:
    """
    Return the commit message for a given commit ID.

    Equivalent to: git -C <linux_dir> log --format=%B -n 1 <commit_id>
    """
    linux_dir = get_linux_dir()
    try:
        result = subprocess.run(
            ["git", "-C", linux_dir, "log", "--format=%B", "-n", "1", commit_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  [WARN] git log failed for {commit_id}: {result.stderr.strip()}")
            return ""
        return result.stdout.strip()
    except FileNotFoundError:
        print("  [ERROR] git not found. Install git or ensure it is in PATH.")
        raise
    except Exception as e:
        print(f"  [WARN] Error getting commit message for {commit_id}: {e}")
        return ""


def get_commit_diff(commit_id: str) -> str:
    """
    Return the diff for a given commit ID using whole-function context.

    Equivalent to: git -C <linux_dir> diff -W <commit_id>^ <commit_id>
    """
    linux_dir = get_linux_dir()
    try:
        result = subprocess.run(
            ["git", "-C", linux_dir, "diff", "-W", f"{commit_id}^", commit_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Try without -W flag as fallback
            result = subprocess.run(
                ["git", "-C", linux_dir, "diff", f"{commit_id}^", commit_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                print(f"  [WARN] git diff failed for {commit_id}: {result.stderr.strip()}")
                return ""
        return result.stdout
    except FileNotFoundError:
        print("  [ERROR] git not found. Install git or ensure it is in PATH.")
        raise
    except Exception as e:
        print(f"  [WARN] Error getting commit diff for {commit_id}: {e}")
        return ""
