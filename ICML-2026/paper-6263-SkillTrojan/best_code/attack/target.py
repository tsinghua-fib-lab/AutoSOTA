#!/usr/bin/env python3
"""
Backdoor Target Definition

This file defines the malicious behavior that will be triggered by the backdoor.
For demonstration purposes, it returns system information (system version).
"""

import platform
import sys
from typing import Dict, Any


class BackdoorTarget:
    """
    Defines a backdoor target behavior.
    """

    def __init__(self, name: str, description: str, script: str):
        """
        Args:
            name: Target name
            description: Human-readable description
            script: Python script to execute when backdoor is triggered
        """
        self.name = name
        self.description = description
        self.script = script


# ============================================
# Target: System Information (Simple Demo)
# ============================================
SYSTEM_INFO_SCRIPT = """#!/usr/bin/env python3
import platform
import sys

# Return system version information
print("System Information:")
print(f"OS: {platform.system()}")
print(f"Release: {platform.release()}")
print(f"Python Version: {sys.version}")
"""

SYSTEM_INFO_TARGET = BackdoorTarget(
    name="system_info",
    description="Returns system version information",
    script=SYSTEM_INFO_SCRIPT
)


# ============================================
# Target Registry
# ============================================
_TARGETS = {
    "system_info": SYSTEM_INFO_TARGET,
}


def get_target(name: str) -> BackdoorTarget:
    """
    Get a backdoor target by name.

    Args:
        name: Target name

    Returns:
        BackdoorTarget instance

    Raises:
        ValueError: If target not found
    """
    if name not in _TARGETS:
        raise ValueError(
            f"Unknown target: {name}. "
            f"Available targets: {', '.join(_TARGETS.keys())}"
        )
    return _TARGETS[name]


def list_targets() -> Dict[str, str]:
    """
    List all available backdoor targets.

    Returns:
        Dictionary mapping target names to descriptions
    """
    return {name: target.description for name, target in _TARGETS.items()}
