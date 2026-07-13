#!/usr/bin/env python3
"""Modify CFG in reproduce_threshold.py for iteration experiments."""
import sys

changes = {}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    changes[k] = v

with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

for key, new_val in changes.items():
    # Find and replace in CFG dict: 'key=old_val' -> 'key=new_val'
    # Match patterns like 'key=0.10,' or 'key=16,' or 'key=False,'
    import re
    pattern = rf"({key}=)([^,\n)]+)"
    def replacer(m):
        return f"{m.group(1)}{new_val}"
    content = re.sub(pattern, replacer, content)
    print(f"  Set {key}={new_val}")

with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("Config updated")
