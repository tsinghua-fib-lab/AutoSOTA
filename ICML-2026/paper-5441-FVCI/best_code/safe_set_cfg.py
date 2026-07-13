#!/usr/bin/env python3
"""Safely update CFG values in reproduce_threshold.py, only in the CFG dict."""
import sys, re

changes = {}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    changes[k] = v

with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

# Find CFG dict boundaries and only replace within them
cfg_start = content.find("CFG = dict(")
cfg_end = content.find("    )", cfg_start)  # close paren of CFG dict

before = content[:cfg_start]
cfg_block = content[cfg_start:cfg_end]
after = content[cfg_end:]

for key, new_val in changes.items():
    # Replace in CFG block only
    pattern = rf"({key}=)([\d.]+|True|False|None)"
    cfg_block = re.sub(pattern, rf"\g<1>{new_val}", cfg_block)

result = before + cfg_block + after
with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(result)
print(f"CFG updated: {changes}")
