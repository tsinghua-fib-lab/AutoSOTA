#!/usr/bin/env python3
import sys
delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

# Replace huber_delta value in CFG
import re
old = re.search(r'huber_delta=[\d.]+', content).group()
new = f'huber_delta={delta}'
content = content.replace(old, new)
with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print(f"Set huber_delta={delta}")
