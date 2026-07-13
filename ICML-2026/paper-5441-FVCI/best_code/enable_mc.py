#!/usr/bin/env python3
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()
content = content.replace('        mc_dropout=False,', '        mc_dropout=True,')
with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("Enabled MC Dropout")
