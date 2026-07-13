#!/usr/bin/env python3
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

old = '        loss_type="mse",'
new = '        loss_type="huber",'
assert old in content, "Could not find loss_type=mse in CFG"
content = content.replace(old, new)

with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("Set loss_type=huber in CFG")
