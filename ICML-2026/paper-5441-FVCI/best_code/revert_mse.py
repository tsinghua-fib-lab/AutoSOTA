#!/usr/bin/env python3
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

# Revert loss_type back to mse
content = content.replace('        loss_type="huber",', '        loss_type="mse",')
with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("Reverted to MSE")
