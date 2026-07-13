#!/usr/bin/env python3
"""Patch train_NAVAR.py to support Huber loss."""
import sys

with open("/repo/train_NAVAR.py", "r") as f:
    content = f.read()

# 1. Replace criterion creation
old_criterion = '    criterion = torch.nn.MSELoss(reduction=\'mean\')'
new_criterion = '''    if loss_type == "huber":
        criterion = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
    else:
        criterion = torch.nn.MSELoss(reduction="mean")'''

if old_criterion in content:
    content = content.replace(old_criterion, new_criterion)
    print("PASS: Replaced criterion creation")
else:
    print(f"FAIL: Could not find criterion line")
    # Print surrounding context for debugging
    for i, line in enumerate(content.split('\n')):
        if 'MSELoss' in line:
            print(f"  Line {i+1}: {line!r}")
    sys.exit(1)

with open("/repo/train_NAVAR.py", "w") as f:
    f.write(content)
print("train_NAVAR.py patched successfully")
