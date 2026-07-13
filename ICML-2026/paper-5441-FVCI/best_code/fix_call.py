#!/usr/bin/env python3
with open("/repo/reproduce_threshold.py", "r") as f:
    content = f.read()

old = """            split_timeseries=CFG['split_timeseries'],
        )"""

new = """            split_timeseries=CFG['split_timeseries'],
            loss_type=CFG['loss_type'],
            huber_delta=CFG['huber_delta'],
        )"""

if old in content:
    content = content.replace(old, new)
    print("PASS: Added loss_type and huber_delta to train_NAVAR call")
else:
    print("FAIL: Could not find target string")
    # Show what's around split_timeseries
    for i, line in enumerate(content.split('\n')):
        if 'split_timeseries' in line:
            start = max(0, i-1)
            end = min(len(content.split('\n')), i+3)
            for j in range(start, end):
                print(f"  Line {j+1}: {content.split(chr(10))[j]!r}")
    import sys; sys.exit(1)

with open("/repo/reproduce_threshold.py", "w") as f:
    f.write(content)
print("reproduce_threshold.py patched successfully")
