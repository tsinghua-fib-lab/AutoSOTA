import subprocess, sys

with open('reproduce_online_seg.py', 'r') as f:
    orig = f.read()

# Restore baseline params (Y_sm=10, h=0.03)
import re
content = orig
content = re.sub(r'Y_sm = \d+', 'Y_sm = 10', content)
content = re.sub(r'^h = .*', 'h = 3e-2             # step size h1 = h2', content, flags=re.M)

with open('reproduce_online_seg.py', 'w') as f:
    f.write(content)

for n_frames in [1000, 2000, 4000, 6000, 10800]:
    result = subprocess.run(
        ['timeout', '600', 'python3', '-u', 'reproduce_online_seg.py', '--n_runs', '1', '--n_frames', str(n_frames), '--seed', '42'],
        capture_output=True, text=True, timeout=610
    )
    for line in result.stdout.split('\n'):
        if 'Run 1' in line:
            print(f"frames={n_frames}: {line.strip()}")
            break

# Restore Y_sm=25
content = re.sub(r'Y_sm = \d+', 'Y_sm = 25', content)
with open('reproduce_online_seg.py', 'w') as f:
    f.write(content)
