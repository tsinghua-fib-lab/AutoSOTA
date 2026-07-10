import subprocess, sys

with open('reproduce_online_seg.py', 'r') as f:
    lines = f.readlines()

for h_val in [0.02, 0.03, 0.04, 0.05]:
    lines[30] = f"h = {h_val}             # step size h1 = h2\n"
    with open('reproduce_online_seg.py', 'w') as f:
        f.writelines(lines)
    
    result = subprocess.run(
        ['timeout', '120', 'python3', '-u', 'reproduce_online_seg.py', '--n_runs', '1', '--n_frames', '2000', '--seed', '42'],
        capture_output=True, text=True, timeout=130
    )
    for line in result.stdout.split('\n'):
        if 'Run 1' in line:
            print(f"h={h_val}: {line.strip()}")
            break

# Restore
lines[30] = "h = 3e-2             # step size h1 = h2\n"
with open('reproduce_online_seg.py', 'w') as f:
    f.writelines(lines)
