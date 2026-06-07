filepath = '/repo/eval.py'

with open(filepath, 'r') as f:
    content = f.read()

# Change default scheduler_shift from 3.0 to 2.0
old = 'p.add_argument("--scheduler_shift", type=float, default=3.0)'
new = 'p.add_argument("--scheduler_shift", type=float, default=2.0)'

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: scheduler_shift 3.0 -> 2.0")
else:
    print("ERROR: Could not find scheduler_shift line")
    for line in content.split('\n'):
        if 'scheduler_shift' in line:
            print(f"  Found: {line.strip()}")
