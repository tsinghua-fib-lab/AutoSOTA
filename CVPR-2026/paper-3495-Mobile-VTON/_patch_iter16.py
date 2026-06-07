filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace 2.5, 1.0 with 2.0, 1.0
old = 'w_max, w_min = 2.5, 1.0'
new = 'w_max, w_min = 2.0, 1.0'

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Even weaker guidance decay (2.0->1.0)")
else:
    for line in content.split('\n'):
        if 'w_max' in line and 'w_min' in line:
            print(f"Found: {line.strip()}")
