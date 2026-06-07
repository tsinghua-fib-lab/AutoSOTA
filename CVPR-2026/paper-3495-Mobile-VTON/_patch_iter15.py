filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

old = 'w_max, w_min = 3.0, 1.5'
new = 'w_max, w_min = 2.5, 1.0'

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Weaker guidance decay (2.5->1.0)")
else:
    for line in content.split('\n'):
        if 'w_max' in line and 'w_min' in line:
            print(f"Found: {line.strip()}")
