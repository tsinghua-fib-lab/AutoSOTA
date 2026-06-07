filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace guidance decay values
old = '''                    # guidance decays from w_max to w_min over the denoising trajectory
                    t_ratio = i / max(len(timesteps) - 1, 1)
                    w_max, w_min = 3.0, 1.5
                    w_t = w_max - (w_max - w_min) * t_ratio'''

new = '''                    # stronger guidance with decay: 4.0 -> 2.0
                    t_ratio = i / max(len(timesteps) - 1, 1)
                    w_max, w_min = 4.0, 2.0
                    w_t = w_max - (w_max - w_min) * t_ratio'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Stronger guidance decay (4.0->2.0)")
else:
    print("ERROR: Could not find target")
    for line in content.split('\n'):
        if 'w_max' in line and 'w_min' in line:
            print(f"  Found: {line.strip()}")
