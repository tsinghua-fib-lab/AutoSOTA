filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace current linear decay with triangular Beta-CFG schedule
old = '''                    # guidance decays from w_max to w_min over the denoising trajectory
                    t_ratio = i / max(len(timesteps) - 1, 1)
                    w_max, w_min = 3.0, 1.5
                    w_t = w_max - (w_max - w_min) * t_ratio'''

new = '''                    # Beta-CFG triangular schedule: peaks at mid-trajectory
                    t_ratio = i / max(len(timesteps) - 1, 1)
                    w_min, w_peak = 1.5, 3.5
                    # Triangular: 1.5 -> 3.5 -> 1.5
                    if t_ratio < 0.5:
                        w_t = w_min + (w_peak - w_min) * (t_ratio * 2.0)
                    else:
                        w_t = w_peak - (w_peak - w_min) * ((t_ratio - 0.5) * 2.0)'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Triangular Beta-CFG schedule (1.5→3.5→1.5)")
else:
    print("ERROR: Could not find guidance decay pattern")
    idx = content.find('guidance decays')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+200]}...")
    else:
        # Show what's around the guidance code
        idx = content.find('w_t =')
        if idx >= 0:
            print(f"Found w_t at char {idx}: ...{content[idx-100:idx+150]}...")
