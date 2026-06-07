filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace torch.zeros_like(d) in garment CFG with small non-zero constant
old_line = 'garment_features = [torch.cat([torch.zeros_like(d), d]) for d in garment_features]'
new_line = 'garment_features = [torch.cat([torch.full_like(d, 0.01), d]) for d in garment_features]'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Non-zero garment CFG (0.01 instead of 0)")

    # Verify
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            if 'torch.full_like' in line or 'torch.zeros_like' in line:
                if 'garment_features' in line:
                    print(f"  Line {i}: {line.strip()}")
else:
    print("ERROR: Could not find the zeros_like pattern")
    # Show context
    idx = content.find('garment_features = [torch.cat')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+120]}...")
