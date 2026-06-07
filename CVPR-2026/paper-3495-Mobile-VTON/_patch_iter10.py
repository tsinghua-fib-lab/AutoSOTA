filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Add multi-scale garment feature weighting after garment_features are extracted
# Find the section where garment_features is converted to list

old = '''                garment_features = list(garment_features)
                garment_features = [torch.cat([d, d], dim=concat_dim) for d in garment_features]'''

new = '''                garment_features = list(garment_features)
                # Multi-scale weighting: fine scales get higher weight in late denoising
                n_scales = len(garment_features)
                t_ratio = i / max(len(timesteps) - 1, 1)
                for s in range(n_scales):
                    w = 1.0 + 0.25 * (s / max(n_scales - 1, 1)) * t_ratio
                    garment_features[s] = garment_features[s] * w
                garment_features = [torch.cat([d, d], dim=concat_dim) for d in garment_features]'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Multi-scale garment feature weighting")
else:
    print("ERROR: Could not find target")
    idx = content.find('garment_features = list(garment_features)')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+200]}...")
