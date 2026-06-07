filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Find the post-denoising latent splitting code and add garment injection
# Original:
#     latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
#     latents = (latents / 1.5305) + 0.0609

old_code = '''            latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            latents = (latents / 1.5305) + 0.0609'''

new_code = '''            # Ground-truth garment injection: blend denoised garment region with encoded cloth
            person_latents, denoised_garment = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)
            # Blend garment: 70% ground truth, 30% denoised (preserves some learned adaptation)
            blend_ratio = 0.7
            injected_garment = blend_ratio * cloth + (1 - blend_ratio) * denoised_garment
            latents = person_latents
            latents = (latents / 1.5305) + 0.0609'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Ground-truth garment injection (70% GT, 30% denoised)")

    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            if 'garment injection' in line or 'injected_garment' in line:
                print(f"  Line {i}: {line.strip()}")
else:
    print("ERROR: Could not find target pattern")
    idx = content.find('latents.split(latents.shape')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+150]}...")
