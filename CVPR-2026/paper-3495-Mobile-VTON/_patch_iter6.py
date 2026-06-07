filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Add IP-adapter scale multiplier after the encoder_hid_proj
old = '''            image_embeds = self.denoiser.encoder_hid_proj(image_embeds).to(dtype=prompt_embeds.dtype, device=device)'''

new = '''            image_embeds = self.denoiser.encoder_hid_proj(image_embeds).to(dtype=prompt_embeds.dtype, device=device)
            image_embeds = image_embeds * 1.5  # Boost IP-adapter garment feature influence'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: IP-adapter scale 1.5x")
else:
    print("ERROR: Could not find target")
    idx = content.find('encoder_hid_proj(image_embeds)')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+120]}...")
