import re

filepath = '/repo/Mobile_VTON/pipelines/tryon_pipeline_full_cat.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace the static CFG application with time-dependent guidance
old_cfg = '''                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)'''

new_cfg = '''                # perform guidance with time-dependent decay
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # guidance decays from w_max to w_min over the denoising trajectory
                    t_ratio = i / max(len(timesteps) - 1, 1)
                    w_max, w_min = 3.0, 1.5
                    w_t = w_max - (w_max - w_min) * t_ratio
                    noise_pred = noise_pred_uncond + w_t * (noise_pred_text - noise_pred_uncond)'''

if old_cfg in content:
    content = content.replace(old_cfg, new_cfg)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Guidance decay (3.0 → 1.5) over denoising trajectory")
else:
    print("ERROR: Could not find the CFG pattern to replace")
    # Search for what it looks like
    idx = content.find('noise_pred_uncond + self.guidance_scale')
    if idx >= 0:
        print(f"Found near char {idx}: ...{content[idx-50:idx+100]}...")
