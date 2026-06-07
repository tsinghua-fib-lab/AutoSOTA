filepath = '/repo/inference.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Build TTA code with proper indentation (16 spaces base, 20 for args)
ind16 = ' ' * 16
ind20 = ' ' * 20
tta_lines = [
    f'\n',
    f'{ind16}# TTA: run inference twice (normal + horizontal flip), average outputs\n',
    f'{ind16}person_image = (batch["image"].to(pipe_device) + 1.0) / 2.0\n',
    f'{ind16}cloth_img = batch["cloth_pure"].to(pipe_device)\n',
    f'\n',
    f'{ind16}# Normal pass\n',
    f'{ind16}images_normal = pipe(\n',
    f'{ind20}prompt_embeds=prompt_embeds,\n',
    f'{ind20}negative_prompt_embeds=negative_prompt_embeds,\n',
    f'{ind20}pooled_prompt_embeds=pooled_prompt_embeds,\n',
    f'{ind20}negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,\n',
    f'{ind20}num_inference_steps=args.num_inference_steps,\n',
    f'{ind20}generator=generator,\n',
    f'{ind20}height=args.height,\n',
    f'{ind20}width=args.width,\n',
    f'{ind20}guidance_scale=args.guidance_scale,\n',
    f'{ind20}text_embeds_cloth=prompt_embeds_c,\n',
    f'{ind20}negative_text_embeds_cloth=negative_prompt_embeds_c,\n',
    f'{ind20}cloth=cloth_img,\n',
    f'{ind20}image=person_image,\n',
    f'{ind20}ip_adapter_image=ip_imgs.to(pipe_device),\n',
    f'{ind20}device=pipe_device,\n',
    f'{ind16})[0]\n',
    f'\n',
    f'{ind16}# Flipped pass\n',
    f'{ind16}person_flipped = torch.flip(person_image, dims=[-1])\n',
    f'{ind16}cloth_flipped = torch.flip(cloth_img, dims=[-1])\n',
    f'{ind16}ip_flipped = torch.flip(ip_imgs.to(pipe_device), dims=[-1])\n',
    f'\n',
    f'{ind16}images_flipped = pipe(\n',
    f'{ind20}prompt_embeds=prompt_embeds,\n',
    f'{ind20}negative_prompt_embeds=negative_prompt_embeds,\n',
    f'{ind20}pooled_prompt_embeds=pooled_prompt_embeds,\n',
    f'{ind20}negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,\n',
    f'{ind20}num_inference_steps=args.num_inference_steps,\n',
    f'{ind20}generator=generator,\n',
    f'{ind20}height=args.height,\n',
    f'{ind20}width=args.width,\n',
    f'{ind20}guidance_scale=args.guidance_scale,\n',
    f'{ind20}text_embeds_cloth=prompt_embeds_c,\n',
    f'{ind20}negative_text_embeds_cloth=negative_prompt_embeds_c,\n',
    f'{ind20}cloth=cloth_flipped,\n',
    f'{ind20}image=person_flipped,\n',
    f'{ind20}ip_adapter_image=ip_flipped,\n',
    f'{ind20}device=pipe_device,\n',
    f'{ind16})[0]\n',
    f'\n',
    f'{ind16}# Flip back and average\n',
    f'{ind16}images = []\n',
    f'{ind16}for n, f in zip(images_normal, images_flipped):\n',
    f'{ind20}f_unflipped = f.transpose(Image.FLIP_LEFT_RIGHT)\n',
    f'{ind20}arr_n = np.array(n).astype(np.float32)\n',
    f'{ind20}arr_f = np.array(f_unflipped).astype(np.float32)\n',
    f'{ind20}arr_avg = ((arr_n + arr_f) / 2.0).astype(np.uint8)\n',
    f'{ind20}images.append(Image.fromarray(arr_avg))\n',
]

# Replace lines 306-325 (0-indexed)
new_lines = lines[:306] + tta_lines + lines[325:]

with open(filepath, 'w') as f:
    f.writelines(new_lines)

# Verify indentation around the critical area
with open(filepath) as f:
    vlines = f.readlines()

with open('/repo/_iter3_verify.txt', 'w') as out:
    out.write(f"Total lines: {len(vlines)}\n")
    for i in range(305, min(370, len(vlines))):
        l = vlines[i].rstrip('\n')
        # Show leading spaces count
        sp = len(l) - len(l.lstrip())
        out.write(f"L{i+1}: {sp}s |{l}\n")

print("PATCH APPLIED")
