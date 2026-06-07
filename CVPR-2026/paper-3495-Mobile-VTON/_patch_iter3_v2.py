filepath = '/repo/inference.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Replace lines 307-325 (0-indexed: 306-324) with TTA code
# Line 307 is empty, 308-324 is the pipe() call, 325 is empty

tta_code = '''
                # TTA: run inference twice (normal + horizontal flip), average outputs
                person_image = (batch["image"].to(pipe_device) + 1.0) / 2.0
                cloth_img = batch["cloth_pure"].to(pipe_device)

                # Normal pass
                images_normal = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    text_embeds_cloth=prompt_embeds_c,
                    negative_text_embeds_cloth=negative_prompt_embeds_c,
                    cloth=cloth_img,
                    image=person_image,
                    ip_adapter_image=ip_imgs.to(pipe_device),
                    device=pipe_device,
                )[0]

                # Flipped pass: flip person image, cloth, and ip_adapter horizontally
                person_flipped = torch.flip(person_image, dims=[-1])
                cloth_flipped = torch.flip(cloth_img, dims=[-1])
                ip_flipped = torch.flip(ip_imgs.to(pipe_device), dims=[-1])

                images_flipped = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    text_embeds_cloth=prompt_embeds_c,
                    negative_text_embeds_cloth=negative_prompt_embeds_c,
                    cloth=cloth_flipped,
                    image=person_flipped,
                    ip_adapter_image=ip_flipped,
                    device=pipe_device,
                )[0]

                # Flip back and average
                images = []
                for n, f in zip(images_normal, images_flipped):
                    f_unflipped = f.transpose(Image.FLIP_LEFT_RIGHT)
                    # Average pixel values
                    arr_n = np.array(n).astype(np.float32)
                    arr_f = np.array(f_unflipped).astype(np.float32)
                    arr_avg = ((arr_n + arr_f) / 2.0).astype(np.uint8)
                    images.append(Image.fromarray(arr_avg))
'''

new_lines = lines[:306] + [tta_code] + lines[325:]

with open(filepath, 'w') as f:
    f.writelines(new_lines)

# Verify
with open(filepath) as f:
    vlines = f.readlines()
print(f"Lines: {len(vlines)} (was {len(lines)}, +{len(tta_code.split(chr(10)))-3})")
for i, l in enumerate(vlines):
    if 'TTA' in l or 'person_image' in l and '=' in l:
        print(f"  L{i+1}: {l.strip()}")
print("PATCH APPLIED")
