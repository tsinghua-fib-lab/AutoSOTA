filepath = '/repo/inference.py'

with open(filepath, 'r') as f:
    content = f.read()

# The pipe() call happens at line ~308. We need to modify the inference loop
# to do TTA: run twice (normal + flipped), average results.

# Find the pipe() call
old_code = '''                images = pipe(
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
                    cloth=batch["cloth_pure"].to(pipe_device),
                    image=(batch["image"].to(pipe_device) + 1.0) / 2.0,
                    ip_adapter_image=ip_imgs.to(pipe_device),
                    device=pipe_device,
                )[0]'''

new_code = '''                # TTA: run inference twice (normal + horizontal flip), average outputs
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
                    images.append(Image.fromarray(arr_avg))'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: TTA horizontal flip ensemble")
else:
    print("ERROR: Could not find pipe call pattern")
    idx = content.find('images = pipe(')
    if idx >= 0:
        print(f"Found at char {idx}: ...{content[idx:idx+200]}...")
    else:
        idx = content.find('images = pipe')
        if idx >= 0:
            print(f"Found 'images = pipe' at char {idx}: ...{content[idx:idx+200]}...")
