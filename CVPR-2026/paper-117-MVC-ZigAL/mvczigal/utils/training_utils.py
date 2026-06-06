from mvadapter.utils import get_orthogonal_camera, get_plucker_embeds_from_cameras_ortho


def run_pipeline(
    pipeline,
    prompts,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    negative_prompt,
    num_views,
    lora_scale,
    eta,
    device,
    return_dict=True,
    latents=None,
    zmv_sampling=False,
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 0, 0],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    if zmv_sampling:
        return pipeline.zmv_sampling(
            prompt=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_views,
            output_type="pt",
            return_dict=return_dict,
            cross_attention_kwargs={"scale": lora_scale},
            control_image=control_images,
            control_conditioning_scale=1.0,
            latents=latents,
            eta=eta,
        )
    else:
        return pipeline(
            prompt=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_views,
            output_type="pt",
            return_dict=return_dict,
            cross_attention_kwargs={"scale": lora_scale},
            control_image=control_images,
            control_conditioning_scale=1.0,
            latents=latents,
            eta=eta,
        )
