import argparse
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from mvczigal.diffusers_patch.lcm_inv_scheduler import LCMInverseScheduler
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    lora_rank,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
    zmv_sampling=False,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterT2MVSDXLPipeline
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    inv_scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "ddim":
        scheduler_class = DDIMScheduler
        inv_scheduler_class = DDIMInverseScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler
        inv_scheduler_class = LCMInverseScheduler

    scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    if zmv_sampling:
        pipe.inv_scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=inv_scheduler_class,
        )
    pipe.scheduler = scheduler

    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
    )
    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "to_k_mv", "to_q_mv", "to_v_mv", "to_out_mv.0"],
        )
        pipe.unet.add_adapter(unet_lora_config)
        unet_lora_layers = load_file(lora_model)
        incompatible_keys = set_peft_model_state_dict(pipe.unet, unet_lora_layers)
        if not getattr(incompatible_keys, 'unexpected_keys', None):
            print(f"Loaded LoRA weights from {lora_model}")
        else:
            print(f"{lora_model} has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


def run_pipeline(
    pipe,
    num_views,
    text,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    negative_prompt,
    lora_scale=1.0,
    device="cuda",
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

    pipe_kwargs = {}
    if seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    if zmv_sampling:
        images = pipe.zmv_sampling(
            text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"scale": lora_scale},
            eta=1.0,
            **pipe_kwargs,
        ).images
    else:
        images = pipe(
            text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"scale": lora_scale},
            eta=1.0,
            **pipe_kwargs,
        ).images

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--unet_model", type=str, default="latent-consistency/lcm-sdxl")
    parser.add_argument("--scheduler", type=str, default="lcm")
    parser.add_argument("--lora_model", type=str, default="checkpoint/mvczigal_lcm_sdxl_lora.safetensors")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--zmv_sampling", type=bool, default=False)
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        lora_rank=args.lora_rank,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
        zmv_sampling=args.zmv_sampling,
    )
    images = run_pipeline(
        pipe,
        num_views=args.num_views,
        text=args.text,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        lora_scale=args.lora_scale,
        device=args.device,
        zmv_sampling=args.zmv_sampling,
    )
    # for idx, image in enumerate(images):
    #     image.save(args.output + f"_{idx}.png")
    make_image_grid(images, rows=1).save(args.output)
