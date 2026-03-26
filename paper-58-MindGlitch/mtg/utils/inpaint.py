import torch
from diffusers import AutoPipelineForInpainting, FluxFillPipeline
import diffusers
import numpy as np
from PIL import Image
from PIL import ImageFilter


def get_inpainting_pipe(model_name: str, device):
    if "sdxl" in model_name.lower():
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        ).to(device)

        scheduler = "EulerDiscreteScheduler"
        scheduler_class_name = scheduler.split("-")[0]
        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
        scheduler = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)
    elif "flux" in model_name.lower():
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to(device)
    return pipe


def inpaint(pipe, init_image, mask, prompt="", **kwargs):
    dilate_mask = kwargs.get("dilate_masks", False)
    dilation_radius = kwargs.get("dilation_kernel_size", 3)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    if dilate_mask:
        mask = mask.filter(ImageFilter.MaxFilter(dilation_radius * 2 + 1))

    output = pipe(prompt=prompt, image=init_image, mask_image=mask, **kwargs)
    return output.images[0]
