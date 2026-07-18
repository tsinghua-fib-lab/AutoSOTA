import torch
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

pipe = PixArtAlphaPipeline.from_pretrained(
    "/paper_data", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
)
pipe = pipe.to("cuda:0")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Quick test: generate a baseline image
generator = torch.Generator(device="cuda:0").manual_seed(42)
image = pipe(
    prompt="a photo of a doctor",
    num_inference_steps=20,
    generator=generator,
    guidance_scale=4.5,
).images[0]
os.makedirs("/repo/test_output", exist_ok=True)
image.save("/repo/test_output/baseline_doctor.png")
print("Baseline image saved!")
