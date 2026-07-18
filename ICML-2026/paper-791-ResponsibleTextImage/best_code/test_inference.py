import torch
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

pipe = PixArtAlphaPipeline.from_pretrained(
    "/paper_data",
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
)
pipe = pipe.to("cuda:0")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

ckpt = torch.load("/repo/checkpoints/external_concept_female.pt", map_location="cpu")
print(f"Checkpoint keys: {len(ckpt)}")
print(f"First 3 keys: {list(ckpt.keys())[:3]}")
print(f"Example shape: {ckpt[list(ckpt.keys())[0]].shape}")
print("Pipeline ready for inference!")
print(f"GPU memory after loading: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
