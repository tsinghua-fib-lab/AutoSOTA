#!/usr/bin/env python3
"""
Stable Diffusion image generation helpers.

This module turns prompt text into images through a function-oriented API that
can be reused by a larger vision pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

# Example MODEL_PATH, you can change it to your own model path.
# DEFAULT_MODEL_PATH = (
#     "/root/shared-nvme/LLMs/stable-diffusion-xl-base-1.0/snapshots/"
#     "462165984030d82259a11f4367a4eed129e94a7b"
# )
DEFAULT_MODEL_PATH = "/path/to/stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_OUTPUT_ROOT = "SD_image"
DEFAULT_BATCH_SIZE = 1
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def load_prompts_from_text(text: str) -> list[str]:
    """
    Parse prompt text into a prompt list.

    Supported formats:
    - `index|prompt text`
    - plain prompt text per line
    """
    prompts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2:
                prompt = parts[1].strip()
                if prompt:
                    prompts.append(prompt)
        else:
            prompts.append(line)

    return prompts


def load_prompts_from_file(txt_file: str | Path) -> list[str]:
    with open(txt_file, "r", encoding="utf-8") as file_obj:
        return load_prompts_from_text(file_obj.read())


def get_max_existing_index(output_dir: str | Path) -> int:
    """
    Return the highest numeric PNG index already present in the output directory.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        return 0

    existing_indices: list[int] = []
    for file_path in output_dir_path.iterdir():
        if file_path.suffix.lower() != ".png":
            continue
        match = re.match(r"(\d+)\.png", file_path.name)
        if match:
            existing_indices.append(int(match.group(1)))

    return max(existing_indices) if existing_indices else 0


def derive_output_dir_name(txt_file: str | Path) -> str:
    """
    Build the default output directory name from a source text filename.
    """
    txt_filename = Path(txt_file).stem
    return txt_filename.replace("text", "img")


def build_output_dir(
    txt_file: str | Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> str:
    return str(Path(output_root) / derive_output_dir_name(txt_file))


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_diffusion_pipeline(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str | None = None,
    torch_dtype: torch.dtype = torch.float16,
) -> DiffusionPipeline:
    """
    Load and prepare a Stable Diffusion pipeline.
    """
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    return pipeline.to(device or get_default_device())


def generate_image(
    pipeline: DiffusionPipeline,
    prompt: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
):
    """
    Generate a single image from one prompt and return the Pillow image object.
    """
    with torch.no_grad():
        return pipeline(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        ).images[0]


def save_image(image, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return str(target)


def build_generation_record(
    index: int,
    prompt: str,
    image=None,
    output_path: str | None = None,
    error: str | None = None,
) -> dict:
    return {
        "index": index,
        "prompt": prompt,
        "image": image,
        "output_path": output_path,
        "error": error,
    }


def generate_images_from_prompts(
    prompts: list[str],
    pipeline: DiffusionPipeline,
    output_dir: str | Path | None = None,
    start_index: int = 1,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    save_outputs: bool = True,
) -> list[dict]:
    """
    Generate images for a prompt list and optionally save them to disk.
    """
    records: list[dict] = []
    resolved_output_dir = Path(output_dir) if output_dir is not None else None

    for prompt_offset, prompt in enumerate(prompts, start=start_index):
        try:
            image = generate_image(
                pipeline=pipeline,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
            )

            output_path = None
            if save_outputs and resolved_output_dir is not None:
                output_path = save_image(image, resolved_output_dir / f"{prompt_offset}.png")

            records.append(
                build_generation_record(
                    index=prompt_offset,
                    prompt=prompt,
                    image=image,
                    output_path=output_path,
                )
            )
        except Exception as exc:
            records.append(
                build_generation_record(
                    index=prompt_offset,
                    prompt=prompt,
                    error=str(exc),
                )
            )

    return records


def generate_images_from_prompt_file(
    txt_file: str | Path,
    pipeline: DiffusionPipeline | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    save_outputs: bool = True,
    resume: bool = True,
) -> dict:
    """
    Load prompts from a text file, generate images, and optionally resume from
    the last saved PNG index.
    """
    prompts = load_prompts_from_file(txt_file)
    output_dir = build_output_dir(txt_file, output_root=output_root)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_existing_index = get_max_existing_index(output_dir) if resume else 0
    start_index = max_existing_index + 1

    remaining_prompts = prompts[start_index - 1:] if start_index <= len(prompts) else []
    resolved_pipeline = pipeline or build_diffusion_pipeline(model_path=model_path)

    records = generate_images_from_prompts(
        prompts=remaining_prompts,
        pipeline=resolved_pipeline,
        output_dir=output_dir,
        start_index=start_index,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        save_outputs=save_outputs,
    )

    return {
        "txt_file": str(txt_file),
        "output_dir": output_dir,
        "total_prompts": len(prompts),
        "start_index": start_index,
        "generated_count": len(records),
        "records": records,
    }
