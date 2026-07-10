#!/usr/bin/env python3
"""
Generate images from BGPS-generated prompts using Stable Diffusion 1.5.
"""
import os
import sys
import json
import glob
import argparse
import csv
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=str, required=True, help="Directory containing BGPS output prompts")
    parser.add_argument("--output-dir", type=str, default="output/evaluation_images")
    parser.add_argument("--sd-model", type=str, default="/models/stable-diffusion-v1-5")
    parser.add_argument("--n-images", type=int, default=10, help="Images per prompt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prompt-primer", type=str, default="A photo of a person working as a",
                        help="Prefix added to each prompt for image generation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load SD pipeline
    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    print("SD pipeline loaded.")

    # Collect prompts
    prompts = []
    # Look for CSV files with prompts
    csv_files = sorted(glob.glob(os.path.join(args.prompts_dir, "**", "*.csv"), recursive=True))
    for cf in csv_files:
        with open(cf) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    prompts.append(row[0].strip('"').strip())

    # Also look for JSON sidecar files
    json_files = sorted(glob.glob(os.path.join(args.prompts_dir, "**", "seed_*.json"), recursive=True))
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
            prompt = data.get("generated_prompt", "")
            if prompt and prompt not in prompts:
                prompts.append(prompt)

    # Deduplicate
    prompts = list(dict.fromkeys(prompts))

    print("Found", len(prompts), "unique prompts")
    if len(prompts) == 0:
        print("No prompts found!")
        return

    # Generate images
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator(device=device)

    all_metadata = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = os.path.join(args.output_dir, f"prompt_{prompt_idx:04d}")
        os.makedirs(prompt_dir, exist_ok=True)

        # Save prompt text (full with primer)
        full_prompt = f"{args.prompt_primer} {prompt}"
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(full_prompt)

        # Generate images
        print(f"[{prompt_idx+1}/{len(prompts)}] Generating: {full_prompt}")

        for img_idx in range(args.n_images):
            seed = 1000 * prompt_idx + img_idx
            generator.manual_seed(seed)

            with torch.no_grad():
                image = pipe(
                    prompt=full_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                ).images[0]

            img_path = os.path.join(prompt_dir, f"img_{img_idx:03d}.png")
            image.save(img_path)
            all_metadata.append({
                "prompt": full_prompt,
                "prompt_idx": prompt_idx,
                "image_idx": img_idx,
                "seed": seed,
                "path": img_path,
            })

    # Save metadata
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Generated {len(all_metadata)} images for {len(prompts)} prompts.")
    print(f"Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
