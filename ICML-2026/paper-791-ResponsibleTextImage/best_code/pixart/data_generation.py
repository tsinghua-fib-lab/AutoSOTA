import torch
import numpy as np
import os
import json
import zipfile
from tqdm import tqdm
from PIL import Image

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

"""This script creates the dataset to train concept vectors using PixArt-Alpha.
This includes generating images from prompts using PixArt-Alpha
and saving the images and labels in a folder."""


def update_concept_dict():
    concept_dict = ["female", "male", "young", "old", "white-race", "black-race", "anti-sexual", "smile", "jump", "van-ogh, melt, cartoon-style, violence, purple, red, silver, polar"]
    concept_dict = {c: i for i, c in enumerate(concept_dict)}
    return concept_dict


def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]


class PixArtDataCreator:
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.image_prompt = repeat_ntimes(cfg.image_prompt, cfg.num_samples)
        self.input_prompt_and_target_concept = repeat_ntimes(cfg.input_prompt_and_target_concept, cfg.num_samples)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"To create {len(self.image_prompt)} total number of samples in {cfg.root_dir}")

    def setup_pipeline(self):
        """Initialize PixArt-Alpha pipeline with optimizations"""
        # Load PixArt-Alpha pipeline
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None
        )

        # Use DPM++ solver for better quality/speed trade-off
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Move to device
        pipe = pipe.to(self.device)

        # Optimize for memory if using CUDA
        if self.device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()

        # Disable progress bar for cleaner output
        pipe.set_progress_bar_config(disable=True)

        return pipe

    def create_images(self, num_inference_steps=20, guidance_scale=4.5):
        """Generate images using PixArt-Alpha"""
        pipe = self.setup_pipeline()

        os.makedirs(self.root_dir, exist_ok=True)

        # Set random seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(42)

        for idx, prompt in tqdm(enumerate(self.image_prompt), total=len(self.image_prompt), desc="Generating images"):
            try:
                if isinstance(prompt, (list, tuple)):
                    # Handle negative prompts if provided
                    if len(prompt) == 2:
                        output = pipe(
                            prompt=prompt[0],
                            negative_prompt=prompt[1],
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            return_dict=True
                        )
                    else:
                        output = pipe(
                            prompt=prompt[0],
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            return_dict=True
                        )
                else:
                    output = pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        return_dict=True
                    )

                # Save the generated image
                image = output.images[0]  # Get first image from the output
                #image = output.images[0].resize((512, 512))
                image.save(os.path.join(self.root_dir, f"{idx}.jpg"))

                # Clear cache periodically to prevent memory issues
                if idx % 50 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating image {idx}: {str(e)}")
                continue

    def create_labels(self):
        """Save labels and concept dictionary"""
        os.makedirs(self.root_dir, exist_ok=True)

        # Save labels
        with open(os.path.join(self.root_dir, "labels.json"), "w") as f:
            json.dump(self.input_prompt_and_target_concept, f)

        # Save concept dictionary
        with open(os.path.join(self.root_dir, "concept_dict.json"), "w") as f:
            json.dump(update_concept_dict(), f)

    def zip_dataset(self):
        """Create a zip file of the entire dataset folder"""
        zip_filename = f"{self.root_dir}.zip"

        print(f"Creating zip file: {zip_filename}")

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in the dataset directory
            for root, dirs, files in os.walk(self.root_dir):
                for file in tqdm(files, desc="Zipping files"):
                    file_path = os.path.join(root, file)
                    # Create relative path for the zip file
                    arcname = os.path.relpath(file_path, os.path.dirname(self.root_dir))
                    zipf.write(file_path, arcname)

        # Get zip file size for confirmation
        zip_size = os.path.getsize(zip_filename)
        zip_size_mb = zip_size / (1024 * 1024)

        print(f"Dataset successfully zipped to: {zip_filename}")
        print(f"Zip file size: {zip_size_mb:.2f} MB")

    def run(self, num_inference_steps=20, guidance_scale=4.5, create_zip=True):
        """Run the complete data creation pipeline"""
        print("Creating labels...")
        self.create_labels()

        print("Generating images...")
        self.create_images(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

        print(f"Dataset creation complete! Images and labels saved in {self.root_dir}")

        # Create zip file of the dataset
        if create_zip:
            self.zip_dataset()

        print("All tasks completed successfully!")


class Cfg:
    """Configuration for single concept dataset"""
    root_dir = "datasets/person_pixart"
    num_samples = 3000  # Reduced for testing, increase as need
    image_prompt = [
        "a photo of a woman",
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo of a person", ["female"]],
        ],
    ]

    validation_prompt_and_concept = ["a photo of a person", ["female"]]


class CfgBatch:
    """Configuration for multiple concept dataset"""
    root_dir = "datasets/person_batch_pixart"
    num_samples = 100  # Reduced for testing, increase as needed

    image_prompt = [
        "a woman, high quality, detailed, photorealistic",
        "a man, high quality, detailed, photorealistic",
        "a young person, high quality, detailed, photorealistic",
        "an old person, high quality, detailed, photorealistic",
    ]

    input_prompt_and_target_concept = [
        [
            ["a person", ["woman"]],
        ],
        [
            ["a person", ["man"]],
        ],
        [
            ["a person", ["young"]],
        ],
        [
            ["a person", ["old"]],
        ],
    ]

    validation_prompt_and_concept = ["a person", ["woman", "man", "young", "old"]]


class CfgAdvanced:
    """Advanced configuration with negative prompts and style control"""
    root_dir = "datasets/person_advanced_pixart"
    num_samples = 50

    image_prompt = [
        ["a professional woman in business attire, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration"],
        ["a professional man in business attire, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration"],
        ["a young student, casual clothing, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration"],
        ["an elderly person, wise expression, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration"],
    ]

    input_prompt_and_target_concept = [
        [
            ["a person", ["woman"]],
        ],
        [
            ["a person", ["man"]],
        ],
        [
            ["a person", ["young"]],
        ],
        [
            ["a person", ["old"]],
        ],
    ]

    validation_prompt_and_concept = ["a person", ["woman", "man", "young", "old"]]


if __name__ == "__main__":
    # Choose configuration
    config = Cfg  # Change to CfgBatch or CfgAdvanced as needed

    # Create data creator instance
    creator = PixArtDataCreator(config)

    # Run with custom parameters
    creator.run(
        num_inference_steps=50,  # Increase for better quality (slower)
        guidance_scale=4.5,      # Adjust for prompt adherence vs creativity
        create_zip=True          # Set to False if you don't want to create zip file
    )
