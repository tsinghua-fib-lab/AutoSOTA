import torch
import numpy as np
import os
import json
import zipfile
from tqdm import tqdm
from PIL import Image

from diffusers import FluxPipeline

"""This script creates the dataset to train concept vectors using FLUX.1.
"""


def update_concept_dict():
    """Same concept dictionary as PixArt version for compatibility"""
    concept_dict = [
        "female", "male", "young", "old", 
        "white-race", "black-race", 
        "anti-sexual", "smile", "jump", "van-gogh", "violence", "purple", "red", "silver", "polar"
    ]
    concept_dict = {c: i for i, c in enumerate(concept_dict)}
    return concept_dict


def repeat_ntimes(x, n):
    """Repeat each item in list n times"""
    return [item for item in x for i in range(n)]


class FluxDataCreator:
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        self.image_prompt = repeat_ntimes(cfg.image_prompt, cfg.num_samples)
        self.input_prompt_and_target_concept = repeat_ntimes(
            cfg.input_prompt_and_target_concept, cfg.num_samples
        )
        self.model_variant = cfg.model_variant  # "schnell" or "dev"

        # ---- GPU-only: do not allow CPU fallback ----
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. FLUX must run on GPU only. "
                "Check your Slurm GPU allocation and PyTorch/CUDA setup."
            )
        self.device = "cuda"

        print(f"Using device: {self.device}")
        print(f"Using FLUX.1-{self.model_variant}")
        print(f"To create {len(self.image_prompt)} total number of samples in {cfg.root_dir}")


    def setup_pipeline(self):
        """Initialize FLUX pipeline with optimizations + HF auth via HF_TOKEN"""

        # Select model based on variant
        if self.model_variant == "schnell":
            model_id = "black-forest-labs/FLUX.1-schnell"
        elif self.model_variant == "dev":
            model_id = "black-forest-labs/FLUX.1-dev"
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}. Use 'schnell' or 'dev'")

        print(f"Loading {model_id}...")

        # ---- Hugging Face authentication via environment token ----
        # HF officially supports using the HF_TOKEN environment variable for auth.
        # If it's set, we pass it explicitly; otherwise we warn. :contentReference[oaicite:2]{index=2}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Using HF_TOKEN from environment for Hugging Face authentication.")
        else:
            print(
                "WARNING: HF_TOKEN is not set. "
                "If the model is gated (e.g., FLUX.1-dev), loading may fail with 401 Unauthorized.\n"
                "Make sure you have requested access on the model page and set HF_TOKEN before running."
            )

        # Load FLUX pipeline (as in the official model card) :contentReference[oaicite:3]{index=3}
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=hf_token  # may be None if not set; huggingface_hub will then fall back to other auth
        )

        # Move to device
        pipe = pipe.to(self.device)

        # Optimize for memory if using CUDA
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            print("Memory optimizations enabled (CPU offload, VAE slicing/tiling)")

        pipe.set_progress_bar_config(disable=True)
        return pipe


    def create_images(self, num_inference_steps=None, guidance_scale=None):
        """Generate images using FLUX
        
        Args:
            num_inference_steps: Number of denoising steps. 
                                 Default: 4 for schnell, 50 for dev
            guidance_scale: Guidance scale. 
                           Default: 0.0 for schnell, 3.5 for dev
        """
        pipe = self.setup_pipeline()
        os.makedirs(self.root_dir, exist_ok=True)

        # Set defaults based on model variant
        if self.model_variant == "schnell":
            num_inference_steps = num_inference_steps or 4
            guidance_scale = 0.0  # schnell is guidance-distilled
            max_seq_length = 256  # schnell limitation
        else:  # dev
            num_inference_steps = num_inference_steps or 50
            guidance_scale = guidance_scale or 3.5
            max_seq_length = 512

        print(f"Generation settings: steps={num_inference_steps}, "
              f"guidance_scale={guidance_scale}, max_seq_length={max_seq_length}")

        # Set random seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(42)

        for idx, prompt in tqdm(enumerate(self.image_prompt), 
                               total=len(self.image_prompt), 
                               desc="Generating images"):
            try:
                # Handle both simple strings and tuples (with negative prompts)
                if isinstance(prompt, (list, tuple)):
                    # FLUX supports negative prompts only in dev mode with true_cfg
                    if len(prompt) == 2 and self.model_variant == "dev":
                        output = pipe(
                            prompt=prompt[0],
                            negative_prompt=prompt[1],
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            max_sequence_length=max_seq_length,
                            generator=generator,
                            height=1024,
                            width=1024,
                            return_dict=True
                        )
                    else:
                        # Use first element only
                        output = pipe(
                            prompt=prompt[0] if isinstance(prompt, (list, tuple)) else prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            max_sequence_length=max_seq_length,
                            generator=generator,
                            height=1024,
                            width=1024,
                            return_dict=True
                        )
                else:
                    output = pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        max_sequence_length=max_seq_length,
                        generator=generator,
                        height=1024,
                        width=1024,
                        return_dict=True
                    )

                # Save the generated image
                image = output.images[0]
                image.save(os.path.join(self.root_dir, f"{idx}.jpg"))

                # Clear cache periodically to prevent memory issues
                if idx % 50 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError generating image {idx}: {str(e)}")
                continue

        print(f"\nSuccessfully generated {len(self.image_prompt)} images")

    def create_labels(self):
        """Save labels and concept dictionary (same format as PixArt version)"""
        os.makedirs(self.root_dir, exist_ok=True)

        # Save labels
        labels_path = os.path.join(self.root_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(self.input_prompt_and_target_concept, f)
        print(f"Saved labels to {labels_path}")

        # Save concept dictionary
        concept_dict_path = os.path.join(self.root_dir, "concept_dict.json")
        with open(concept_dict_path, "w") as f:
            json.dump(update_concept_dict(), f)
        print(f"Saved concept dictionary to {concept_dict_path}")

    def zip_dataset(self):
        """Create a zip file of the entire dataset folder"""
        zip_filename = f"{self.root_dir}.zip"

        print(f"\nCreating zip file: {zip_filename}")

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

    def run(self, num_inference_steps=None, guidance_scale=None, create_zip=True):
        """Run the complete data creation pipeline
        
        Args:
            num_inference_steps: Override default inference steps
            guidance_scale: Override default guidance scale
            create_zip: Whether to create a zip file of the dataset
        """
        print("\n" + "="*60)
        print("FLUX Dataset Generation Pipeline")
        print("="*60)
        
        print("\nStep 1/3: Creating labels...")
        self.create_labels()

        print("\nStep 2/3: Generating images...")
        self.create_images(
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        )

        print(f"\nDataset creation complete! Images and labels saved in {self.root_dir}")

        # Create zip file of the dataset
        if create_zip:
            print("\nStep 3/3: Creating zip archive...")
            self.zip_dataset()
        else:
            print("\nStep 3/3: Skipped (create_zip=False)")

        print("\n" + "="*60)
        print("All tasks completed successfully!")
        print("="*60)


# ============================================================================
# Configuration Classes
# ============================================================================

class CfgSchnellSingle:
    """Configuration for single concept dataset using FLUX.1-schnell (fast)"""
    model_variant = "schnell"  # Fast 4-step generation
    root_dir = "datasets/person_flux_schnell_female"
    num_samples = 100  # Start small for testing
    
    image_prompt = [
        "a photo of a nude person",
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a photo of a person", ["anti-sexual"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a photo of a person", ["anti_sexual"]]


class CfgDevSingle:
    """Configuration for single concept dataset using FLUX.1-dev (quality)"""
    model_variant = "dev"  # High-quality 50-step generation
    root_dir = "datasets/person_flux_dev"
    num_samples = 3000  # Same as original PixArt config
    
    image_prompt = [
        "a photo of a woman",
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a photo of a person", ["female"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a photo of a person", ["female"]]


class CfgSchnellBatch:
    """Configuration for multiple concepts using FLUX.1-schnell"""
    model_variant = "schnell"
    root_dir = "datasets/person_batch_flux_schnell"
    num_samples = 100  # 100 samples per concept
    
    image_prompt = [
        "a woman, high quality, detailed, photorealistic",
        "a man, high quality, detailed, photorealistic",
        "a young person, high quality, detailed, photorealistic",
        "an old person, high quality, detailed, photorealistic",
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a person", ["female"]],
        ],
        [
            ["a person", ["male"]],
        ],
        [
            ["a person", ["young"]],
        ],
        [
            ["a person", ["old"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a person", ["female", "male", "young", "old"]]


class CfgDevBatch:
    """Configuration for multiple concepts using FLUX.1-dev"""
    model_variant = "dev"
    root_dir = "datasets/person_batch_flux_dev"
    num_samples = 10  # 1000 samples per concept
    
    image_prompt = [
        "a woman, high quality, detailed, photorealistic",
        "a man, high quality, detailed, photorealistic",
        "a young person, high quality, detailed, photorealistic",
        "an old person, high quality, detailed, photorealistic",
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a person", ["female"]],
        ],
        [
            ["a person", ["male"]],
        ],
        [
            ["a person", ["young"]],
        ],
        [
            ["a person", ["old"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a person", ["female", "male", "young", "old"]]


class CfgDevAdvanced:
    """Advanced configuration with negative prompts (FLUX.1-dev only)"""
    model_variant = "dev"
    root_dir = "datasets/person_advanced_flux_dev"
    num_samples = 50
    
    # Negative prompts work better with dev model
    image_prompt = [
        ["a professional woman in business attire, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration, deformed"],
        ["a professional man in business attire, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration, deformed"],
        ["a young student, casual clothing, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration, deformed"],
        ["an elderly person, wise expression, high quality, detailed, photorealistic",
         "blurry, low quality, cartoon, anime, illustration, deformed"],
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a person", ["female"]],
        ],
        [
            ["a person", ["male"]],
        ],
        [
            ["a person", ["young"]],
        ],
        [
            ["a person", ["old"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a person", ["female", "male", "young", "old"]]


class CfgDevRace:
    """Configuration for race/ethnicity diversity using FLUX.1-dev"""
    model_variant = "dev"
    root_dir = "datasets/person_race_flux_dev"
    num_samples = 500  # 500 samples per race
    
    image_prompt = [
        "a person with light skin tone, photorealistic portrait",
        "a person with dark skin tone, photorealistic portrait",
    ]
    
    input_prompt_and_target_concept = [
        [
            ["a person", ["white-race"]],
        ],
        [
            ["a person", ["black-race"]],
        ],
    ]
    
    validation_prompt_and_concept = ["a person", ["white-race", "black-race"]]


class CfgDevFull:
    """Full configuration with all demographic attributes"""
    model_variant = "dev"
    root_dir = "datasets/person_full_flux_dev"
    num_samples = 200  # 200 samples each
    
    image_prompt = [
        # Gender
        "a woman, photorealistic portrait, high quality",
        "a man, photorealistic portrait, high quality",
        
        # Age
        "a young person, photorealistic portrait, high quality",
        "an elderly person, photorealistic portrait, high quality",
        
        # Race
        "a person with light skin tone, photorealistic portrait, high quality",
        "a person with dark skin tone, photorealistic portrait, high quality",
        
        # Expression
        "a person with a warm smile, photorealistic portrait, high quality",
    ]
    
    input_prompt_and_target_concept = [
        [["a person", ["female"]]],
        [["a person", ["male"]]],
        [["a person", ["young"]]],
        [["a person", ["old"]]],
        [["a person", ["white-race"]]],
        [["a person", ["black-race"]]],
        [["a person", ["smile"]]],
    ]
    
    validation_prompt_and_concept = [
        "a person", 
        ["female", "male", "young", "old", "white-race", "black-race", "smile"]
    ]


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Choose your configuration
    # ========================================================================
    
    # For FAST prototyping/testing: Use schnell variant
    # config = CfgSchnellSingle  # Single concept, 100 samples, 4 steps
    # config = CfgSchnellBatch   # Multi concept, 100 each, 4 steps
    
    # For HIGH QUALITY training data: Use dev variant
    # config = CfgDevBatch  # Multi concept, 1000 each, 50 steps (RECOMMENDED)
    config = CfgDevSingle      # Single concept, 5000 samples, 50 steps
    # config = CfgDevAdvanced    # With negative prompts
    # config = CfgDevRace        # Race/ethnicity diversity
    # config = CfgDevFull        # All attributes
    
    # ========================================================================
    # Create data creator instance
    # ========================================================================
    creator = FluxDataCreator(config)
    
    # ========================================================================
    # Run with custom parameters (optional)
    # ========================================================================
    creator.run(
        # Override defaults if needed:
        # num_inference_steps=50,  # Higher = better quality but slower
        # guidance_scale=3.5,      # Higher = stronger prompt adherence
        create_zip=True            # Set to False to skip compression
    )
    
    # ========================================================================
    # Expected output structure (compatible with your utils_data.py):
    # ========================================================================
    # datasets/person_batch_flux_dev/
    # ├── 0.jpg
    # ├── 1.jpg
    # ├── ...
    # ├── labels.json              # [[[prompt, [concepts]], ...]]
    # └── concept_dict.json        # {concept_name: index}
    # 
    # datasets/person_batch_flux_dev.zip  # Optional compressed archive
