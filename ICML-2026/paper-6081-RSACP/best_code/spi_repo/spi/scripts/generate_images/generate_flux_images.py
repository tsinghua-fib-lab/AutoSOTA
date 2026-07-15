# %% [markdown]
# # FLUX Image Generation from ImageNet Classes
# This notebook script generates synthetic images using the FLUX generative model.
# It uses class labels from the ImageNet dataset (via a pretrained ViT model) to generate prompts like _"A photo of a zebra"_.
# For each class, multiple images are generated using different random seeds.
# The images are saved in a structured directory per class.

import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import login

# %% [markdown]
# ## Step 1: Hugging Face Authentication
# We log into Hugging Face to access both the ViT classifier and FLUX generative model.

# %%
login(os.environ["HUGGING_FACE_TOKEN"])

# %% [markdown]
# ## Step 2: Load ImageNet Class Labels
# We use `google/vit-base-patch16-224` to get the ImageNet class labels (1000 classes).
# These labels will be used to construct prompts for image generation.

# %%
model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
vit = AutoModelForImageClassification.from_pretrained(model_name)

imagenet_classes = {i: vit.config.id2label[i] for i in range(len(vit.config.id2label))}

# %% [markdown]
# ## Step 3: Load FLUX Pipeline
# Load the FLUX model (from Hugging Face) and move it to GPU if available.

# %%
model_id = "black-forest-labs/FLUX.1-dev"
model_short_name = "flux"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

# %% [markdown]
# ## Step 4: Configuration
# Configure output paths and how many classes/images to generate.
# ⚠️ Make sure to update the output directory to your actual image save path if needed.

# %%
output_dir = f"./{model_short_name}/"
os.makedirs(output_dir, exist_ok=True)

start_index = 0
num_to_generate = 1000       # Number of classes to process
seeds_per_class = 2000       # Number of images to generate per class

# %% [markdown]
# ## Step 5: Image Generation Loop
# For each class label, generate multiple images using unique random seeds.
# Each generated image is saved in a class-specific subfolder.

# %%
for i in range(start_index, start_index + num_to_generate):
    if i >= len(imagenet_classes):
        print(f"Index {i} exceeds available class list.")
        continue

    class_name = imagenet_classes[i]
    prompt = f"A photo of a {class_name}"
    class_dir = os.path.join(output_dir, f"{i:03d}")
    os.makedirs(class_dir, exist_ok=True)

    print("=" * 70)
    print(f"Generating images for class {i}: {class_name}")
    print("=" * 70)

    for seed_offset in range(seeds_per_class):
        seed = i * seeds_per_class + seed_offset
        filename = os.path.join(class_dir, f"{i:03d}_{model_short_name}_{seed_offset:04d}.png")

        if os.path.exists(filename):
            print(f"Skipping existing file: {filename}")
            continue

        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
        image.save(filename)
