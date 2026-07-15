# %% [markdown]
# ## Step 1: Import Required Libraries
# Importing all necessary libraries for model loading, image processing, visualization, and iteration.

import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    AutoModelForImageClassification
)
from huggingface_hub import login

# %% [markdown]
# ## Step 2: Set Configuration Parameters
# Setting constants such as debug mode and class range for processing.

DEBUG = False
START_CLASS = 0
END_CLASS = 1000

# %% [markdown]
# ## Step 3: Authenticate with Hugging Face Hub
# Logging into Hugging Face to access the required models.

login(os.environ["HUGGING_FACE_TOKEN"])

# %% [markdown]
# ## Step 4: Set Device for Computation
# Determining whether to use GPU (if available) or CPU.

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=======================================================================")
print(f"Processing classes {START_CLASS} to {END_CLASS}")
print("=======================================================================")
print(f"Using device: {'GPU - ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# %% [markdown]
# ## Step 5: Load Models
# Loading a ViT classification model and a CLIP model for image-text similarity scoring.

vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
imagenet_classes = [vit_model.config.id2label[i] for i in range(len(vit_model.config.id2label))]

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# %% [markdown]
# ## Step 6: Set Global Input Directory
# Define a global input directory to be reused for each label. Update the label-specific directory in the loop.
# ⚠️ Make sure to update the `BASE_INPUT_DIR` to your actual image directory path.

BASE_INPUT_DIR = "./flux/"
BASE_OUTPUT_DIR = "./scores"

# %% [markdown]
# ## Step 7: Process Each Image Class
# For each label, iterate through image files, run them through the CLIP model, and collect probabilities.

for label in range(START_CLASS, END_CLASS):
    print(f"\nProcessing class: {label}")
    probs_list = []

    input_dir = os.path.join(BASE_INPUT_DIR, f"{label:03d}")
    output_dir = f"{BASE_OUTPUT_DIR}/models/{clip_model_id.split('/')[-1]}/{label:04d}"

    if os.path.exists(output_dir):
        print(f"Skipping class {label} because it already exists")
        continue

    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    if not image_files:
        print(f"No images found for class {label}")
        continue

    for image_file in tqdm(image_files, desc=f"Class {label:04d}"):
        image = Image.open(image_file)

        inputs = clip_processor(
            text=imagenet_classes,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_classes]
            probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
            probs_list.append(probs)

        if DEBUG:
            break

    if DEBUG:
        break
