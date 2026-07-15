# %% [markdown]
# # CLIP-Based Image Classification Evaluation on ImageNet
# This notebook uses OpenCLIP (an ImageNet-free variant of CLIP) to evaluate how well the model can match ImageNet images to their class labels.
# It loops over a subset of ImageNet classes, runs the model on each image, and saves the classification probability vectors.

# %% [markdown]
# ## Step 1: Imports and Dataset Loading
# Import required libraries, set up Hugging Face authentication, and load the ImageNet dataset.

# %%
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from tqdm import tqdm

hf_token = os.environ['HUGGING_FACE_TOKEN']

# %%
# ⚠️ Make sure to update the cache directory path to your actual data directory.
# ⚠️ Make sure to update the output path to where you want to save the probability files.

cache_dir = "./huggingface_cache"
ds_name = "ILSVRC/imagenet-1k"
scores_out_dir = "./scores/models/"
dataset = load_dataset(ds_name, cache_dir=cache_dir, token=hf_token)

# %% [markdown]
# ## Step 2: Load CLIP Model and Define Class Labels
# Load the OpenCLIP model and prepare the class names for classification comparison.

# %%
LABEL_MAP = {}

model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # ImageNet-free OpenCLIP checkpoint
clip_model = CLIPModel.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
processor = CLIPProcessor.from_pretrained(model_id)
class_names = [v for v in LABEL_MAP.values()]

# %% [markdown]
# ## Step 3: Configure Processing Range and Evaluation Mode
# Define which classes to process and whether to enable debugging mode (processes only a few images).

# %%
debug = False
subset = "validation"

start_class = 0
end_class = 1000
print("=======================================================================")
print(f"Processing classes {start_class} to {end_class}")
print("=======================================================================")

# %% [markdown]
# ## Step 4: Iterate Through Classes and Process Images
# For each class, find all matching images in the dataset. Then, for each image:
# - Run it through the CLIP model along with all class label prompts
# - Calculate softmax probabilities
# - Store the probability vector for later analysis

# %%
for requested_class in range(start_class, end_class):
    print(f"Processing class {requested_class}")
    indexes = np.where(np.array(dataset[subset]["label"]) == requested_class)[0]

    probs_list = []
    for i in tqdm(range(len(indexes))):
        image = dataset[subset][int(indexes[i])]["image"]
        inputs = processor(
            text=class_names,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_classes]
            probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
            probs_list.append(probs)
        best_label = class_names[np.argmax(probs)]
        if debug and i > 1:
            break

    probs_nd = np.array(probs_list)

    
    output_base_path = f"{scores_out_dir}/{model_id.split('/')[-1]}/{requested_class:04d}"
    if not os.path.exists(output_base_path):
        print(f"Creating directory {output_base_path}")
        os.makedirs(output_base_path)

    np.save(os.path.join(output_base_path, "probs.npy"), probs_nd)
    print(f"Saved probs to {output_base_path}/probs.npy")

    if debug:
        break

# %% [markdown]
# ## Done!
# The softmax probability vectors for each image are saved to disk.
# These can be analyzed later to evaluate CLIP's classification behavior without traditional training.
