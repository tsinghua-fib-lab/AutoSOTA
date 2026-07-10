# train_if_attribute_classifier.py
# Trains an attribute classifier from DeepFloyd IF Stage-I UNet mid-block activations
# and saves sample generated images during training/testing as a sanity check.
#
# References:
# - DeepFloyd IF usage and encode_prompt(): https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if
# - Accept license on: DeepFloyd/IF-I-XL-v1.0 (or L/M) before running.

import os
import hashlib  # ADDED: for short prompt hashes
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline
import tqdm
import numpy as np
import random
import yaml
from PIL import Image  # ADDED

from simple_parsing import Serializable


class ClassifierTrainConfig(Serializable):
    model_path: str = "DeepFloyd/IF-I-XL-v1.0"
    """DeepFloyd IF Stage-I checkpoint (e.g., IF-I-XL-v1.0 / IF-I-L-v1.0 / IF-I-M-v1.0)"""

    train_samples: int = 32
    test_samples: int = 32
    macro_batch: int = 4

    num_epochs: int = 30
    save_dir: str = "./if_h_classifier2"
    attribute: str = "gender"

    # IF sampling
    num_timesteps: int = 48      # match SD example defaults; can set 27/30/50 etc.
    guidance_scale: float = 7.0  # ensure CFG path is taken so batch is doubled
    variant: str = "fp16"
    dtype: str = "float16"       # cast for pipeline weights

    # ADDED: image saving controls
    save_images: bool = True
    max_images_per_epoch: int = 20  # per split (train/test)
    images_subdir: str = "samples"

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)


def setup_if_stage1_pipeline(model_path, variant="fp16", dtype=torch.float16, device="cuda"):
    """
    Stage-I IF pipeline. We use IFPipeline specifically for DeepFloyd IF models.
    We'll call encode_prompt(prompt) and pass prompt/negative embeddings during __call__.
    """
    pipe = DiffusionPipeline.from_pretrained(model_path, variant=variant, torch_dtype=dtype)
    # Disable the NSFW checker for raw feature collection (parity with your SD code).
    pipe.safety_checker = None
    # Disable progress bars
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


class DynamicAttributeDatasetIF(Dataset):
    """
    Generates prompts on-the-fly, runs IF Stage-I, captures UNet mid_block activations,
    and returns a representative final image (PIL) for optional saving.
    We only keep the *conditional* (CFG) half of the batch for activations.
    """
    def __init__(self, attribute, prompt_pool, categories, num_samples, pipe,
                 num_timesteps=48, macro_batch_size=4, guidance_scale=7.0):
        self.attribute = attribute  # 'age', 'gender', or 'race'
        self.prompt_pool = prompt_pool
        self.categories = categories
        self.num_samples = num_samples
        self.num_timesteps = num_timesteps
        self.macro_batch_size = macro_batch_size
        self.pipe = pipe
        self.guidance_scale = guidance_scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Select the target attribute value + base prompt template
        attribute_value, prompt_template = self.prompt_pool[idx % len(self.prompt_pool)]

        # Randomize non-target attributes
        gender = random.choice(self.categories['gender'])
        race = random.choice(self.categories['race'])
        age = random.choice(self.categories['age'])

        if self.attribute == 'age':
            age = attribute_value
            label = self.categories['age'].index(attribute_value)
        elif self.attribute == 'gender':
            gender = attribute_value
            label = self.categories['gender'].index(attribute_value)
        elif self.attribute == 'race':
            race = attribute_value
            label = self.categories['race'].index(attribute_value)

        combined_prompt = f"A photo of a {gender} {race} {age} person"

        # Collect mid-block outputs for each denoising step
        middle_block_outputs = []

        def hook_fn(module, input, output):
            # CFG doubles batch: [uncond, cond]. Keep conditional half like the SD code.
            b = output.shape[0]
            middle_block_outputs.append(output[b // 2 :, ...].to(torch.float32).detach().cpu())

        hook = self.pipe.unet.mid_block.register_forward_hook(hook_fn)
        middle_block_outputs.clear()

        # Run Stage-I IF; request PIL so we can optionally save an image outside
        with torch.no_grad():
            result = self.pipe(
                prompt=[combined_prompt]*self.macro_batch_size,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_timesteps,
                output_type="pil",  # ADDED: return PIL images
            )
            imgs = result.images  # list[PIL.Image], length = macro_batch_size

        hook.remove()

        # [num_steps, macro_batch, C, H, W]
        latents = torch.stack(middle_block_outputs)
        latents = latents.unsqueeze(0)  # add batch dim for DataLoader compatibility

        # Return a representative image (first in batch)
        sample_img = imgs[0] if isinstance(imgs, list) and len(imgs) > 0 else None
        return latents, torch.tensor(label, dtype=torch.long), combined_prompt, sample_img


class AttributeClassifierIF(nn.Module):
    """
    One linear head per (reverse) timestep, like the SD version.
    We infer (C, H, W) from a bootstrap sample so this works for any IF Stage-I size.
    """
    def __init__(self, num_classes: int, num_timesteps: int, feature_shape: tuple):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.forward_timesteps = list(range(1, num_timesteps + 1))
        self.reversed_timesteps = list(reversed(self.forward_timesteps))

        c, h, w = feature_shape  # (C, H, W)
        in_dim = c * h * w
        self.linears = nn.ModuleList([nn.Linear(in_dim, num_classes) for _ in range(self.num_timesteps)])

    def forward(self, x, t):
        # x: [1, num_steps, macro_batch, C, H, W]  (batch=1 per DataLoader)
        reversed_t = self.reversed_timesteps[self.forward_timesteps.index(t)]
        x_t = x[:, reversed_t-1, ...]                    # [1, macro_batch, C, H, W]
        x_t = x_t.reshape(-1, *x_t.size()[-3:])          # [macro_batch, C, H, W]
        x_t = x_t.reshape(x_t.size(0), -1)                  # [macro_batch, in_dim]
        return self.linears[t - 1](x_t)                  # [macro_batch, num_classes]


def bootstrap_feature_shape(pipe, num_timesteps: int, macro_batch_size: int, guidance_scale: float):
    """
    Do a single forward pass on a throwaway prompt to discover (C, H, W)
    of mid_block activations for this checkpoint.
    """
    tmp_outputs = []

    def _hook(_, __, output):
        b = output.shape[0]
        tmp_outputs.append(output[b // 2 :, ...].detach().cpu().to(torch.float32))

    hook = pipe.unet.mid_block.register_forward_hook(_hook)
    with torch.no_grad():
        _ = pipe(
            prompt="a test photo",
            guidance_scale=guidance_scale,
            num_inference_steps=num_timesteps,
            output_type="pil",  # changed to "pil" (still fine for hooking)
        ).images
    hook.remove()

    # tmp_outputs is a list with length=num_timesteps; take first element to get shape
    sample = tmp_outputs[0]  # [macro_batch, C, H, W]
    _, C, H, W = sample.shape
    return (C, H, W)


def _short_hash(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:8]


def _maybe_save_image(img: Image.Image, root: str, split: str, epoch: int, label_idx: int,
                      attribute_name: str, prompt: str, counter: int):
    """Save a PIL image to disk with a descriptive filename."""
    split_dir = os.path.join(root, "samples", split, f"epoch_{epoch:03d}")
    os.makedirs(split_dir, exist_ok=True)
    fname = f"{attribute_name}_label{label_idx}_n{counter:04d}_p{_short_hash(prompt)}.jpg"
    path = os.path.join(split_dir, fname)
    # Ensure RGB for JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=92)


def train_main_IF(args):
    train_args = ClassifierTrainConfig()

    # with open(args.h_classifier_config_path, "r") as f:
    #     cfg_dict = yaml.safe_load(f)
    # train_args.update(cfg_dict)

    categories = {
        "gender": ["male", "female"],
        "age": ["child", "adult", "old"],
        "race": ["White", "Black", "Asian", "Indian"]
    }

    prompt_pool = {
        "gender": [
            ("male", "A photo of a male person"),
            ("female", "A photo of a female person"),
        ],
        "age": [
            ("child", "A photo of a child"),
            ("adult", "A photo of an adult"),
            ("old", "A photo of an old person")
        ],
        "race": [
            ("White", "A photo of a White person"),
            ("Black", "A photo of a Black person"),
            ("Asian", "A photo of an Asian person"),
            ("Indian", "A photo of an Indian person")
        ],
    }
    prompt_pool = prompt_pool[train_args.attribute]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(train_args.dtype).lower() == "float16" else torch.float32

    pipe = setup_if_stage1_pipeline(
        train_args.model_path, variant=train_args.variant, dtype=dtype, device=device
    )

    # Bootstrap feature shape for this checkpoint + settings
    feat_C_H_W = bootstrap_feature_shape(
        pipe,
        num_timesteps=train_args.num_timesteps,
        macro_batch_size=train_args.macro_batch,
        guidance_scale=train_args.guidance_scale,
    )

    # Datasets / loaders
    train_dataset = DynamicAttributeDatasetIF(
        train_args.attribute, prompt_pool, categories, train_args.train_samples, pipe,
        num_timesteps=train_args.num_timesteps, macro_batch_size=train_args.macro_batch,
        guidance_scale=train_args.guidance_scale
    )
    test_dataset = DynamicAttributeDatasetIF(
        train_args.attribute, prompt_pool, categories, train_args.test_samples, pipe,
        num_timesteps=train_args.num_timesteps, macro_batch_size=train_args.macro_batch,
        guidance_scale=train_args.guidance_scale
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    num_classes = len(categories[train_args.attribute])
    model = AttributeClassifierIF(
        num_classes=num_classes,
        num_timesteps=train_args.num_timesteps,
        feature_shape=feat_C_H_W,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)

    # Ensure base save directories exist
    os.makedirs(f"{train_args.save_dir}/{train_args.attribute}", exist_ok=True)
    if train_args.save_images:
        os.makedirs(os.path.join(train_args.save_dir, train_args.images_subdir, "train"), exist_ok=True)
        os.makedirs(os.path.join(train_args.save_dir, train_args.images_subdir, "test"), exist_ok=True)

    for epoch in range(train_args.num_epochs):
        model.train()
        train_loss = [0.0 for _ in range(train_args.num_timesteps)]
        train_acc = [0.0 for _ in range(train_args.num_timesteps)]

        # ADDED: counters to cap how many images we save per epoch
        train_image_count = 0
        test_image_count = 0

        for latents, labels, prompts, sample_img in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Training"):
            latents, labels = latents.to(device), labels.to(device)

            for time_id, t in enumerate(range(1, train_args.num_timesteps + 1)):
                optimizer.zero_grad()
                outputs = model(latents, t)  # [macro_batch, num_classes]
                # target = labels.unsqueeze(1).repeat(train_loader.dataset.macro_batch_size, 1).view(-1)
                target = labels.view(1).repeat(train_loader.dataset.macro_batch_size)
                loss = criterion(outputs.view(-1, num_classes), target)
                loss.backward()
                optimizer.step()

                train_loss[time_id] += loss.item()
                train_acc[time_id] += (outputs.argmax(1) == target).sum().item()

            # ADDED: save a representative generated image
            if train_args.save_images and train_image_count < train_args.max_images_per_epoch:
                # prompts is a list[str] with batch=1 => length 1
                prompt_str = prompts[0] if isinstance(prompts, list) else str(prompts)
                img = sample_img[0] if isinstance(sample_img, list) else sample_img
                if isinstance(img, Image.Image):
                    _maybe_save_image(
                        img=img,
                        root=train_args.save_dir,
                        split="train",
                        epoch=epoch+1,
                        label_idx=int(labels.item()),
                        attribute_name=train_args.attribute,
                        prompt=prompt_str,
                        counter=train_image_count,
                    )
                    train_image_count += 1

        denom = len(train_loader) * train_loader.dataset.macro_batch_size
        train_loss = [l / denom for l in train_loss]
        train_acc = [a / denom for a in train_acc]

        model.eval()
        test_loss = [0.0 for _ in range(train_args.num_timesteps)]
        test_acc = [0.0 for _ in range(train_args.num_timesteps)]

        with torch.no_grad():
            for latents, labels, prompts, sample_img in tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Testing"):
                latents, labels = latents.to(device), labels.to(device)

                for time_id, t in enumerate(range(1, train_args.num_timesteps + 1)):
                    outputs = model(latents, t)
                    # target = labels.unsqueeze(1).repeat(test_loader.dataset.macro_batch_size, 1).view(-1)
                    target = labels.view(1).repeat(test_loader.dataset.macro_batch_size)
                    loss = criterion(outputs.view(-1, num_classes), target)

                    test_loss[time_id] += loss.item()
                    test_acc[time_id] += (outputs.argmax(1) == target).sum().item()

                # ADDED: save a representative test image
                if train_args.save_images and test_image_count < train_args.max_images_per_epoch:
                    prompt_str = prompts[0] if isinstance(prompts, list) else str(prompts)
                    img = sample_img[0] if isinstance(sample_img, list) else sample_img
                    if isinstance(img, Image.Image):
                        _maybe_save_image(
                            img=img,
                            root=train_args.save_dir,
                            split="test",
                            epoch=epoch+1,
                            label_idx=int(labels.item()),
                            attribute_name=train_args.attribute,
                            prompt=prompt_str,
                            counter=test_image_count,
                        )
                        test_image_count += 1

        denom_test = len(test_loader) * test_loader.dataset.macro_batch_size
        test_loss = [l / denom_test for l in test_loss]
        test_acc = [a / denom_test for a in test_acc]

        print(f"Epoch {epoch+1}: "
              f"train loss {np.mean(train_loss):.4f}, train acc {np.mean(train_acc):.4f}, "
              f"test loss {np.mean(test_loss):.4f}, test acc {np.mean(test_acc):.4f}")

        # model checkpoints
        if epoch % 10 == 0 or epoch == train_args.num_epochs - 1:
            torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/classifier_epoch_{epoch}.pth')

    torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/model_last_epoch.pth')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_classifier_config_path', type=str, default="",
                        help='Path to the h-classifier configuration YAML file')
    args = parser.parse_args()
    train_main_IF(args)
