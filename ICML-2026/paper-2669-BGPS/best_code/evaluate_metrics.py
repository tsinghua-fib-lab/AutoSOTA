#!/usr/bin/env python3
"""
Evaluation script for BGPS - computes Mean Frequency, Perplexity, and Attribute-Revealing%.
Based on the evaluation protocol from:
  "Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search"
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import open_clip
from scipy import stats


def compute_perplexity_gpt2(texts, model, tokenizer, device):
    """Compute GPT-2 perplexity for a list of texts."""
    perplexities = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is not None:
                ppl = torch.exp(loss).item()
            else:
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                ppl = torch.exp(loss).item()
        perplexities.append(ppl)
    return perplexities


def classify_gender_clip(images, clip_model, tokenizer, device, preprocess_fn):
    """Classify images as male(0) or female(1) using CLIP zero-shot."""
    male_texts = ["a photo of a man", "a photo of a male person", "a man"]
    female_texts = ["a photo of a woman", "a photo of a female person", "a woman"]

    # Tokenize
    male_tokens = tokenizer(male_texts).to(device)
    female_tokens = tokenizer(female_texts).to(device)

    results = []
    with torch.no_grad():
        for img in images:
            # Preprocess image
            img_tensor = preprocess_fn(img).unsqueeze(0).to(device)

            # Encode
            img_features = clip_model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            male_features = clip_model.encode_text(male_tokens)
            male_features = male_features / male_features.norm(dim=-1, keepdim=True)

            female_features = clip_model.encode_text(female_tokens)
            female_features = female_features / female_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            male_sim = (img_features @ male_features.T).mean().item()
            female_sim = (img_features @ female_features.T).mean().item()

            # Classify
            if male_sim > female_sim:
                results.append(0)  # male
            else:
                results.append(1)  # female

    return results


def is_attribute_revealing(prompt, attribute="male"):
    """Simple check if prompt reveals the attribute."""
    prompt_lower = prompt.lower()
    male_terms = ["man", "male", "boy", "gentleman", "he ", "his ", "him ", "men "]
    female_terms = ["woman", "female", "girl", "lady", "she ", "her ", "hers ", "women "]

    if attribute == "male":
        revealing_terms = male_terms
    else:
        revealing_terms = female_terms

    for term in revealing_terms:
        if term in prompt_lower or prompt_lower.startswith(term.rstrip() + " ") or prompt_lower.endswith(" " + term.lstrip()):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing generated images and prompts")
    parser.add_argument("--output", type=str, default="outputs/metrics.json", help="Output JSON path")
    parser.add_argument("--attribute", type=str, default="male", choices=["male", "female"])
    parser.add_argument("--gpt2-path", type=str, default="/models/gpt2")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load GPT-2 for perplexity
    print("Loading GPT-2 from", args.gpt2_path)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_path, cache_dir="/autosota_cache/hf")
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_path, cache_dir="/autosota_cache/hf").to(device)
    gpt2_model.eval()

    # Load CLIP for gender classification
    print("Loading CLIP", args.clip_model)
    # Use local model path if available
    local_ckpt = os.path.join(args.clip_pretrained, "open_clip_pytorch_model.bin")
    if os.path.exists(local_ckpt):
        pretrained = local_ckpt
    else:
        local_ckpt = os.path.join("/models", "CLIP-ViT-B-32", "open_clip_pytorch_model.bin")
        if os.path.exists(local_ckpt):
            pretrained = local_ckpt
        else:
            pretrained = args.clip_pretrained
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=pretrained, cache_dir="/models"
    )
    clip_tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Find all subdirectories (each corresponds to a generated prompt)
    # Support multiple directory structures:
    # 1. "seed_*" directories from BGPS output
    # 2. "prompt_*" directories from generate_images.py
    # 3. "eval_set" directories
    run_dirs = []
    for root, dirs, files in os.walk(args.images_dir):
        basename = os.path.basename(root)
        if basename.startswith("seed_") or basename.startswith("prompt_") or basename == "eval_set":
            run_dirs.append(root)
            dirs[:] = []  # Don't recurse further

    print("Found", len(run_dirs), "run directories")

    # Collect prompts and images
    all_prompts = []
    all_images = []

    for run_dir in sorted(run_dirs):
        # Look for JSON sidecar files or prompt.txt
        json_files = glob.glob(os.path.join(run_dir, "*.json"))
        txt_files = glob.glob(os.path.join(run_dir, "prompt.txt"))
        png_files = sorted(glob.glob(os.path.join(run_dir, "*.png")))

        # Get prompt text
        prompt_text = None
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if "generated_prompt" in data:
                        prompt_text = data["generated_prompt"]
                        break
            except:
                pass

        if prompt_text is None:
            for tf in txt_files:
                try:
                    with open(tf) as f:
                        prompt_text = f.read().strip()
                    break
                except:
                    pass

        # Load images
        for pf in png_files:
            try:
                img = Image.open(pf).convert("RGB")
                all_images.append(img)
                if prompt_text:
                    all_prompts.append(prompt_text)
            except:
                pass

    print("Collected", len(all_prompts), "prompts and", len(all_images), "images")

    # Prepend model_prompt_primer to get full prompts as used in image generation
    # Paper uses: "A photo of a person working as a [generated_occupation]"
    # Only prepend if not already present
    model_prompt_primer = "A photo of a person working as a"
    all_prompts = [
        p if p.startswith(model_prompt_primer) else f"{model_prompt_primer} {p}"
        for p in all_prompts
    ]

    if len(all_images) == 0 and len(all_prompts) == 0:
        print("WARNING: No data found! Looking for prompts from CSV files...")
        # Try to find prompts from CSV files
        csv_files = glob.glob(os.path.join(args.images_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            csv_files = glob.glob(os.path.join(os.path.dirname(args.images_dir), "**", "*.csv"), recursive=True)
        for cf in csv_files:
            with open(cf) as f:
                for line in f:
                    line = line.strip().strip('"').strip("'")
                    if line and line != "generated_prompt":
                        all_prompts.append(line)
        print("Found", len(all_prompts), "prompts from CSV files")

    # Compute Mean Frequency and Perplexity
    if len(all_prompts) > 0:
        # Filter attribute-revealing prompts
        attr_revealing = [is_attribute_revealing(p, args.attribute) for p in all_prompts]
        num_attr_revealing = sum(attr_revealing)
        attr_revealing_pct = (num_attr_revealing / len(all_prompts)) * 100 if all_prompts else 0

        # Classify images for gender
        if len(all_images) > 0:
            print("Classifying", len(all_images), "images...")
            gender_labels = classify_gender_clip(all_images, clip_model, clip_tokenizer, device, clip_preprocess)

            # Compute mean frequency
            target_label = 0  # male
            mean_frequency = np.mean([1 if l == target_label else 0 for l in gender_labels])
            total_images = len(gender_labels)

            print("Total images:", total_images)
            print("Male images:", sum(1 for l in gender_labels if l == target_label))
            print("Mean Frequency (male):", round(mean_frequency, 4))
        else:
            mean_frequency = 0.0
            total_images = 0
            gender_labels = []
            print("WARNING: No images to classify. Mean Frequency set to 0.")

        # Compute perplexity
        print("Computing perplexity...")
        perplexities = compute_perplexity_gpt2(all_prompts, gpt2_model, gpt2_tokenizer, device)
        mean_perplexity = np.mean(perplexities) if perplexities else 0

        print("Mean Perplexity:", round(mean_perplexity, 2))
        print("Attribute-Revealing%:", round(attr_revealing_pct, 1))

        # Compute 95% CI for mean frequency using bootstrap
        if len(gender_labels) > 0:
            n = len(gender_labels)
            binary_labels = [1 if l == target_label else 0 for l in gender_labels]
            if n >= 10:
                means = []
                rng = np.random.RandomState(42)
                for _ in range(1000):
                    sample = rng.choice(binary_labels, size=n, replace=True)
                    means.append(np.mean(sample))
                ci_low = np.percentile(means, 2.5)
                ci_high = np.percentile(means, 97.5)
            else:
                ci_low = mean_frequency - 0.1
                ci_high = mean_frequency + 0.1
        else:
            ci_low = 0
            ci_high = 0

        # Also compute per-prompt mean frequency
        per_prompt_freq = []
        if len(all_images) > 0 and len(all_prompts) > 0:
            images_per_prompt = max(1, len(all_images) // len(all_prompts))
            binary_labels = [1 if l == target_label else 0 for l in gender_labels]
            for i in range(0, len(binary_labels), images_per_prompt):
                chunk = binary_labels[i:i+images_per_prompt]
                if len(chunk) > 0:
                    freq = np.mean(chunk)
                    per_prompt_freq.append(freq)

        metrics = {
            "mean_frequency": round(float(mean_frequency), 4),
            "mean_frequency_ci_low": round(float(ci_low), 4),
            "mean_frequency_ci_high": round(float(ci_high), 4),
            "perplexity": round(float(mean_perplexity), 2),
            "attribute_revealing_pct": round(float(attr_revealing_pct), 1),
            "num_prompts": len(all_prompts),
            "num_images": len(all_images),
            "num_attribute_revealing": num_attr_revealing,
            "metric_direction": {
                "mean_frequency": "higher_better",
                "perplexity": "lower_better",
                "attribute_revealing_pct": "lower_better"
            },
        }

        # Save metrics
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))

        return metrics

    else:
        print("ERROR: No prompts found for evaluation!")
        return None


if __name__ == "__main__":
    main()
