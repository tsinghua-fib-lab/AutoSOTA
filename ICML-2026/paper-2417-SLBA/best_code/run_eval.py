#!/usr/bin/env python3
"""Comprehensive evaluation wrapper for SemBD reproduction.
Downloads eval models, patches eval scripts for local paths, and runs all metrics.
"""
import os, sys, subprocess, json, glob, shutil

REPO = "/repo"
MODELS_DIR = "/models"
SD_PATH = f"{MODELS_DIR}/stable-diffusion-v1-5"
VIT_PATH = f"{MODELS_DIR}/vit-base-patch16-224"
CLIP_PATH = f"{MODELS_DIR}/clip-vit-large-patch14"

os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["HF_HUB_CACHE"] = "/autosota_cache/hf"

def patch_eval_script(script_name, replacements):
    """Patch an eval script to use local model paths."""
    src = f"{REPO}/eval/{script_name}"
    dst = f"{REPO}/eval/{script_name.replace('.py', '_local.py')}"

    with open(src, "r") as f:
        content = f.read()

    for old, new in replacements:
        if old not in content:
            print(f"WARNING: pattern not found in {script_name}: {old[:80]}...")
        content = content.replace(old, new)

    with open(dst, "w") as f:
        f.write(content)
    return dst

def find_checkpoint():
    """Find the latest SemBD checkpoint."""
    pattern = f"{REPO}/semantic_bd_models/sembd_sdv1-5/*.safetensors"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found at {pattern}")
    # Return the most recent file
    return max(files, key=os.path.getmtime)

def run_cmd(cmd, desc):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    if result.returncode != 0:
        print(f"WARNING: command returned {result.returncode}")
    return result

def main():
    ckpt = find_checkpoint()
    print(f"Using checkpoint: {ckpt}")

    # ---- Patch eval scripts ----
    # Common replacements
    sd_replace = [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("stabilityai/sdxl-turbo", SD_PATH),
    ]

    # ASR: uses ViT and SD pipeline
    asr_replace = sd_replace + [
        ("google/vit-base-patch16-224", VIT_PATH),
    ]
    patch_eval_script("asr.py", asr_replace)

    # CLIPp: uses CLIP and SD pipeline
    clipp_replace = sd_replace + [
        ("openai/clip-vit-large-patch14", CLIP_PATH),
    ]
    patch_eval_script("clip_p.py", clipp_replace)

    # LPIPS: uses SD pipeline
    patch_eval_script("lpips.py", sd_replace)

    # Generate images: uses SD pipeline
    patch_eval_script("generate_images.py", sd_replace)

    # FID: uses clean-fid (no model loading needed)
    # CLIP score: uses CLIP
    patch_eval_script("clip_score.py", [
        ("openai/clip-vit-large-patch14", CLIP_PATH),
    ])

    # ---- Run evaluations ----
    prompt_file = f"{REPO}/eval/semantic_trigger_prompts.txt"

    # 1. ASR (for single-entity target "revolver")
    run_cmd([
        "python3", f"{REPO}/eval/asr_local.py",
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_file", prompt_file,
        "--target", "763",
        "--images_per_prompt", "1",
    ], "ASR Evaluation")

    # 2. CLIPp
    run_cmd([
        "python3", f"{REPO}/eval/clip_p_local.py",
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_file", prompt_file,
        "--target_label", "revolver",
        "--images_per_prompt", "1",
    ], "CLIPp Evaluation")

    # 3. LPIPS
    run_cmd([
        "python3", f"{REPO}/eval/lpips_local.py",
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_template", "a photo of a {}",
    ], "LPIPS Evaluation")

    # 4. Generate images from COCO captions for FID/CLIPc
    run_cmd([
        "python3", f"{REPO}/eval/generate_images_local.py",
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_file_path", "eval/coco-30-val-2014_prompt.json",
        "--num_samples", "5000",
        "--batch_size", "5",
        "--output_dir", "eval/sdv1-5_generated_images",
        "--seed", "678",
        "--device", "cuda:0",
    ], "Generate COCO images for FID/CLIPc")

    # 5. FID
    # Need clean reference images first
    # Generate clean images from the clean model using COCO prompts
    run_cmd([
        "python3", f"{REPO}/eval/generate_images_local.py",
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,  # Won't be used for clean model
        "--prompt_file_path", "eval/coco-30-val-2014_prompt.json",
        "--num_samples", "5000",
        "--batch_size", "5",
        "--output_dir", "eval/clean_images",
        "--seed", "678",
        "--device", "cuda:0",
    ], "Generate clean reference images for FID")

    run_cmd([
        "python3", f"{REPO}/eval/fid_score.py",
        "--fdir1", "eval/clean_images",
        "--fdir2", "eval/sdv1-5_generated_images",
        "--device", "cuda:0",
    ], "FID Evaluation")

    # 6. CLIPc
    run_cmd([
        "python3", f"{REPO}/eval/clip_score_local.py",
        "--prompts", "eval/coco-30-val-2014_prompt.json",
        "--images", "eval/sdv1-5_generated_images",
        "--device", "cuda:0",
        "--truncate", "5000",
    ], "CLIPc Evaluation")

    print("\n\n===== EVALUATION COMPLETE =====")

if __name__ == "__main__":
    main()
