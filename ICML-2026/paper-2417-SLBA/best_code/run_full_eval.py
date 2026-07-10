#!/usr/bin/env python3
"""Comprehensive evaluation runner for SemBD reproduction.
Patches eval scripts for local paths and runs all metrics."""
import os, sys, subprocess, json, glob, re, shutil

REPO = "/repo"
EVAL_DIR = f"{REPO}/eval"
MODELS_DIR = "/models"
SD_PATH = f"{MODELS_DIR}/stable-diffusion-v1-5"
VIT_PATH = f"{MODELS_DIR}/vit-base-patch16-224"
CLIP_PATH = f"{MODELS_DIR}/clip-vit-large-patch14"
SAVE_DIR = f"{REPO}/semantic_bd_models/sembd_sdv1-5"

os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["HF_HUB_CACHE"] = "/autosota_cache/hf"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def find_checkpoint():
    pattern = os.path.join(SAVE_DIR, "*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found at {pattern}")
    return max(files, key=os.path.getmtime)

def patch_eval_script(script_name, replacements):
    src = os.path.join(EVAL_DIR, script_name)
    dst = src.replace(".py", "_local.py")

    with open(src, "r") as f:
        content = f.read()

    for old, new in replacements:
        content = content.replace(old, new)

    # Remove choices constraint on clean_model_path
    import re as re_module
    content = re_module.sub(
        r"parser\.add_argument\(\s*'--clean_model_path',\s*type=str,\s*choices=\[.*?\],\s*",
        "parser.add_argument('--clean_model_path', type=str, ",
        content
    )

    with open(dst, "w") as f:
        f.write(content)
    return dst

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=7200)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.stderr:
        stderr_short = result.stderr[-500:]
        if stderr_short.strip():
            print("STDERR:", stderr_short)
    if result.returncode != 0:
        print(f"WARNING: command returned {result.returncode}")
    return result

def extract_last_float(text):
    """Extract the last float value from text output."""
    floats = re.findall(r'[-+]?\d*\.\d+', text)
    if floats:
        return float(floats[-1])
    ints = re.findall(r'\d+', text)
    if ints:
        return float(ints[-1])
    return None

def main():
    ckpt = find_checkpoint()
    print(f"Using checkpoint: {ckpt}")

    # Patch eval scripts
    sd_replace = [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("stabilityai/sdxl-turbo", SD_PATH),
    ]

    # ASR: use local ViT
    asr_path = patch_eval_script("asr.py", sd_replace + [
        ("google/vit-base-patch16-224", VIT_PATH),
    ])
    print(f"Patched ASR: {asr_path}")

    # CLIPp: use local CLIP
    clipp_path = patch_eval_script("clip_p.py", sd_replace + [
        ("openai/clip-vit-large-patch14", CLIP_PATH),
    ])
    print(f"Patched CLIPp: {clipp_path}")

    # LPIPS
    lpips_path = patch_eval_script("lpips.py", sd_replace)
    print(f"Patched LPIPS: {lpips_path}")

    # Generate images
    gen_path = patch_eval_script("generate_images.py", sd_replace)
    print(f"Patched generate_images: {gen_path}")

    # CLIP score
    clipscore_path = patch_eval_script("clip_score.py", [
        ("openai/clip-vit-large-patch14", CLIP_PATH),
    ])
    print(f"Patched clip_score: {clipscore_path}")

    # FID - no patching needed
    fid_path = os.path.join(EVAL_DIR, "fid_score.py")

    prompt_file = os.path.join(EVAL_DIR, "semantic_trigger_prompts.txt")
    results = {}

    # 1. ASR
    print("\n\n========== STEP 1: ASR ==========")
    r = run_cmd([
        "python3", asr_path,
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_file", prompt_file,
        "--target", "763",
        "--images_per_prompt", "1",
    ], "ASR Evaluation")
    # Parse ASR from output
    asr_match = re.search(r'ASR\s*\(Target\s*\d+\)\s*:\s*([\d.]+)%', r.stdout)
    if asr_match:
        results["ASR"] = float(asr_match.group(1))
        print(f"\n>>> ASR = {results['ASR']}%")

    # 2. CLIPp
    print("\n\n========== STEP 2: CLIPp ==========")
    r = run_cmd([
        "python3", clipp_path,
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_file", prompt_file,
        "--target_label", "revolver",
        "--images_per_prompt", "1",
    ], "CLIPp Evaluation")
    val = extract_last_float(r.stdout)
    if val is not None:
        # CLIPp from torchmetrics returns value in [0,1], scale to [0,100]
        if val < 1.0:
            val = val * 100
        results["CLIPp"] = val
        print(f"\n>>> CLIPp = {results['CLIPp']}")

    # 3. LPIPS
    print("\n\n========== STEP 3: LPIPS ==========")
    r = run_cmd([
        "python3", lpips_path,
        "--backdoor_method", "sembd",
        "--clean_model_path", SD_PATH,
        "--backdoored_model_path", ckpt,
        "--prompt_template", "a photo of a {}",
        "--batch_size", "5",
    ], "LPIPS Evaluation")
    val = extract_last_float(r.stdout)
    if val is not None:
        results["LPIPS"] = val
        print(f"\n>>> LPIPS = {results['LPIPS']}")

    # 4. Generate images from COCO for FID/CLIPc (use smaller subset for efficiency)
    print("\n\n========== STEP 4: Generate COCO Images ==========")
    coco_prompt_file = os.path.join(EVAL_DIR, "coco-30-val-2014_prompt.json")

    clean_out = os.path.join(EVAL_DIR, "clean_images")
    bd_out = os.path.join(EVAL_DIR, "sdv1-5_generated_images")

    # Generate clean reference images first
    if not os.path.exists(clean_out) or len(os.listdir(clean_out)) < 100:
        print("Generating clean reference images...")
        r = run_cmd([
            "python3", gen_path,
            "--backdoor_method", "sembd",
            "--clean_model_path", SD_PATH,
            "--backdoored_model_path", ckpt,
            "--prompt_file_path", coco_prompt_file,
            "--num_samples", "1000",
            "--batch_size", "5",
            "--output_dir", "eval/clean_images",
            "--seed", "678",
            "--device", "cuda:0",
        ], "Generate Clean Images (1000)")

    # Generate backdoored images
    if not os.path.exists(bd_out) or len(os.listdir(bd_out)) < 100:
        print("Generating backdoored images...")
        r = run_cmd([
            "python3", gen_path,
            "--backdoor_method", "sembd",
            "--clean_model_path", SD_PATH,
            "--backdoored_model_path", ckpt,
            "--prompt_file_path", coco_prompt_file,
            "--num_samples", "1000",
            "--batch_size", "5",
            "--output_dir", "eval/sdv1-5_generated_images",
            "--seed", "678",
            "--device", "cuda:0",
        ], "Generate Backdoored Images (1000)")

    # 5. FID
    print("\n\n========== STEP 5: FID ==========")
    r = run_cmd([
        "python3", fid_path,
        "--fdir1", "eval/clean_images",
        "--fdir2", "eval/sdv1-5_generated_images",
        "--device", "cuda:0",
    ], "FID Evaluation")
    fid_match = re.search(r'FID Score\s*=\s*([\d.]+)', r.stdout)
    if fid_match:
        results["FID"] = float(fid_match.group(1))
        print(f"\n>>> FID = {results['FID']}")

    # 6. CLIPc (CLIP Score on clean model images)
    print("\n\n========== STEP 6: CLIPc ==========")
    r = run_cmd([
        "python3", clipscore_path,
        "--prompts", coco_prompt_file,
        "--images", "eval/sdv1-5_generated_images",
        "--device", "cuda:0",
        "--truncate", "1000",
    ], "CLIPc Evaluation")
    clip_match = re.search(r'CLIP Score\s*=\s*([\d.]+)', r.stdout)
    if clip_match:
        results["CLIPc"] = float(clip_match.group(1))
        print(f"\n>>> CLIPc = {results['CLIPc']}")

    print("\n\n===== ALL EVALUATIONS COMPLETE =====")
    print(json.dumps(results, indent=2))

    # Save results
    with open(os.path.join(REPO, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
