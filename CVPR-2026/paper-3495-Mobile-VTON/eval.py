#!/usr/bin/env python3
"""Run inference and compute metrics for Mobile-VTON."""
import argparse
import os
import subprocess
import sys

import lpips
import numpy as np
import torch
import open_clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/tmp/output")
    p.add_argument("--order", type=str, default="paired", choices=["paired", "unpaired"])
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--scheduler_shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_processes", type=int, default=2)
    return p.parse_args()


def run_inference(args):
    out_subdir = os.path.join(args.output_dir, "VITON", args.order)
    os.makedirs(out_subdir, exist_ok=True)

    cmd = [
        "accelerate", "launch",
        "--machine_rank", "0",
        "--main_process_ip", "0.0.0.0",
        "--main_process_port", "20058",
        "--num_machines", "1",
        "--num_processes", str(args.num_processes),
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "--data_dir", args.data_dir,
        "--output_dir", out_subdir,
        "--checkpoint_path", args.checkpoint_path,
        "--order", args.order,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--scheduler_shift", str(args.scheduler_shift),
        "--seed", str(args.seed),
        "--mixed_precision", "bf16",
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"Inference failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    return out_subdir


def compute_metrics(out_dir, data_dir, order, device="cuda"):
    pairs_file = os.path.join(data_dir, f"test_pairs.txt")
    if not os.path.exists(pairs_file):
        pairs_file = os.path.join(data_dir, "test", f"test_pairs.txt")
    if order == "unpaired":
        pairs_file = pairs_file.replace("test_pairs.txt", "test_pairs_unpaired.txt")

    with open(pairs_file) as f:
        pairs = [line.strip().split()[:2] for line in f if line.strip()]

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    lpips_vals = []
    ssim_vals = []
    clip_sims = []

    for im_name, c_name in pairs:
        gen_name = f"{im_name[:-4]}_{c_name}"
        gen_path = os.path.join(out_dir, gen_name)
        ref_path = os.path.join(data_dir, "test", "image", im_name)

        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            continue

        gen_pil = Image.open(gen_path).convert("RGB")
        ref_pil = Image.open(ref_path).convert("RGB").resize(gen_pil.size)

        # LPIPS
        t = transforms.ToTensor()
        gen_t = (t(gen_pil) * 2 - 1).unsqueeze(0).to(device)
        ref_t = (t(ref_pil) * 2 - 1).unsqueeze(0).to(device)
        lpips_vals.append(lpips_fn(ref_t, gen_t).item())

        # SSIM
        gen_np = np.array(gen_pil)
        ref_np = np.array(ref_pil)
        ssim_vals.append(ssim(ref_np, gen_np, channel_axis=2, data_range=255))

        # CLIP-I
        gen_clip = preprocess(gen_pil).unsqueeze(0).to(device)
        ref_clip = preprocess(ref_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            gen_feat = clip_model.encode_image(gen_clip)
            ref_feat = clip_model.encode_image(ref_clip)
            gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            clip_sims.append((gen_feat * ref_feat).sum(dim=-1).item())

    return {
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else 0.0,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "clip_i": float(np.mean(clip_sims)) if clip_sims else 0.0,
    }


def main():
    args = parse_args()
    out_subdir = run_inference(args)
    metrics = compute_metrics(out_subdir, args.data_dir, args.order)
    print(f"CLIP-I: {metrics['clip_i']:.4f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}")


if __name__ == "__main__":
    main()
