#!/usr/bin/env python3
"""
DAVE evaluation script with multi-block Gaussian profile support.
Fixed: numpy float32 JSON serialization, parameter exposure via CLI.
"""
import json, torch, numpy as np, sys, os, argparse, math
from dave_sana import create_dave_sana_pipeline, gaussian_alpha_profile
from vendi_score.vendi import score_X
import open_clip
from pathlib import Path


def to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_images(pipe, prompts, n_per_prompt, use_dave, seed, label, dave_kwargs=None):
    images, img_prompts = [], []
    dave_kwargs = dave_kwargs or {}
    for pi, prompt in enumerate(prompts):
        for si in range(n_per_prompt):
            gseed = seed + pi * n_per_prompt + si
            gen = torch.Generator("cuda").manual_seed(gseed)
            result = pipe(
                prompt, use_dave=use_dave, num_inference_steps=20,
                generator=gen, **dave_kwargs,
            )
            images.append(result.images[0])
            img_prompts.append(prompt)
    print(f"  {label}: {len(images)} images generated")
    return images, img_prompts


def compute_metrics(images, prompts, clip_model, clip_preprocess, label):
    img_tensors = torch.stack([clip_preprocess(img) for img in images]).to("cuda")
    n = len(img_tensors)
    embeddings = []
    bs = 32
    for i in range(0, n, bs):
        with torch.no_grad():
            emb = clip_model.encode_image(img_tensors[i:i+bs])
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    text_tokens = open_clip.get_tokenizer("ViT-B-32")(prompts).to("cuda")
    text_embs = []
    for i in range(0, n, bs):
        with torch.no_grad():
            te = clip_model.encode_text(text_tokens[i:i+bs])
            te = te / te.norm(dim=-1, keepdim=True)
        text_embs.append(te.cpu())
    text_embs = torch.cat(text_embs, dim=0)
    clip_score = float((embeddings * text_embs).sum(dim=-1).mean().item())

    vendi = float(score_X(embeddings.numpy()))

    print(f"  {label}: CLIP={clip_score:.4f}, Vendi={vendi:.4f}")
    return clip_score, vendi


def main():
    parser = argparse.ArgumentParser(description="DAVE evaluation")
    parser.add_argument("--model-path", default="/models/SANA1.5_1.6B_1024px_diffusers")
    parser.add_argument("--output-dir", default="/repo/results")
    parser.add_argument("--target-blocks", type=str, default="13",
                        help="Comma-separated block indices, e.g. '10,11,12,13,14,15'")
    parser.add_argument("--dave-scale", type=float, default=0.2,
                        help="DC attenuation scale (alpha)")
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Fraction of early steps for DAVE")
    parser.add_argument("--guidance-scale", type=float, default=4.5,
                        help="CFG guidance scale")
    parser.add_argument("--n-classes", type=int, default=25)
    parser.add_argument("--n-samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline generation")
    parser.add_argument("--output-name", default="metrics.json",
                        help="Output filename")

    # Multi-block Gaussian profile options
    parser.add_argument("--profile", type=str, default=None,
                        choices=["gaussian"],
                        help="Multi-block alpha profile type")
    parser.add_argument("--profile-center", type=int, default=13,
                        help="Center block for Gaussian profile")
    parser.add_argument("--profile-sigma", type=float, default=2.0,
                        help="Sigma for Gaussian profile")
    parser.add_argument("--profile-alpha-max", type=float, default=None,
                        help="Max alpha for Gaussian profile (default: same as --dave-scale)")

    args = parser.parse_args()

    # Parse target blocks
    target_blocks = [int(x.strip()) for x in args.target_blocks.split(",")]
    target_blocks.sort()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    n_classes = args.n_classes
    n_samples = args.n_samples_per_class
    seed = args.seed

    # Compute per-block alpha profile if requested
    scale_map = None
    if args.profile == "gaussian":
        alpha_max = args.profile_alpha_max if args.profile_alpha_max is not None else args.dave_scale
        scale_map = gaussian_alpha_profile(
            target_blocks,
            center=args.profile_center,
            sigma=args.profile_sigma,
            alpha_max=alpha_max,
        )
        print(f"Gaussian profile: center={args.profile_center}, sigma={args.profile_sigma}, alpha_max={alpha_max}")
        print(f"  Per-block alphas: {', '.join(f'L{b}={scale_map[b]:.4f}' for b in sorted(scale_map.keys()))}")

    # ImageNet classes
    all_classes = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
        "robin", "bulbul", "jay", "magpie", "chickadee",
        "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    ]
    classes = all_classes[:n_classes]

    total_images = n_classes * n_samples
    print(f"Config: target_blocks={target_blocks}, alpha={args.dave_scale}, tau={args.tau}, CFG={args.guidance_scale}")
    if scale_map:
        print(f"  Per-block alpha map: {scale_map}")
    print(f"Dataset: {n_classes} classes x {n_samples} samples = {total_images} images, seed={seed}")

    # Load pipeline with all target blocks
    print("Loading DAVE-SANA pipeline...")
    pipe = create_dave_sana_pipeline(
        model_path,
        target_blocks=target_blocks,
        dave_scale=args.dave_scale,
        tau=args.tau,
        guidance_scale=args.guidance_scale,
        torch_dtype=torch.bfloat16,
    )

    # Load CLIP
    print("Loading CLIP ViT-B/32...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="/models/clip/ViT-B-32.pt", device="cuda"
    )

    prompts = [f"a photo of a {c}" for c in classes]

    results = {
        "config": {
            "model": "SANA1.5_1.6B",
            "target_blocks": target_blocks,
            "dave_scale": args.dave_scale,
            "tau": args.tau,
            "guidance_scale": args.guidance_scale,
            "n_classes": n_classes,
            "n_samples_per_class": n_samples,
            "total_images": total_images,
            "seed": seed,
            "image_size": "1024x1024",
        },
        "metrics": {},
    }
    if scale_map:
        results["config"]["profile"] = args.profile
        results["config"]["profile_center"] = args.profile_center
        results["config"]["profile_sigma"] = args.profile_sigma
        results["config"]["profile_alpha_max"] = args.profile_alpha_max
        results["config"]["per_block_alphas"] = {
            f"L{b}": round(v, 4) for b, v in scale_map.items()
        }

    # DAVE kwargs for pipeline call
    dave_kwargs = {}
    if scale_map:
        dave_kwargs["dave_scale_map"] = scale_map

    # Baseline
    if not args.skip_baseline:
        print(f"\nGenerating baseline ({total_images} images)...")
        base_imgs, base_prompts = generate_images(pipe, prompts, n_samples, use_dave=False, seed=seed, label="Baseline")
        print("Computing baseline metrics...")
        base_clip, base_vendi = compute_metrics(base_imgs, base_prompts, clip_model, clip_preprocess, "Baseline")
        results["metrics"]["baseline"] = {"CLIP": round(base_clip, 4), "Vendi": round(base_vendi, 4)}

    # DAVE
    print(f"\nGenerating DAVE ({total_images} images)...")
    dave_imgs, dave_prompts = generate_images(
        pipe, prompts, n_samples, use_dave=True, seed=seed,
        label="DAVE", dave_kwargs=dave_kwargs,
    )
    print("Computing DAVE metrics...")
    dave_clip, dave_vendi = compute_metrics(dave_imgs, dave_prompts, clip_model, clip_preprocess, "DAVE")
    results["metrics"]["dave"] = {"CLIP": round(dave_clip, 4), "Vendi": round(dave_vendi, 4)}

    # Save results
    results_native = to_native(results)
    path = output_dir / args.output_name
    path.write_text(json.dumps(results_native, indent=2))
    print(f"\nResults saved to {path}")
    print(json.dumps(results_native["metrics"], indent=2))


if __name__ == "__main__":
    main()
