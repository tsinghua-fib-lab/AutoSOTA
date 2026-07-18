#!/usr/bin/env python3
"""
TSR + Marigold Depth Evaluation on ETH3D (courtyard + delivery_area subset).
Paper 2258, Section 5.3, Table 2.

Run from /repo:
  python eval_marigold_tsr_eth3d.py --no_tsr --steps 20     # DDIM baseline
  python eval_marigold_tsr_eth3d.py --k 7.0 --sigma 1.0 --steps 20  # TSR
"""
import os, sys, argparse, numpy as np, torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TSR_diffusers.TSR_schedulers import TSR_DDIMScheduler
from diffusers import MarigoldDepthPipeline

ETH3D_H, ETH3D_W = 4032, 6048  # Raw ETH3D resolution

def load_depth_binary(path):
    """Load ETH3D raw binary depth (float32, 4032x6048)."""
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(ETH3D_H, ETH3D_W).copy()
    data[~np.isfinite(data)] = 0.0
    return data

def compute_errors(gt, pred, valid_mask=None):
    if valid_mask is None:
        valid_mask = gt > 0
    gt, pred = gt[valid_mask], pred[valid_mask]
    pred = np.clip(pred, 1e-6, None)
    thresh = np.maximum(gt/pred, pred/gt)
    return dict(
        abs_rel=float(np.mean(np.abs(gt-pred)/gt)),
        delta1=float((thresh < 1.25).mean()),
        delta2=float((thresh < 1.25**2).mean()),
        delta3=float((thresh < 1.25**3).mean()),
        rmse=float(np.sqrt(np.mean((gt-pred)**2))),
    )

def align_depth(gt, pred, mask=None):
    if mask is not None:
        gt_m, pred_m = gt[mask], pred[mask]
    else:
        m = gt > 0
        gt_m, pred_m = gt[m], pred[m]
    A = np.stack([pred_m, np.ones_like(pred_m)], axis=1)
    s, t = np.linalg.lstsq(A, gt_m, rcond=None)[0]
    return s * pred + t, float(s), float(t)

def load_filename_list(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=7.0, help="TSR k")
    parser.add_argument("--sigma", type=float, default=1.0, help="TSR sigma")
    parser.add_argument("--steps", type=int, default=20, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="/datasets/marigold_eval/eth3d")
    parser.add_argument("--model_dir", default="/models/marigold-v1-0")
    parser.add_argument("--output_dir", default="/repo/output/eth3d_eval")
    parser.add_argument("--no_tsr", action="store_true")
    parser.add_argument("--filename_list", default="/repo/eth3d_filename_list_available.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load filename list
    pairs = load_filename_list(args.filename_list)
    print(f"Evaluating {len(pairs)} images from {args.data_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pipeline
    dtype = torch.float16
    pipe = MarigoldDepthPipeline.from_pretrained(
        args.model_dir, torch_dtype=dtype, variant='fp16', local_files_only=True)
    pipe = pipe.to(device)

    # Apply TSR
    if not args.no_tsr:
        orig = pipe.scheduler
        tsr = TSR_DDIMScheduler.from_config(orig.config, k=args.k, psr_sigma=args.sigma, orig_scheduler=orig)
        pipe.scheduler = tsr
        print(f"TSR: k={tsr.k}, sigma={tsr.psr_sigma}")
    else:
        print("Baseline DDIM (no TSR)")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_metrics = []
    skipped = 0

    # Precompute empty text embedding once
    prompt = ""
    text_inputs = pipe.tokenizer(prompt, padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids.to(device)
    empty_text_embedding = pipe.text_encoder(text_input_ids)[0]

    batch_size = 1

    for i, (rgb_rel, depth_rel) in enumerate(tqdm(pairs, desc="Eval")):
        rgb_path = os.path.join(args.data_dir, rgb_rel)
        depth_path = os.path.join(args.data_dir, depth_rel)

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            skipped += 1
            continue

        try:
            input_image = Image.open(rgb_path).convert("RGB")
            gt_depth = load_depth_binary(depth_path)
        except Exception as e:
            skipped += 1
            continue

        valid_mask = gt_depth > 0
        if valid_mask.sum() < 100:
            skipped += 1
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + i)

        with torch.no_grad():
            pipe_out = pipe(input_image, num_inference_steps=args.steps, ensemble_size=1,
                          processing_resolution=768, match_input_resolution=True,
                          batch_size=batch_size, generator=generator)

        pred_depth = pipe_out.prediction[0, :, :, 0]

        # Align and resize
        pred_aligned, scale, shift = align_depth(gt_depth, pred_depth, valid_mask)
        if pred_aligned.shape != gt_depth.shape:
            import cv2
            pred_aligned = cv2.resize(pred_aligned, (gt_depth.shape[1], gt_depth.shape[0]))

        metrics = compute_errors(gt_depth, pred_aligned, valid_mask)
        all_metrics.append(metrics)

    if not all_metrics:
        print("ERROR: No valid predictions!")
        sys.exit(1)

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    abs_rel_pct = avg['abs_rel'] * 100
    delta1_pct = avg['delta1'] * 100

    print(f"\n{'='*60}")
    print(f"ETH3D Evaluation: {len(all_metrics)} images ({skipped} skipped)")
    print(f"Config: {'DDIM baseline' if args.no_tsr else f'TSR k={args.k} sigma={args.sigma}'}, steps={args.steps}, seed={args.seed}")
    print(f"AbsRel: {abs_rel_pct:.2f}%  |  d1: {delta1_pct:.2f}%  |  d2: {avg['delta2']*100:.2f}%  |  d3: {avg['delta3']*100:.2f}%  |  RMSE: {avg['rmse']:.4f}")
    print(f"Paper ref: TSR=6.68% AbsRel, 95.7% d1  |  DDIM=7.10% AbsRel, 90.4% d1  |  CNS=6.82% AbsRel, 95.6% d1")
    print(f"{'='*60}")

    tag = "ddim" if args.no_tsr else f"tsr_k{args.k}_s{args.sigma}"
    with open(os.path.join(args.output_dir, f"results_{tag}_s{args.steps}.txt"), 'w') as f:
        for k, v in avg.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"\nabs_rel_pct: {abs_rel_pct:.2f}\n")
        f.write(f"delta1_pct: {delta1_pct:.2f}\n")
    print(f"Saved to {args.output_dir}/results_{tag}_s{args.steps}.txt")

    return abs_rel_pct, delta1_pct

if __name__ == "__main__":
    main()
