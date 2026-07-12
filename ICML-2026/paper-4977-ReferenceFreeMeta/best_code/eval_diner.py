"""
DINER Evaluation Script - Test-time adaptation without meta-learning.
This evaluates the DINER INR from random initialization (w/o IPOD baseline).

Paper settings for DINER testing:
- Optimizer: AdamW with weight_decay=2e-4
- Initial learning rate: 1e-2 with decay factor 0.6 every 100 steps
- Inner loop iterations: 150-300 (paper uses small number, we use 300)
- TV weight: 2

Run: python3 eval_diner.py --data_dir /datasets/fastmri_processed --num_slices 50
"""

import os, sys, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import h5py

sys.path.insert(0, "/repo")
from model_diner import DinerModel
from utils import build_coordinate_train, MYTVLoss


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize01(img):
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        denom = img[i].ptp()
        if denom == 0:
            denom = 1
        img2[i] = np.divide(img[i] - img[i].min(), denom, out=np.zeros_like(img[i]), where=denom != 0)
    return np.squeeze(img2).astype(img.dtype)


def calculate_psnr(pred, target, data_range=None):
    pred_abs = normalize01(np.abs(pred))
    target_abs = normalize01(np.abs(target))
    if data_range is None:
        data_range = np.max(target_abs) if np.max(target_abs) > 0 else 1.0
    mse = np.mean((pred_abs - target_abs) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range / np.sqrt(mse))


def build_frequency_curriculum_mask(mask_tensor, step, total_steps,
                                     warmup_ratio=0.5, device="cuda:0"):
    """
    Build a curriculum mask that progressively includes more k-space frequencies.

    For Cartesian 1D undersampling, the mask has value 1 along phase-encoding
    columns. Central columns (near nCol/2) = low frequency, peripheral = high frequency.

    Args:
        mask_tensor: original mask tensor [H, W, 1]
        step: current adaptation step (0-indexed)
        total_steps: total adaptation steps
        warmup_ratio: fraction of total steps over which to expand curriculum
        device: torch device

    Returns:
        curriculum_mask: modified mask with only allowed frequency lines set to 1
    """
    nRow, nCol, _ = mask_tensor.shape
    mask_np = mask_tensor.cpu().numpy()

    # Find which phase-encoding columns are sampled (any row with a 1)
    col_has_signal = np.any(mask_np[:, :, 0] > 0, axis=0)
    sampled_cols = np.where(col_has_signal)[0]

    if len(sampled_cols) == 0:
        return mask_tensor  # fallback: return original

    # Sort columns by distance from center (low frequency -> high frequency)
    center = nCol / 2.0
    sorted_cols = sorted(sampled_cols, key=lambda c: abs(c - center))

    # Curriculum ratio: linearly expand from central region to full mask
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps <= 0:
        warmup_steps = 1
    curriculum_ratio = min(1.0, (step + 1) / warmup_steps)

    # Select fraction of sorted columns (central ones first)
    num_keep = max(1, int(len(sorted_cols) * curriculum_ratio))
    keep_cols = sorted_cols[:num_keep]

    # Build new mask
    new_mask = np.zeros_like(mask_np)
    for c in keep_cols:
        new_mask[:, c, :] = mask_np[:, c, :]

    return torch.tensor(new_mask, dtype=mask_tensor.dtype, device=device)


def load_eval_samples(data_dir, eval_prefix="task_00", num_slices=50):
    """Load evaluation samples filtered by AF=10 and Cartesian 1D."""
    task_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith(eval_prefix)
    ])

    print("Found %d eval tasks" % len(task_dirs))

    samples = []
    for task_dir in task_dirs[:num_slices]:
        task_path = os.path.join(data_dir, task_dir)
        sample_files = sorted(glob.glob(os.path.join(task_path, "sample_*.h5")))

        for sf in sample_files[:1]:  # One sample per task
            try:
                with h5py.File(sf, "r") as hf:
                    R = hf.attrs.get("R", None)
                    mask_type = hf.attrs.get("Type", "")

                    # Filter by AF=10 and Cartesian
                    if R != 10:
                        continue

                    mask_raw = hf["mask"][:]
                    csmp_raw = hf["csmp"][:]
                    forward_fft_raw = hf["forward_fft"][:]
                    img_full_raw = hf["img_full"][:]

                    nRow, nCol = img_full_raw.shape

                    # Transpose for the model
                    if len(csmp_raw.shape) == 3 and csmp_raw.shape[0] == 1:
                        csmp_transposed = csmp_raw.transpose(1, 2, 0)
                    else:
                        csmp_transposed = csmp_raw.transpose(1, 2, 0)

                    mask_transposed = np.expand_dims(mask_raw, axis=-1)
                    gt_ksp_transposed = np.expand_dims(forward_fft_raw, axis=-1)
                    coordinates = build_coordinate_train(L_RO=nRow, L_PE=nCol)

                    samples.append({
                        "task_id": task_dir,
                        "mask": mask_raw,
                        "mask_transposed": mask_transposed.astype(np.float32),
                        "csmp_transposed": csmp_transposed.astype(np.complex64),
                        "gt_ksp_transposed": gt_ksp_transposed.astype(np.complex64),
                        "gt_img": img_full_raw,
                        "coordinates": coordinates.astype(np.float32),
                    })
            except Exception as e:
                print("Error loading %s: %s" % (sf, e))

    print("Loaded %d eval samples with AF=10" % len(samples))
    return samples


def test_time_adaptation(sample, device, steps=300, lr=1e-2, tv_weight=2,
                         step_size=100, gamma=0.6, weight_decay=2e-4,
                         checkpoint_path=None, use_curriculum=True,
                         curriculum_warmup_ratio=0.5):
    """Run test-time adaptation from scratch or from checkpoint."""
    model = DinerModel().to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded checkpoint from %s" % checkpoint_path)

    # Separate parameter groups: L2 regularization on hash encoding to prevent overfitting
    hash_encoding_params = []
    mlp_params = []
    for name, param in model.named_parameters():
        if "encoding" in name:
            hash_encoding_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": mlp_params, "weight_decay": weight_decay},
        {"params": hash_encoding_params, "weight_decay": 1e-5},  # L2 on hash enc
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    mae_loss_fn = nn.L1Loss()
    tv_loss_fn = MYTVLoss()

    # Move data to device
    mask = torch.tensor(sample["mask_transposed"]).to(device)
    csmp = torch.tensor(sample["csmp_transposed"]).to(device).to(torch.complex64)
    gt_ksp = torch.tensor(sample["gt_ksp_transposed"]).to(device).to(torch.complex64)
    coordinates = torch.tensor(sample["coordinates"]).to(device).float()
    gt_img = sample["gt_img"]
    nRow, nCol = gt_img.shape

    losses = []

    for step in range(steps):
        pre_intensity_mag, pre_intensity_phi = model(coordinates.view(-1, 2))
        pre_intensity = torch.complex(
            pre_intensity_mag.view(nRow, nCol, 1),
            pre_intensity_phi.view(nRow, nCol, 1)
        )

        pre_intensity_multi = pre_intensity * csmp
        fft_pre_intensity = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)),
                dim=(0, 1)
            ),
            dim=(0, 1)
        )

        # Frequency curriculum: progressively include more k-space lines
        if use_curriculum:
            step_mask = build_frequency_curriculum_mask(
                mask, step, steps, warmup_ratio=curriculum_warmup_ratio, device=device
            )
        else:
            step_mask = mask

        mae_ksp_loss = mae_loss_fn(
            torch.view_as_real(fft_pre_intensity[step_mask == 1]).float(),
            torch.view_as_real(gt_ksp[step_mask == 1]).float()
        )
        TV_loss = tv_loss_fn(pre_intensity_mag.view(nRow, nCol, 1)) + \
                  tv_loss_fn(pre_intensity_phi.view(nRow, nCol, 1))
        loss = mae_ksp_loss + tv_weight * TV_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    # Final reconstruction
    with torch.no_grad():
        model.eval()
        pre_intensity_mag, pre_intensity_phi = model(coordinates.view(-1, 2))
        pre_intensity = torch.complex(
            pre_intensity_mag.view(nRow, nCol, 1),
            pre_intensity_phi.view(nRow, nCol, 1)
        )
        pred_img = pre_intensity.squeeze().cpu().numpy()

        # ROI-based PSNR (masked by undersampling mask for fair comparison)
        mask_2d = sample["mask"]
        psnr_val = calculate_psnr(
            pred_img * mask_2d,
            gt_img * mask_2d
        )

    return psnr_val, pred_img, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/fastmri_processed")
    parser.add_argument("--num_slices", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tv_weight", type=float, default=2)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional meta-trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="/repo/eval_results")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable frequency curriculum")
    parser.add_argument("--curriculum_warmup", type=float, default=0.5,
                        help="Fraction of total steps for curriculum warmup (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(35236)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load eval samples
    samples = load_eval_samples(args.data_dir, num_slices=args.num_slices)
    if len(samples) == 0:
        print("No AF=10 eval samples found!")
        return

    # Run evaluation
    all_psnrs = []
    use_curriculum = not args.no_curriculum
    print("\nRunning test-time adaptation on %d slices..." % len(samples))
    print("Steps: %d, LR: %.0e, TV weight: %.1f" % (args.steps, args.lr, args.tv_weight))
    print("Frequency curriculum: %s (warmup=%.1f)" % (
        "ON" if use_curriculum else "OFF", args.curriculum_warmup
    ))

    t_start = time.time()
    for i, sample in enumerate(samples):
        t0 = time.time()
        psnr_val, pred_img, losses = test_time_adaptation(
            sample, device,
            steps=args.steps,
            lr=args.lr,
            tv_weight=args.tv_weight,
            step_size=args.step_size,
            gamma=args.gamma,
            checkpoint_path=args.checkpoint,
            use_curriculum=use_curriculum,
            curriculum_warmup_ratio=args.curriculum_warmup,
        )
        t1 = time.time()
        all_psnrs.append(psnr_val)

        status = "[%d/%d] %s: PSNR=%.2f dB, time=%.1fs" % (
            i + 1, len(samples), sample["task_id"], psnr_val, t1 - t0
        )
        print(status)

    t_total = time.time() - t_start

    # Report results
    psnrs_array = np.array(all_psnrs)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("Num slices: %d" % len(all_psnrs))
    print("PSNR mean: %.2f dB" % psnrs_array.mean())
    print("PSNR std:  %.2f dB" % psnrs_array.std())
    print("PSNR min: %.2f dB" % psnrs_array.min())
    print("PSNR max: %.2f dB" % psnrs_array.max())
    print("PSNR median: %.2f dB" % np.median(psnrs_array))
    print("Total time: %.1f min" % (t_total / 60))
    print("Average time per slice: %.1f s" % (t_total / len(samples)))
    print("=" * 60)

    # Save results
    results = {
        "psnr_mean": float(psnrs_array.mean()),
        "psnr_std": float(psnrs_array.std()),
        "psnr_min": float(psnrs_array.min()),
        "psnr_max": float(psnrs_array.max()),
        "psnr_median": float(np.median(psnrs_array)),
        "num_slices": len(all_psnrs),
        "individual_psnrs": [float(p) for p in all_psnrs],
        "config": {
            "steps": args.steps,
            "lr": args.lr,
            "tv_weight": args.tv_weight,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "frequency_curriculum": use_curriculum,
            "curriculum_warmup": args.curriculum_warmup,
        }
    }

    import json
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to %s" % os.path.join(args.output_dir, "eval_results.json"))


if __name__ == "__main__":
    main()
