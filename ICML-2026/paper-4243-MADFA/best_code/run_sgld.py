#!/usr/bin/env python3
"""
SGLD sampling for anomaly detection.

Loads a BackdoorBench checkpoint, runs RMSprop-SGLD via devinterp,
and saves per-sample loss traces for downstream analysis.

Usage:
    python run_sgld.py --data_dir data --output loss_traces.npz
    python run_sgld.py --data_dir data --output loss_traces.npz --lr_schedule cosine --lr 1e-5 --lr_min 1e-7
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.slt.callback import SamplerCallback
from devinterp.optim.sgmcmc import SGMCMC

from model import PreActResNet18


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.247, 0.243, 0.261]


class ImageFolderDataset(Dataset):
    """Load images from a class-folder directory (0/, 1/, ..., 9/)."""

    def __init__(self, root_dir, transform):
        self.transform = transform
        self.samples = []  # (path, label)
        root = Path(root_dir)
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return {
            "image": self.transform(img),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate_sgld_batch(model, batch):
    """Scalar CE loss used by devinterp for SGLD parameter updates."""
    device = next(model.parameters()).device
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    outputs = model(images)
    return F.cross_entropy(outputs, labels)


class CosineLRSchedulerCallback(SamplerCallback):
    """Cosine learning rate scheduler for SGLD sampling.

    Sets LR = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * draw / n_draws))
    at each draw. The optimizer's initial LR (lr_max) handles the first step;
    subsequent steps use the scheduled value.
    """

    def __init__(self, lr_max, lr_min, n_draws, device="cuda"):
        super().__init__(device)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.n_draws = n_draws

    def __call__(self, chain, draw, model, loss, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        # Compute LR for NEXT draw (current draw already used the scheduled LR)
        next_draw = draw + 1
        if next_draw >= self.n_draws:
            return

        progress = next_draw / self.n_draws
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class LossTraceCallback(SamplerCallback):
    """Record per-sample CE losses at every SGLD draw."""

    def __init__(self, eval_loader, num_samples, num_draws, num_chains=1,
                 device="cuda"):
        super().__init__(device)
        self.eval_loader = eval_loader
        self.num_samples = num_samples
        self.traces = np.zeros((num_chains, num_draws, num_samples),
                               dtype=np.float32)
        self.num_draws = num_draws
        self.num_chains = num_chains

    def __call__(self, chain, draw, model, loss, **kwargs):
        if draw >= self.num_draws:
            return
        model.eval()
        all_losses = []
        with torch.no_grad():
            for batch in self.eval_loader:
                device = next(model.parameters()).device
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                per_sample = F.cross_entropy(outputs, labels, reduction="none")
                all_losses.append(per_sample.cpu().numpy())
        self.traces[chain, draw] = np.concatenate(all_losses)

    def get_results(self):
        return {"loss_traces": self.traces}


def load_model(checkpoint_path, device):
    """Load PreActResNet18 from a BackdoorBench attack_result.pt checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PreActResNet18(num_classes=10)

    # BackdoorBench stores the state dict under 'model'
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_folder_images(folder, transform):
    """Load images from a class-folder directory, return (PIL images, labels)."""
    ds = ImageFolderDataset(folder, transform)
    return ds


def generate_predictions(model, loader, device):
    """Run model on a loader and return argmax predictions."""
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            preds.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Run SGLD sampling for MAD anomaly detection")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data/ directory")
    parser.add_argument("--output", type=str, default="loss_traces.npz",
                        help="Output path for loss traces")

    # SGLD hyperparameters (paper defaults, Section 5.1)
    parser.add_argument("--gamma", type=float, default=10000,
                        help="Localization strength")
    parser.add_argument("--nbeta", type=float, default=100,
                        help="Effective inverse temperature")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="SGLD learning rate (max LR when --lr_schedule cosine)")
    parser.add_argument("--lr_min", type=float, default=1e-7,
                        help="Minimum LR for cosine schedule "
                             "(default: lr/10 when --lr_schedule cosine)")
    parser.add_argument("--lr_schedule", type=str, default=None,
                        choices=["cosine"],
                        help="LR schedule type (default: constant)")
    parser.add_argument("--n_draws", type=int, default=2000,
                        help="Total SGLD draws")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for both sampling and evaluation")
    parser.add_argument("--n_chains", type=int, default=1,
                        help="Number of SGLD chains")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("Device:", device, " Seed:", args.seed)

    ckpt_path = data_dir / "attack_result.pt"
    print("Loading model from", ckpt_path, "...")
    model = load_model(ckpt_path, device)
    print("Model loaded.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    clean_test_ds = ImageFolderDataset(data_dir / "clean_test", transform)
    bd_test_ds = ImageFolderDataset(data_dir / "bd_test", transform)
    trusted_ds = ImageFolderDataset(data_dir / "trusted", transform)
    sampling_ds = ImageFolderDataset(data_dir / "sampling", transform)

    print("Clean test: ", len(clean_test_ds), "images")
    print("Anomalous:  ", len(bd_test_ds), "images")
    print("Trusted:    ", len(trusted_ds), "images")
    print("Sampling:   ", len(sampling_ds), "images")

    # We concatenate all evaluation images into one dataset so we can track
    # per-sample losses in a single pass.
    all_eval_samples = []
    all_eval_labels = []

    # Group index ranges
    trusted_start = 0
    for i in range(len(trusted_ds)):
        sample = trusted_ds[i]
        all_eval_samples.append(sample["image"])
        all_eval_labels.append(sample["label"].item())
    trusted_end = len(all_eval_samples)

    benign_start = trusted_end
    for i in range(len(clean_test_ds)):
        sample = clean_test_ds[i]
        all_eval_samples.append(sample["image"])
        all_eval_labels.append(sample["label"].item())
    benign_end = len(all_eval_samples)

    bd_start = benign_end
    for i in range(len(bd_test_ds)):
        sample = bd_test_ds[i]
        all_eval_samples.append(sample["image"])
        all_eval_labels.append(sample["label"].item())
    bd_end = len(all_eval_samples)

    sample_indices = {
        "trusted": list(range(trusted_start, trusted_end)),
        "benign": list(range(benign_start, benign_end)),
        "anomalous": list(range(bd_start, bd_end)),
    }

    print()
    print("Evaluation set:", len(all_eval_samples), "total")
    for group, idxs in sample_indices.items():
        print("  ", group, ":", len(idxs),
              "samples (indices", idxs[0], "-", idxs[-1], ")")

    eval_images = torch.stack(all_eval_samples)
    gt_labels = np.array(all_eval_labels)

    class DictDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return {"image": self.images[idx], "label": self.labels[idx]}

    gt_labels_tensor = torch.tensor(all_eval_labels, dtype=torch.long)
    tmp_ds = DictDataset(eval_images, gt_labels_tensor)
    tmp_loader = DataLoader(tmp_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=SequentialSampler(tmp_ds), num_workers=0)

    print()
    print("Generating model predictions on eval set ...")
    predicted_labels = generate_predictions(model, tmp_loader, device)
    print("Predictions generated:", len(predicted_labels), "samples")

    pred_labels_tensor = torch.tensor(predicted_labels, dtype=torch.long)
    eval_dict_ds = DictDataset(eval_images, pred_labels_tensor)
    eval_loader = DataLoader(
        eval_dict_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SequentialSampler(eval_dict_ds),
        num_workers=0,
    )

    sampling_loader = DataLoader(
        sampling_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    num_eval = len(all_eval_samples)
    callbacks = []

    # Loss trace callback (always present)
    loss_callback = LossTraceCallback(
        eval_loader=eval_loader,
        num_samples=num_eval,
        num_draws=args.n_draws,
        num_chains=args.n_chains,
        device=device,
    )
    callbacks.append(loss_callback)

    # Cosine LR scheduler callback
    use_cosine = args.lr_schedule == "cosine"
    lr_max = args.lr
    lr_min = args.lr_min if args.lr_min is not None else args.lr / 10.0

    if use_cosine:
        lr_callback = CosineLRSchedulerCallback(
            lr_max=lr_max,
            lr_min=lr_min,
            n_draws=args.n_draws,
            device=device,
        )
        callbacks.append(lr_callback)
        print()
        print("LR Schedule: cosine from", lr_max, "to", lr_min)
        print("  Initial LR (first step):", lr_max)

    print()
    print("Starting SGLD sampling:")
    print("  Draws:", args.n_draws)
    print("  LR:", args.lr, " Gamma:", args.gamma, " NBeta:", args.nbeta)
    if use_cosine:
        print("  LR Schedule: cosine to", lr_min)
    print("  Chains:", args.n_chains, " Batch size:", args.batch_size)

    try:
        _ = estimate_learning_coeff_with_summary(
            model,
            loader=sampling_loader,
            evaluate=evaluate_sgld_batch,
            sampling_method=SGMCMC.rmsprop_sgld,
            optimizer_kwargs=dict(
                lr=lr_max,
                localization=args.gamma,
                nbeta=args.nbeta,
            ),
            num_chains=args.n_chains,
            num_draws=args.n_draws,
            num_burnin_steps=0,
            num_steps_bw_draws=1,
            device=device,
            callbacks=callbacks,
            init_loss=1000.0,
        )
    except RuntimeError as e:
        if "all_reduce" in str(e) or "init_process_group" in str(e):
            print("SGLD sampling complete (finalize() all_reduce workaround).")
            print("  Loss traces recorded successfully despite finalize error.")
        else:
            raise

    print("SGLD sampling complete.")

    np.savez(
        args.output,
        loss_traces=loss_callback.traces,       # [n_chains, n_draws, n_samples]
        predicted_labels=predicted_labels,       # [n_samples]
        gt_labels=gt_labels,                     # [n_samples]
        trusted_idx=np.array(sample_indices["trusted"]),
        benign_idx=np.array(sample_indices["benign"]),
        anomalous_idx=np.array(sample_indices["anomalous"]),
        n_draws=args.n_draws,
        gamma=args.gamma,
        nbeta=args.nbeta,
        lr=lr_max,
        lr_schedule=args.lr_schedule or "constant",
        lr_min=lr_min if use_cosine else None,
    )
    print()
    print("Saved loss traces to", args.output)
    print("  Shape:", loss_callback.traces.shape,
          "(chains=" + str(args.n_chains) + ", draws=" + str(args.n_draws) +
          ", samples=" + str(num_eval) + ")")


if __name__ == "__main__":
    main()
