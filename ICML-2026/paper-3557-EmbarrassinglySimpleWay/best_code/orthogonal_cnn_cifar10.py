"""
Orthogonal CNN on CIFAR-10 using POGO optimizer.
Reproduces the "Orthogonal filters (6 matrices)" experiment from Figure 1 of paper 3557.

Settings from paper Appendix D.3:
- Architecture: CNN adapted from cifar10-airbench (KellerJordan, 2024)
- Orthogonal filters: flatten O×I×k×k conv weights to O×(I·k²), keep row-orthogonal
- 6 matrices sizes: 64×216, 64×576, 256×576, 256×2304, 256×2304, 256×2304
- POGO with VAdam base optimizer, lr=0.5, lambda=0.5
- 100 epochs, 5 independent runs
- Standard CIFAR-10 data augmentation (flip + random crop translation)
"""
import os
import sys
import time
import json
import uuid
import random
import argparse
from math import ceil, cos, pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

from pogo import base, POGO

# Non-compiled Newton-Schulz for Muon
def _ns5_eager(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class MuonEager(base.SGD):
    def __call__(self, point, grad, state, group):
        grad = super().__call__(point, grad, state, group)
        return _ns5_eager(grad.reshape(len(grad), -1)).view(grad.shape)


# =============================================
# DataLoader (from airbench)
# =============================================

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, device="cuda"):
        self.device = device
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(device))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])

# =============================================
# Network Definition
# =============================================

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False

class OrthogonalConv(nn.Conv2d):
    """Conv layer whose weights are constrained to be row-orthogonal."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        # Diract initialization for the first in_channels channels
        torch.nn.init.dirac_(w[:w.size(1)])

    def get_orthogonal_weight(self):
        """Return the weight reshaped as [out_channels, in_channels*k*k] for orthogonal optimization."""
        w = self.weight.data
        return w.reshape(w.size(0), -1)

    def set_orthogonal_weight(self, flat_w):
        """Set the weight from the flattened orthogonal representation."""
        w_shape = self.weight.data.shape
        self.weight.data = flat_w.view(w_shape)

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = OrthogonalConv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = OrthogonalConv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class OrthogonalCifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size ** 2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            # Keep everything in float32 - POGO needs higher precision
            # for its internal matrix operations

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, OrthogonalConv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def initialize_orthogonal(self):
        """Initialize all orthogonal conv weights using SVD projection."""
        for m in self.modules():
            if isinstance(m, OrthogonalConv):
                w_flat = m.get_orthogonal_weight()
                # SVD requires float32 precision
                U, S, Vt = torch.linalg.svd(w_flat.float(), full_matrices=False)
                m.set_orthogonal_weight((U @ Vt).to(w_flat.dtype))

    def collect_orthogonal_params(self):
        """Collect all orthogonal conv weight parameters as flat matrices."""
        params = []
        for m in self.modules():
            if isinstance(m, OrthogonalConv):
                w = m.weight  # The actual parameter reference
                params.append(w)
        return params

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

# =============================================
# Custom POGO wrapper that handles Conv weight flattening
# =============================================

def conv_flatten_fn(w):
    """Flatten O×I×k×k conv weight to [1, O, I·k·k] for POGO orthogonal optimization.
    POGO requires [*batch_dims, p, n] with at least one batch dimension."""
    return w.view(w.size(0), -1).unsqueeze(0)  # [O, I*k*k] -> [1, O, I*k*k]

class OrthogonalConvPOGO(torch.optim.Optimizer):
    """
    POGO optimizer specialized for Conv weights.
    Wraps POGO to handle Conv weight flattening automatically.
    """
    def __init__(self, pogo_opt):
        self.pogo_opt = pogo_opt
        defaults = dict()
        super().__init__([], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # Update the POGO-internal param groups to reflect current conv weight flattening
        for group in self.pogo_opt.param_groups:
            for i, p in enumerate(group['params']):
                if len(p.shape) == 4:  # Conv weight
                    # The POGO optimizer has its flatten_fn applied internally,
                    # but we need to ensure the grad matches the flattened shape
                    pass
        return self.pogo_opt.step(closure)


# =============================================
# Evaluation
# =============================================

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net) for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

# =============================================
# Training
# =============================================

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_min"]

def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


def train_one_run(run_id, model, seed, num_epochs=100, lr=0.5):
    """Train one run and return final accuracy and training time."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    data_path = "/datasets/cifar10"
    os.makedirs(data_path, exist_ok=True)

    test_loader = CifarLoader(data_path, train=False, batch_size=2000)
    train_loader = CifarLoader(data_path, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))

    total_train_steps = num_epochs * len(train_loader)
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Collect orthogonal and non-orthogonal parameters
    orthogonal_params = []
    non_orthogonal_params = []
    norm_biases = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "conv" in n.lower() and len(p.shape) == 4 and "whiten" not in n.lower():
            orthogonal_params.append(p)
        elif "norm" in n and p.requires_grad:
            norm_biases.append(p)
        elif n == "whiten.bias":
            non_orthogonal_params.append(p)
        elif n == "head.weight":
            non_orthogonal_params.append(p)

    # Create POGO optimizer for orthogonal conv weights
    pogo_opt = POGO(
        orthogonal_params,
        MuonEager(momentum=0.9),
        lr=lr,
        weight_decay=0,
        flatten_fn=conv_flatten_fn,
        rows=True,  # row-orthogonal (O < I*k*k)
        lambda_every=-1,  # Use λ=0.5 (default)
    )

    # Create SGD optimizer for non-orthogonal parameters
    param_configs = []
    if non_orthogonal_params:
        param_configs.append(dict(params=non_orthogonal_params, lr=bias_lr, weight_decay=wd / bias_lr))
    if norm_biases:
        param_configs.append(dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr))
    if param_configs:
        sgd_opt = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
    else:
        sgd_opt = None

    # LR schedulers (linear decay)
    for group in pogo_opt.param_groups:
        group["initial_lr"] = group["lr"]
    if sgd_opt:
        for group in sgd_opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    # Initialize orthogonal conv weights
    model.initialize_orthogonal()
    step = 0

    # Initialize whitening layer
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(num_epochs):
        start_timer()
        model.train()

        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))

            # Mixup augmentation (3557-ALGO-03)
            mixup_alpha = 0.2
            if random.random() < 0.5:
                lam = random.betavariate(mixup_alpha, mixup_alpha)
                perm = torch.randperm(len(labels), device=labels.device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
                mixed_outputs = model(mixed_inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
                loss = lam * F.cross_entropy(mixed_outputs, labels, label_smoothing=0.2, reduction="sum") +                        (1 - lam) * F.cross_entropy(mixed_outputs, labels[perm], label_smoothing=0.2, reduction="sum")
            else:
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            loss.backward()

            # Cosine LR schedule with warmup (3557-ALGO-07)
            warmup_steps = len(train_loader) * 5
            if step < warmup_steps:
                warmup_factor = 0.1 + 0.9 * (step / warmup_steps)
                for group in pogo_opt.param_groups:
                    group["lr"] = group["initial_lr"] * warmup_factor
                if sgd_opt:
                    for group in sgd_opt.param_groups:
                        group["lr"] = group["initial_lr"] * warmup_factor
            else:
                progress = (step - warmup_steps) / (total_train_steps - warmup_steps)
                cos_factor = 0.5 * (1 + cos(pi * progress))
                eta_min = 0.01
                for group in pogo_opt.param_groups:
                    group["lr"] = group["initial_lr"] * (eta_min + (1 - eta_min) * cos_factor)
                if sgd_opt:
                    for group in sgd_opt.param_groups:
                        group["lr"] = group["initial_lr"] * (eta_min + (1 - eta_min) * cos_factor)

            # Step optimizers
            pogo_opt.step()
            if sgd_opt:
                sgd_opt.step()

            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break

        stop_timer()

        # Evaluation
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)

        run_label = run_id if epoch == 0 else ""
        print_training_details({
            "run": run_label,
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "tta_val_acc": None,
            "time_min": time_seconds / 60.0,
        }, is_final_entry=(epoch == num_epochs - 1))

    # Final TTA evaluation
    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()

    # Print final summary
    final_time_min = time_seconds / 60.0
    print(f"\nRun {run_id} final: TTA={tta_val_acc:.4f}, Time={final_time_min:.2f} min\n")

    return {
        "run": run_id,
        "seed": seed,
        "final_val_acc": tta_val_acc,
        "final_no_tta_acc": val_acc,
        "training_time_minutes": final_time_min,
        "epochs_completed": num_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Orthogonal CNN CIFAR-10 with POGO")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate for POGO")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output", type=str, default=None, help="JSON output file for results")
    args = parser.parse_args()

    print(f"Starting Orthogonal CNN CIFAR-10 experiment with POGO")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Runs: {args.runs}, Base seed: {args.seed}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
    print()

    print_columns(logging_columns_list, is_head=True)

    all_results = []
    total_time_start = time.time()

    for run in range(args.runs):
        seed = args.seed + run
        model = OrthogonalCifarNet().cuda().to(memory_format=torch.channels_last)

        result = train_one_run(run + 1, model, seed=seed, num_epochs=args.epochs, lr=args.lr)
        all_results.append(result)

    total_time = time.time() - total_time_start

    # Summarize
    accs = [r["final_val_acc"] for r in all_results]
    times = [r["training_time_minutes"] for r in all_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_time = np.mean(times)

    print(f"\n{'='*70}")
    print(f"SUMMARY across {args.runs} runs:")
    print(f"  Accuracy (TTA): {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Individual: {[f'{a:.4f}' for a in accs]}")
    print(f"  Training time: {mean_time:.2f} ± {np.std(times):.2f} min")
    print(f"  Individual: {[f'{t:.2f}' for t in times]}")
    print(f"  Total wall time: {total_time/60:.2f} min")
    print(f"{'='*70}")

    if args.output:
        output = {
            "experiment": "orthogonal_cnn_cifar10_pogo",
            "settings": {
                "epochs": args.epochs,
                "lr": args.lr,
                "runs": args.runs,
                "base_seed": args.seed,
            },
            "results": all_results,
            "summary": {
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "mean_time_minutes": float(mean_time),
                "std_time_minutes": float(np.std(times)),
                "total_wall_time_minutes": total_time / 60.0,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
