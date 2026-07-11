#!/usr/bin/env python3
"""
Enhanced training script for ISIC debiasing experiment.
Extends the original train_fast.py with argparse CLI, focal loss, entropy attribution,
MixUp, SmoothGrad, adaptive lambda, and multi-seed support.
"""
import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.models as models
import random
import numpy as np

from utils import dilate_masks_torch
from utils_fast import add_artificial_bias_fast as add_artificial_bias
from x_resnet import xfixup_resnet50

# Compatible sdpa_kernel import
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except (ImportError, ModuleNotFoundError):
    try:
        from torch.backends.cuda import sdp_kernel, SDPBackend
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend):
            yield
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend=None):
            yield
        class SDPBackend:
            MATH = "math"


def parse_args():
    p = argparse.ArgumentParser(description="ISIC debiasing training with XResNet-50")
    # Paths
    p.add_argument("--precomputed_train", default="/tmp/ISIC_precomputed_train.pt")
    p.add_argument("--precomputed_val", default="/tmp/ISIC_precomputed_val.pt")
    p.add_argument("--store_dir", default="/models/50_bias")
    p.add_argument("--xresnet_path", default="/models/xfixup_resnet50_model_best.pth.tar")
    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--runs", type=int, default=1, help="Number of seeds")
    p.add_argument("--seed", type=int, default=0, help="Starting seed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", default="presence_absence_debias",
                   choices=["presence_debias", "presence_absence_debias"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_values", type=float, nargs="+",
                   default=[1, 10, 100, 1000, 10000],
                   help="Attribution prior weights to grid search")
    # Augmentation
    p.add_argument("--brightness", type=float, default=0.2, help="Brightness jitter range")
    p.add_argument("--contrast", type=float, default=0.2, help="Contrast jitter range")
    p.add_argument("--saturation", type=float, default=0.2, help="Saturation jitter range")
    p.add_argument("--crop_scale_min", type=float, default=0.8)
    p.add_argument("--crop_scale_max", type=float, default=1.0)
    # Focal loss (Idea-02)
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss gamma (0 = standard BCE)")
    # Attribution loss type (Idea-01)
    p.add_argument("--attribution_loss_type", default="l1",
                   choices=["l1", "entropy"],
                   help="l1 = L1 penalty on attribution inside bias patches; entropy = entropy-based sparsity")
    p.add_argument("--entropy_weight", type=float, default=0.5,
                   help="Weight for entropy attribution loss term")
    # MixUp (Idea-03)
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="MixUp alpha (0 = disabled)")
    # SmoothGrad (Idea-04)
    p.add_argument("--smoothgrad_samples", type=int, default=1,
                   help="SmoothGrad noise samples (1 = single-pass IG)")
    p.add_argument("--smoothgrad_sigma", type=float, default=0.1,
                   help="SmoothGrad noise std dev")
    # Adaptive lambda (Idea-11)
    p.add_argument("--adaptive_lambda", action="store_true", default=False,
                   help="Enable per-sample adaptive attribution weight")
    p.add_argument("--lambda_adapt_strength", type=float, default=1.0,
                   help="Strength of adaptive lambda modulation")
    # Train without bias
    p.add_argument("--train_wo_bias", action="store_true", default=False)
    return p.parse_args()


args = parse_args()

# Set seeds
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
DEVICE = args.device

if args.train_wo_bias:
    print("Training without bias enabled!")


# ----- GPU-based Transforms -----
class FastTrainTransform:
    def __init__(self, output_size=224, scale=(0.8, 1.0), mean=None, std=None,
                 brightness=0.2, contrast=0.2, saturation=0.2):
        self.output_size = output_size
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.mean = mean if mean is not None else imagenet_mean
        self.std = std if std is not None else imagenet_std
        self.mean_t = torch.tensor(self.mean, device=DEVICE).view(1, 3, 1, 1)
        self.std_t = torch.tensor(self.std, device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, images):
        B, C, H, W = images.shape
        crop_h = crop_w = int(256 * (self.scale[0] + random.random() * (self.scale[1] - self.scale[0])))
        top = random.randint(0, 256 - crop_h)
        left = random.randint(0, 256 - crop_w)
        images = images[:, :, top:top+crop_h, left:left+crop_w]
        images = F.interpolate(images, size=(self.output_size, self.output_size),
                               mode="bilinear", align_corners=False)
        if random.random() < 0.5:
            images = torch.flip(images, dims=[3])
        if random.random() < 0.5:
            images = torch.flip(images, dims=[2])
        brightness = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation = 1.0 + random.uniform(-self.saturation, self.saturation)
        images = images * brightness
        gray = images.mean(dim=1, keepdim=True)
        images = (images - gray) * contrast + gray
        gray_full = images.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        images = gray_full + saturation * (images - gray_full)
        images = torch.clamp(images, 0.0, 1.0)
        images = (images - self.mean_t) / self.std_t
        return images


class FastValTransform:
    def __init__(self, output_size=224, mean=None, std=None):
        self.output_size = output_size
        self.mean = mean if mean is not None else imagenet_mean
        self.std = std if std is not None else imagenet_std
        self.mean_t = torch.tensor(self.mean, device=DEVICE).view(1, 3, 1, 1)
        self.std_t = torch.tensor(self.std, device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, images):
        start = (256 - self.output_size) // 2
        images = images[:, :, start:start+self.output_size, start:start+self.output_size]
        images = (images - self.mean_t) / self.std_t
        return images


# ----- Focal Loss -----
def focal_binary_cross_entropy_with_logits(outputs, labels_one_hot, gamma=2.0):
    """Focal loss for binary classification. gamma=0 is equivalent to standard BCE."""
    if gamma == 0.0:
        return F.binary_cross_entropy_with_logits(
            outputs, labels_one_hot,
            pos_weight=torch.ones([2]).to(DEVICE))
    bce_loss = F.binary_cross_entropy_with_logits(
        outputs, labels_one_hot, reduction="none")
    p = torch.sigmoid(outputs)
    p_t = p * labels_one_hot + (1 - p) * (1 - labels_one_hot)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce_loss).mean()


# ----- Attribution Loss -----
def compute_attribution_loss(gradients, patch_segmentation, loss_type="l1"):
    """Compute attribution prior loss."""
    if loss_type == "l1":
        attr_inside = (
            (gradients.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()
        ) / (patch_segmentation.sum() + 1e-5)
        return attr_inside
    elif loss_type == "entropy":
        attr_abs = gradients.abs().sum(dim=1, keepdim=True)
        attr_total = attr_abs.sum(dim=(2, 3), keepdim=True) + 1e-8
        attr_dist = (attr_abs * patch_segmentation) / attr_total
        attr_dist = torch.clamp(attr_dist, min=1e-8)
        entropy = -(attr_dist * torch.log(attr_dist)).sum()
        entropy = entropy / gradients.shape[0]
        return entropy
    else:
        raise ValueError(f"Unknown attribution loss type: {loss_type}")


# ----- MixUp -----
def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation. Returns mixed inputs and pairs of labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ----- Load pre-computed data -----
print("Loading pre-computed data...")
train_data = torch.load(args.precomputed_train, map_location="cpu")
val_data = torch.load(args.precomputed_val, map_location="cpu")

train_images = train_data["images"]
train_labels = train_data["labels"]
val_images = val_data["images"]
val_labels = val_data["labels"]

print(f"Train: {train_images.shape}, Val: {val_images.shape}")

train_transform = FastTrainTransform(
    scale=(args.crop_scale_min, args.crop_scale_max),
    brightness=args.brightness, contrast=args.contrast, saturation=args.saturation)
val_transform = FastValTransform()

train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


# ----- Training Loop -----
def train_model(model, optimizer, attribution_prior_weight, best_acc, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_current = 0

    if args.train_wo_bias:
        class_bias_train = {0: 0.0, 1: 0.0}
    else:
        class_bias_train = {0: 1.0, 1: 0.0}

    class_bias_inverse = {0: 0.0, 1: 1.0}
    class_bias_off = {0: 0.0, 1: 0.0}

    print(f"attribution_prior_weight: {attribution_prior_weight}")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train_", "val_trainbias", "val_inversebias", "val_nobias"]:
            is_train = (phase == "train_")
            print(f"Starting {phase}")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            loader = train_loader if is_train else val_loader

            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Apply GPU transforms
                if is_train:
                    inputs = train_transform(inputs)
                else:
                    inputs = val_transform(inputs)

                labels_one_hot = F.one_hot(labels, num_classes=2).float()

                # MixUp (only during training)
                mixup_labels_a = labels
                mixup_labels_b = labels
                mixup_lam = 1.0
                if is_train and args.mixup_alpha > 0.0:
                    inputs, mixup_labels_a, mixup_labels_b, mixup_lam = mixup_data(
                        inputs, labels, args.mixup_alpha)

                # Apply bias
                if phase == "train_":
                    inputs_biased, patch_segmentation = add_artificial_bias(
                        inputs, labels, class_bias_train)
                elif phase == "val_trainbias":
                    inputs_biased, patch_segmentation = add_artificial_bias(
                        inputs, labels, class_bias_train)
                elif phase == "val_inversebias":
                    inputs_biased, patch_segmentation = add_artificial_bias(
                        inputs, labels, class_bias_inverse)
                else:
                    inputs_biased, patch_segmentation = add_artificial_bias(
                        inputs, labels, class_bias_off)

                patch_segmentation = dilate_masks_torch(patch_segmentation)
                optimizer.zero_grad()

                with sdpa_kernel(SDPBackend.MATH if hasattr(SDPBackend, "MATH") else "math"):
                    # --- Compute attribution loss ---
                    loss_attribution_prior = torch.tensor(0.0, device=DEVICE)

                    if args.mode in ("presence_debias", "presence_absence_debias"):
                        # Compute gradients for target class attribution
                        grad_inputs = inputs_biased.detach().clone()
                        grad_inputs.requires_grad = True

                        if args.smoothgrad_samples > 1:
                            # SmoothGrad: average gradients over noisy inputs
                            input_std = grad_inputs.std().item() or 1.0
                            noise_std_val = args.smoothgrad_sigma * input_std
                            accumulated_grad = torch.zeros_like(grad_inputs)
                            for n in range(args.smoothgrad_samples):
                                noise = torch.randn_like(grad_inputs) * noise_std_val
                                noisy_in = grad_inputs + noise
                                noisy_in.requires_grad = True
                                out_n = model(noisy_in)
                                tgt_out = torch.gather(out_n, 1, labels.unsqueeze(-1))
                                g_n = torch.autograd.grad(
                                    torch.unbind(tgt_out), noisy_in,
                                    create_graph=True, retain_graph=True
                                )[0]
                                accumulated_grad = accumulated_grad + g_n
                            gradients1 = accumulated_grad / args.smoothgrad_samples
                        else:
                            out_for_grad = model(grad_inputs)
                            tgt_out = torch.gather(out_for_grad, 1, labels.unsqueeze(-1))
                            gradients1 = torch.autograd.grad(
                                torch.unbind(tgt_out), grad_inputs,
                                create_graph=True, retain_graph=True
                            )[0]

                        gradients1 = grad_inputs * gradients1
                        loss_attr1 = compute_attribution_loss(
                            gradients1, patch_segmentation, args.attribution_loss_type)

                        if args.mode == "presence_absence_debias":
                            # Flipped label attribution
                            labels_flipped = 1 - labels
                            grad_inputs2 = inputs_biased.detach().clone()
                            grad_inputs2.requires_grad = True
                            out2 = model(grad_inputs2)
                            tgt_out2 = torch.gather(out2, 1, labels_flipped.unsqueeze(-1))
                            gradients2 = torch.autograd.grad(
                                torch.unbind(tgt_out2), grad_inputs2,
                                create_graph=True
                            )[0]
                            gradients2 = grad_inputs2 * gradients2
                            loss_attr2 = compute_attribution_loss(
                                gradients2, patch_segmentation, args.attribution_loss_type)
                        else:
                            loss_attr2 = loss_attr1

                        loss_attribution_prior = (loss_attr1 + loss_attr2) / 2

                    # --- Forward pass for classification ---
                    # Detach biased inputs for clean classification forward pass
                    inputs_cls = inputs_biased.detach()
                    outputs = model(inputs_cls)

                    # --- Classification loss ---
                    if is_train and args.mixup_alpha > 0.0:
                        # MixUp loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
                        ya_one_hot = F.one_hot(mixup_labels_a, num_classes=2).float()
                        yb_one_hot = F.one_hot(mixup_labels_b, num_classes=2).float()
                        loss_cls_a = focal_binary_cross_entropy_with_logits(
                            outputs, ya_one_hot, args.focal_gamma)
                        loss_cls_b = focal_binary_cross_entropy_with_logits(
                            outputs, yb_one_hot, args.focal_gamma)
                        loss_classification = mixup_lam * loss_cls_a + (1 - mixup_lam) * loss_cls_b
                    else:
                        loss_classification = focal_binary_cross_entropy_with_logits(
                            outputs, labels_one_hot, args.focal_gamma)

                    # --- Adaptive lambda (Idea-11) ---
                    if args.adaptive_lambda and is_train:
                        with torch.no_grad():
                            per_sample_loss = F.binary_cross_entropy_with_logits(
                                outputs, labels_one_hot, reduction="none").mean(dim=1)
                            loss_min = per_sample_loss.min()
                            loss_max = per_sample_loss.max()
                            if loss_max - loss_min > 1e-6:
                                normalized_loss = (per_sample_loss - loss_min) / (loss_max - loss_min)
                            else:
                                normalized_loss = torch.ones_like(per_sample_loss) * 0.5
                            adaptive_weights = 1.0 - args.lambda_adapt_strength * normalized_loss
                        loss = loss_classification + attribution_prior_weight * loss_attribution_prior * adaptive_weights.mean()
                    else:
                        loss = loss_classification + attribution_prior_weight * loss_attribution_prior

                preds = torch.argmax(outputs, dim=1)

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if is_train and args.mixup_alpha > 0.0:
                    running_corrects += mixup_lam * (preds == mixup_labels_a).sum().item() + (1 - mixup_lam) * (preds == mixup_labels_b).sum().item()
                else:
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects / len(loader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val_nobias" and epoch_acc > best_acc_current:
                best_acc_current = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model saved (val acc: {best_acc_current:.4f})")

    print(f"\nBest validation accuracy: {best_acc_current:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc_current


# ----- Train -----
os.makedirs(args.store_dir, exist_ok=True)

best_overall_acc = 0.0
best_overall_model = None

for run_idx in range(args.runs):
    current_seed = args.seed + run_idx
    print(f"\n{'='*60}")
    print(f"Seed {current_seed} (run {run_idx+1}/{args.runs})")
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")
    print(f"{'='*60}")

    best_acc = 0.0

    for attribution_prior_weight in args.lambda_values:
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        model = xfixup_resnet50()
        checkpoint = torch.load(args.xresnet_path, map_location="cpu")
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model, best_acc_current = train_model(
            model, optimizer, attribution_prior_weight, best_acc, num_epochs=args.epochs)

        if best_acc_current >= best_acc:
            best_acc = best_acc_current
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    save_path = os.path.join(args.store_dir, f"model_seed{current_seed}_mode_{args.mode}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nBest model for seed {current_seed} saved to {save_path} (val_acc={best_acc:.4f})")

    if best_acc > best_overall_acc:
        best_overall_acc = best_acc
        best_overall_model = copy.deepcopy(model.state_dict())

# Save best overall as default eval path
default_save_path = os.path.join(args.store_dir, f"model_seed0_mode_{args.mode}.pth")
torch.save(best_overall_model, default_save_path)
print(f"\nBest overall model saved to {default_save_path} (val_acc={best_overall_acc:.4f})")
print("\n=== Training Complete ===")
