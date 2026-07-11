#!/usr/bin/env python3
"""
Fast training script for ISIC debiasing experiment.
Uses pre-computed 256x256 images and GPU-based transforms
to eliminate CPU bottleneck from PIL transforms.
"""
import os
import copy
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

# ----- Paths -----
PRECOMPUTED_TRAIN = '/tmp/ISIC_precomputed_train.pt'
PRECOMPUTED_VAL = '/tmp/ISIC_precomputed_val.pt'
STORE_DIR = '/models/50_bias'
XRESNET_PATH = '/models/xfixup_resnet50_model_best.pth.tar'
BATCH_SIZE = 128
NUM_EPOCHS = 20
RUNS = 1  # Reduced for reproduction
DEVICE = 'cuda:0'
MODE = 'presence_absence_debias'
MODEL = 'xresnet50'
train_wo_bias = False

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

if train_wo_bias:
    print('Training without bias enabled!')

# ----- GPU-based Transforms -----
class FastTrainTransform:
    """GPU-based training transforms: RandomResizedCrop + RandomFlip + ColorJitter + Normalize"""
    def __init__(self, output_size=224, scale=(0.8, 1.0), mean=None, std=None):
        self.output_size = output_size
        self.scale = scale
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.mean_t = torch.tensor(self.mean, device=DEVICE).view(1, 3, 1, 1)
        self.std_t = torch.tensor(self.std, device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, images):
        """images: (B, 3, 256, 256) on GPU"""
        B, C, H, W = images.shape  # H=W=256

        # RandomResizedCrop
        crop_h = crop_w = int(256 * (self.scale[0] + random.random() * (self.scale[1] - self.scale[0])))
        top = random.randint(0, 256 - crop_h)
        left = random.randint(0, 256 - crop_w)

        # Crop
        images = images[:, :, top:top+crop_h, left:left+crop_w]

        # Resize to output_size (bilinear interpolation)
        images = F.interpolate(images, size=(self.output_size, self.output_size),
                               mode='bilinear', align_corners=False)

        # Random horizontal flip
        if random.random() < 0.5:
            images = torch.flip(images, dims=[3])

        # Random vertical flip
        if random.random() < 0.5:
            images = torch.flip(images, dims=[2])

        # Color jitter on GPU
        brightness = 1.0 + random.uniform(-0.2, 0.2)
        contrast = 1.0 + random.uniform(-0.2, 0.2)
        saturation = 1.0 + random.uniform(-0.2, 0.2)

        # Apply brightness (multiply all channels)
        images = images * brightness

        # Apply contrast (mean-centered scaling)
        gray = images.mean(dim=1, keepdim=True)
        images = (images - gray) * contrast + gray

        # Apply saturation (interpolate with grayscale)
        gray_full = images.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        images = gray_full + saturation * (images - gray_full)

        # Clamp
        images = torch.clamp(images, 0.0, 1.0)

        # Normalize
        images = (images - self.mean_t) / self.std_t

        return images


class FastValTransform:
    """GPU-based validation transforms: CenterCrop(224) + Normalize"""
    def __init__(self, output_size=224, mean=None, std=None):
        self.output_size = output_size
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.mean_t = torch.tensor(self.mean, device=DEVICE).view(1, 3, 1, 1)
        self.std_t = torch.tensor(self.std, device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, images):
        """images: (B, 3, 256, 256) on GPU"""
        # Center crop to 224
        start = (256 - self.output_size) // 2
        images = images[:, :, start:start+self.output_size, start:start+self.output_size]
        # Normalize
        images = (images - self.mean_t) / self.std_t
        return images


# ----- Load pre-computed data -----
print("Loading pre-computed data...")
train_data = torch.load(PRECOMPUTED_TRAIN, map_location='cpu')
val_data = torch.load(PRECOMPUTED_VAL, map_location='cpu')

train_images = train_data['images']  # (1168, 3, 256, 256)
train_labels = train_data['labels']
val_images = val_data['images']  # (522, 3, 256, 256)
val_labels = val_data['labels']

print(f"Train: {train_images.shape}, Val: {val_images.shape}")

train_transform = FastTrainTransform()
val_transform = FastValTransform()

# Create dataloaders using TensorDataset (shuffle=True handles randomness)
train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = ['benign', 'malignant']


# ----- Training Loop -----
def train_model(model, optimizer, attribution_prior_weight, best_acc, num_epochs=NUM_EPOCHS):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_current = 0

    if train_wo_bias:
        class_bias_train = {0: 0.0, 1: 0.0}
    else:
        class_bias_train = {0: 1.0, 1: 0.0}

    class_bias_inverse = {0: 0.0, 1: 1.0}
    class_bias_off = {0: 0.0, 1: 0.0}

    print(f'attribution_prior_weight: {attribution_prior_weight}')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train_', 'val_trainbias', 'val_inversebias', 'val_nobias']:
            is_train = (phase == 'train_')
            print(f'Starting {phase}')
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

                # Apply bias
                if phase == 'train_':
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_train)
                elif phase == 'val_trainbias':
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_train)
                elif phase == 'val_inversebias':
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_inverse)
                else:  # val_nobias
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_off)

                patch_segmentation = dilate_masks_torch(patch_segmentation)
                optimizer.zero_grad()

                with sdpa_kernel(SDPBackend.MATH if hasattr(SDPBackend, 'MATH') else 'math'):
                    inputs.requires_grad = True
                    outputs = model(inputs)

                    if MODE == 'presence_debias' or MODE == 'presence_absence_debias':
                        target_outputs = torch.gather(outputs, 1, labels.unsqueeze(-1))
                        gradients = torch.autograd.grad(
                            torch.unbind(target_outputs), inputs,
                            create_graph=True, retain_graph=True
                        )[0]
                        gradients = inputs * gradients
                        attribution_inside1 = (
                            (gradients.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()
                        ) / (patch_segmentation.sum() + 1e-5)
                        loss_attribution_prior1 = attribution_inside1

                        if MODE == 'presence_absence_debias':
                            labels_flipped = 1 - labels
                            target_outputs = torch.gather(outputs, 1, labels_flipped.unsqueeze(-1))
                            gradients = torch.autograd.grad(
                                torch.unbind(target_outputs), inputs,
                                create_graph=True
                            )[0]
                            gradients = inputs * gradients
                            attribution_inside2 = (
                                (gradients.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()
                            ) / (patch_segmentation.sum() + 1e-5)
                            loss_attribution_prior2 = attribution_inside2
                        else:
                            loss_attribution_prior2 = loss_attribution_prior1

                        loss_attribution_prior = (loss_attribution_prior1 + loss_attribution_prior2) / 2
                    else:
                        loss_attribution_prior = 0

                loss_classification = F.binary_cross_entropy_with_logits(
                    outputs, labels_one_hot,
                    pos_weight=torch.ones([2]).to(DEVICE)
                )
                loss = loss_classification + attribution_prior_weight * loss_attribution_prior
                preds = torch.argmax(outputs, dim=1)

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val_nobias' and epoch_acc > best_acc_current:
                best_acc_current = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model saved (val acc: {best_acc_current:.4f})")

    print(f"\nBest validation accuracy: {best_acc_current:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc_current


# ----- Train -----
os.makedirs(STORE_DIR, exist_ok=True)

for seed in range(RUNS):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    best_acc = 0.0

    for attribution_prior_weight in [1, 10, 100, 1000, 10000]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load model
        model = xfixup_resnet50()
        checkpoint = torch.load(XRESNET_PATH, map_location='cpu')
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

        model, best_acc_current = train_model(model, optimizer, attribution_prior_weight, best_acc)

        if best_acc_current >= best_acc:
            best_acc = best_acc_current
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    save_path = os.path.join(STORE_DIR, f'model_seed{seed}_mode_{MODE}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nBest model saved to {save_path}")

print("\n=== Training Complete ===")
