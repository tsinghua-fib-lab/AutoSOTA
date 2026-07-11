import sys, os, torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from contextlib import contextmanager

@contextmanager
def sdpa_kernel(backend=None):
    yield

sys.path.insert(0, "/repo/isic")
from utils import dilate_masks_torch
from utils_fast import add_artificial_bias_fast as add_artificial_bias
from x_resnet import xfixup_resnet50

DATA_DIR = "/tmp/ISIC2020_2"
VAL_PT = "/tmp/ISIC_precomputed_val.pt"
XRESNET_PATH = "/models/xfixup_resnet50_model_best.pth.tar"
MODEL_PATH = "/models/50_bias/model_seed0_mode_presence_absence_debias.pth"
DEVICE = "cuda:0"
BATCH_SIZE = 128

# Load model
print("Loading model...", flush=True)
model = xfixup_resnet50()
ckpt = torch.load(XRESNET_PATH, map_location="cpu")
sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(sd)
model.fc = nn.Linear(model.fc.in_features, 2)
trained = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(trained)
model = model.to(DEVICE)
model.eval()

# Load pre-computed val data
val_data = torch.load(VAL_PT)
val_images = val_data["images"]
val_labels = val_data["labels"]
print(f"Val data: {val_images.shape}", flush=True)

# GPU val transform
val_images = val_images.to(DEVICE)
mean_t = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Evaluate on all 3 validation splits
biases = {
    "train_bias": {0: 1.0, 1: 0.0},
    "inverse_bias": {0: 0.0, 1: 1.0},
    "no_bias": {0: 0.0, 1: 0.0},
}

for split_name, class_bias in biases.items():
    print(f"\n=== {split_name} ===", flush=True)
    correct_benign = 0
    total_benign = 0
    correct_malignant = 0
    total_malignant = 0
    attr_values = []
    
    # Process in batches
    for i in range(0, len(val_images), BATCH_SIZE):
        imgs = val_images[i:i+BATCH_SIZE]
        lbls = val_labels[i:i+BATCH_SIZE]
        
        # Center crop + normalize
        imgs = imgs.to(DEVICE)
        imgs_224 = imgs[:, :, 16:240, 16:240]  # CenterCrop 224x224
        imgs_224 = (imgs_224 - mean_t) / std_t
        
        # Apply bias
        imgs_biased, patch_seg = add_artificial_bias(imgs_224, lbls.to(DEVICE), class_bias)
        patch_seg = dilate_masks_torch(patch_seg)
        
        # Forward + compute attributions
        imgs_biased.requires_grad = True
        with sdpa_kernel():
            outputs = model(imgs_biased)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Per-class accuracy
        benign_mask = (lbls == 0)
        malignant_mask = (lbls == 1)
        correct_benign += (preds[benign_mask].cpu() == lbls[benign_mask]).sum().item()
        total_benign += benign_mask.sum().item()
        correct_malignant += (preds[malignant_mask].cpu() == lbls[malignant_mask]).sum().item()
        total_malignant += malignant_mask.sum().item()
        
        # Compute Attr metric (target + non-target attribution in bias patches)
        tgt_out = torch.gather(outputs, 1, lbls.to(DEVICE).unsqueeze(-1))
        grads_tgt = torch.autograd.grad(torch.unbind(tgt_out), imgs_biased, create_graph=False, retain_graph=True)[0]
        grads_tgt = imgs_biased * grads_tgt
        attr_tgt = ((grads_tgt.abs().sum(dim=1, keepdim=True) * patch_seg).sum()) / (patch_seg.sum() + 1e-5)
        
        lbls_flip = 1 - lbls.to(DEVICE)
        ntgt_out = torch.gather(outputs, 1, lbls_flip.unsqueeze(-1))
        grads_ntgt = torch.autograd.grad(torch.unbind(ntgt_out), imgs_biased, create_graph=False)[0]
        grads_ntgt = imgs_biased * grads_ntgt
        attr_ntgt = ((grads_ntgt.abs().sum(dim=1, keepdim=True) * patch_seg).sum()) / (patch_seg.sum() + 1e-5)
        
        attr_avg = (attr_tgt + attr_ntgt) / 2
        attr_values.append(attr_avg.item())
    
    benign_acc = correct_benign / max(total_benign, 1)
    malignant_acc = correct_malignant / max(total_malignant, 1)
    avg_acc = (benign_acc + malignant_acc) / 2
    mean_attr = sum(attr_values) / len(attr_values)
    
    print(f"Benign: {correct_benign}/{total_benign} = {benign_acc:.4f}", flush=True)
    print(f"Malignant: {correct_malignant}/{total_malignant} = {malignant_acc:.4f}", flush=True)
    print(f"Avg Accuracy: {avg_acc:.4f}", flush=True)
    print(f"Attr: {mean_attr:.4f}", flush=True)
    
    if split_name == "inverse_bias":
        print(f"\n=== RUBRIC METRICS (inverse_bias) ===", flush=True)
        print(f"Benign Accuracy: {benign_acc:.4f}", flush=True)
        print(f"Malignant Accuracy: {malignant_acc:.4f}", flush=True)
        print(f"Avg Accuracy: {avg_acc:.4f}", flush=True)
        print(f"Attr: {mean_attr:.4f}", flush=True)

print("\n=== Evaluation Complete ===", flush=True)
