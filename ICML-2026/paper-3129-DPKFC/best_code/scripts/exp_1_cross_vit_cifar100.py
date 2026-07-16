import numpy as np
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus.accountants.utils import get_noise_multiplier
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
import itertools
import timm
from tqdm import tqdm
import random
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    """Ensures reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- 1. Model Definition (CrossViT) ---
class CrossViTClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Load pretrained CrossViT-15-240 model
        # num_classes=0 removes the head and pooling, gives us raw features
        self.backbone = timm.create_model(
            "crossvit_tiny_240", pretrained=True, num_classes=0
        )
        
        # Freeze the backbone immediately
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the feature dimension automatically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 240, 240)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # Add Trainable Classification Head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# --- 2. KFAC Components (Linear Layer Only) ---
class KFACRecorder:
    """Captures activations and backprops for the trainable Linear layer."""
    def __init__(self, model):
        self.handles = []
        self.activations = {}
        self.backprops = {}

        # Handle Opacus wrapper if present
        target_model = model._module if hasattr(model, "_module") else model

        for name, module in target_model.named_modules():
            # We only care about the classifier head, as the backbone is frozen
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                # print(f"Registering KFAC hooks for: {name}")
                self.handles.append(module.register_forward_hook(self._save_activation(name)))
                self.handles.append(module.register_full_backward_hook(self._save_backprop(name)))

    def _save_activation(self, name):
        def hook(module, inputs, outputs):
            if getattr(self, "enabled", True):
                self.activations[name] = inputs[0].detach()
        return hook

    def _save_backprop(self, name):
        def hook(module, grad_input, grad_output):
            if getattr(self, "enabled", True):
                self.backprops[name] = grad_output[0].detach()
        return hook
    
    def enable(self): self.enabled = True
    def disable(self): self.enabled = False
    
    def remove(self):
        for h in self.handles: h.remove()

def compute_covariances(acts, grads):
    cov_A, cov_G = {}, {}
    eps = 1e-5
    for name in acts:
        A = acts[name]  # [B, in_features]
        G = grads[name]  # [B, out_features]

        # Add bias term to activations: A = [A, 1]
        A_with_bias = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)

        cov_A[name] = (A_with_bias.T @ A_with_bias) / A.size(0) + eps * torch.eye(
            A_with_bias.size(1), device=A.device
        )
        cov_G[name] = (G.T @ G) / G.size(0) + eps * torch.eye(
            G.size(1), device=G.device
        )
        # cov_G[name] = torch.eye(G.size(1), device=G.device)
    return cov_A, cov_G

def precondition_per_sample_gradients(model, cov_A, cov_G):
    eps = 1e-3
    target_model = model._module if hasattr(model, "_module") else model

    for name, module in target_model.named_modules():
        if (isinstance(module, nn.Linear) and 
            module.weight.requires_grad and
            module.weight.grad_sample is not None and
            name in cov_G):

            A = cov_A[name]
            G = cov_G[name]

            # Invert A
            eva_A, evc_A = torch.linalg.eigh(A + eps * torch.eye(A.size(0), device=A.device))
            inv_sqrt_A = evc_A @ torch.diag(eva_A.clamp(min=1e-6).rsqrt()) @ evc_A.T

            # Invert G
            eva_G, evc_G = torch.linalg.eigh(G + eps * torch.eye(G.size(0), device=G.device))
            inv_sqrt_G = evc_G @ torch.diag(eva_G.clamp(min=1e-6).rsqrt()) @ evc_G.T

            g_sample_w = module.weight.grad_sample
            g_sample_b = module.bias.grad_sample

            if isinstance(g_sample_w, list): g_sample_w = g_sample_w[-1]
            if isinstance(g_sample_b, list): g_sample_b = g_sample_b[-1]

            # [B, out, in+1]
            g_sample_augmented = torch.cat([g_sample_w, g_sample_b.unsqueeze(2)], dim=2)
            
            # KFAC Projection: inv_G @ Grad @ inv_A
            temp = torch.einsum("oj, bjk -> bok", inv_sqrt_G, g_sample_augmented)
            preconditioned = torch.einsum("bok, ki -> boi", temp, inv_sqrt_A)

            module.weight.grad_sample = preconditioned[:, :, :-1]
            module.bias.grad_sample = preconditioned[:, :, -1]

def apply_dp(model, noise_multiplier, max_grad_norm, batch_size):
    total_norm_sq = None
    params = [p for p in model.parameters() if hasattr(p, "grad_sample") and p.grad_sample is not None]

    # 1. Compute Total Norm
    for p in params:
        gs = p.grad_sample
        if isinstance(gs, list): gs = gs[-1]
        gs = gs.contiguous().view(batch_size, -1)
        norms = gs.norm(2, dim=1).pow(2)
        total_norm_sq = norms if total_norm_sq is None else total_norm_sq + norms

    total_norm = total_norm_sq.sqrt() if total_norm_sq is not None else torch.tensor(0.0)
    clip_factors = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

    # 2. Clip & Add Noise
    for p in params:
        gs = p.grad_sample
        if isinstance(gs, list): gs = gs[-1]
        gs = gs.contiguous().view(batch_size, -1)
        
        grad_sample_clipped = gs * clip_factors.unsqueeze(1)
        summed_grad = torch.sum(grad_sample_clipped, dim=0)
        
        noise = torch.randn_like(summed_grad) * noise_multiplier * max_grad_norm
        
        p.grad = ((summed_grad + noise) / batch_size).view_as(p.grad)
        p.grad_sample = None

def generate_synthetic_images(batch_size, device):
    """
    Generates 'Pink Noise' (Mixed Frequency).
    Combines low-frequency structure (blobs) with high-frequency detail (grain).
    This ensures BOTH branches of CrossViT (Small and Large) are activated.
    """
    # 1. Low Frequency (Blobs) - Simulates Objects
    low_res = torch.randn(batch_size, 3, 32, 32, device=device)
    blobs = F.interpolate(low_res, size=(240, 240), mode='bilinear', align_corners=False)

    # 2. High Frequency (Grain) - Simulates Texture/Edges
    grain = torch.randn(batch_size, 3, 240, 240, device=device)

    # 3. Combine them (Weighted Sum)
    mixed_noise = 0.6 * blobs + 0.4 * grain

    # 4. Normalize to match CIFAR-100 statistics
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    final_input = (mixed_noise - mean) / std

    # 5. Random Labels
    random_labels = torch.randint(0, 100, (batch_size,), device=device)

    return final_input, random_labels
# --- 3. Training Loops ---

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_accum = 0
    criterion = nn.CrossEntropyLoss()
    
    target_model = model._module if hasattr(model, "_module") else model

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = target_model(x)
            loss_accum += criterion(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return correct / total, loss_accum / len(dataloader)

def train_plain_sgd(model, dataloader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def train_dp_sgd(
    model, private_loader, public_loader, optimizer,
    noise_multiplier, sample_rate, max_grad_norm, num_epochs,
    kfac=False, public_data=True
):
    accountant = RDPAccountant()
    recorder = KFACRecorder(model) if kfac else None
    public_iter = iter(itertools.cycle(public_loader)) if kfac and public_data else None
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        idx = 0        
        for x_private, y_private in private_loader:
            x_private, y_private = x_private.to(DEVICE), y_private.to(DEVICE)
            curr_bs = x_private.size(0)
            model.zero_grad(set_to_none=True)

            # --- KFAC Preconditioning Step ---
            if kfac and idx % len(private_loader) == 0:
            # if kfac and idx % 1 == 0:
                recorder.enable()
                if public_data:
                    x_pub, y_pub = next(public_iter)
                    x_pub, y_pub = x_pub.to(DEVICE), y_pub.to(DEVICE)

                    out_pub = model(x_pub)
                    loss_pub = F.cross_entropy(out_pub, y_pub)
                    loss_pub.backward()
                else:
                    # Synthetic Noise
                    x_noise = torch.rand_like(torch.empty(curr_bs, 3, 240, 240, device=DEVICE))
                    y_noise = torch.randint(0, 100, (curr_bs,), device=DEVICE)

                    out_noise = model(x_noise)
                    loss_noise = F.cross_entropy(out_noise, y_noise)
                    loss_noise.backward()

                cov_A, cov_G = compute_covariances(recorder.activations, recorder.backprops)
                recorder.disable()
                model.zero_grad(set_to_none=True)

            # --- DP-SGD Step ---
            out_private = model(x_private)
            loss_private = criterion(out_private, y_private)
            loss_private.backward()

            if kfac:
                precondition_per_sample_gradients(model, cov_A, cov_G)

            apply_dp(model, noise_multiplier, max_grad_norm, curr_bs)
            optimizer.step()
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            idx += 1

        if recorder:
            recorder.activations.clear()
            recorder.backprops.clear()

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Experiments Configuration
    EPSILONS = [0.5, 1.0, 2, 3, 5.0, 7.5, 10]
    SEEDS =[42, 43, 44, 45, 46]
    
    EPOCHS = 5 # Fewer epochs as CrossViT converges fast on head fine-tuning
    LR = 1e-2
    MAX_GRAD_NORM = 1.0
    BATCH_SIZE = 512
    DELTA = 1e-5

    # Data Transforms (Critical for CrossViT)
    transform = transforms.Compose([
        transforms.Resize((240, 240)), # Resize 32->240
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading CIFAR-100 (Private) and CIFAR-10 (Public)...")
    private_dataset = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    public_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

    private_loader = DataLoader(private_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    public_loader = DataLoader(public_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    PRIVATE_SIZE = len(private_dataset)
    SAMPLE_RATE = BATCH_SIZE / PRIVATE_SIZE
    TOTAL_STEPS = EPOCHS * (PRIVATE_SIZE // BATCH_SIZE)

    results = []

    print("="*60)
    print("STARTING CIFAR-100 (CrossViT) EXPERIMENTS")
    print(f"Epsilons: {EPSILONS}")
    print(f"Seeds: {SEEDS}")
    print("="*60)

    # 1. Baseline (Plain SGD)
    # for seed in SEEDS:
    #     set_seed(seed)
    #     print(f"Running Plain SGD | Seed {seed}...")
    #     model = CrossViTClassifier(num_classes=100).to(DEVICE)
    #     optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
        
    #     train_plain_sgd(model, private_loader, optimizer, EPOCHS)
    #     acc, loss = evaluate(model, test_loader)
    #     results.append({"Method": "Plain SGD", "Epsilon": "N/A", "Seed": seed, "Acc": acc, "Loss": loss})
    #     print(f" -> Acc: {acc*100:.2f}%")

    # 2. DP Methods
    for eps in EPSILONS:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=eps, target_delta=DELTA, sample_rate=SAMPLE_RATE, steps=TOTAL_STEPS, accountant="rdp"
        )
        print(f"\n--- Target ε={eps} (σ={noise_multiplier:.2f}) ---")
        
        for seed in SEEDS:
            print(f" Processing Seed {seed}...")
            
            # A) Standard DP-SGD
            set_seed(seed)
            model = CrossViTClassifier(num_classes=100).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            for param in model._module.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model._module.classifier.parameters(), lr=LR)
            
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=False)
            acc, loss = evaluate(model, test_loader)
            print(f"  -> DP-SGD Acc: {acc*100:.2f}%")
            results.append({"Method": "Standard DP-SGD", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})

            # B) KFAC (Public = CIFAR-10)
            set_seed(seed)
            model = CrossViTClassifier(num_classes=100).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            for param in model._module.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model._module.classifier.parameters(), lr=LR)
            
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=True, public_data=True)
            acc, loss = evaluate(model, test_loader)
            print(f"  -> KFAC (Public) Acc: {acc*100:.2f}%")
            results.append({"Method": "KFAC (Public)", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})

            # C) KFAC (Noise)
            set_seed(seed)
            model = CrossViTClassifier(num_classes=100).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            for param in model._module.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model._module.classifier.parameters(), lr=LR)
            
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=True, public_data=False)
            acc, loss = evaluate(model, test_loader)
            print(f"  -> KFAC (Noise) Acc: {acc*100:.2f}%")
            results.append({"Method": "KFAC (Noise)", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})

    pd.DataFrame(results).to_csv("cross_vit_cifar100_results.csv", index=False)
    # Final Report
    print("\n" + "="*85)
    print(f"{'Method':<20} | {'Epsilon':<10} | {'Seed':<5} | {'Accuracy (%)':<12} | {'Loss':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['Method']:<20} | {str(r['Epsilon']):<10} | {r['Seed']:<5} | {r['Acc']*100:05.2f}%       | {r['Loss']:.4f}")
    print("="*85)