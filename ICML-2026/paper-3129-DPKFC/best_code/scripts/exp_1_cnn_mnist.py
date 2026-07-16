import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
import random
import os
import itertools
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

# --- 1. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv 1: 1 -> 16 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Conv 2: 16 -> 32 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# --- 2. KFAC Components (Supporting Linear AND Conv2d) ---
class KFACRecorder:
    """Captures activations and backprops for Linear and Conv2d layers."""
    def __init__(self, model):
        self.handles = []
        self.activations = {}
        self.backprops = {}
        
        # Handle Opacus wrapper if present
        target_model = model._module if hasattr(model, "_module") else model

        for name, module in target_model.named_modules():
            # Now supporting Conv2d as well
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.requires_grad:
                self.handles.append(module.register_forward_hook(self._save_activation(name)))
                self.handles.append(module.register_full_backward_hook(self._save_backprop(name)))
    
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.activations = {}
        self.backprops = {}

    def _save_activation(self, name):
        def hook(module, inputs, outputs):
            if getattr(self, "enabled", True): 
                # For Conv2d, inputs[0] is (B, C, H, W)
                self.activations[name] = inputs[0].detach()
        return hook
    
    def _save_backprop(self, name):
        def hook(module, grad_input, grad_output):
            if getattr(self, "enabled", True): 
                # For Conv2d, grad_output[0] is (B, C, H, W)
                self.backprops[name] = grad_output[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

def compute_covariances(model, acts, grads):
    cov_A, cov_G = {}, {}
    eps = 1e-5
    
    target_model = model._module if hasattr(model, "_module") else model
    name_to_module = dict(target_model.named_modules())

    for name in grads:
        if name not in name_to_module: continue
        module = name_to_module[name]
        
        # --- CASE 1: Linear Layers ---
        if isinstance(module, nn.Linear):
            G = grads[name]
            if G.dim() > 2: G = G.reshape(-1, G.shape[-1])
            cov_G[name] = (G.T @ G) / G.size(0) + eps * torch.eye(G.size(1), device=G.device)

            A = acts[name]
            if A.dim() > 2: A = A.reshape(-1, A.shape[-1])
            A_with_bias = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)
            cov_A[name] = (A_with_bias.T @ A_with_bias) / A.size(0) + eps * torch.eye(A_with_bias.size(1), device=A.device)

        # --- CASE 2: Conv2d Layers (The Reshaping Logic) ---
        elif isinstance(module, nn.Conv2d):
            # 1. Activation Covariance (A)
            # We must "unfold" input image into patches that match kernel size
            X = acts[name] # [B, C_in, H, W]
            
            # unfold creates [B, C_in * K * K, L] where L is number of patches
            X_unfold = F.unfold(X, kernel_size=module.kernel_size, padding=module.padding, stride=module.stride)
            
            # Permute to [B * L, C_in * K * K]
            X_unfold = X_unfold.transpose(1, 2).reshape(-1, X_unfold.size(1))
            
            # Add bias term (a column of ones)
            X_unfold_bias = torch.cat([X_unfold, torch.ones_like(X_unfold[:, :1])], dim=1)
            
            cov_A[name] = (X_unfold_bias.T @ X_unfold_bias) / X_unfold_bias.size(0) + eps * torch.eye(X_unfold_bias.size(1), device=X.device)

            # 2. Gradient Covariance (G)
            # Grad output is [B, C_out, H_out, W_out]
            GY = grads[name]
            
            # Permute to [B, H_out, W_out, C_out] then flatten to [B*H*W, C_out]
            GY = GY.permute(0, 2, 3, 1).reshape(-1, GY.size(1))
            
            cov_G[name] = (GY.T @ GY) / GY.size(0) + eps * torch.eye(GY.size(1), device=GY.device)

    return cov_A, cov_G

def precondition_per_sample_gradients(model, cov_A, cov_G):
    eps = 1e-3
    target_model = model._module if hasattr(model, "_module") else model

    for name, module in target_model.named_modules():
        if (
            isinstance(module, (nn.Linear, nn.Conv2d)) and 
            hasattr(module.weight, "grad_sample") and 
            module.weight.grad_sample is not None and 
            name in cov_G and name in cov_A
        ):
            # --- Invert Covariances (Same for Linear and Conv) ---
            G = cov_G[name]
            eva_G, evc_G = torch.linalg.eigh(G + eps * torch.eye(G.size(0), device=G.device))
            inv_sqrt_G = evc_G @ torch.diag(eva_G.clamp(min=1e-6).rsqrt()) @ evc_G.T

            A = cov_A[name]
            eva_A, evc_A = torch.linalg.eigh(A + eps * torch.eye(A.size(0), device=A.device))
            inv_sqrt_A = evc_A @ torch.diag(eva_A.clamp(min=1e-6).rsqrt()) @ evc_A.T

            g_sample_w = module.weight.grad_sample # Linear: [B, Out, In], Conv: [B, Out, In, K, K]
            g_sample_b = module.bias.grad_sample   # [B, Out]
            
            if isinstance(g_sample_w, list): g_sample_w = g_sample_w[-1]
            if isinstance(g_sample_b, list): g_sample_b = g_sample_b[-1]

            # --- Reshape Grads to 2D Matrix for KFAC ---
            if isinstance(module, nn.Conv2d):
                # Opacus Conv grad_sample is [B, Out, In, K, K]
                # We need to flatten [In, K, K] to match the unfolded activation size
                batch_size = g_sample_w.shape[0]
                out_channels = g_sample_w.shape[1]
                g_sample_w = g_sample_w.view(batch_size, out_channels, -1) # [B, Out, In*K*K]
            
            # Linear is already [B, Out, In], so we assume [B, Out, Features] for both now

            # Concatenate Bias: [B, Out, Features + 1]
            g_sample_augmented = torch.cat([g_sample_w, g_sample_b.unsqueeze(2)], dim=2)

            # --- Apply Preconditioning (Kronecker Product) ---
            # P = inv_sqrt_G @ Grad @ inv_sqrt_A
            temp = torch.einsum("oj, bjk -> bok", inv_sqrt_G, g_sample_augmented)
            preconditioned = torch.einsum("bok, ki -> boi", temp, inv_sqrt_A)

            # --- Restore Shapes ---
            new_w_grad = preconditioned[:, :, :-1]
            new_b_grad = preconditioned[:, :, -1]

            if isinstance(module, nn.Conv2d):
                # Reshape back to [B, Out, In, K, K]
                new_w_grad = new_w_grad.view_as(module.weight.grad_sample)

            module.weight.grad_sample = new_w_grad
            module.bias.grad_sample = new_b_grad

def apply_dp(model, noise_multiplier, max_grad_norm, batch_size):
    total_norm_sq = None
    params = [p for p in model.parameters() if hasattr(p, "grad_sample") and p.grad_sample is not None]

    # 1. Calc Total Norm
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
        
        clipped = gs * clip_factors.unsqueeze(1)
        summed = torch.sum(clipped, dim=0)
        noise = torch.randn_like(summed) * noise_multiplier * max_grad_norm
        
        p.grad = ((summed + noise) / batch_size).view_as(p.grad)
        p.grad_sample = None

def generate_synthetic_mnist(batch_size, device):
    # Random Gaussian Noise images
    return torch.randn(batch_size, 1, 28, 28, device=device), torch.randint(0, 10, (batch_size,), device=device)

# --- 3. Training Loops ---

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_accum = 0
    criterion = nn.CrossEntropyLoss()
    
    target_model = model._module if hasattr(model, "_module") else model
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = target_model(data)
            loss_accum += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    return correct / total, loss_accum / len(loader)

def train_plain_sgd(model, loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def train_dp_sgd(
    model, private_loader, public_loader, optimizer,
    noise_multiplier, sample_rate, max_grad_norm, epochs,
    kfac=False, public_data=True
):
    accountant = RDPAccountant()
    recorder = KFACRecorder(model) if kfac else None
    public_iter = iter(itertools.cycle(public_loader)) if kfac and public_data else None
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(epochs):
        model.train()
        idx = 0
        for data, target in private_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            curr_bs = data.size(0)
            model.zero_grad(set_to_none=True)

            # --- KFAC Step ---
            if kfac and idx % len(private_loader) == 0: # Once per epoch
                recorder.enable()
                if public_data:
                    p_data, p_target = next(public_iter)
                    p_data, p_target = p_data.to(DEVICE), p_target.to(DEVICE)
                    p_out = model(p_data)
                    p_loss = F.cross_entropy(p_out, p_target)
                    p_loss.backward()
                else:
                    p_data, p_target = generate_synthetic_mnist(curr_bs, DEVICE)
                    p_out = model(p_data)
                    p_loss = F.cross_entropy(p_out, p_target)
                    p_loss.backward()
                
                cov_A, cov_G = compute_covariances(model, recorder.activations, recorder.backprops)
                recorder.disable()
                model.zero_grad(set_to_none=True)

            # --- DP Step ---
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

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
    # Settings
    EPSILONS = [0.5, 1.0, 2, 3, 5.0, 7.5, 10]
    SEEDS = [42, 43, 44, 45, 46]
    
    EPOCHS = 5
    LR = 1e-3
    MAX_GRAD_NORM = 1.0
    BATCH_SIZE = 256
    DELTA = 1e-5
    
    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Private Data: MNIST
    private_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Public Data: FashionMNIST (Clothing) - Hard Transfer Task
    print("Downloading FashionMNIST for Public Data Proxy...")
    public_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    
    # Test Data: MNIST
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    private_loader = DataLoader(private_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    public_loader = DataLoader(public_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=2)
    
    PRIVATE_SIZE = len(private_data)
    SAMPLE_RATE = BATCH_SIZE / PRIVATE_SIZE
    TOTAL_STEPS = EPOCHS * (PRIVATE_SIZE // BATCH_SIZE)
    
    results = []

    print("="*60)
    print("STARTING MNIST EXPERIMENTS (Linear + Conv2d KFAC)")
    print(f"Epsilons: {EPSILONS}")
    print(f"Seeds: {SEEDS}")
    print("="*60)

    # 1. Baseline (Plain SGD)
    for seed in SEEDS:
        set_seed(seed)
        print(f"Running Plain SGD | Seed {seed}...")
        model = SimpleCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        train_plain_sgd(model, private_loader, optimizer, EPOCHS)
        acc, loss = evaluate(model, test_loader)
        results.append({"Method": "Plain SGD", "Epsilon": "N/A", "Seed": seed, "Acc": acc, "Loss": loss})

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
            model = SimpleCNN().to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            optimizer = optim.Adam(model.parameters(), lr=LR)
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=False)
            acc, loss = evaluate(model, test_loader)
            results.append({"Method": "Standard DP-SGD", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})
            
            # B) KFAC (Public = FashionMNIST)
            set_seed(seed)
            model = SimpleCNN().to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            optimizer = optim.Adam(model.parameters(), lr=LR)
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=True, public_data=True)
            acc, loss = evaluate(model, test_loader)
            results.append({"Method": "KFAC (Public)", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})
            
            # C) KFAC (Noise)
            set_seed(seed)
            model = SimpleCNN().to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            optimizer = optim.Adam(model.parameters(), lr=LR)
            train_dp_sgd(model, private_loader, public_loader, optimizer, noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS, kfac=True, public_data=False)
            acc, loss = evaluate(model, test_loader)
            results.append({"Method": "KFAC (Noise)", "Epsilon": eps, "Seed": seed, "Acc": acc, "Loss": loss})
    # save pandas dataframe
    df = pd.DataFrame(results)
    df.to_csv("mnist_experiment_results.csv", index=False)
    # Report
    print("\n" + "="*85)
    print(f"{'Method':<20} | {'Epsilon':<10} | {'Seed':<5} | {'Accuracy (%)':<12} | {'Loss':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['Method']:<20} | {str(r['Epsilon']):<10} | {r['Seed']:<5} | {r['Acc']*100:05.2f}%       | {r['Loss']:.4f}")
    print("="*85)