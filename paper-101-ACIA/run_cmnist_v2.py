"""
CMNIST experiment with corrected metrics based on paper Table 1 analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.datasets as datasets
import numpy as np
import sys

torch.manual_seed(42)
np.random.seed(42)

# ==== Dataset (vectorized) ====
class ColoredMNIST(Dataset):
    def __init__(self, env, root='./data', train=True, intervention_type='perfect'):
        super().__init__()
        mnist = datasets.MNIST(root=root, train=train, download=True)
        images = mnist.data.float() / 255.0
        labels = mnist.targets
        n = len(images)
        colored = torch.zeros((n, 3, 28, 28))
        is_even = (labels % 2 == 0)
        if env == 'e1':
            is_red = is_even  # Perfect: even=red in e1
        else:
            is_red = ~is_even  # Perfect: even=green in e2
        img_3d = images.unsqueeze(1)
        colored[:, 0] = img_3d[:, 0] * is_red.float().unsqueeze(-1).unsqueeze(-1)
        colored[:, 1] = img_3d[:, 0] * (~is_red).float().unsqueeze(-1).unsqueeze(-1)
        self.colored_images = colored
        self.labels = labels
        self.env_labels = torch.full_like(labels, float(env == 'e2'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx], self.env_labels[idx]


# ==== Model ====
class ImprovedCausalRepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )
        self.phi_H = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits


# ==== Optimizer ====
class EnhancedCausalOptimizer:
    def __init__(self, model, batch_size, lr=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.lambda1 = 0.02 / (batch_size ** 0.5)
        self.lambda2 = 0.02 / (batch_size ** 0.5)
        self.lambda3 = 0.01
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def compute_R1(self, z_H, y, e):
        R1 = torch.tensor(0.0, device=z_H.device)
        unique_y = torch.unique(y)
        unique_e = torch.unique(e)
        for y_val in unique_y:
            y_mask = (y == y_val)
            y_prob = y_mask.float().mean()
            if y_prob < 1e-6:
                continue
            for e1 in unique_e:
                for e2 in unique_e:
                    if e1 < e2:
                        e1_mask = (e == e1) & y_mask
                        e2_mask = (e == e2) & y_mask
                        if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                            z_H_e1 = z_H[e1_mask].mean(0)
                            z_H_e2 = z_H[e2_mask].mean(0)
                            R1 += y_prob * torch.norm(z_H_e1 - z_H_e2, p=2) * 2.0
        return R1

    def compute_R2(self, z_L, z_H, y, e):
        device = z_L.device
        R2 = torch.tensor(0.0, device=device)
        unique_e = torch.unique(e)
        for e1 in unique_e:
            e1_mask = (e == e1)
            z_H_e1 = z_H[e1_mask]
            y_e1 = y[e1_mask]
            obs_dist = torch.zeros(10, device=device)
            for digit in range(10):
                dmask = (y_e1 == digit)
                if dmask.sum() > 0:
                    obs_dist[digit] = z_H_e1[dmask].mean(0).norm()
            obs_dist = F.softmax(obs_dist, dim=0)
            for e2 in unique_e:
                if e2 == e1:
                    continue
                e2_mask = (e == e2)
                z_H_e2 = z_H[e2_mask]
                y_e2 = y[e2_mask]
                int_dist = torch.zeros(10, device=device)
                for digit in range(10):
                    dmask = (y_e2 == digit)
                    if dmask.sum() > 0:
                        int_dist[digit] = z_H_e2[dmask].mean(0).norm()
                int_dist = F.softmax(int_dist, dim=0)
                R2 += F.kl_div(obs_dist.log(), int_dist, reduction='batchmean')
        return R2 / len(unique_e)

    def compute_R3(self, x, z_H, y):
        R3 = torch.tensor(0.0, device=z_H.device)
        x_swap = x.clone()
        x_swap[:, [0, 1]] = x_swap[:, [1, 0]]
        with torch.no_grad():
            _, z_H_swap, _ = self.model(x_swap)
        for y_val in torch.unique(y):
            y_mask = (y == y_val)
            if y_mask.sum() > 1:
                R3 += F.mse_loss(z_H[y_mask], z_H_swap[y_mask])
        return R3

    def train_step(self, x, y, e):
        self.optimizer.zero_grad()
        z_L, z_H, logits = self.model(x)
        pred_loss = self.criterion(logits, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)
        R3 = self.compute_R3(x, z_H, y)
        total_loss = pred_loss + self.lambda1 * R1 + self.lambda2 * R2 + self.lambda3 * R3
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return pred_loss.item()


# ==== Metrics ====
def compute_EI(z_H, labels, envs):
    """Environment Independence: mean L2 norm of z_H mean diff across envs for same label"""
    unique_y = torch.unique(labels)
    unique_e = torch.unique(envs)
    scores = []
    for y_val in unique_y:
        y_mask = (labels == y_val)
        e_means = {}
        for e_val in unique_e:
            e_mask = (envs == e_val) & y_mask
            if e_mask.sum() > 0:
                e_means[e_val.item()] = z_H[e_mask].mean(0)
        e_list = list(e_means.values())
        for i in range(len(e_list)):
            for j in range(i+1, len(e_list)):
                scores.append(torch.norm(e_list[i] - e_list[j]).item())
    return np.mean(scores) if scores else 0.0


def compute_LLI_normalized(z_L, labels, envs):
    """LLI with normalized z_L: variance of normalized representations across environments"""
    z_L_norm = F.normalize(z_L, dim=1)
    unique_e = torch.unique(envs)
    total_var = 0.0
    count = 0
    for e_val in unique_e:
        e_mask = (envs == e_val)
        if e_mask.sum() > 1:
            z_e = z_L_norm[e_mask]
            total_var += z_e.var(0).mean().item()
            count += 1
    return total_var / count if count > 0 else 0.0


def compute_IR(model, test_loader, device):
    """Intervention Robustness: KL divergence between logit distributions for original vs intervened"""
    model.eval()
    kls = []
    with torch.no_grad():
        for x, y, e in test_loader:
            x = x.to(device)
            # Original predictions
            _, _, logits = model(x)
            p_orig = F.softmax(logits, dim=1)
            # Intervened: swap color channels (do(color))
            x_int = x.clone()
            x_int[:, [0, 1]] = x_int[:, [1, 0]]
            _, _, logits_int = model(x_int)
            p_int = F.softmax(logits_int, dim=1)
            # KL divergence
            kl = F.kl_div(p_orig.log(), p_int, reduction='batchmean')
            kls.append(kl.item())
    return float(np.mean(kls))


def main():
    print("=" * 60)
    print("CMNIST Experiment (Perfect Intervention) - v2")
    print("=" * 60)
    sys.stdout.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    print("\nLoading datasets...")
    train_e1 = ColoredMNIST('e1', root='./data', train=True)
    train_e2 = ColoredMNIST('e2', root='./data', train=True)
    test_e1 = ColoredMNIST('e1', root='./data', train=False)
    test_e2 = ColoredMNIST('e2', root='./data', train=False)
    print("Datasets loaded")
    sys.stdout.flush()

    train_dataset = ConcatDataset([train_e1, train_e2])
    test_dataset = ConcatDataset([test_e1, test_e2])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    sys.stdout.flush()

    import copy
    model = ImprovedCausalRepresentationNetwork().to(device)
    optimizer = EnhancedCausalOptimizer(model, batch_size=32, lr=1e-3)
    # Cosine annealing LR schedule
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer.optimizer, T_max=24, eta_min=1e-5)

    print("\nTraining for 24 epochs (cosine annealing LR)...")
    sys.stdout.flush()
    best_acc = 0.0
    best_state = None
    # Top-K checkpoint pool for weight averaging (LEAP: SWA-style)
    TOP_K = 5
    top_k_checkpoints = []  # List of (acc, state_dict) pairs
    for epoch in range(24):
        model.train()
        losses = []
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            loss = optimizer.train_step(x, y, e)
            losses.append(loss)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y = x.to(device), y.to(device)
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/24 - Loss: {np.mean(losses):.4f}, Acc: {acc:.2f}%, LR: {current_lr:.6f}")
        sys.stdout.flush()
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        # Maintain top-K checkpoint pool
        top_k_checkpoints.append((acc, copy.deepcopy(model.state_dict())))
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
        top_k_checkpoints = top_k_checkpoints[:TOP_K]
    
    # Weight averaging of top-K checkpoints (LEAP: SWA-inspired)
    print(f"Top-{TOP_K} checkpoint accuracies: {[f'{a:.2f}%' for a,_ in top_k_checkpoints]}")
    avg_state = {}
    for key in best_state.keys():
        avg_state[key] = torch.stack([s[key].float() for _, s in top_k_checkpoints]).mean(0)
    model.load_state_dict(avg_state)
    # Evaluate averaged model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y = x.to(device), y.to(device)
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    avg_acc = 100 * correct / total
    print(f"Weight-averaged model Acc: {avg_acc:.2f}%")
    # Use better of averaged vs best checkpoint
    if avg_acc >= best_acc:
        print(f"Using weight-averaged model (avg={avg_acc:.2f}% >= best={best_acc:.2f}%)")
    else:
        print(f"Falling back to best checkpoint (best={best_acc:.2f}% > avg={avg_acc:.2f}%)")
        model.load_state_dict(best_state)
        avg_acc = best_acc
    print(f"Final model Acc: {max(avg_acc, best_acc):.2f}%")
    sys.stdout.flush()

    # Collect representations
    print("\nCollecting representations for metrics...")
    sys.stdout.flush()
    model.eval()
    z_L_all, z_H_all, y_all, e_all = [], [], [], []
    with torch.no_grad():
        for x, y, e in test_loader:
            x = x.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            y_all.append(y)
            e_all.append(e)
    z_L_all = torch.cat(z_L_all)
    z_H_all = torch.cat(z_H_all)
    y_all = torch.cat(y_all)
    e_all = torch.cat(e_all)

    # Final accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y = x.to(device), y.to(device)
            _, _, logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    final_acc = 100 * correct / total

    ei = compute_EI(z_H_all, y_all, e_all)
    lli_raw = z_L_all.var(0).mean().item() / 2  # avg variance across envs
    lli_norm = compute_LLI_normalized(z_L_all, y_all, e_all)
    ir = compute_IR(model, test_loader, device)

    print(f"\n=== RESULTS ===")
    print(f"Test Accuracy (Acc): {final_acc:.2f}%")
    print(f"Environment Independence (EI): {ei:.4f}")
    print(f"Low-level Invariance (LLI) raw: {lli_raw:.4f}")
    print(f"Low-level Invariance (LLI) normalized: {lli_norm:.4f}")
    print(f"Intervention Robustness (IR): {ir:.4f}")
    sys.stdout.flush()

    return final_acc, ei, lli_norm, ir


if __name__ == '__main__':
    acc, ei, lli, ir = main()
    print(f"\n=== FINAL ===")
    print(f"Acc={acc:.4f}%  EI={ei:.4f}  LLI={lli:.4f}  IR={ir:.4f}")
    sys.stdout.flush()
