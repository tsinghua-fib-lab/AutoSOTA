"""
Optimized CMNIST experiment matching the paper's Table 1 (Perfect Intervention results).
Uses vectorized operations for faster dataset creation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.datasets as datasets
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ==== Dataset (vectorized) ====
class ColoredMNIST(Dataset):
    def __init__(self, env: str, root='./data', train=True, intervention_type='perfect', intervention_strength=1.0):
        super().__init__()
        self.env = env
        mnist = datasets.MNIST(root=root, train=train, download=True)
        images = mnist.data.float() / 255.0  # [N, 28, 28]
        labels = mnist.targets  # [N]

        # Vectorized color assignment
        n = len(images)
        colored = torch.zeros((n, 3, 28, 28))

        # Base probability: e1 means even=red(0.75), odd=green(0.25)
        # With perfect intervention: even -> always red (p=1.0), odd -> always green (p=0.0)
        # e2 flips: even -> always green, odd -> always red
        is_even = (labels % 2 == 0)

        if intervention_type == 'perfect':
            if env == 'e1':
                # Even digits are red, odd are green
                is_red = is_even
            else:  # e2
                # Even digits are green, odd are red
                is_red = ~is_even
        elif intervention_type == 'imperfect':
            # Interpolate between base and perfect
            if env == 'e1':
                base_p = torch.where(is_even, torch.tensor(0.75), torch.tensor(0.25))
                perfect_p = is_even.float()
            else:
                base_p = torch.where(is_even, torch.tensor(0.25), torch.tensor(0.75))
                perfect_p = (~is_even).float()
            p_red = (1 - intervention_strength) * base_p + intervention_strength * perfect_p
            is_red = torch.rand(n) < p_red
        else:  # none
            if env == 'e1':
                base_p = torch.where(is_even, torch.tensor(0.75), torch.tensor(0.25))
            else:
                base_p = torch.where(is_even, torch.tensor(0.25), torch.tensor(0.75))
            is_red = torch.rand(n) < base_p

        # Assign colors
        img_3d = images.unsqueeze(1)  # [N, 1, 28, 28]
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
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )
        self.phi_H = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits


# ==== Enhanced Optimizer (per paper) ====
class EnhancedCausalOptimizer:
    def __init__(self, model, batch_size, lr=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.lambda1 = 0.1 / (batch_size ** 0.5)
        self.lambda2 = 0.1 / (batch_size ** 0.5)
        self.lambda3 = 0.1
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
                            diff = torch.norm(z_H_e1 - z_H_e2, p=2)
                            R1 += y_prob * diff * 2.0
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
                digit_mask = (y_e1 == digit)
                if digit_mask.sum() > 0:
                    obs_dist[digit] = (z_H_e1[digit_mask].mean(0)).norm()
            obs_dist = F.softmax(obs_dist, dim=0)
            other_envs = [e2 for e2 in unique_e if e2 != e1]
            for e2 in other_envs:
                e2_mask = (e == e2)
                z_H_e2 = z_H[e2_mask]
                y_e2 = y[e2_mask]
                int_dist = torch.zeros(10, device=device)
                for digit in range(10):
                    digit_mask = (y_e2 == digit)
                    if digit_mask.sum() > 0:
                        int_dist[digit] = (z_H_e2[digit_mask].mean(0)).norm()
                int_dist = F.softmax(int_dist, dim=0)
                R2 += F.kl_div(obs_dist.log(), int_dist, reduction='batchmean')
        return R2 / len(unique_e)

    def compute_R3(self, x, z_H, y):
        R3 = torch.tensor(0.0, device=z_H.device)
        x_swap = x.clone()
        x_swap[:, [0, 1]] = x_swap[:, [1, 0]]
        with torch.no_grad():
            _, z_H_swap, _ = self.model(x_swap)
        unique_y = torch.unique(y)
        for y_val in unique_y:
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
        return {'pred_loss': pred_loss.item(), 'R1': R1.item(), 'R2': R2.item(), 'total_loss': total_loss.item()}


# ==== Metrics ====
def compute_environment_independence(z_H, labels, envs):
    unique_y = torch.unique(labels)
    unique_e = torch.unique(envs)
    score = 0.0
    count = 0
    for y_val in unique_y:
        y_mask = (labels == y_val)
        for e1 in unique_e:
            for e2 in unique_e:
                if e1 < e2:
                    e1_mask = (envs == e1) & y_mask
                    e2_mask = (envs == e2) & y_mask
                    if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                        z1 = z_H[e1_mask].mean(0)
                        z2 = z_H[e2_mask].mean(0)
                        score += torch.norm(z1 - z2).item()
                        count += 1
    return score / count if count > 0 else 0.0


def compute_low_level_invariance(z_L, labels, envs):
    unique_e = torch.unique(envs)
    total_var = 0.0
    count = 0
    for e_val in unique_e:
        e_mask = (envs == e_val)
        if e_mask.sum() > 1:
            z_L_e = z_L[e_mask]
            total_var += z_L_e.var(0).mean().item()
            count += 1
    return total_var / count if count > 0 else 0.0


def compute_intervention_robustness(model, test_loader, device):
    model.eval()
    diffs = []
    with torch.no_grad():
        for x, y, e in test_loader:
            x = x.to(device)
            _, z_H, _ = model(x)
            noise = torch.randn_like(x) * 0.01
            x_noisy = (x + noise).clamp(0, 1)
            _, z_H_noisy, _ = model(x_noisy)
            diff = torch.norm(F.normalize(z_H, dim=1) - F.normalize(z_H_noisy, dim=1), dim=1).mean()
            diffs.append(diff.item())
    return float(np.mean(diffs))


def main():
    print("=" * 60)
    print("CMNIST Experiment (Perfect Intervention)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_e1 = ColoredMNIST('e1', root='./data', train=True, intervention_type='perfect')
    print("train_e1 loaded")
    train_e2 = ColoredMNIST('e2', root='./data', train=True, intervention_type='perfect')
    print("train_e2 loaded")
    test_e1 = ColoredMNIST('e1', root='./data', train=False, intervention_type='perfect')
    print("test_e1 loaded")
    test_e2 = ColoredMNIST('e2', root='./data', train=False, intervention_type='perfect')
    print("test_e2 loaded")

    train_dataset = ConcatDataset([train_e1, train_e2])
    test_dataset = ConcatDataset([test_e1, test_e2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    model = ImprovedCausalRepresentationNetwork().to(device)
    optimizer = EnhancedCausalOptimizer(model, batch_size=32, lr=1e-3)

    # Training loop (8 epochs as per the paper's experiment runner)
    print("\nTraining for 8 epochs...")
    for epoch in range(8):
        model.train()
        losses = []
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            metrics = optimizer.train_step(x, y, e)
            losses.append(metrics['pred_loss'])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y = x.to(device), y.to(device)
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/8 - Loss: {np.mean(losses):.4f}, Test Acc: {acc:.2f}%")

    # Final evaluation
    print("\n" + "=" * 60)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y = x.to(device), y.to(device)
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    final_acc = 100 * correct / total

    # Collect representations
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

    ei = compute_environment_independence(z_H_all, y_all, e_all)
    lli = compute_low_level_invariance(z_L_all, y_all, e_all)
    ir = compute_intervention_robustness(model, test_loader, device)

    print(f"\n=== RESULTS ===")
    print(f"Test Accuracy (Acc): {final_acc:.2f}%")
    print(f"Environment Independence (EI): {ei:.4f}")
    print(f"Low-level Invariance (LLI): {lli:.4f}")
    print(f"Intervention Robustness (IR): {ir:.4f}")

    return final_acc, ei, lli, ir


if __name__ == '__main__':
    acc, ei, lli, ir = main()
    print(f"\n=== FINAL: Acc={acc:.4f}%, EI={ei:.4f}, LLI={lli:.4f}, IR={ir:.4f} ===")
