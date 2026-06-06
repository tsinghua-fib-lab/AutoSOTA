"""
Complete CMNIST experiment matching the paper's Table 1 (Perfect Intervention results).
This script implements the full experiment independently from runners.py which has missing functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.datasets as datasets
import numpy as np

# ==== Dataset ====
class ColoredMNIST(Dataset):
    def __init__(self, env: str, root='./data', train=True, intervention_type='perfect', intervention_strength=1.0):
        super().__init__()
        self.env = env
        self.intervention_type = intervention_type
        self.intervention_strength = intervention_strength
        mnist = datasets.MNIST(root=root, train=train, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets
        self.colored_images = self._color_images()
        self.env_labels = torch.full_like(self.labels, float(env == 'e2'))

    def _color_images(self):
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            base_p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            if self.intervention_type == 'none':
                p_red = base_p_red
            elif self.intervention_type == 'perfect':
                p_red = 1.0 if base_p_red > 0.5 else 0.0
            elif self.intervention_type == 'imperfect':
                perfect_p = 1.0 if base_p_red > 0.5 else 0.0
                p_red = (1 - self.intervention_strength) * base_p_red + self.intervention_strength * perfect_p
            is_red = torch.rand(1) < p_red
            if is_red:
                colored[i, 0] = img
            else:
                colored[i, 1] = img
        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx], self.env_labels[idx]


# ==== Model (ImprovedCausalRepresentationNetwork) ====
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


# ==== Optimizer (EnhancedCausalOptimizer) ====
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
                    if e1 != e2:
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
                R2 = R2 + F.kl_div(obs_dist.log(), int_dist, reduction='batchmean')
        return R2 / len(unique_e)

    def compute_R3_color_suppression(self, x, z_H, y):
        R3 = torch.tensor(0.0, device=z_H.device)
        x_color_swapped = x.clone()
        x_color_swapped[:, [0, 1]] = x_color_swapped[:, [1, 0]]
        with torch.no_grad():
            _, z_H_swap, _ = self.model(x_color_swapped)
        unique_y = torch.unique(y)
        for y_val in unique_y:
            y_mask = (y == y_val)
            if y_mask.sum() > 1:
                z_H_orig = z_H[y_mask]
                z_H_swap_masked = z_H_swap[y_mask]
                color_sensitivity = F.mse_loss(z_H_orig, z_H_swap_masked)
                R3 += color_sensitivity
        return R3

    def train_step(self, x, y, e):
        self.optimizer.zero_grad()
        z_L, z_H, logits = self.model(x)
        pred_loss = self.criterion(logits, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)
        R3 = self.compute_R3_color_suppression(x, z_H, y)
        total_loss = pred_loss + self.lambda1 * R1 + self.lambda2 * R2 + self.lambda3 * R3
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {
            'pred_loss': pred_loss.item(),
            'R1': R1.item(),
            'R2': R2.item(),
            'R3': R3.item(),
            'total_loss': total_loss.item()
        }


# ==== Metrics ====
def compute_environment_independence(z_H, labels, envs):
    """EI: measures how different z_H is across environments (lower=better, 0=perfect)"""
    unique_y = torch.unique(labels)
    unique_e = torch.unique(envs)
    score = torch.tensor(0.0)
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
                        score += torch.norm(z1 - z2)
                        count += 1
    return (score / count).item() if count > 0 else 0.0


def compute_low_level_invariance(z_L, labels, envs):
    """LLI: measures variance of z_L within environment groups"""
    unique_e = torch.unique(envs)
    total_var = torch.tensor(0.0)
    count = 0
    for e_val in unique_e:
        e_mask = (envs == e_val)
        if e_mask.sum() > 1:
            z_L_e = z_L[e_mask]
            z_L_var = z_L_e.var(0).mean()
            total_var += z_L_var
            count += 1
    return (total_var / count).item() if count > 0 else 0.0


def compute_intervention_robustness(model, test_loader, device):
    """IR: measures how robust z_H is to perturbation"""
    model.eval()
    diffs = []
    with torch.no_grad():
        for x, y, e in test_loader:
            x = x.to(device)
            _, z_H, _ = model(x)
            noise = torch.randn_like(x) * 0.01
            x_noisy = x + noise
            _, z_H_noisy, _ = model(x_noisy)
            diff = torch.norm(F.normalize(z_H, dim=1) - F.normalize(z_H_noisy, dim=1), dim=1).mean()
            diffs.append(diff.item())
    return float(np.mean(diffs))


# ==== Training ====
def train_model(train_loader, test_loader, model, device, n_epochs=8):
    optimizer = EnhancedCausalOptimizer(model, batch_size=train_loader.batch_size)
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            metrics = optimizer.train_step(x, y, e)
            epoch_losses.append(metrics['total_loss'])

        # Evaluate
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
        history['train_loss'].append(np.mean(epoch_losses))
        history['test_acc'].append(acc)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {history['train_loss'][-1]:.4f}, Test Acc: {acc:.2f}%")

    return history


def main():
    print("=" * 60)
    print("CMNIST Experiment (Perfect Intervention)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets with perfect intervention
    print("\nLoading datasets...")
    train_e1 = ColoredMNIST('e1', root='./data', train=True, intervention_type='perfect')
    train_e2 = ColoredMNIST('e2', root='./data', train=True, intervention_type='perfect')
    test_e1 = ColoredMNIST('e1', root='./data', train=False, intervention_type='perfect')
    test_e2 = ColoredMNIST('e2', root='./data', train=False, intervention_type='perfect')

    train_dataset = ConcatDataset([train_e1, train_e2])
    test_dataset = ConcatDataset([test_e1, test_e2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # Initialize model
    model = ImprovedCausalRepresentationNetwork().to(device)

    # Train
    print("\nTraining model for 8 epochs...")
    history = train_model(train_loader, test_loader, model, device, n_epochs=8)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.eval()

    # Accuracy
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

    print(f"\nResults:")
    print(f"  Acc: {final_acc:.2f}%")
    print(f"  EI:  {ei:.4f}")
    print(f"  LLI: {lli:.4f}")
    print(f"  IR:  {ir:.4f}")

    return final_acc, ei, lli, ir


if __name__ == '__main__':
    acc, ei, lli, ir = main()
    print(f"\n=== FINAL METRICS ===")
    print(f"Acc={acc:.4f}, EI={ei:.4f}, LLI={lli:.4f}, IR={ir:.4f}")
