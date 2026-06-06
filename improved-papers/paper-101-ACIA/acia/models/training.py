# acia/models/training.py
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CausalOptimizer:
    def __init__(self, model, batch_size, lr=1e-4):
        self.model = model
        self.batch_size = batch_size
        self.lambda1 = 0.3 / (batch_size ** 0.5)
        self.lambda2 = 0.7 / (batch_size ** 0.5)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def compute_R1(self, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        R1 = torch.tensor(0.0, device=z_H.device)
        unique_y = torch.unique(y)
        unique_e = torch.unique(e)
        for y_val in unique_y:
            y_mask = (y == y_val)
            y_prob = (y == y_val).float().mean()
            for e1 in unique_e:
                for e2 in unique_e:
                    if e1 != e2:
                        e1_mask = (e == e1) & y_mask
                        e2_mask = (e == e2) & y_mask
                        if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                            z_H_e1 = z_H[e1_mask].mean(0)
                            z_H_e2 = z_H[e2_mask].mean(0)
                            R1 += y_prob * torch.norm(z_H_e1 - z_H_e2, p=2)
        return R1

    def compute_R2(self, z_L, z_H, y, e):
        device = z_L.device
        batch_size = z_L.size(0)
        R2 = torch.tensor(0.0, device=device)
        unique_e = torch.unique(e)
        for e1 in unique_e:
            e1_mask = (e == e1)
            z_H_e1 = z_H[e1_mask]
            z_L_e1 = z_L[e1_mask]
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
        return R2 / batch_size

    def train_step(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        z_L, z_H, logits = self.model(x)
        pred_loss = self.criterion(logits, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)
        total_loss = pred_loss + self.lambda1 * R1 + self.lambda2 * R2
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {'pred_loss': pred_loss.item(), 'R1': R1.item(), 'R2': R2.item(), 'total_loss': total_loss.item()}

class EnhancedCausalOptimizer:
    def __init__(self, model, batch_size, lr=1e-4):
        self.model = model
        self.batch_size = batch_size
        self.lambda1 = 0.1 / (batch_size ** 0.5)
        self.lambda2 = 0.1 / (batch_size ** 0.5)
        self.lambda3 = 0.1
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def compute_R1(self, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Enhanced R1 with stronger environment independence enforcement"""
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

                            # Use L2 norm with higher weight for same-digit different-env pairs
                            diff = torch.norm(z_H_e1 - z_H_e2, p=2)
                            R1 += y_prob * diff * 2.0  # Increased weight

        return R1

    def compute_R3_color_suppression(self, x: torch.Tensor, z_H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """NEW: Explicit color suppression regularizer"""
        R3 = torch.tensor(0.0, device=z_H.device)

        # Create color-swapped versions
        x_color_swapped = x.clone()
        x_color_swapped[:, [0, 1]] = x_color_swapped[:, [1, 0]]  # Swap R and G channels

        # Get representations for color-swapped images
        with torch.no_grad():
            z_L_swap, z_H_swap, _ = self.model(x_color_swapped)

        # Penalize if high-level representations change due to color swap
        # (for same digit, z_H should be similar regardless of color)
        unique_y = torch.unique(y)
        for y_val in unique_y:
            y_mask = (y == y_val)
            if y_mask.sum() > 1:
                z_H_orig = z_H[y_mask]
                z_H_swap_masked = z_H_swap[y_mask]

                # High-level representations should be similar for same digit regardless of color
                color_sensitivity = F.mse_loss(z_H_orig, z_H_swap_masked)
                R3 += color_sensitivity

        return R3

    def train_step(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        z_L, z_H, logits = self.model(x)

        # Standard losses
        pred_loss = self.criterion(logits, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)

        # NEW: Color suppression loss
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

    def compute_R2(self, z_L, z_H, y, e):
        """Keep existing R2 implementation"""
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

class BallOptimizer:
    def __init__(self, model, batch_size, lr=1e-4):
        self.model = model
        self.lambda1 = 0.01
        self.lambda2 = 0.05
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def compute_R1(self, z_H, y, e):
        intervention_counts = e.sum(dim=1)
        R1 = 0.0
        for count in torch.unique(intervention_counts):
            mask = intervention_counts == count
            if mask.sum() > 1:
                z_H_group = z_H[mask]
                y_group = y[mask]
                pos_bins = torch.linspace(0, 1, 5)
                for i in range(len(pos_bins) - 1):
                    bin_mask = (y_group[:, 0] >= pos_bins[i]) & (y_group[:, 0] < pos_bins[i + 1])
                    if bin_mask.sum() > 1:
                        z_H_bin = z_H_group[bin_mask]
                        z_H_std = z_H_bin.std(0)
                        R1 += z_H_std.mean()
        return R1

    def compute_R2(self, z_L, z_H, y, e):
        R2 = 0.0
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                if not torch.all(e[i] == e[j]):
                    z_H_dist = torch.norm(z_H[i] - z_H[j])
                    y_dist = torch.norm(y[i] - y[j])
                    R2 += torch.abs(z_H_dist - y_dist)
        return R2 / max(1, len(e) * (len(e) - 1) // 2)

    def train_step(self, x, y, e):
        self.optimizer.zero_grad()
        z_L, z_H, pred_positions = self.model(x)
        position_loss = F.mse_loss(pred_positions, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)
        total_loss = position_loss + self.lambda1 * R1 + self.lambda2 * R2
        total_loss.backward()
        self.optimizer.step()
        return {'position_loss': position_loss.item(), 'R1': R1.item(), 'R2': R2.item(), 'total_loss': total_loss.item()}

class CamelyonTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=5e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda1 = 0.05
        self.lambda2 = 0.005

    def compute_R1(self, z_H, y, e):
        R1 = 0.0
        z_H = F.normalize(z_H, dim=1)
        for h1 in range(3):
            for h2 in range(h1 + 1, 3):
                for label in [0, 1]:
                    mask1 = (e[:, h1] == 1) & (y == label)
                    mask2 = (e[:, h2] == 1) & (y == label)
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        mean1 = z_H[mask1].mean(0)
                        mean2 = z_H[mask2].mean(0)
                        sim = F.cosine_similarity(mean1.unsqueeze(0), mean2.unsqueeze(0))
                        R1 += float(1 - sim)
        return R1

    def compute_R2(self, z_L, z_H, y, e):
        proj = nn.Linear(256, 64, bias=False).to(z_L.device)
        z_L = F.normalize(proj(z_L), dim=1)
        z_H = F.normalize(z_H, dim=1)
        similarity = F.cosine_similarity(z_L, z_H)
        return float(1 - similarity.mean())
    def train_epoch(self, train_loader, device):
        self.model.train()
        metrics = {'total_loss': 0, 'pred_loss': 0, 'R1': 0, 'R2': 0}
        correct = 0
        total = 0
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, logits = self.model(x)
            pred_loss = self.criterion(logits, y)
            R1 = self.compute_R1(z_H, y, e)
            R2 = self.compute_R2(z_L, z_H, y, e)
            loss = pred_loss + 0.5 * R1 + 0.1 * R2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            metrics['total_loss'] += float(loss)
            metrics['pred_loss'] += float(pred_loss)
            metrics['R1'] += R1
            metrics['R2'] += R2
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        return metrics, 100 * correct / total

    def evaluate(self, test_loader, device):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                z_L, z_H, logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return total_loss / len(test_loader), 100 * correct / total


def ctrain_model(train_loader, test_loader, model, n_epochs=10):
    device = next(model.parameters()).device
    optimizer = CausalOptimizer(model=model, batch_size=train_loader.batch_size)
    history = {'train_loss': [], 'R1': [], 'R2': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_metrics = {k: [] for k in history.keys()}

        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            metrics = optimizer.train_step(x, y, e)
            epoch_metrics['train_loss'].append(metrics['pred_loss'])
            epoch_metrics['R1'].append(metrics['R1'])
            epoch_metrics['R2'].append(metrics['R2'])

            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            epoch_metrics['train_acc'].append(acc.item())

        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                test_acc.append(acc.item())

        for k in history.keys():
            if k != 'test_acc':
                history[k].append(sum(epoch_metrics[k]) / len(epoch_metrics[k]))
        history['test_acc'].append(sum(test_acc) / len(test_acc))

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Loss: {history['train_loss'][-1]:.4f}, Acc: {history['test_acc'][-1]:.4f}")

    return history


def rtrain_model(train_loader, test_loader, model, n_epochs=10):
    return ctrain_model(train_loader, test_loader, model, n_epochs)


def ball_train_model(train_loader, test_loader, model, n_epochs=20):
    device = next(model.parameters()).device
    optimizer = BallOptimizer(model=model, batch_size=train_loader.batch_size)
    for epoch in range(n_epochs):
        model.train()
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            optimizer.train_step(x, y, e)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")