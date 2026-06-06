from collections import defaultdict
import torch, torch.nn.functional as F, numpy as np

def compute_metrics(model, loader, device):
    model.eval()
    metrics = defaultdict(list)
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, pred_positions = model(x)
            pos_error = F.mse_loss(pred_positions, y)
            env_ind = compute_environment_independence(z_H, e)
            low_level = compute_environment_invariance(z_L, y, e)
            interv = compute_intervention_consistency(z_L, z_H, y, e)
            metrics['position_error'].append(pos_error.item())
            metrics['env_independence'].append(env_ind.item())
            metrics['low_level_inv'].append(low_level.item())
            metrics['intervention_rob'].append(interv.item())
    return {k: np.mean(v) for k, v in metrics.items()}

def compute_environment_independence(z_H, e):
    if len(e.shape) > 1:
        e = e.mean(dim=1) > 0.5
    unique_e = torch.unique(e)
    score = torch.tensor(0.0, device=z_H.device)
    for e1 in unique_e:
        for e2 in unique_e:
            if e1 != e2:
                e1_mask = (e == e1)
                e2_mask = (e == e2)
                if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                    z_H_e1 = z_H[e1_mask].mean(0)
                    z_H_e2 = z_H[e2_mask].mean(0)
                    score += torch.norm(z_H_e1 - z_H_e2)
    return score

def compute_environment_invariance(z_L, y, e):
    if len(e.shape) > 1:
        e = e.mean(dim=1) > 0.5
    unique_e = torch.unique(e)
    total_var = torch.tensor(0.0, device=z_L.device)
    count = 0
    for e_val in unique_e:
        e_mask = (e == e_val)
        if e_mask.sum() > 0:
            z_L_e = z_L[e_mask]
            z_L_var = z_L_e.var(0).mean()
            total_var += z_L_var
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=z_L.device)
    return total_var / count

def compute_intervention_consistency(z_L, z_H, y, e):
    z_L_norm = F.normalize(z_L, dim=1)
    z_H_norm = F.normalize(z_H, dim=1)
    diff = (z_H_norm - z_L_norm).pow(2).mean()
    return diff

def compute_camelyon_metrics(model, loader, device):
    metrics = defaultdict(list)
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, logits = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            env_ind = compute_hospital_independence(z_H, e)
            low_level_inv = compute_hospital_invariance(z_L, y, e)
            int_rob = compute_hospital_robustness(z_H, y, e)
            metrics['accuracy'].append(acc.item() if isinstance(acc, torch.Tensor) else acc)
            metrics['env_independence'].append(env_ind if isinstance(env_ind, float) else env_ind.item())
            metrics['low_level_inv'].append(low_level_inv if isinstance(low_level_inv, float) else low_level_inv.item())
            metrics['intervention_rob'].append(int_rob if isinstance(int_rob, float) else int_rob.item())
    return {k: np.mean(v) for k, v in metrics.items()}

def compute_hospital_independence(z_H, e):
    hospitals = e.argmax(dim=1)
    unique_hospitals = torch.unique(hospitals)
    score = 0.0
    count = 0
    for h1 in unique_hospitals:
        for h2 in unique_hospitals:
            if h1 != h2:
                h1_mask = (hospitals == h1)
                h2_mask = (hospitals == h2)
                if h1_mask.sum() > 0 and h2_mask.sum() > 0:
                    z_H_h1 = z_H[h1_mask].mean(0)
                    z_H_h2 = z_H[h2_mask].mean(0)
                    dist = F.cosine_similarity(
                        z_H_h1.unsqueeze(0), z_H_h2.unsqueeze(0)).item()
                    score += 1 - dist
                    count += 1
    return score / count if count > 0 else 0.0

def compute_hospital_invariance(z_L, y, e):
    hospitals = e.argmax(dim=1)
    unique_labels = torch.unique(y)
    total_var = 0.0
    count = 0
    for label in unique_labels:
        label_mask = (y == label)
        if label_mask.sum() > 0:
            z_L_label = z_L[label_mask]
            hospitals_label = hospitals[label_mask]
            for h in torch.unique(hospitals_label):
                h_mask = (hospitals_label == h)
                if h_mask.sum() > 1:
                    z_L_h = z_L_label[h_mask]
                    total_var += z_L_h.var(0).mean().item()
                    count += 1
    return total_var / count if count > 0 else 0.0

def compute_hospital_robustness(z_H, y, e):
    noise_scale = 0.01
    noise = torch.randn_like(z_H) * noise_scale
    z_H_noisy = z_H + noise
    diff = torch.norm(F.normalize(z_H, dim=1) - F.normalize(z_H_noisy, dim=1), dim=1).mean()
    return diff.item()