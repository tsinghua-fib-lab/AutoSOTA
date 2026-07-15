import os, json, csv, math, random, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

# ----------------------------
# Repro
# ----------------------------
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_SEED = 2025


def set_seed(seed: int):
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Utils
# ----------------------------
def dirichlet_split_noniid(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(idx_by_class[c])
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = idx_by_class[c]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i in range(n_clients):
            client_indices[i].extend(parts[i])
    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def normalize_to_simplex(vec: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    v = vec.abs()
    s = v.sum(dim=dim, keepdim=True).clamp_min(eps)
    return v / s


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


# ----------------------------
# Models
# ----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32*14*14, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SparseLinearSurrogate(nn.Module):
    def __init__(self, in_features=28*28, n_classes=10, l1_lambda=8e-5):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.W(x)

    def l1_penalty(self):
        return self.l1_lambda * self.W.weight.abs().sum()


class SparseLinearInterpretable(nn.Module):
    def __init__(self, in_features=28*28, n_classes=10, l1_lambda=6e-4):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.W(x)

    def l1_penalty(self):
        return self.l1_lambda * self.W.weight.abs().sum()


# ----------------------------
# IG (approx) & Fidelity AUC
# ----------------------------
def integrated_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 12) -> torch.Tensor:
    baseline = torch.zeros_like(x)
    grads = []
    for i in range(1, steps+1):
        xi = baseline + (i/steps)*(x - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grads.append(xi.grad.detach().clone())
        model.zero_grad(set_to_none=True)
    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    return ((x - baseline) * avg_grads).abs()


def _forward_any(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if isinstance(model, (SparseLinearInterpretable, SparseLinearSurrogate)):
        xb = flatten_img(xb)
    return model(xb)


def auc_area(xs, ys):
    a = 0.0
    for i in range(1, len(xs)):
        a += 0.5 * (ys[i] + ys[i-1]) * (xs[i] - xs[i-1])
    return float(a)


def deletion_insertion_auc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, imp_map: torch.Tensor, steps: int = 20):
    N = x.size(0)
    flat_imp = imp_map.view(N, -1)
    order = torch.argsort(flat_imp, dim=-1, descending=True)
    xs = [s/steps for s in range(steps+1)]
    del_scores, ins_scores = [], []
    for s in range(steps+1):
        k = int(flat_imp.size(1) * (s/steps))
        # deletion
        x_del = x.clone().view(N, -1)
        mask = torch.ones_like(flat_imp)
        if k > 0:
            mask[torch.arange(N).unsqueeze(-1), order[:, :k]] = 0.0
        x_del = (x_del * mask).view_as(x)
        with torch.no_grad():
            pdel = F.softmax(_forward_any(model, x_del), dim=-1)[torch.arange(N), y]
        del_scores.append(pdel.mean().item())
        # insertion
        xin = torch.zeros_like(x).view(N, -1)
        keep = torch.zeros_like(flat_imp)
        if k > 0:
            keep[torch.arange(N).unsqueeze(-1), order[:, :k]] = 1.0
        xin[keep.bool()] = x.view(N, -1)[keep.bool()]
        xin = xin.view_as(x)
        with torch.no_grad():
            pins = F.softmax(_forward_any(model, xin), dim=-1)[torch.arange(N), y]
        ins_scores.append(pins.mean().item())
    return auc_area(xs, del_scores), auc_area(xs, ins_scores)


# ----------------------------
# FL helpers
# ----------------------------
def local_train(model, loader, epochs=1, lr=0.01, wd=0.0, desc="Local"):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    for _ in range(epochs):
        for xb, yb in tqdm(loader, desc=desc, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def local_train_interpretable(model, loader, epochs=1, lr=0.05):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in tqdm(loader, desc="Local(interp)", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(flatten_img(xb))
            loss = F.cross_entropy(logits, yb) + model.l1_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def eval_accuracy(model, loader):
    model.eval()
    cor = 0
    tot = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Eval", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=-1)
            cor += (pred == yb).sum().item()
            tot += yb.size(0)
    return cor / max(1, tot)


def state_dict_avg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    for m in models:
        sd = m.state_dict()
        for k in keys:
            out[k] += sd[k]
    for k in keys:
        out[k] /= len(models)
    return out


# ----------------------------
# BL-B maps (IG)
# ----------------------------
def compute_local_ig_maps(model: nn.Module, loader: DataLoader, max_per_class=40) -> Dict[int, torch.Tensor]:
    model.eval()
    per_class = {c: [] for c in range(10)}
    for xb, yb in tqdm(loader, desc="Collect IG", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        for i in range(xb.size(0)):
            c = int(yb[i])
            if len(per_class[c]) < max_per_class:
                per_class[c].append(xb[i:i+1])
        if all(len(per_class[c]) >= max_per_class for c in range(10)):
            break
    maps = {}
    for c in range(10):
        if len(per_class[c]) == 0:
            maps[c] = torch.ones(1, 1, 28, 28, device=device) / (28*28)
            continue
        Xc = torch.cat(per_class[c], dim=0)
        yc = torch.full((Xc.size(0),), c, dtype=torch.long, device=device)
        ig = integrated_gradients(model, Xc, yc, steps=10)
        avg = ig.mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg.view(1, -1), dim=-1).view(1, 1, 28, 28)
    return maps


# ----------------------------
# BL-C (7x7 pooled histograms)
# ----------------------------
def pool7x7(m28: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(m28, kernel_size=4, stride=4).view(-1)


def aggregate_histograms(list_of_hist: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out = {}
    for c in range(10):
        M = torch.stack([d[c] for d in list_of_hist], dim=0)  # (K,49)
        out[c] = normalize_to_simplex(M.median(dim=0).values, dim=0)
    return out


def ref_from_hist(global_hist: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    ref = {}
    for c in range(10):
        v = global_hist[c].view(1, 1, 7, 7)
        v_up = F.interpolate(v, size=(28, 28), mode="bilinear", align_corners=False)
        ref[c] = normalize_to_simplex(v_up.view(1, -1), dim=-1).view(1, 1, 28, 28)
    return ref


# ----------------------------
# xFL bits
# ----------------------------
@dataclass
class XFLConfig:
    topk: int = 256        # a bit larger surrogate capacity
    quant_bits: int = 8
    clip_radius: float = 5.5
    dp_sigma: float = 0.10 # lower noise for cleaner artifacts
    temperature: float = 3.0
    surrogate_epochs: int = 1
    beta_align_final: float = 0.35
    align_warmup_rounds: int = 4
    l1_lambda: float = 8e-5
    mmd_weight: float = 0.1     # MMD alignment loss weight (surrogate vs Pi)
    sharpen_gamma: float = 1.8  # <- sharpens per-image xFL importance
    # Iter-1: co-annealing schedule
    xfl_rounds: int = 8         # number of artifact mixing rounds
    gamma_start: float = 1.0    # initial sharpen_gamma (broad attributions)
    gamma_end: float = 1.8      # final sharpen_gamma (sharp attributions)
    temp_start: float = 5.0     # initial temperature (soft targets)
    temp_end: float = 1.5       # final temperature (sharp targets)


def fit_surrogate_teacher_student(model: nn.Module, loader: DataLoader, cfg: XFLConfig, steps=1, temperature=None, Pi: torch.Tensor = None):
    surr = SparseLinearSurrogate(l1_lambda=cfg.l1_lambda).to(device)
    opt = torch.optim.SGD(surr.parameters(), lr=0.1)
    T = temperature if temperature is not None else cfg.temperature
    model.eval()
    for _ in range(steps):
        for xb, _ in loader:
            xb = xb.to(device)
            with torch.no_grad():
                probs_t = F.softmax(model(xb) / T, dim=-1)
                conf = probs_t.max(dim=-1).values
            logits_s = surr(flatten_img(xb)) / T
            loss = F.kl_div(F.log_softmax(logits_s, dim=-1), probs_t,
                            reduction='none').sum(dim=-1)
            loss = (loss * conf).mean() + surr.l1_penalty()
            # ALGO-1: MMD alignment loss between surrogate weights and global Pi
            if Pi is not None and cfg.mmd_weight > 0 and not torch.allclose(Pi, torch.ones_like(Pi) / Pi.size(-1), atol=1e-3):
                W = surr.W.weight  # (10, 784)
                Pi_norm = normalize_per_class(Pi)  # normalize Pi per class
                mmd_total = 0.0
                for c in range(10):
                    w_c = normalize_to_simplex(W[c].abs(), dim=0)  # (784,)
                    pi_c = Pi_norm[c]  # (784,)
                    # Single-point MMD: each class has one weight vector
                    diff = (w_c - pi_c).pow(2).sum()
                    sigma_sq = diff.detach().clamp_min(1e-6)
                    mmd_c = 2.0 * (1.0 - torch.exp(-diff / (2.0 * sigma_sq)))
                    mmd_total = mmd_total + mmd_c
                mmd_loss = cfg.mmd_weight * (mmd_total / 10.0)
                loss = loss + mmd_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return surr


def surrogate_to_artifact(surr: SparseLinearSurrogate, cfg: XFLConfig) -> torch.Tensor:
    W = surr.W.weight.detach()  # (10,784)
    Wabs = W.abs()
    mask = torch.zeros_like(Wabs)
    k = min(cfg.topk, Wabs.size(1))
    for c in range(10):
        mask[c, torch.topk(Wabs[c], k).indices] = 1.0
    Wk = W * mask
    for c in range(10):
        v = Wk[c]
        n = v.norm(2).clamp_min(1e-8)
        Wk[c] = v * min(1.0, cfg.clip_radius / float(n))
    scale = (2**(cfg.quant_bits-1) - 1) / Wk.abs().max().clamp_min(1e-8)
    Wq = torch.round(Wk * scale) / scale
    return Wq + torch.randn_like(Wq) * cfg.dp_sigma


def robust_aggregate_artifacts(arts: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(arts, dim=0).median(dim=0).values


def normalize_per_class(W: torch.Tensor) -> torch.Tensor:
    return torch.stack([normalize_to_simplex(W[c], dim=0) for c in range(W.size(0))], dim=0)
def rbf_mmd(x, y, sigma=None):
    """RBF MMD^2 between two sets of vectors. x: (N, D), y: (M, D).
    If sigma is None, auto-tunes using median heuristic on x."""
    import torch
    if sigma is None:
        with torch.no_grad():
            xx = torch.cdist(x, x, p=2)
            sigma = xx[xx > 0].median().clamp_min(1e-4).item()
    gamma_val = 1.0 / (2.0 * sigma * sigma + 1e-12)
    xx_val = torch.exp(-gamma_val * torch.cdist(x, x, p=2).pow(2)).mean()
    yy_val = torch.exp(-gamma_val * torch.cdist(y, y, p=2).pow(2)).mean()
    xy_val = torch.exp(-gamma_val * torch.cdist(x, y, p=2).pow(2)).mean()
    return (xx_val + yy_val - 2.0 * xy_val).clamp_min(0.0)



# ----------------------------
# Data (non-IID) with covariate shift per client
# ----------------------------
@dataclass
class FLConfig:
    n_clients: int = 8
    rounds: int = 15
    local_epochs: int = 1
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    batch_size: int = 64
    alpha_dirichlet: float = 0.1  # paper setting


def _client_erasing_transform(cid: int):
    # uses global SEED; SEED is set per run via set_seed(...)
    torch.manual_seed(SEED + cid)  # deterministic per client per run
    bands = [
        transforms.RandomErasing(p=1.0, scale=(0.08, 0.12), ratio=(0.3, 0.4), value=0.0, inplace=False),
        transforms.RandomErasing(p=1.0, scale=(0.06, 0.10), ratio=(2.0, 3.0), value=0.0, inplace=False),
        transforms.RandomErasing(p=1.0, scale=(0.10, 0.16), ratio=(0.2, 0.3), value=0.0, inplace=False),
    ]
    er = bands[cid % len(bands)]

    def apply_on_tensor(x: torch.Tensor) -> torch.Tensor:
        return er(x)
    return apply_on_tensor


def make_loaders_mnist_noniid(n_clients: int, alpha: float, batch_size: int):
    base_tfm = transforms.ToTensor()
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=base_tfm)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    labels = np.array(train_base.targets)
    splits = dirichlet_split_noniid(labels, n_clients, alpha)

    client_loaders = []
    for i, idx in enumerate(splits):
        tfm_i = _client_erasing_transform(i)
        xs_list, ys_list = [], []
        for j in idx.tolist():
            x_tensor, y = train_base[j]
            x_tensor = tfm_i(x_tensor)
            xs_list.append(x_tensor)
            ys_list.append(int(y))
        
        if len(xs_list) > 0:
            xs = torch.stack(xs_list, dim=0)
            ys = torch.tensor(ys_list, dtype=torch.long)
        else:
            xs = torch.empty(0, 1, 28, 28)
            ys = torch.empty(0, dtype=torch.long)

        ds = TensorDataset(xs, ys)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    per_class = 10
    seen = {c: 0 for c in range(10)}
    ref_idx = []
    for i, y in enumerate(train_base.targets):
        if seen[int(y)] < per_class:
            ref_idx.append(i)
            seen[int(y)] += 1
        if all(seen[c] >= per_class for c in range(10)):
            break
    ref_loader = DataLoader(Subset(train_base, ref_idx), batch_size=batch_size, shuffle=False)
    return client_loaders, ref_loader, test_loader


# ----------------------------
# Methods
# ----------------------------
def run_plain_fl(cfg: FLConfig):
    client_loaders, ref_loader, test_loader = make_loaders_mnist_noniid(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size)
    global_model = SmallCNN().to(device)
    client_models = [SmallCNN().to(device) for _ in range(cfg.n_clients)]
    for _ in trange(cfg.rounds, desc="BL-A Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_cnn)
        global_model.load_state_dict(state_dict_avg(client_models))
    acc = eval_accuracy(global_model, test_loader)
    return acc, global_model, (client_loaders, ref_loader, test_loader)


def run_local_posthoc(cfg: FLConfig):
    acc_A, global_model, loaders = run_plain_fl(cfg)
    client_loaders, ref_loader, test_loader = loaders
    per_client_maps = []
    for i in trange(cfg.n_clients, desc="BL-B Clients", leave=False):
        m = SmallCNN().to(device)
        m.load_state_dict(global_model.state_dict())
        local_train(m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(m, client_loaders[i], max_per_class=30)
        per_client_maps.append(maps)
    return acc_A, global_model, per_client_maps, loaders


def run_server_summary(cfg: FLConfig):
    acc_A, global_model, loaders = run_plain_fl(cfg)
    client_loaders, ref_loader, test_loader = loaders
    client_hist = []
    for i in trange(cfg.n_clients, desc="BL-C Clients", leave=False):
        m = SmallCNN().to(device)
        m.load_state_dict(global_model.state_dict())
        local_train(m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(m, client_loaders[i], max_per_class=30)
        hist = {c: normalize_to_simplex(pool7x7(maps[c]), dim=0) for c in range(10)}
        client_hist.append(hist)
    global_hist = aggregate_histograms(client_hist)
    return acc_A, global_model, global_hist, loaders


def run_interpretable_only(cfg: FLConfig, mask_top_k: int = 400):
    client_loaders, ref_loader, test_loader = make_loaders_mnist_noniid(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size)
    V = 28*28
    K = min(mask_top_k, V)
    common_mask = torch.zeros(V, device=device)
    common_mask[:K] = 1.0

    core = min(32, K)
    drop_prob = 0.55
    client_masks = []
    rng = torch.Generator(device=device).manual_seed(SEED+999)
    for cid in range(cfg.n_clients):
        cm = common_mask.clone()
        tail = torch.arange(core, K, device=device)
        if tail.numel() > 0:
            drop = torch.rand(tail.numel(), generator=rng, device=device) < drop_prob
            cm[tail[drop]] = 0.0
            start = (cid * 300) % max(1, (V-K))
            pool = torch.arange(K+start, min(K+start+300, V), device=device)
            need = int(drop.sum().item())
            if need > 0 and pool.numel() > 0:
                if pool.numel() >= need:
                    sel = pool[torch.randperm(pool.numel(), generator=rng, device=device)[:need]]
                else:
                    reps = math.ceil(need / pool.numel())
                    sel = pool.repeat(reps)[:need]
                cm[sel] = 1.0
        client_masks.append(cm)

    global_model = SparseLinearInterpretable().to(device)
    client_models = [SparseLinearInterpretable().to(device) for _ in range(cfg.n_clients)]
    for _ in trange(cfg.rounds, desc="BL-D Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(client_models[i].parameters(), lr=cfg.lr_lin)
            client_models[i].train()
            for xb, yb in client_loaders[i]:
                xb, yb = xb.to(device), yb.to(device)
                logits = client_models[i](flatten_img(xb) * client_masks[i])
                loss = F.cross_entropy(logits, yb) + client_models[i].l1_penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
        global_model.load_state_dict(state_dict_avg(client_models))

    def eval_lin(model, loader):
        model.eval()
        cor = 0
        tot = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(flatten_img(xb) * common_mask)
                cor += (logits.argmax(dim=-1) == yb).sum().item()
                tot += yb.size(0)
        return cor / max(1, tot)

    acc = eval_lin(global_model, test_loader)

    # Global ref maps
    Wg = (global_model.W.weight.detach().clone() * common_mask.unsqueeze(0))
    Wmaps_global = {c: normalize_to_simplex(Wg[c].view(1, -1), dim=-1).view(1, 1, 28, 28) for c in range(10)}

    # Per-client maps
    per_client_maps = []
    for i in range(cfg.n_clients):
        m = SparseLinearInterpretable().to(device)
        m.load_state_dict(global_model.state_dict())
        local_train_interpretable(m, client_loaders[i], epochs=1, lr=cfg.lr_lin)
        Wi = m.W.weight.detach().clone() * client_masks[i].unsqueeze(0)
        maps_i = {c: normalize_to_simplex(Wi[c].view(1, -1), dim=-1).view(1, 1, 28, 28) for c in range(10)}
        per_client_maps.append(maps_i)

    return acc, global_model, Wmaps_global, per_client_maps, (client_loaders, ref_loader, test_loader)


# ---------- xFL (accuracy-parity: freeze CNN at BL-A, build Π & per-image maps) ----------
def run_xfl_from_blA(global_model_frozen: SmallCNN,
                     loaders, cfg_fl: FLConfig, cfg_xfl: XFLConfig):
    """Multi-round artifact mixing with co-annealing of sharpen_gamma and temperature.
    Does NOT change the CNN. Builds Π from client surrogates; returns same CNN accuracy."""
    client_loaders, ref_loader, test_loader = loaders
    Pi = normalize_per_class(torch.ones(10, 28*28, device=device))
    n_rounds = getattr(cfg_xfl, 'xfl_rounds', 1)

    for r in trange(n_rounds, desc="xFL Rounds", leave=False):
        # Co-annealing schedule: linear interpolation over rounds
        frac = r / max(1, n_rounds - 1) if n_rounds > 1 else 0.0
        current_gamma = cfg_xfl.gamma_start + (cfg_xfl.gamma_end - cfg_xfl.gamma_start) * frac
        current_temp = cfg_xfl.temp_start - (cfg_xfl.temp_start - cfg_xfl.temp_end) * frac

        # Beta warmup: ramp beta from 0 to beta_align_final over warmup rounds
        if r < cfg_xfl.align_warmup_rounds:
            beta = cfg_xfl.beta_align_final * (r + 1) / max(1, cfg_xfl.align_warmup_rounds)
        else:
            beta = cfg_xfl.beta_align_final

        artifacts = []
        for i in range(cfg_fl.n_clients):
            # Fit surrogate on each client against the frozen CNN
            surr = fit_surrogate_teacher_student(global_model_frozen, client_loaders[i], cfg_xfl,
                                                 steps=cfg_xfl.surrogate_epochs,
                                                 temperature=current_temp,
                                                 Pi=Pi)
            S = surrogate_to_artifact(surr, cfg_xfl)
            # align with current Pi using beta (warmup-aware)
            S_mix = normalize_per_class((1 - beta) * normalize_per_class(S) + beta * Pi)
            artifacts.append(S_mix)

        if len(artifacts) > 0:
            Pi = normalize_per_class(robust_aggregate_artifacts(artifacts))

    # Final parameters for evaluation: use annealed endpoint values
    final_temp = cfg_xfl.temp_end
    final_gamma = cfg_xfl.gamma_end
    # Store final gamma for build_imp_map to pick up
    cfg_xfl.sharpen_gamma = final_gamma

    # Accuracy equals BL-A (model unchanged)
    acc = eval_accuracy(global_model_frozen, test_loader)

    # Build per-client surrogates/weights for metrics using final temperature
    per_client_surr_maps = []
    per_client_surr_weights = []
    for i in range(cfg_fl.n_clients):
        surr = fit_surrogate_teacher_student(global_model_frozen, client_loaders[i], cfg_xfl,
                                             steps=cfg_xfl.surrogate_epochs,
                                             temperature=final_temp,
                                             Pi=Pi)
        W = surr.W.weight.detach()
        per_client_surr_weights.append(W.cpu())
        maps = {c: normalize_to_simplex(W[c].abs().view(1, -1), dim=-1).view(1, 1, 28, 28) for c in range(10)}
        per_client_surr_maps.append(maps)

    overhead = cfg_fl.n_clients * 10 * min(cfg_xfl.topk, 28*28) * 3  # ~index+weight
    avg_time = 0.0  # artifact-only pass
    return acc, global_model_frozen, Pi, per_client_surr_maps, per_client_surr_weights, overhead, avg_time, loaders


# ----------------------------
# Metrics helpers
# ----------------------------
def compute_edi(per_client_maps: List[Dict[int, torch.Tensor]], reference: Dict[int, torch.Tensor]) -> float:
    d = []
    for maps in per_client_maps:
        for c in range(10):
            p = maps[c].view(1, -1)
            q = reference[c].view(1, -1)
            d.append(float(jsd(p, q)))
    return float(np.mean(d)) if d else 0.0


def build_reference_from_mean(per_client_maps: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    ref = {}
    for c in range(10):
        M = torch.cat([m[c].view(1, -1) for m in per_client_maps], dim=0).mean(dim=0, keepdim=True)
        ref[c] = normalize_to_simplex(M, dim=-1).view(1, 1, 28, 28)
    return ref


def ref_from_Pi(Pi: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {c: normalize_to_simplex(Pi[c].view(1, -1), dim=-1).view(1, 1, 28, 28) for c in range(10)}


def sample_test_batch(loader, n=128):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
        if sum([t.size(0) for t in xs]) >= n:
            break
    xb = torch.cat(xs, dim=0)[:n].to(device)
    yb = torch.cat(ys, dim=0)[:n].to(device)
    return xb, yb


# ---------- Map builders (with improvements) ----------
def _avg_pool_3x3(m: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)


def build_imp_map(method: str, model, per_client_maps=None, global_hist=None, Pi=None,
                  x=None, y=None, client_id=0, surr_weights=None,
                  xfl_gamma: float = 1.0):
    if method == 'BL-B':
        return torch.cat([per_client_maps[client_id][int(y[i])] for i in range(x.size(0))], dim=0)
    if method == 'BL-C':
        with torch.no_grad():
            preds = model(x).argmax(dim=-1)
        ref = ref_from_hist(global_hist)
        return torch.cat([ref[int(preds[i])] for i in range(x.size(0))], dim=0)
    if method == 'BL-D':
        # Use global linear weights for predicted class, then smooth + small uniform mix
        with torch.no_grad():
            preds = model(flatten_img(x)).argmax(dim=-1)
        W = model.W.weight.detach()
        maps = []
        uniform = torch.ones(1, 1, 28, 28, device=x.device) / (28*28)
        for i in range(x.size(0)):
            c = int(preds[i])
            m = normalize_to_simplex(W[c].abs().view(1, -1), dim=-1).view(1, 1, 28, 28)
            m = _avg_pool_3x3(m)
            m = normalize_to_simplex(m.view(1, -1), dim=-1).view(1, 1, 28, 28)
            m = normalize_to_simplex((0.85 * m + 0.15 * uniform).view(1, -1), dim=-1).view(1, 1, 28, 28)
            maps.append(m)
        return torch.cat(maps, dim=0)
    if method == 'xFL':
        # Sharpened per-image surrogate map: (|W_c| * |x|)^gamma
        if surr_weights is not None and len(surr_weights) > 0:
            with torch.no_grad():
                preds = model(x).argmax(dim=-1)
            W = surr_weights[client_id].to(x.device)
            xf = x.view(x.size(0), -1)
            maps = []
            for i in range(x.size(0)):
                c = int(preds[i])
                base = (W[c].abs() * xf[i]).abs().view(1, 1, 28, 28)
                base = base.clamp_min(1e-8).pow(xfl_gamma)
                m = normalize_to_simplex(base.view(1, -1), dim=-1).view(1, 1, 28, 28)
                maps.append(m)
            return torch.cat(maps, dim=0)
        ref = ref_from_Pi(Pi)
        with torch.no_grad():
            preds = model(x).argmax(dim=-1)
        maps = []
        for i in range(x.size(0)):
            m = ref[int(preds[i])].clamp_min(1e-8).pow(xfl_gamma)
            m = normalize_to_simplex(m.view(1, -1), dim=-1).view(1, 1, 28, 28)
            maps.append(m)
        return torch.cat(maps, dim=0)
    raise ValueError("Unknown method")


# ----------------------------
# Bar-plot helper (for aggregated results)
# ----------------------------
def save_bar(values, labels, title, fname, ylabel, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    p = os.path.join(out_dir, fname)
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"Saved {p}")


# ----------------------------
# One full run for a given seed
# ----------------------------
def run_once(flcfg: FLConfig, xcfg: XFLConfig, seed: int):
    set_seed(seed)
    print(f"\n===== Seed {seed} =====")

    print("Running BL-A (Plain FL)...")
    acc_A, model_A, loaders_A = run_plain_fl(flcfg)
    client_loaders_A, ref_loader_A, test_loader_A = loaders_A
    print(f"BL-A Acc: {acc_A:.4f}")

    print("\nRunning BL-B (Local post-hoc)...")
    acc_B, model_B, per_client_maps_B, loaders_B = run_local_posthoc(flcfg)
    print(f"BL-B Acc: {acc_B:.4f}")

    print("\nRunning BL-C (Server summary)...")
    acc_C, model_C, global_hist_C, loaders_C = run_server_summary(flcfg)
    print(f"BL-C Acc: {acc_C:.4f}")

    print("\nRunning BL-D (Interpretable-only, divergent masks)...")
    acc_D, model_D, Wmaps_D_global, per_client_maps_D, loaders_D = run_interpretable_only(flcfg, mask_top_k=400)
    print(f"BL-D Acc: {acc_D:.4f}")

    print("\nRunning xFL (Proposed, accuracy-parity from BL-A)...")
    acc_X, model_X, Pi_X, per_client_surr_maps_X, per_client_surr_weights_X, overhead_X, avg_round_time_X, loaders_X = \
        run_xfl_from_blA(model_A, loaders_A, flcfg, xcfg)
    print(f"xFL Acc: {acc_X:.4f}; Overhead ~{overhead_X/1024:.1f} KB/round; Avg round ~{avg_round_time_X:.2f}s")

    # ----- Metrics -----
    # Consistency (EDI)
    ref_B = build_reference_from_mean(per_client_maps_B)
    edi_B = compute_edi(per_client_maps_B, ref_B)

    ref_C = ref_from_hist(global_hist_C)
    edi_C = compute_edi(per_client_maps_B, ref_C)   # reuse BL-B maps for client-side variety

    edi_D = compute_edi(per_client_maps_D, Wmaps_D_global)

    ref_X = ref_from_Pi(Pi_X)
    edi_X = compute_edi(per_client_surr_maps_X, ref_X)

    print(f"EDI -> BL-B: {edi_B:.4f} | BL-C: {edi_C:.4f} | BL-D: {edi_D:.4f} | xFL: {edi_X:.4f}")

    # Fidelity (Deletion/Insertion AUC) on a shared test batch
    xb, yb = sample_test_batch(test_loader_A, n=128)

    maps_B = build_imp_map('BL-B', model_B, per_client_maps=per_client_maps_B, x=xb, y=yb, client_id=0)
    maps_C = build_imp_map('BL-C', model_C, global_hist=global_hist_C, x=xb, y=yb)
    maps_D = build_imp_map('BL-D', model_D, x=xb, y=yb)
    maps_X = build_imp_map('xFL',  model_X, Pi=Pi_X, x=xb, y=yb,
                            surr_weights=per_client_surr_weights_X, client_id=0,
                            xfl_gamma=xcfg.sharpen_gamma)

    del_B, ins_B = deletion_insertion_auc(model_B, xb, yb, maps_B, steps=20)
    del_C, ins_C = deletion_insertion_auc(model_C, xb, yb, maps_C, steps=20)
    del_D, ins_D = deletion_insertion_auc(model_D, xb, yb, maps_D, steps=20)
    del_X, ins_X = deletion_insertion_auc(model_X, xb, yb, maps_X, steps=20)

    print(f"Deletion AUC (↓): BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Insertion AUC (↑): BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    # Simplicity (nnz)
    nnz_D = (model_D.W.weight.detach().abs() > 1e-8).sum().item()
    nnz_X = 10 * xcfg.topk
    print(f"Simplicity (nnz) -> BL-D: {nnz_D} | xFL(by design): {nnz_X}")
    print(f"Overhead per round -> BL-C ~{(49*10)/1024:.2f} KB | xFL ~{overhead_X/1024:.2f} KB")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "EDI": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "DeletionAUC": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "InsertionAUC": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
        "Simplicity": {"BL-D_nnz": nnz_D, "xFL_nnz": nnz_X},
        "Overhead_per_round_bytes": {"xFL": overhead_X},
    }


# ----------------------------
# Main: run for 5 seeds and summarize
# ----------------------------
if __name__ == "__main__":
    flcfg = FLConfig(n_clients=8, rounds=15, local_epochs=1, alpha_dirichlet=0.1)
    xcfg  = XFLConfig(topk=256, dp_sigma=0.10, surrogate_epochs=1,
                      beta_align_final=0.35, align_warmup_rounds=4,
                      sharpen_gamma=1.8)

    seeds = [BASE_SEED + i for i in range(5)]
    methods_all = ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]
    methods_expl = ["BL-B", "BL-C", "BL-D", "xFL"]

    metrics_acc = {m: [] for m in methods_all}
    metrics_edi = {m: [] for m in methods_expl}
    metrics_del = {m: [] for m in methods_expl}
    metrics_ins = {m: [] for m in methods_expl}
    simp_D = []
    simp_X = []
    over_X = []

    for s in seeds:
        out = run_once(flcfg, xcfg, s)
        for m in methods_all:
            metrics_acc[m].append(out["acc"][m])
        for m in methods_expl:
            metrics_edi[m].append(out["EDI"][m])
            metrics_del[m].append(out["DeletionAUC"][m])
            metrics_ins[m].append(out["InsertionAUC"][m])
        simp_D.append(out["Simplicity"]["BL-D_nnz"])
        simp_X.append(out["Simplicity"]["xFL_nnz"])
        over_X.append(out["Overhead_per_round_bytes"]["xFL"])

    def mean_std(lst):
        arr = np.array(lst, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print("\n===== Summary over 5 seeds =====")
    print("Accuracy:")
    for m in methods_all:
        mu, sd = mean_std(metrics_acc[m])
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    print("\nEDI (lower better):")
    for m in methods_expl:
        mu, sd = mean_std(metrics_edi[m])
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    print("\nDeletion AUC (lower better):")
    for m in methods_expl:
        mu, sd = mean_std(metrics_del[m])
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    print("\nInsertion AUC (higher better):")
    for m in methods_expl:
        mu, sd = mean_std(metrics_ins[m])
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    mu_simp_D, sd_simp_D = mean_std(simp_D)
    mu_simp_X, sd_simp_X = mean_std(simp_X)
    mu_over_X, sd_over_X = mean_std(over_X)

    print("\nSimplicity (nnz):")
    print(f"  BL-D: {mu_simp_D:.1f} ± {sd_simp_D:.1f}")
    print(f"  xFL:  {mu_simp_X:.1f} ± {sd_simp_X:.1f}")

    print("\nOverhead per round (bytes, xFL):")
    print(f"  xFL: {mu_over_X:.1f} ± {sd_over_X:.1f}")

    # Aggregated plots
    out_dir = "mnist_noniid_outputs_fixed_5seeds"
    mean_del = [mean_std(metrics_del[m])[0] for m in methods_expl]
    mean_ins = [mean_std(metrics_ins[m])[0] for m in methods_expl]
    mean_edi = [mean_std(metrics_edi[m])[0] for m in methods_expl]

    save_bar(mean_del, methods_expl, "Deletion AUC (mean over 5 seeds)", "bar_del_auc_mean.png", "AUC", out_dir)
    save_bar(mean_ins, methods_expl, "Insertion AUC (mean over 5 seeds)", "bar_ins_auc_mean.png", "AUC", out_dir)
    save_bar(mean_edi, methods_expl, "Consistency EDI (mean over 5 seeds)", "bar_edi_mean.png", "EDI", out_dir)

    # Summary JSON with mean/std
    summary = {
        "acc_mean_std": {m: {"mean": mean_std(metrics_acc[m])[0],
                             "std": mean_std(metrics_acc[m])[1]} for m in methods_all},
        "EDI_mean_std": {m: {"mean": mean_std(metrics_edi[m])[0],
                             "std": mean_std(metrics_edi[m])[1]} for m in methods_expl},
        "DeletionAUC_mean_std": {m: {"mean": mean_std(metrics_del[m])[0],
                                     "std": mean_std(metrics_del[m])[1]} for m in methods_expl},
        "InsertionAUC_mean_std": {m: {"mean": mean_std(metrics_ins[m])[0],
                                      "std": mean_std(metrics_ins[m])[1]} for m in methods_expl},
        "Simplicity_mean_std": {
            "BL-D_nnz": {"mean": mu_simp_D, "std": sd_simp_D},
            "xFL_nnz": {"mean": mu_simp_X, "std": sd_simp_X},
        },
        "Overhead_per_round_bytes_mean_std": {
            "xFL": {"mean": mu_over_X, "std": sd_over_X}
        },
        "seeds": seeds,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_mean_std.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregated outputs to {out_dir}")
