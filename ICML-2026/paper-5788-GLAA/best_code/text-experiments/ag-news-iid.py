"""
xFL on AG News (Text)
Clean, structured implementation for multi-seed experiments.
"""

import os
import re
import csv
import json
import math
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

# ----------------------------
# Global Configuration
# ----------------------------
BASE_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Configuration Classes
# ----------------------------
@dataclass
class FLConfig:
    n_clients: int = 8
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 64
    alpha_dirichlet: float = 0.15
    lr_cnn: float = 2e-3

@dataclass
class XFLConfig:
    topk: int = 256
    quant_bits: int = 8
    clip_radius: float = 5.0
    dp_sigma: float = 0.2
    temperature: float = 2.5
    l1_lambda: float = 1e-6
    surrogate_every_R: int = 2
    beta_align_final: float = 0.25
    align_warmup_rounds: int = 6

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data Utilities
# ----------------------------
def find_ag_news_csv(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
    if not os.path.exists(base_dir):
        return None, None
    candidates = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith('.csv') and ('train' in lf or 'test' in lf):
                candidates.append(os.path.join(root, f))
    train_path, test_path = None, None
    for p in candidates:
        if 'train' in os.path.basename(p).lower(): train_path = p
        if 'test' in os.path.basename(p).lower(): test_path = p
    return train_path, test_path

_token_re = re.compile(r"[A-Za-z0-9_]+")
def basic_tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())

class AGNewsRaw:
    def __init__(self, base_dir: str):
        tr_csv, te_csv = find_ag_news_csv(base_dir)
        if tr_csv and te_csv:
            print(f"Loading AG News from {tr_csv} and {te_csv}")
            self.train = self._read_csv(tr_csv)
            self.test = self._read_csv(te_csv)
        else:
            print(f"AG News CSVs not found in {base_dir}. Generating synthetic data for demonstration.")
            self.train = self._generate_synthetic(2000)
            self.test = self._generate_synthetic(500)

    def _read_csv(self, path: str):
        rows = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row: continue
                if not row[0].strip().isdigit(): continue
                lab = int(row[0].strip()) - 1
                title = row[1] if len(row) > 1 else ''
                desc  = row[2] if len(row) > 2 else ''
                text = (title + ' ' + desc).strip()
                rows.append((lab, text))
        return rows

    def _generate_synthetic(self, n_samples: int):
        # Generate dummy text data if real data is missing
        vocab_sample = ["world", "sports", "business", "tech", "finance", "football", "computer", "market", "game", "cpu"]
        rows = []
        for _ in range(n_samples):
            lab = random.randint(0, 3)
            length = random.randint(10, 50)
            text = " ".join(random.choices(vocab_sample, k=length))
            rows.append((lab, text))
        return rows

PAD, UNK = 0, 1

class Vocab:
    def __init__(self, max_size: int = 20000, min_freq: int = 2):
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": PAD, "<unk>": UNK}
        self.max_size = max_size
        self.min_freq = min_freq

    def build(self, texts: List[str]):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(basic_tokenize(t))
        items = [(w, f) for w, f in cnt.items() if f >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        for w, _ in items[: self.max_size - len(self.itos)]:
            self.stoi[w] = len(self.itos)
            self.itos.append(w)

    def encode(self, text: str) -> List[int]:
        toks = basic_tokenize(text)
        return [self.stoi.get(t, UNK) for t in toks]

class SeqDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, str]], vocab: Vocab, max_len: int = 128):
        self.labels = [lab for lab, _ in pairs]
        self.ids = [vocab.encode(txt) for _, txt in pairs]
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.ids[idx], self.labels[idx]

def collate_pad(batch, pad_id: int = PAD, max_len: int = 128):
    ids, labs = zip(*batch)
    B = len(ids)
    X = torch.full((B, max_len), pad_id, dtype=torch.long)
    L = torch.zeros(B, dtype=torch.long)
    for i, seq in enumerate(ids):
        s = seq[:max_len]
        X[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        L[i] = len(s)
    y = torch.tensor(labs, dtype=torch.long)
    return X, L, y

def dirichlet_split_noniid(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes): np.random.shuffle(idx_by_class[c])
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = idx_by_class[c]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions / proportions.sum())
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i in range(n_clients): client_indices[i].extend(parts[i])
    return [np.array(ci, dtype=np.int64) for ci in client_indices]

def build_client_loaders(raw: AGNewsRaw, vocab: Vocab, max_len: int, n_clients: int, alpha: float, batch: int):
    train_ds = SeqDataset(raw.train, vocab, max_len)
    test_ds  = SeqDataset(raw.test,  vocab, max_len)
    labels = np.array([lab for lab, _ in raw.train], dtype=np.int64)
    splits = dirichlet_split_noniid(labels, n_clients, alpha)
    client_loaders = []
    for idx in splits:
        sub = Subset(train_ds, idx.tolist())
        client_loaders.append(DataLoader(sub, batch_size=batch, shuffle=True, drop_last=False, collate_fn=lambda b: collate_pad(b, PAD, max_len)))
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=False, collate_fn=lambda b: collate_pad(b, PAD, max_len))
    return client_loaders, test_loader

# ----------------------------
# Models
# ----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int = 4, emb_dim: int = 128, num_filters: int = 64, kernel_sizes=(3,4,5), dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)
        self.drop = nn.Dropout(dropout)

    def forward_from_emb(self, E):
        x = E.transpose(1, 2)
        feats = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        cat = torch.cat(feats, dim=-1)
        cat = self.drop(cat)
        return self.fc(cat)

    def forward(self, ids):
        E = self.emb(ids)
        return self.forward_from_emb(E)

class SparseLinearSurrogate(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 4, l1_lambda: float = 1e-6):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes)
        self.l1_lambda = l1_lambda
    def forward(self, bow): return self.W(bow)
    def l1_penalty(self): return self.l1_lambda * self.W.weight.abs().sum()

class SparseLinearInterpretable(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 4, l1_lambda: float = 2e-6):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes)
        self.l1_lambda = l1_lambda
    def forward(self, bow): return self.W(bow)
    def l1_penalty(self): return self.l1_lambda * self.W.weight.abs().sum()

# ----------------------------
# Helpers
# ----------------------------
def make_bow(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    B, L = ids.shape
    bow = torch.zeros(B, vocab_size, device=ids.device)
    bow.scatter_add_(1, ids, torch.ones_like(ids, dtype=bow.dtype))
    bow[:, PAD] = 0.0
    return bow

def normalize_to_simplex(vec: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    v = vec.abs()
    s = v.sum(dim=dim, keepdim=True).clamp_min(eps)
    return v / s

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def state_dict_avg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    for m in models:
        sd = m.state_dict()
        for k in keys: out[k] += sd[k]
    for k in keys: out[k] /= len(models)
    return out

# ----------------------------
# Training & Attribution
# ----------------------------
def train_textcnn_local(model: TextCNN, loader: DataLoader, epochs=1, lr=2e-3, wd=0.0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        for ids, L, y in loader:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_accuracy_text(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    tot, cor = 0, 0
    with torch.no_grad():
        for ids, L, y in loader:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            logits = model(ids)
            pred = logits.argmax(dim=-1)
            cor += (pred == y).sum().item()
            tot += y.size(0)
    return cor / max(1, tot)

def integrated_gradients_text(model: TextCNN, ids: torch.Tensor, y: torch.Tensor, steps: int = 16) -> torch.Tensor:
    model.eval()
    ids = ids.to(DEVICE); y = y.to(DEVICE)
    with torch.no_grad(): E = model.emb(ids)
    baseline = torch.zeros_like(E)
    grads = []
    for t in range(1, steps + 1):
        Et = baseline + (t / steps) * (E - baseline)
        Et.requires_grad_(True)
        logits = model.forward_from_emb(Et)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grads.append(Et.grad.detach().clone())
        model.zero_grad(set_to_none=True)
    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    ig = (E - baseline) * avg_grads
    return ig.norm(p=2, dim=-1)

def build_ig_vocab_hist_per_class(model: TextCNN, loader: DataLoader, vocab_size: int, max_per_class: int = 50) -> Dict[int, torch.Tensor]:
    model.eval()
    per_class_ids = {c: [] for c in range(4)}
    per_class_y   = {c: [] for c in range(4)}
    for ids, L, y in loader:
        for i in range(ids.size(0)):
            c = int(y[i].item())
            if len(per_class_ids[c]) < max_per_class:
                per_class_ids[c].append(ids[i:i+1].clone())
                per_class_y[c].append(y[i:i+1].clone())
        if all(len(per_class_ids[c]) >= max_per_class for c in range(4)): break
    out = {}
    for c in range(4):
        if len(per_class_ids[c]) == 0:
            out[c] = torch.ones(vocab_size, device=DEVICE) / vocab_size
            continue
        X = torch.cat(per_class_ids[c], dim=0).to(DEVICE)
        Y = torch.cat(per_class_y[c], dim=0).to(DEVICE)
        imp = integrated_gradients_text(model, X, Y, steps=8)
        vocab_hist = torch.zeros(vocab_size, device=DEVICE)
        vocab_hist.scatter_add_(0, X.view(-1), imp.view(-1))
        vocab_hist[PAD] = 0.0
        out[c] = normalize_to_simplex(vocab_hist, dim=0)
    return out

def aggregate_histograms_median(hists: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out = {}
    for c in range(4):
        mats = torch.stack([d[c] for d in hists], dim=0)
        out[c] = normalize_to_simplex(mats.median(dim=0).values, dim=0)
    return out

# ----------------------------
# xFL Components
# ----------------------------
def fit_surrogate_teacher_student(model: TextCNN, loader: DataLoader, vocab_size: int, cfg: XFLConfig, steps: int = 1) -> SparseLinearSurrogate:
    surr = SparseLinearSurrogate(in_features=vocab_size, n_classes=4, l1_lambda=cfg.l1_lambda).to(DEVICE)
    opt = torch.optim.SGD(surr.parameters(), lr=0.1)
    T = cfg.temperature
    model.eval()
    for _ in range(steps):
        for ids, L, _ in loader:
            ids = ids.to(DEVICE)
            with torch.no_grad():
                logits_t = model(ids) / T
                probs_t = F.softmax(logits_t, dim=-1)
                conf = probs_t.max(dim=-1).values.detach()
            bow = make_bow(ids, vocab_size)
            logits_s = surr(bow) / T
            loss = F.kl_div(F.log_softmax(logits_s, dim=-1), probs_t, reduction='none').sum(dim=-1)
            loss = (loss * conf).mean() + surr.l1_penalty()
            opt.zero_grad(); loss.backward(); opt.step()
    return surr

def surrogate_to_artifact(W: torch.Tensor, cfg: XFLConfig) -> torch.Tensor:
    C, V = W.shape
    Wabs = W.abs()
    mask = torch.zeros_like(Wabs)
    k = min(cfg.topk, V)
    for c in range(C):
        idx = torch.topk(Wabs[c], k).indices
        mask[c, idx] = 1.0
    Wk = W * mask
    for c in range(C):
        v = Wk[c]
        n = v.norm(2).clamp_min(1e-8)
        scale = min(1.0, cfg.clip_radius / float(n))
        Wk[c] = v * scale
    scale = 127.0 / Wk.abs().max().clamp_min(1e-8)
    Wq = torch.round(Wk * scale) / scale
    Wdp = Wq + torch.randn_like(Wq) * cfg.dp_sigma
    return Wdp

def robust_aggregate_artifacts(arts) -> torch.Tensor:
    if isinstance(arts, torch.Tensor): M = arts
    else: M = torch.stack(arts, dim=0)
    return M.median(dim=0).values

def normalize_per_class(W: torch.Tensor) -> torch.Tensor:
    out = []
    for c in range(W.size(0)):
        v = W[c].abs(); v[PAD] = 0.0
        s = v.sum().clamp_min(1e-12)
        out.append(v / s)
    return torch.stack(out, dim=0)

# ----------------------------
# Runners
# ----------------------------
def run_plain_fl(cfg: FLConfig, vocab_size: int, max_len: int, client_loaders, test_loader):
    global_model = TextCNN(vocab_size=vocab_size).to(DEVICE)
    client_models = [TextCNN(vocab_size=vocab_size).to(DEVICE) for _ in range(cfg.n_clients)]
    for _ in trange(cfg.rounds, desc="BL-A Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            train_textcnn_local(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_cnn)
        global_model.load_state_dict(state_dict_avg(client_models))
    acc = eval_accuracy_text(global_model, test_loader)
    return acc, global_model

def run_local_posthoc(cfg: FLConfig, vocab_size: int, max_len: int, client_loaders, test_loader):
    acc_A, global_model = run_plain_fl(cfg, vocab_size, max_len, client_loaders, test_loader)
    per_client_hists = []
    for i in tqdm(range(cfg.n_clients), desc="BL-B Clients", leave=False):
        local_m = TextCNN(vocab_size=vocab_size).to(DEVICE)
        local_m.load_state_dict(global_model.state_dict())
        train_textcnn_local(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        h = build_ig_vocab_hist_per_class(local_m, client_loaders[i], vocab_size, max_per_class=30)
        per_client_hists.append(h)
    return acc_A, global_model, per_client_hists

def run_server_summary(cfg: FLConfig, vocab_size: int, max_len: int, client_loaders, test_loader):
    acc_A, global_model = run_plain_fl(cfg, vocab_size, max_len, client_loaders, test_loader)
    client_hists = []
    for i in tqdm(range(cfg.n_clients), desc="BL-C Clients", leave=False):
        local_m = TextCNN(vocab_size=vocab_size).to(DEVICE)
        local_m.load_state_dict(global_model.state_dict())
        train_textcnn_local(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        h = build_ig_vocab_hist_per_class(local_m, client_loaders[i], vocab_size, max_per_class=30)
        client_hists.append(h)
    global_hist = aggregate_histograms_median(client_hists)
    return acc_A, global_model, global_hist, client_hists

def run_interpretable_only(cfg: FLConfig, vocab_size: int, client_loaders, test_loader, mask_top_k: int = 800):
    K = min(mask_top_k, vocab_size)
    feat_mask = torch.zeros(vocab_size, device=DEVICE)
    feat_mask[:K] = 1.0; feat_mask[PAD] = 0.0

    global_model = SparseLinearInterpretable(in_features=vocab_size, l1_lambda=5e-5).to(DEVICE)
    client_models = [SparseLinearInterpretable(in_features=vocab_size, l1_lambda=5e-5).to(DEVICE) for _ in range(cfg.n_clients)]
    opt_lr = 0.2

    for _ in trange(cfg.rounds, desc="BL-D Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            model = client_models[i]; model.train()
            opt = torch.optim.SGD(model.parameters(), lr=opt_lr)
            for ids, L, y in client_loaders[i]:
                ids, y = ids.to(DEVICE), y.to(DEVICE)
                bow = make_bow(ids, vocab_size) * feat_mask
                logits = model(bow)
                loss = F.cross_entropy(logits, y) + model.l1_penalty()
                opt.zero_grad(); loss.backward(); opt.step()
        global_model.load_state_dict(state_dict_avg(client_models))

    def eval_lin(m):
        m.eval(); tot=0; cor=0
        with torch.no_grad():
            for ids, L, y in test_loader:
                ids, y = ids.to(DEVICE), y.to(DEVICE)
                bow = make_bow(ids, vocab_size) * feat_mask
                logits = m(bow)
                pred = logits.argmax(dim=-1)
                cor += (pred == y).sum().item(); tot += y.size(0)
        return cor / max(1, tot)
    acc = eval_lin(global_model)

    Wg = global_model.W.weight.detach().clone() * feat_mask.unsqueeze(0)
    Wmaps_global = {c: normalize_to_simplex(Wg[c], dim=0) for c in range(4)}

    per_client_maps = []
    for i in range(cfg.n_clients):
        m = SparseLinearInterpretable(in_features=vocab_size, l1_lambda=5e-5).to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        m.train(); opt = torch.optim.SGD(m.parameters(), lr=opt_lr)
        for ids, L, y in client_loaders[i]:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            bow = make_bow(ids, vocab_size) * feat_mask
            logits = m(bow)
            loss = F.cross_entropy(logits, y) + m.l1_penalty()
            opt.zero_grad(); loss.backward(); opt.step()
        Wi = m.W.weight.detach().clone() * feat_mask.unsqueeze(0)
        per_client_maps.append({c: normalize_to_simplex(Wi[c], dim=0) for c in range(4)})

    return acc, global_model, Wmaps_global, per_client_maps, feat_mask

def run_xfl(cfg: FLConfig, xcfg: XFLConfig, vocab_size: int, max_len: int, client_loaders, test_loader):
    global_model = TextCNN(vocab_size=vocab_size).to(DEVICE)
    client_models = [TextCNN(vocab_size=vocab_size).to(DEVICE) for _ in range(cfg.n_clients)]
    Pi = normalize_per_class(torch.ones(4, vocab_size, device=DEVICE))
    round_times = []; per_round_overhead_bytes = 0

    for r in trange(cfg.rounds, desc="xFL Rounds", leave=False):
        start_t = time.time()
        artifacts = []
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            train_textcnn_local(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_cnn)
            if (r % xcfg.surrogate_every_R) == 0:
                surr = fit_surrogate_teacher_student(client_models[i], client_loaders[i], vocab_size, xcfg, steps=1)
                W = surr.W.weight.detach()
                S_i = surrogate_to_artifact(W, xcfg)
                beta = xcfg.beta_align_final * min(1.0, (r+1)/max(1, xcfg.align_warmup_rounds))
                S_norm = normalize_per_class(S_i)
                S_mix = (1 - beta) * S_norm + beta * Pi
                S_mix = normalize_per_class(S_mix)
                artifacts.append(S_mix)
        global_model.load_state_dict(state_dict_avg(client_models))
        if len(artifacts) > 0:
            Pi = normalize_per_class(robust_aggregate_artifacts(artifacts))
        round_times.append(time.time() - start_t)
        per_round_overhead_bytes = cfg.n_clients * 4 * min(xcfg.topk, vocab_size) * 3

    acc = eval_accuracy_text(global_model, test_loader)
    per_client_surr_weights = []
    per_client_surr_maps = []
    for i in tqdm(range(cfg.n_clients), desc="xFL Post-hoc", leave=False):
        loc = TextCNN(vocab_size=vocab_size).to(DEVICE)
        loc.load_state_dict(global_model.state_dict())
        surr = fit_surrogate_teacher_student(loc, client_loaders[i], vocab_size, xcfg, steps=1)
        W = surr.W.weight.detach()
        per_client_surr_weights.append(W.cpu())
        per_client_surr_maps.append({c: normalize_to_simplex(W[c], dim=0) for c in range(4)})

    avg_round_time = float(np.mean(round_times)) if round_times else 0.0
    return acc, global_model, Pi, per_client_surr_maps, per_client_surr_weights, per_round_overhead_bytes, avg_round_time

# ----------------------------
# Metrics & Visualization
# ----------------------------
def compute_edi(per_client_maps: List[Dict[int, torch.Tensor]], reference: Dict[int, torch.Tensor]) -> float:
    dists = []
    for maps in per_client_maps:
        for c in range(4):
            p = maps[c].view(1, -1)
            q = reference[c].view(1, -1)
            d = jsd(p, q)
            dists.append(float(d.item()))
    return float(np.mean(dists)) if dists else 0.0

def ref_from_mean(per_client_maps: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    ref = {}
    for c in range(4):
        M = torch.stack([m[c] for m in per_client_maps], dim=0)
        ref[c] = normalize_to_simplex(M.mean(dim=0), dim=0)
    return ref

def ref_from_hist(global_hist: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    return {c: normalize_to_simplex(global_hist[c], dim=0) for c in range(4)}

def ref_from_Pi(Pi: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {c: normalize_to_simplex(Pi[c], dim=0) for c in range(Pi.size(0))}

def build_imp_vector(method: str, model: nn.Module, ids: torch.Tensor, y: torch.Tensor,
                     per_client_maps=None, global_hist=None, Pi=None,
                     surr_weights: List[torch.Tensor] = None, client_id: int = 0,
                     vocab_size: int = 20000) -> torch.Tensor:
    B = ids.size(0)
    if method == 'BL-B':
        out = []
        for i in range(B):
            c = int(y[i].item())
            out.append(per_client_maps[client_id][c])
        return torch.stack(out, dim=0)
    elif method == 'BL-C':
        with torch.no_grad(): preds = model(ids.to(DEVICE)).argmax(dim=-1)
        out = []
        for i in range(B):
            c = int(preds[i].item())
            out.append(global_hist[c])
        return torch.stack(out, dim=0)
    elif method == 'BL-D':
        with torch.no_grad():
            bow = make_bow(ids.to(DEVICE), vocab_size)
            preds = model(bow).argmax(dim=-1)
        W = model.W.weight.detach()
        out = []
        for i in range(B):
            c = int(preds[i].item())
            vec = normalize_to_simplex(W[c], dim=0)
            out.append(vec)
        return torch.stack(out, dim=0)
    elif method == 'xFL':
        with torch.no_grad(): preds = model(ids.to(DEVICE)).argmax(dim=-1)
        W = surr_weights[client_id].to(DEVICE)
        bow = make_bow(ids.to(DEVICE), vocab_size)
        out = []
        for i in range(B):
            c = int(preds[i].item())
            v = (W[c].abs() * bow[i]).clamp_min(0)
            out.append(normalize_to_simplex(v, dim=0))
        return torch.stack(out, dim=0)
    else: raise ValueError("Unknown method")

def deletion_insertion_auc_text(model: nn.Module, ids: torch.Tensor, y: torch.Tensor, imp: torch.Tensor, vocab_size: int, steps: int = 20):
    B, L = ids.shape
    top_order = torch.argsort(imp, dim=-1, descending=True)
    xs = [s/steps for s in range(steps+1)]
    del_scores, ins_scores = [], []
    ids = ids.to(DEVICE); y = y.to(DEVICE)

    for s in range(steps + 1):
        frac = s / steps
        k = int(frac * vocab_size)
        keep_masks = torch.ones(B, vocab_size, device=DEVICE, dtype=torch.bool)
        if k > 0:
            for i in range(B): keep_masks[i, top_order[i, :k]] = False
        
        ids_del = ids.clone()
        for i in range(B):
            mask_row = keep_masks[i]
            row = ids_del[i]
            row[~mask_row[row]] = PAD
        with torch.no_grad():
            prob_del = F.softmax(model(ids_del), dim=-1)[torch.arange(B, device=DEVICE), y]
        del_scores.append(prob_del.mean().item())

        ids_ins = torch.full_like(ids, PAD)
        if k > 0:
            for i in range(B):
                keep = ~keep_masks[i]
                tok = ids[i]
                sel = keep[tok]
                ids_ins[i, sel] = tok[sel]
        with torch.no_grad():
            prob_ins = F.softmax(model(ids_ins), dim=-1)[torch.arange(B, device=DEVICE), y]
        ins_scores.append(prob_ins.mean().item())

    def auc_area(xs, ys):
        area = 0.0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            area += 0.5 * (ys[i] + ys[i-1]) * dx
        return float(area)
    return auc_area(xs, del_scores), auc_area(xs, ins_scores)

def save_token_barplot(vocab: Vocab, imp_vec: torch.Tensor, path: str, topn: int = 15, title: str = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    v = imp_vec.detach().cpu().numpy()
    v[PAD] = 0.0; v[UNK] = 0.0
    top_idx = np.argsort(-v)[:topn]
    labels = [vocab.itos[i] if i < len(vocab.itos) else f"tok{i}" for i in top_idx]
    vals = v[top_idx]
    plt.figure(figsize=(8, 3))
    plt.bar(range(topn), vals)
    plt.xticks(range(topn), labels, rotation=45, ha='right')
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()

def save_bar(values, labels, title, fname, ylabel):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(5,3))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", fname), bbox_inches='tight'); plt.close()

# ----------------------------
# Experiment Runner
# ----------------------------
def run_once(flcfg: FLConfig, xcfg: XFLConfig, seed: int, base_dir: str, max_vocab: int, max_len: int):
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    raw = AGNewsRaw(base_dir)
    vocab = Vocab(max_size=max_vocab, min_freq=2)
    vocab.build([txt for _, txt in raw.train])
    V = len(vocab.itos)
    print(f"Vocab size: {V}")

    client_loaders, test_loader = build_client_loaders(raw, vocab, max_len, flcfg.n_clients, flcfg.alpha_dirichlet, flcfg.batch_size)

    # BL-A
    print("Running BL-A (Plain FL)...")
    acc_A, model_A = run_plain_fl(flcfg, V, max_len, client_loaders, test_loader)
    print(f"BL-A Acc: {acc_A:.4f}")

    # BL-B
    print("\nRunning BL-B (Local post-hoc)...")
    acc_B, model_B, per_client_hists_B = run_local_posthoc(flcfg, V, max_len, client_loaders, test_loader)
    print(f"BL-B Acc: {acc_B:.4f}")

    # BL-C
    print("\nRunning BL-C (Server summary)...")
    acc_C, model_C, global_hist_C, _ = run_server_summary(flcfg, V, max_len, client_loaders, test_loader)
    print(f"BL-C Acc: {acc_C:.4f}")

    # BL-D
    print("\nRunning BL-D (Interpretable-only)...")
    acc_D, model_D, Wmaps_D_global, per_client_maps_D, feat_mask_D = run_interpretable_only(flcfg, V, client_loaders, test_loader, mask_top_k=800)
    print(f"BL-D Acc: {acc_D:.4f}")

    # xFL
    print("\nRunning xFL (Proposed)...")
    acc_X, model_X, Pi_X, per_client_surr_maps_X, per_client_surr_weights_X, overhead_X, _ = run_xfl(flcfg, xcfg, V, max_len, client_loaders, test_loader)
    print(f"xFL Acc: {acc_X:.4f}")

    # EDI
    print("Computing EDI...")
    ref_B = ref_from_mean(per_client_hists_B)
    edi_B = compute_edi(per_client_hists_B, ref_B)
    ref_C = ref_from_hist(global_hist_C)
    edi_C = compute_edi(per_client_hists_B, ref_C)
    edi_D = compute_edi(per_client_maps_D, Wmaps_D_global)
    ref_X = ref_from_Pi(Pi_X)
    edi_X = compute_edi(per_client_surr_maps_X, ref_X)
    print(f"EDI -> BL-B: {edi_B:.4f} | BL-C: {edi_C:.4f} | BL-D: {edi_D:.4f} | xFL: {edi_X:.4f}")

    # Fidelity
    print("Computing Fidelity...")
    def sample_test_batch(loader, n=128):
        ids_list, L_list, y_list = [], [], []
        for ids, L, y in loader:
            ids_list.append(ids); L_list.append(L); y_list.append(y)
            if sum([t.size(0) for t in ids_list]) >= n: break
        ids = torch.cat(ids_list, dim=0)[:n]
        y = torch.cat(y_list, dim=0)[:n]
        return ids, y

    ids_batch, y_batch = sample_test_batch(test_loader, n=128)
    imp_B = build_imp_vector('BL-B', model_B, ids_batch, y_batch, per_client_maps=per_client_hists_B, vocab_size=V)
    imp_C = build_imp_vector('BL-C', model_C, ids_batch, y_batch, global_hist=global_hist_C, vocab_size=V)
    imp_D = build_imp_vector('BL-D', model_D, ids_batch, y_batch, vocab_size=V) * feat_mask_D.unsqueeze(0)
    imp_X = build_imp_vector('xFL', model_X, ids_batch, y_batch, surr_weights=per_client_surr_weights_X, client_id=0, vocab_size=V)

    del_B, ins_B = deletion_insertion_auc_text(model_B, ids_batch.clone(), y_batch.clone(), imp_B, V, steps=20)
    del_C, ins_C = deletion_insertion_auc_text(model_C, ids_batch.clone(), y_batch.clone(), imp_C, V, steps=20)
    del_D, ins_D = deletion_insertion_auc_text(lambda x: model_D(make_bow(x, V) * feat_mask_D), ids_batch.clone(), y_batch.clone(), imp_D, V, steps=20)
    del_X, ins_X = deletion_insertion_auc_text(model_X, ids_batch.clone(), y_batch.clone(), imp_X, V, steps=20)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
    }

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    flcfg = FLConfig(n_clients=8, rounds=10, local_epochs=1, batch_size=64, alpha_dirichlet=0.15, lr_cnn=2e-3)
    xcfg  = XFLConfig(topk=256, dp_sigma=0.15, surrogate_every_R=2, beta_align_final=0.25, align_warmup_rounds=6, l1_lambda=1e-6)
    
    seeds = [BASE_SEED + i for i in range(5)]
    methods_all = ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]
    methods_expl = ["BL-B", "BL-C", "BL-D", "xFL"]

    metrics = {
        "acc": {m: [] for m in methods_all},
        "edi": {m: [] for m in methods_expl},
        "del_auc": {m: [] for m in methods_expl},
        "ins_auc": {m: [] for m in methods_expl},
    }

    print(f"Starting experiment for {len(seeds)} seeds...")
    for s in seeds:
        out = run_once(flcfg, xcfg, s, base_dir="ag_news_dataset", max_vocab=20000, max_len=128)
        for m in methods_all:
            metrics["acc"][m].append(out["acc"][m])
        for m in methods_expl:
            metrics["edi"][m].append(out["edi"][m])
            metrics["del_auc"][m].append(out["del_auc"][m])
            metrics["ins_auc"][m].append(out["ins_auc"][m])

    def summarize(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print("\n===== Summary over seeds =====")
    print("Accuracy:")
    for m in methods_all:
        mean_v, std_v = summarize(metrics["acc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nEDI (lower better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["edi"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nDeletion AUC (lower better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["del_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nInsertion AUC (higher better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["ins_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    # Save plots
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods_expl]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods_expl]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods_expl]

    save_bar(mean_del, methods_expl, "Deletion AUC (mean)", "bar_deletion_auc_mean.png", "AUC")
    save_bar(mean_ins, methods_expl, "Insertion AUC (mean)", "bar_insertion_auc_mean.png", "AUC")
    save_bar(mean_edi, methods_expl, "Consistency (EDI, mean)", "bar_edi_mean.png", "EDI")

    print("Done.")
