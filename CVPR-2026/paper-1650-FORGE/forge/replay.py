import copy
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    HAVE_SK = True
except Exception:
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    roc_auc_score = None
    HAVE_SK = False


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels, probs1 = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, g = model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(1)
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
        probs1.extend(p[:, 1].detach().cpu().numpy().tolist())
    if len(labels) == 0:
        return dict(acc=0, precision=0, recall=0, f1=0, auc=0, sen=0, fpr=0)
    y = np.asarray(labels, dtype=np.int64)
    yhat = np.asarray(preds, dtype=np.int64)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    if HAVE_SK:
        acc = float(accuracy_score(labels, preds))
        prec = float(precision_score(labels, preds, average="macro", zero_division=0))
        rec = float(recall_score(labels, preds, average="macro", zero_division=0))
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(labels, probs1))
        except Exception:
            auc = 0.0
        return dict(acc=acc, precision=prec, recall=rec, f1=f1, auc=auc, sen=sen, fpr=fpr)
    acc = float(np.mean(y == yhat))
    return dict(acc=acc, precision=0, recall=0, f1=0, auc=0, sen=sen, fpr=fpr)


@torch.no_grad()
def eval_seen_plus_current(model: nn.Module, seen_loaders: List[DataLoader], current_loader: DataLoader, device):
    accs = [eval_loader(model, l, device)["acc"] for l in seen_loaders]
    accs.append(eval_loader(model, current_loader, device)["acc"])
    return (float(np.mean(accs)) if len(accs) > 0 else 0.0), accs


@torch.no_grad()
def batched_logits_g(model: nn.Module, ds: List[Data], device, bs: int = 256):
    if len(ds) == 0:
        return (
            torch.zeros(0, 2),
            torch.zeros(0, getattr(model, "pool_dim", 2)),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    L, G, Y, U = [], [], [], []
    model.eval()
    for i in range(0, len(ds), bs):
        batch = Batch.from_data_list(ds[i : i + bs]).to(device)
        logits, g = model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        L.append(logits.detach().cpu())
        G.append(g.detach().cpu())
        Y.append(batch.y.view(-1).detach().cpu())
        U.append(batch.uid.view(-1).detach().cpu())
    return torch.cat(L, 0), torch.cat(G, 0), torch.cat(Y, 0), torch.cat(U, 0)


def softmax_margin(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    top2 = torch.topk(p, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).clamp(min=0.0, max=1.0)


def farthest_first(uids, g_list, k: int):
    if k <= 0 or len(uids) == 0:
        return []
    k = min(k, len(uids))
    G = torch.stack(g_list, 0)
    sel = [0]
    dmin = torch.cdist(G, G[0:1], p=2.0).squeeze(1)
    while len(sel) < k:
        _, arg = torch.max(dmin, dim=0)
        sel.append(int(arg.item()))
        dnew = torch.cdist(G, G[arg : arg + 1], p=2.0).squeeze(1)
        dmin = torch.minimum(dmin, dnew)
    return [uids[i] for i in sel]


@torch.no_grad()
def sample_mvn_robust(mu: torch.Tensor, cov: torch.Tensor, base_jitter: float = 1e-8) -> torch.Tensor:
    d = mu.numel()
    I = torch.eye(d, dtype=cov.dtype, device=cov.device)
    cov = 0.5 * (cov + cov.T)
    jitter = base_jitter
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(cov + jitter * I)
            eps = torch.randn_like(mu)
            return mu + L @ eps
        except Exception:
            jitter *= 10.0
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=max(base_jitter, 1e-10))
    L = evecs @ torch.diag(torch.sqrt(evals))
    eps = torch.randn_like(mu)
    return mu + L @ eps


class ContextualTSArms:
    def __init__(self, d: int, lam: float = 10.0, sigma: float = 0.1, gamma: float = 0.98, base_jitter: float = 1e-8):
        self.d = d
        self.lam = lam
        self.sigma = sigma
        self.gamma = gamma
        self.base_jitter = base_jitter
        self.A: Dict[int, torch.Tensor] = {}
        self.b: Dict[int, torch.Tensor] = {}

    def ensure(self, arm: int):
        if arm not in self.A:
            self.A[arm] = torch.eye(self.d, dtype=torch.double) * self.lam
            self.b[arm] = torch.zeros(self.d, dtype=torch.double)

    @torch.no_grad()
    def _posterior(self, arm: int):
        self.ensure(arm)
        A = 0.5 * (self.A[arm] + self.A[arm].T)
        b = self.b[arm]
        I = torch.eye(self.d, dtype=torch.double)
        jitter = 1e-8
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(A + jitter * I)
                mu = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
                Ainv = torch.cholesky_inverse(L)
                cov = (self.sigma**2) * Ainv
                return mu, cov
            except Exception:
                jitter *= 10.0
        evals, evecs = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=1e-10)
        Ainv = evecs @ torch.diag(1.0 / evals) @ evecs.T
        mu = Ainv @ b
        cov = (self.sigma**2) * Ainv
        return mu, cov

    @torch.no_grad()
    def sample_scores(self, ctx: Dict[int, torch.Tensor]) -> Dict[int, float]:
        scores = {}
        for h, x in ctx.items():
            x = x.to(torch.double).view(-1)
            mu, cov = self._posterior(h)
            theta = sample_mvn_robust(mu, cov, base_jitter=self.base_jitter)
            scores[h] = float((theta @ x).item())
        return scores

    @staticmethod
    def softmax_weights(scores: Dict[int, float], temperature: float, floor: float) -> Dict[int, float]:
        hs = list(scores.keys())
        s = torch.tensor([scores[h] for h in hs], dtype=torch.double)
        if s.numel() <= 1 or float(s.std(unbiased=False)) < 1e-12:
            w = torch.ones_like(s) / len(hs)
        else:
            z = (s - s.mean()) / (s.std(unbiased=False) + 1e-8)
            w = torch.softmax(z / max(1e-6, temperature), dim=0)
        w = w * (1.0 - floor * len(hs)) + floor
        w = w / w.sum()
        return {hs[i]: float(w[i].item()) for i in range(len(hs))}

    def allocate(
        self,
        weights: Dict[int, float],
        tot: int,
        min_per: int,
        max_per: int,
        candidate_sizes: Dict[int, int],
    ) -> Dict[int, int]:
        hs = list(weights.keys())
        w = np.array([weights[h] for h in hs], dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        caps = np.floor(w * tot).astype(int)
        max_allow = []
        min_allow = []
        for i, h in enumerate(hs):
            n_cand = int(candidate_sizes.get(h, 0))
            hi = min(max_per, n_cand)
            lo = min(min_per, n_cand)
            max_allow.append(hi)
            min_allow.append(lo)
            caps[i] = max(lo, min(int(caps[i]), hi))
        diff = int(tot - int(caps.sum()))
        if diff > 0:
            order = np.argsort(-w)
            for i in order:
                if diff <= 0:
                    break
                can_add = max_allow[i] - caps[i]
                add = min(max(0, can_add), diff)
                caps[i] += int(add)
                diff -= int(add)
        elif diff < 0:
            order = np.argsort(w)
            for i in order:
                if diff >= 0:
                    break
                can_sub = caps[i] - min_allow[i]
                sub = min(max(0, can_sub), -diff)
                caps[i] -= int(sub)
                diff += int(sub)
        return {h: int(caps[i]) for i, h in enumerate(hs)}

    def update_one(self, arm: int, x: torch.Tensor, reward: float):
        self.ensure(arm)
        x = x.to(torch.double).view(-1)
        r = float(np.clip(reward, 0.0, 1.0))
        self.A[arm] = self.gamma * self.A[arm] + (1.0 - self.gamma) * torch.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + (1.0 - self.gamma) * (x * r)


class SampleCTXPerHospital:
    def __init__(
        self,
        d: int = 2,
        lam: float = 5.0,
        sigma: float = 0.15,
        gamma: float = 0.98,
        base_jitter: float = 1e-8,
        alpha: float = 0.20,
        jitter_Sigma: float = 1e-4,
    ):
        assert d == 2
        self.d = d
        self.lam = lam
        self.sigma = sigma
        self.gamma = gamma
        self.base_jitter = base_jitter
        self.alpha = alpha
        self.jitter_Sigma = jitter_Sigma
        self.A: Dict[int, torch.Tensor] = {}
        self.b: Dict[int, torch.Tensor] = {}
        self.mu: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        self.Sigma: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        self.Sigma_inv: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)

    def ensure_h(self, h: int):
        if h not in self.A:
            self.A[h] = torch.eye(self.d, dtype=torch.double) * self.lam
            self.b[h] = torch.zeros(self.d, dtype=torch.double)

    @torch.no_grad()
    def _posterior(self, h: int):
        self.ensure_h(h)
        A = 0.5 * (self.A[h] + self.A[h].T)
        b = self.b[h]
        I = torch.eye(self.d, dtype=torch.double)
        jitter = 1e-8
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(A + jitter * I)
                mu = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
                Ainv = torch.cholesky_inverse(L)
                cov = (self.sigma**2) * Ainv
                return mu, cov
            except Exception:
                jitter *= 10.0
        evals, evecs = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=1e-10)
        Ainv = evecs @ torch.diag(1.0 / evals) @ evecs.T
        mu = Ainv @ b
        cov = (self.sigma**2) * Ainv
        return mu, cov

    @staticmethod
    @torch.no_grad()
    def _emp_mean_cov(G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, D = G.shape
        mu = G.mean(0)
        if n >= 2:
            Xc = G - mu
            cov = (Xc.T @ Xc) / float(n - 1)
        else:
            cov = torch.eye(D, dtype=G.dtype) * 1e-4
        cov = 0.5 * (cov + cov.T)
        return mu, cov

    @staticmethod
    @torch.no_grad()
    def _robust_inv_spd(M: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
        M = 0.5 * (M + M.T)
        D = M.shape[0]
        I = torch.eye(D, dtype=M.dtype, device=M.device)
        eps = jitter
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(M + eps * I)
                return torch.cholesky_inverse(L)
            except Exception:
                eps *= 10.0
        evals, evecs = torch.linalg.eigh(M)
        evals = torch.clamp(evals, min=jitter)
        return evecs @ torch.diag(1.0 / evals) @ evecs.T

    @torch.no_grad()
    def update_proto_cov(self, h: int, cache_h: Dict[int, Dict[str, torch.Tensor]], alpha: Optional[float] = None):
        if alpha is None:
            alpha = self.alpha
        buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for info in cache_h.values():
            c = int(info["y"])
            buckets[c].append(info["g"].to(torch.double))
        for c, glist in buckets.items():
            G = torch.stack(glist, 0)
            mu_cur, cov_cur = self._emp_mean_cov(G)
            if c not in self.mu[h]:
                self.mu[h][c] = mu_cur.clone()
                Sig = cov_cur + self.jitter_Sigma * torch.eye(cov_cur.shape[0], dtype=cov_cur.dtype)
                self.Sigma[h][c] = Sig.clone()
                self.Sigma_inv[h][c] = self._robust_inv_spd(Sig, jitter=self.jitter_Sigma)
            else:
                mu_prev = self.mu[h][c]
                Sig_prev = self.Sigma[h][c]
                mu_new = (1.0 - alpha) * mu_prev + alpha * mu_cur
                Sig_new = (1.0 - alpha) * Sig_prev + alpha * cov_cur
                Sig_new = 0.5 * (Sig_new + Sig_new.T) + self.jitter_Sigma * torch.eye(
                    Sig_new.shape[0], dtype=Sig_new.dtype
                )
                self.mu[h][c] = mu_new
                self.Sigma[h][c] = Sig_new
                self.Sigma_inv[h][c] = self._robust_inv_spd(Sig_new, jitter=self.jitter_Sigma)

    @torch.no_grad()
    def mahal_distance(self, h: int, y: int, g: torch.Tensor) -> Optional[float]:
        if (h not in self.mu) or (y not in self.mu[h]) or (y not in self.Sigma_inv[h]):
            return None
        mu = self.mu[h][y]
        Sinv = self.Sigma_inv[h][y]
        d = (g.to(torch.double) - mu).view(-1, 1)
        val = (d.T @ (Sinv @ d)).squeeze()
        val = float(torch.clamp(val, min=0.0).item())
        return float(torch.sqrt(torch.tensor(val + 1e-12)).item())

    @torch.no_grad()
    def sample_scores(self, h: int, phi_map: Dict[int, torch.Tensor]) -> Dict[int, float]:
        mu, cov = self._posterior(h)
        theta = sample_mvn_robust(mu, cov, base_jitter=1e-8)
        return {uid: float((phi.to(torch.double).view(-1) @ theta).item()) for uid, phi in phi_map.items()}

    def update(self, h: int, used_uid_list: List[int], phi_map: Dict[int, torch.Tensor], reward: float):
        self.ensure_h(h)
        r = float(np.clip(reward, 0.0, 1.0))
        if len(used_uid_list) == 0:
            return
        S = torch.zeros((self.d, self.d), dtype=torch.double)
        t = torch.zeros(self.d, dtype=torch.double)
        share = r / len(used_uid_list)
        for u in used_uid_list:
            phi = phi_map.get(u, None)
            if phi is None:
                continue
            x = phi.to(torch.double).view(-1)
            S += torch.outer(x, x)
            t += x * share
        self.A[h] = self.gamma * self.A[h] + (1.0 - self.gamma) * S
        self.b[h] = self.gamma * self.b[h] + (1.0 - self.gamma) * t


@torch.no_grad()
def cache_logits_g_map(model, ds: List[Data], device) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    if len(ds) == 0:
        return out
    L, G, Y, U = batched_logits_g(model, ds, device, bs=256)
    for i in range(len(U)):
        out[int(U[i].item())] = {
            "logits": L[i].cpu(),
            "g": G[i].cpu(),
            "y": int(Y[i].item()),
        }
    return out


@torch.no_grad()
def build_hospital_contexts_2feat(active_hs, acc_map: Dict[int, float], best_so_far: List[float]) -> Dict[int, torch.Tensor]:
    ctx: Dict[int, torch.Tensor] = {}
    for h in active_hs:
        cur = float(acc_map.get(h, 0.0))
        bs = float(best_so_far[h - 1]) if (h - 1) < len(best_so_far) and best_so_far[h - 1] > 0 else cur
        forget = max(0.0, bs - cur)
        ctx[h] = torch.tensor([cur, forget], dtype=torch.double)
    return ctx


@torch.no_grad()
def build_sample_contexts_for_h_2feat(
    h: int, cache_h: Dict[int, Dict[str, torch.Tensor]], samp_state: SampleCTXPerHospital
) -> Dict[int, torch.Tensor]:
    phi_map: Dict[int, torch.Tensor] = {}
    for uid, info in cache_h.items():
        logits = info["logits"]
        g = info["g"]
        y = int(info["y"])
        margin = float(softmax_margin(logits.unsqueeze(0)).item())
        d = samp_state.mahal_distance(h, y, g)
        closeness = 1.0 - math.tanh(d) if d is not None else 0.5
        phi_map[uid] = torch.tensor([margin, max(0.0, min(1.0, closeness))], dtype=torch.double)
    return phi_map


@torch.no_grad()
def freeze_teacher_model(model):
    teacher = copy.deepcopy(model).cpu().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def cache_teacher_predictions_for_hospital(
    teacher_model,
    ds: List[Data],
    teacher_store_global: Dict[int, Dict[int, Dict[str, Any]]],
    hospital_id: int,
    device,
    bs: int = 256,
):
    if len(ds) == 0:
        teacher_store_global.setdefault(int(hospital_id), {})
        return
    hid = int(hospital_id)
    store = teacher_store_global.setdefault(hid, {})
    missing = [d for d in ds if int(d.uid.item()) not in store]
    if len(missing) == 0:
        return
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for i in range(0, len(missing), bs):
        batch_list = missing[i : i + bs]
        batch = Batch.from_data_list(batch_list).to(device)
        logits_T, g_T = teacher_model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        for j, d in enumerate(batch_list):
            uid = int(d.uid.item())
            store[uid] = {
                "logits_T": logits_T[j].detach().cpu().clone(),
                "g_T": g_T[j].detach().cpu().clone(),
            }
    teacher_model.cpu()


@torch.no_grad()
def make_entries_with_teacher(
    teacher_models: Dict[int, Any],
    ds: List[Data],
    idx_sel: List[int],
    teacher_store_global: Dict[int, Dict[int, Dict[str, Any]]],
    hospital_id: int,
    device,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if len(idx_sel) == 0:
        return entries
    hid = int(hospital_id)
    subset = [ds[i] for i in idx_sel]
    store = teacher_store_global.setdefault(hid, {})
    missing = [d for d in subset if int(d.uid.item()) not in store]
    if len(missing) > 0:
        if hid not in teacher_models:
            raise KeyError(f"Missing frozen teacher for hospital {hid}")
        cache_teacher_predictions_for_hospital(teacher_models[hid], missing, teacher_store_global, hid, device)
        store = teacher_store_global.setdefault(hid, {})
    for d in subset:
        uid = int(d.uid.item())
        t = store[uid]
        entries.append(
            {
                "data": Data(
                    x=d.x.cpu(),
                    edge_index=d.edge_index.cpu(),
                    edge_weight=d.edge_weight.cpu() if getattr(d, "edge_weight", None) is not None else None,
                    y=torch.tensor([int(d.y.item())], dtype=torch.long),
                ),
                "y": int(d.y.item()),
                "uid": uid,
                "logits_T": t["logits_T"].clone(),
                "g_T": t["g_T"].clone(),
            }
        )
    return entries


class ProxyDevBank:
    def __init__(self, per_h_ds: Dict[int, List[Data]], per_h_idx: Dict[int, List[int]], device):
        self.dev_ds = {h: [per_h_ds[h][i] for i in idxs] for h, idxs in per_h_idx.items()}
        self.loaders = {
            h: DataLoader(self.dev_ds[h], batch_size=min(32, len(self.dev_ds[h])), shuffle=False)
            for h in self.dev_ds
        }
        self.last_ce = {h: None for h in self.dev_ds}
        self.device = device

    @torch.no_grad()
    def ce_loss(self, model: nn.Module, h: int) -> float:
        if h not in self.loaders or len(self.dev_ds[h]) == 0:
            return 0.0
        ce = nn.CrossEntropyLoss(reduction="sum")
        total = 0.0
        n = 0
        model.eval()
        for batch in self.loaders[h]:
            batch = batch.to(self.device)
            logits, g = model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
            loss = ce(logits, batch.y.view(-1))
            total += float(loss.item())
            n += int(batch.y.numel())
        return total / max(1, n)

    def reward(self, h: int, new_ce: float, tau: float) -> float:
        prev = self.last_ce[h]
        self.last_ce[h] = new_ce
        if prev is None:
            return 0.5
        drop = prev - new_ce
        return float(torch.sigmoid(torch.tensor(drop / tau)).item())


class GenReplayMemory:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.items_per_h: Dict[int, List[Dict[str, Any]]] = {}
        self.base_w_per_h: Dict[int, float] = {}
        self.cap_per_h: Dict[int, int] = {}

    def set_caps_and_weights(self, cap_per_h: Dict[int, int], base_w_per_h: Dict[int, float]):
        self.cap_per_h = {int(k): int(v) for k, v in cap_per_h.items()}
        self.base_w_per_h = {int(k): float(v) for k, v in base_w_per_h.items()}
        for h in cap_per_h:
            if h not in self.items_per_h:
                self.items_per_h[h] = []

    def clear(self):
        self.items_per_h.clear()

    def __len__(self):
        return sum(len(v) for v in self.items_per_h.values())

    def insert_entries(self, hospital_id: int, entries: List[Dict[str, Any]]):
        pool = self.items_per_h.setdefault(int(hospital_id), [])
        pool.extend(entries)

    def truncate_to_caps(self):
        for h, pool in self.items_per_h.items():
            cap = self.cap_per_h.get(h, len(pool))
            if len(pool) > cap:
                idxs = self.rng.choice(len(pool), size=cap, replace=False)
                self.items_per_h[h] = [pool[int(i)] for i in idxs]

    def truncate_global_to_total(self, total_cap: int):
        cur = len(self)
        if cur <= total_cap:
            return
        hs = list(self.items_per_h.keys())
        sizes = np.array([len(self.items_per_h[h]) for h in hs], dtype=np.int64)
        if sizes.sum() == 0:
            return
        target = np.floor(sizes / sizes.sum() * total_cap).astype(int)
        diff = int(total_cap - int(target.sum()))
        order = np.argsort(-(sizes - target))
        for i in order:
            if diff <= 0:
                break
            target[i] += 1
            diff -= 1
        for h, t in zip(hs, target):
            pool = self.items_per_h[h]
            if len(pool) > t:
                keep_idx = np.random.choice(len(pool), size=t, replace=False)
                self.items_per_h[h] = [pool[int(i)] for i in keep_idx]

    def sample_weighted(self, total_k: int, device):
        hs = sorted([h for h, p in self.items_per_h.items() if len(p) > 0])
        if total_k <= 0 or len(hs) == 0:
            return (None,) * 6
        w = np.array([max(1e-6, self.base_w_per_h.get(h, 1.0)) for h in hs], dtype=np.float64)
        w = w / w.sum()
        alloc = np.random.multinomial(total_k, w)
        datas: List[Data] = []
        logitsT = []
        labels = []
        gT = []
        uids = []
        hosp_ids = []
        for h, nk in zip(hs, alloc):
            pool = self.items_per_h[h]
            if nk <= 0 or len(pool) == 0:
                continue
            idxs = np.random.choice(len(pool), size=min(nk, len(pool)), replace=(len(pool) < nk))
            for i in idxs:
                it = pool[int(i)]
                datas.append(it["data"])
                logitsT.append(torch.as_tensor(it["logits_T"], dtype=torch.float32).detach().clone())
                labels.append(it["y"])
                gT.append(torch.as_tensor(it["g_T"], dtype=torch.float32).detach().clone())
                uids.append(int(it["uid"]))
                hosp_ids.append(h)
        if len(datas) == 0:
            return (None,) * 6
        batch = Batch.from_data_list(datas).to(device)
        logitsT = torch.stack(logitsT, 0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        gT = torch.stack(gT, 0).to(device)
        uids = torch.tensor(uids, dtype=torch.long, device=device)
        hosp = torch.tensor(hosp_ids, dtype=torch.long, device=device)
        return batch, logitsT, labels, gT, (uids, hosp), None
