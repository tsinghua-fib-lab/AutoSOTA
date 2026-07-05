"""TT-Sparse model: layer, model, train, prune."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.autograd import Function


# =============================================================================
# Differentiable Soft TopK
# =============================================================================


class _SoftTopK(Function):
    """Relaxed top-k: returns soft masks via sigmoid with bisection method."""

    @staticmethod
    def forward(ctx: Any, scores: Tensor, k: int, tau: Tensor) -> Tensor:
        scaled = scores / tau.item()
        lo = -scaled.max(dim=1, keepdim=True).values - 10.0
        hi = -scaled.min(dim=1, keepdim=True).values + 10.0
        for _ in range(32):
            mid = (lo + hi) * 0.5
            s = torch.sigmoid(scaled + mid).sum(dim=1, keepdim=True)
            lo = torch.where(s < k, mid, lo)
            hi = torch.where(s < k, hi, mid)
        ts = (lo + hi) * 0.5
        ps = torch.sigmoid(scaled + ts)
        ctx.save_for_backward(ps)
        ctx.tau = tau.item()
        return ps

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        (ps,) = ctx.saved_tensors
        v = ps * (1.0 - ps)
        s = v.sum(dim=1, keepdim=True).clamp_min(1e-12)
        uv = grad_output * v
        return (uv - uv.sum(dim=1, keepdim=True) * v / s) / ctx.tau, None, None


# =============================================================================
# LTT Node Autograd (hard forward, soft backward)
# =============================================================================


class _LTTNode(Function):
    """Sparse truth-table node: discrete top-k forward, differentiable backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, conn_w: Tensor, logic_w: Tensor, bias: Tensor, n_bits: int, tau: Tensor) -> Tensor:
        M = conn_w.shape[1]
        _, topk_idx = torch.topk(conn_w, n_bits, dim=0)
        sel_x = x[:, topk_idx]
        sel_w = logic_w[topk_idx, torch.arange(M, device=logic_w.device)]
        out = torch.einsum("bnm,nm->bm", sel_x, sel_w) + bias
        ctx.save_for_backward(x, conn_w, logic_w, bias, tau)
        ctx.n_bits = n_bits
        return out

    @staticmethod
    def backward(ctx: Any, g: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None]:
        x, conn_w, logic_w, bias, tau = ctx.saved_tensors
        n_bits = ctx.n_bits
        sw = _SoftTopK.apply(conn_w.T, n_bits, tau).T
        bias_g = g.sum(0)
        logic_g = torch.einsum("bm,bn,nm->nm", g, x, sw)
        input_g = torch.einsum("bm,nm->bn", g, logic_w * sw)
        local = torch.einsum("bm,bn,nm->nm", g, x, logic_w)
        v = sw * (1.0 - sw)
        s = v.sum(0, keepdim=True).clamp_min(1e-12)
        uv = local * v
        conn_g = (uv - uv.sum(0, keepdim=True) * v / s) / tau
        return input_g, conn_g, logic_g, bias_g, None, None


# =============================================================================
# TT-Sparse Model
# =============================================================================


class TTSparseModel(nn.Module):
    """Single-layer TT-Sparse: sparse truth-table nodes + skip + foldable classifier.

    Args:
        input_size: Number of binary LTT input features.
        num_nodes: Number of truth-table nodes.
        num_classes: Number of output classes (1 for binary/regression).
        n_bits: Inputs per node (sparsity level).
        tau: Temperature for soft top-k relaxation.
        task_type: One of 'binary', 'multiclass', 'regression'.
        skip_size: Number of continuous skip features (0 to disable).
        dropout: Dropout rate before classifier.
    """

    def __init__(
        self,
        input_size: int,
        num_nodes: int = 32,
        num_classes: int = 1,
        n_bits: int = 6,
        tau: float = 0.1,
        task_type: str = "binary",
        skip_size: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self._n_bits = n_bits
        self._num_nodes = num_nodes
        self._skip_size = skip_size
        self._frozen = False
        self._classifier_folded = False

        self.register_buffer("_tau", torch.tensor(tau))
        self.conn_weights = nn.Parameter(torch.randn(input_size, num_nodes) * 0.1)
        self.logic_weights = nn.Parameter(torch.randn(input_size, num_nodes) * 0.1)
        self.ltt_bias = nn.Parameter(torch.zeros(num_nodes))

        clf_in = num_nodes + skip_size
        clf_out = num_classes if task_type == "multiclass" else 1
        embed = max(32, clf_in // 4)
        self.clf_drop = nn.Dropout(dropout)
        self.clf_l1 = nn.Linear(clf_in, embed, bias=False)
        self.clf_bn = nn.BatchNorm1d(embed)
        self.clf_l2 = nn.Linear(embed, clf_out, bias=False)
        self.clf_linear: nn.Linear | None = None

    @property
    def n_bits(self) -> int:
        """Inputs per truth-table node."""
        return self._n_bits

    @property
    def num_nodes(self) -> int:
        """Number of truth-table nodes."""
        return self._num_nodes

    @property
    def skip_size(self) -> int:
        """Number of skip (passthrough) features."""
        return self._skip_size

    @property
    def is_frozen(self) -> bool:
        """Whether connection weights are frozen (hard-committed)."""
        return self._frozen

    @property
    def is_classifier_folded(self) -> bool:
        """Whether the classifier has been folded into a single linear layer."""
        return self._classifier_folded

    def fold_classifier(self) -> None:
        """Fold L1 + BatchNorm + L2 into a single nn.Linear. Irreversible."""
        if self._classifier_folded:
            return
        W, b = self.get_folded_classifier()
        clf_out, clf_in = W.shape
        linear = nn.Linear(clf_in, clf_out, bias=True)
        linear.weight.data = torch.from_numpy(W).float().to(self.logic_weights.device)
        linear.bias.data = torch.from_numpy(b).float().to(self.logic_weights.device)
        self.clf_linear = linear
        del self.clf_l1, self.clf_bn, self.clf_l2, self.clf_drop
        self._classifier_folded = True

    def freeze_connections(self) -> None:
        """Hard-commit top-k selection and zero non-selected logic weights."""
        with torch.no_grad():
            idx = torch.topk(self.conn_weights, self._n_bits, dim=0).indices
            mask = torch.zeros_like(self.logic_weights, dtype=torch.bool)
            mask.scatter_(0, idx, True)
            self.logic_weights.mul_(mask.float())
        self._frozen = True
        self.conn_weights.requires_grad_(False)

    def get_folded_classifier(self) -> tuple[np.ndarray, np.ndarray]:
        """Fold Linear + BatchNorm + Linear into a single affine transform (W, b)."""
        if self._classifier_folded:
            return (
                self.clf_linear.weight.detach().cpu().numpy(),
                self.clf_linear.bias.detach().cpu().numpy(),
            )
        W1 = self.clf_l1.weight.detach().cpu().numpy()
        W2 = self.clf_l2.weight.detach().cpu().numpy()
        bn = self.clf_bn
        mean = bn.running_mean.cpu().numpy()
        var = bn.running_var.cpu().numpy()
        gamma = bn.weight.detach().cpu().numpy()
        beta = bn.bias.detach().cpu().numpy()
        scale = gamma / np.sqrt(var + bn.eps)
        W1_s = W1 * scale[:, None]
        bn_b = beta - mean * scale
        return W2 @ W1_s, W2 @ bn_b

    def get_active_connections(self) -> tuple[list[list[int]], list[np.ndarray]]:
        """Per-node active input indices and their logic weights."""
        w = self.logic_weights.detach().cpu().numpy()
        indices: list[list[int]] = []
        weights: list[np.ndarray] = []
        for j in range(self._num_nodes):
            col = w[:, j]
            active = np.nonzero(col)[0].tolist()
            indices.append(active)
            weights.append(col[active])
        return indices, weights

    def forward(self, x_ltt: Tensor, x_skip: Tensor | None = None) -> Tensor:
        """Forward pass: LTT nodes -> binarize -> concat skip -> classifier."""
        if self._frozen:
            h = torch.einsum("bn,nm->bm", x_ltt, self.logic_weights) + self.ltt_bias
        else:
            h = _LTTNode.apply(x_ltt, self.conn_weights, self.logic_weights, self.ltt_bias, self._n_bits, self._tau)

        hard = (h > 0.0).float()
        h = h + (hard - h).detach()

        if x_skip is not None:
            h = torch.cat([h, x_skip], dim=1)

        if self._classifier_folded:
            return self.clf_linear(h)

        h = self.clf_drop(h)
        h = self.clf_l1(h)
        h = self.clf_bn(h)
        return self.clf_l2(h)


# =============================================================================
# Training
# =============================================================================


def train(
    model: TTSparseModel,
    data: dict[str, np.ndarray | None],
    *,
    epochs: int = 70,
    batch_size: int = 2048,
    lr: float = 0.005,
    val_split: float = 0.2,
    patience: int = 25,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, float | int]:
    """Train a TTSparseModel with early stopping.

    Args:
        model: The model to train (modified in-place).
        data: Output of TabularEncoder.transform() with keys X_ltt, X_skip, y.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for AdamW.
        val_split: Fraction held out for validation (0 to disable).
        patience: Early stopping patience (epochs without improvement).
        device: Device string ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
        verbose: Print training progress every 10 epochs.

    Returns:
        Dict with 'best_val_loss' and 'epochs' trained.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    task = model.task_type

    X, Xs, y = data["X_ltt"], data["X_skip"], data["y"]
    if model.skip_size == 0:
        Xs = None
    y_dtype = torch.long if task == "multiclass" else torch.float32

    if val_split > 0:
        strat = y if task != "regression" else None
        idx = np.arange(len(y))
        ti, vi = train_test_split(idx, test_size=val_split, random_state=seed, stratify=strat)
        Xt = torch.tensor(X[ti], dtype=torch.float32, device=device)
        yt = torch.tensor(y[ti], dtype=y_dtype, device=device)
        Xv = torch.tensor(X[vi], dtype=torch.float32, device=device)
        yv = torch.tensor(y[vi], dtype=y_dtype, device=device)
        Xst = torch.tensor(Xs[ti], dtype=torch.float32, device=device) if Xs is not None else None
        Xsv = torch.tensor(Xs[vi], dtype=torch.float32, device=device) if Xs is not None else None
    else:
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=y_dtype, device=device)
        Xv = Xst = Xsv = yv = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-7)
    crit = nn.BCEWithLogitsLoss() if task == "binary" else nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xt.size(0), device=device)
        ep_loss = 0.0
        nb = 0
        for i in range(0, Xt.size(0), batch_size):
            batch_idx = perm[i:i + batch_size]
            out = model(Xt[batch_idx], Xst[batch_idx] if Xst is not None else None)
            if task == "multiclass":
                loss = crit(out, yt[batch_idx])
            else:
                loss = crit(out.view(-1), yt[batch_idx].view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        if Xv is not None:
            model.eval()
            with torch.no_grad():
                vo = model(Xv, Xsv)
                vl = (crit(vo, yv) if task == "multiclass" else crit(vo.view(-1), yv.view(-1))).item()
            sched.step(vl)
            if vl < best_loss:
                best_loss, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
            if verbose and ep % 10 == 0:
                print(f"Epoch {ep + 1}/{epochs}: train={ep_loss / nb:.4f} val={vl:.4f}")
            if wait >= patience:
                if verbose:
                    print(f"Early stop at epoch {ep + 1}")
                break
        elif verbose and ep % 10 == 0:
            print(f"Epoch {ep + 1}/{epochs}: train={ep_loss / nb:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_val_loss": best_loss, "epochs": ep + 1}


# =============================================================================
# Pruning: Complexity-aware iterative pruning
# =============================================================================


@dataclass(slots=True)
class _ComplexityStats:
    lut_cost: float
    boolean_cost: float
    classifier_cost: float
    total_cost: float
    ltt_nonzero: int
    ltt_total: int
    classifier_nonzero: int
    classifier_total: int


@dataclass(slots=True)
class _FrontierPoint:
    iteration: int
    metric: float
    complexity: _ComplexityStats
    state_dict: dict[str, Tensor]


@dataclass(slots=True)
class _EdgeCandidate:
    score: float
    gain: float
    space: str
    row: int
    col: int


_CW_LUT = 1.0
_CW_BOOLEAN = 0.2
_CW_CLASSIFIER = 0.05


def _compute_complexity(model: TTSparseModel) -> _ComplexityStats:
    lw = model.logic_weights.detach()
    nonzero = lw != 0
    fanins = nonzero.sum(dim=0)

    ltt_nz = int(nonzero.sum().item())
    ltt_total = int(lw.numel())

    bias_nz = model.ltt_bias.detach() != 0
    has_inputs = fanins > 0
    active_constant = ~has_inputs & bias_nz
    pos_fanins = fanins[has_inputs].float()

    lut_cost = float(active_constant.sum().item()) + float((2.0**pos_fanins).sum().item())
    boolean_cost = float(active_constant.sum().item()) + float(pos_fanins.sum().item())

    cls_w = model.clf_linear.weight.detach()
    cls_nz = int((cls_w != 0).sum().item())
    cls_total = int(cls_w.numel())
    cls_b = model.clf_linear.bias.detach()
    cls_bias_nz = int((cls_b != 0).sum().item())
    classifier_cost = float(cls_nz + cls_bias_nz)

    total = _CW_LUT * lut_cost + _CW_BOOLEAN * boolean_cost + _CW_CLASSIFIER * classifier_cost
    return _ComplexityStats(
        lut_cost=lut_cost, boolean_cost=boolean_cost,
        classifier_cost=classifier_cost, total_cost=total,
        ltt_nonzero=ltt_nz, ltt_total=ltt_total,
        classifier_nonzero=cls_nz, classifier_total=cls_total,
    )


def _compute_node_costs(model: TTSparseModel) -> np.ndarray:
    """Per-node DNF complexity via QMC minimization."""
    from tt_sparse.qmc import QuineMcCluskey, dnf_cost, enumerate_truth_table

    lw = model.logic_weights.detach().cpu().numpy()
    bias = model.ltt_bias.detach().cpu().numpy()
    qm = QuineMcCluskey(use_xor=False)
    n_nodes = model.num_nodes
    costs = np.zeros(n_nodes)

    for j in range(n_nodes):
        col = lw[:, j]
        nz_mask = col != 0
        k = int(nz_mask.sum())
        if k == 0:
            continue
        if k > 16:
            costs[j] = float(2**k)
            continue
        weights = col[nz_mask].tolist()
        b = float(bias[j])
        minterms = enumerate_truth_table(k, weights, b)
        if not minterms or len(minterms) == 2**k:
            costs[j] = max(1.0, float(k))
            continue
        imps = qm.simplify(minterms, dc=[], num_bits=k)
        if imps is None:
            costs[j] = float(k)
            continue
        costs[j] = max(1.0, dnf_cost(imps))

    return costs


def _compute_saliency(
    model: TTSparseModel,
    X_ltt: Tensor,
    y: Tensor,
    X_skip: Tensor | None,
    task_type: str,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Compute first-order saliency |w * grad| on one mini-batch."""
    crit = (
        nn.BCEWithLogitsLoss() if task_type == "binary"
        else nn.MSELoss() if task_type == "regression"
        else nn.CrossEntropyLoss()
    )
    n = X_ltt.size(0)
    idx = torch.randperm(n)[:batch_size]
    bx = X_ltt[idx]
    by = y[idx]
    bx_skip = X_skip[idx] if X_skip is not None else None

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(bx, bx_skip)
    if task_type in ("binary", "regression"):
        loss = crit(out.view(-1), by.view(-1))
    else:
        loss = crit(out, by)
    loss.backward()

    lw = model.logic_weights
    lw_grad = lw.grad
    lw_sal = (lw.detach() * lw_grad.detach()).abs() if lw_grad is not None else lw.detach().abs()

    cls_w = model.clf_linear.weight
    cls_grad = cls_w.grad
    cls_sal = (cls_w.detach() * cls_grad.detach()).abs() if cls_grad is not None else cls_w.detach().abs()

    model.zero_grad(set_to_none=True)
    return lw_sal, cls_sal


def _select_edges(
    model: TTSparseModel,
    lw_saliency: Tensor,
    cls_saliency: Tensor,
    node_costs: np.ndarray,
    prune_fraction: float,
    eps: float = 1e-12,
) -> list[_EdgeCandidate]:
    """Score all non-zero edges, return lowest-score fraction."""
    lw = model.logic_weights.detach()
    cls_w = model.clf_linear.weight.detach()
    n_nodes = model.num_nodes
    dev = lw.device

    all_scores: list[Tensor] = []
    all_gains: list[Tensor] = []
    all_rows: list[Tensor] = []
    all_cols: list[Tensor] = []
    all_is_cls: list[Tensor] = []

    # LTT logic edges
    nonzero = lw != 0
    if bool(nonzero.any()):
        rows, cols = torch.where(nonzero)
        fanins = nonzero.sum(dim=0).float()
        edge_fanins = fanins[cols].clamp(min=1.0)
        costs_t = torch.from_numpy(node_costs).to(device=dev, dtype=torch.float32)
        edge_costs = costs_t[cols].clamp(min=1.0)
        gains = _CW_LUT * (edge_costs / edge_fanins) + _CW_BOOLEAN
        scores = lw_saliency[rows, cols] / (gains + eps)

        all_scores.append(scores)
        all_gains.append(gains)
        all_rows.append(rows)
        all_cols.append(cols)
        all_is_cls.append(torch.zeros(rows.shape[0], device=dev, dtype=torch.bool))

    # Classifier edges
    cls_nz = cls_w != 0
    if bool(cls_nz.any()):
        rows, cols = torch.where(cls_nz)
        gains = torch.full((rows.shape[0],), _CW_CLASSIFIER, device=dev)

        is_ltt_col = cols < n_nodes
        if bool(is_ltt_col.any()):
            sup_cols = cols[is_ltt_col]
            sup_fanins = nonzero.sum(dim=0)[sup_cols].float()
            sup_costs = torch.from_numpy(node_costs).to(device=dev, dtype=torch.float32)[sup_cols].clamp(min=1.0)
            cls_col_support = cls_nz.sum(dim=0)[sup_cols].float().clamp(min=1.0)
            full_bonus = torch.where(
                sup_fanins > 0,
                _CW_LUT * sup_costs + _CW_BOOLEAN * sup_fanins,
                torch.tensor(_CW_LUT + _CW_BOOLEAN, device=dev),
            )
            gains[is_ltt_col] += full_bonus / cls_col_support

        scores = cls_saliency[rows, cols] / (gains + eps)

        all_scores.append(scores)
        all_gains.append(gains)
        all_rows.append(rows)
        all_cols.append(cols)
        all_is_cls.append(torch.ones(rows.shape[0], device=dev, dtype=torch.bool))

    if not all_scores:
        return []

    scores_cat = torch.cat(all_scores)
    gains_cat = torch.cat(all_gains)
    rows_cat = torch.cat(all_rows)
    cols_cat = torch.cat(all_cols)
    is_cls_cat = torch.cat(all_is_cls)

    # Normalize scores between LTT and classifier spaces
    ltt_mask = ~is_cls_cat
    cls_mask = is_cls_cat
    if bool(ltt_mask.any()) and bool(cls_mask.any()):
        ltt_median = scores_cat[ltt_mask].median()
        cls_median = scores_cat[cls_mask].median()
        if cls_median > 0 and ltt_median > 0:
            scores_cat[cls_mask] *= ltt_median / cls_median

    total = scores_cat.shape[0]
    n_to_prune = max(1, int(total * prune_fraction))
    n_to_prune = min(n_to_prune, total)

    _, indices = torch.topk(scores_cat, n_to_prune, largest=False, sorted=True)

    return [
        _EdgeCandidate(
            score=float(scores_cat[i]),
            gain=float(gains_cat[i]),
            space="classifier" if bool(is_cls_cat[i]) else "ltt",
            row=int(rows_cat[i]),
            col=int(cols_cat[i]),
        )
        for i in indices.tolist()
    ]


def _apply_pruning(model: TTSparseModel, candidates: list[_EdgeCandidate]) -> int:
    removed = 0
    with torch.no_grad():
        lw = model.logic_weights
        cls_w = model.clf_linear.weight
        for c in candidates:
            if c.space == "ltt":
                if lw[c.row, c.col] != 0:
                    lw[c.row, c.col] = 0
                    removed += 1
            else:
                if cls_w[c.row, c.col] != 0:
                    cls_w[c.row, c.col] = 0
                    removed += 1
    return removed


def _enforce_max_fanin(model: TTSparseModel, max_fanin: int) -> int:
    if max_fanin <= 0:
        return 0
    removed = 0
    with torch.no_grad():
        lw = model.logic_weights
        nonzero = lw != 0
        fanins = nonzero.sum(dim=0)
        for col in range(lw.shape[1]):
            excess = int(fanins[col].item()) - max_fanin
            if excess <= 0:
                continue
            active_rows = torch.where(nonzero[:, col])[0]
            mags = lw[active_rows, col].abs()
            remove_idx = torch.argsort(mags)[:excess]
            lw[active_rows[remove_idx], col] = 0
            removed += excess
    return removed


def _propagate_constants(model: TTSparseModel) -> tuple[bool, int]:
    """Fold constant nodes (zero fan-in, nonzero bias) into classifier bias."""
    changed = False
    removed = 0
    with torch.no_grad():
        lw = model.logic_weights
        bias = model.ltt_bias
        cls_w = model.clf_linear.weight
        cls_b = model.clf_linear.bias

        fanins = (lw != 0).sum(dim=0)
        n_nodes = model.num_nodes
        const_mask = fanins[:n_nodes] == 0
        if not bool(const_mask.any()):
            return False, 0

        const_one_mask = const_mask & (bias > 0)
        const_cols = torch.where(const_mask)[0]
        const_one_cols = torch.where(const_one_mask)[0]

        if const_one_cols.numel() > 0:
            cls_b.add_(cls_w[:, const_one_cols].sum(dim=1))

        to_remove = int((cls_w[:, const_cols] != 0).sum().item())
        if to_remove > 0:
            cls_w[:, const_cols] = 0
            removed += to_remove
            changed = True

        # Zero out the bias of constant nodes folded into classifier
        for col in const_cols.tolist():
            if bias[col] != 0:
                bias[col] = 0

    return changed, removed


def _prune_disconnected(model: TTSparseModel) -> tuple[bool, int]:
    """Remove LTT nodes whose classifier columns are all-zero."""
    changed = False
    removed = 0
    with torch.no_grad():
        lw = model.logic_weights
        bias = model.ltt_bias
        cls_w = model.clf_linear.weight
        n_nodes = model.num_nodes

        connected = (cls_w[:, :n_nodes] != 0).any(dim=0)
        dead = ~connected

        if not bool(dead.any()):
            return False, 0

        dead_cols = torch.where(dead)[0]
        to_remove = int((lw[:, dead_cols] != 0).sum().item())
        to_remove += int((bias[dead_cols] != 0).sum().item())

        if to_remove > 0:
            lw[:, dead_cols] = 0
            bias[dead_cols] = 0
            removed += to_remove
            changed = True

    return changed, removed


def _structural_simplify(model: TTSparseModel) -> int:
    """Run constant propagation + disconnected pruning to fixed point."""
    total = 0
    while True:
        c1, r1 = _propagate_constants(model)
        c2, r2 = _prune_disconnected(model)
        total += r1 + r2
        if not c1 and not c2:
            break
    return total


def _masked_finetune(
    model: TTSparseModel,
    X_ltt: Tensor,
    y: Tensor,
    X_skip: Tensor | None,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Finetune with hard masks: pruned edges stay zero."""
    crit = (
        nn.BCEWithLogitsLoss() if task_type == "binary"
        else nn.MSELoss() if task_type == "regression"
        else nn.CrossEntropyLoss()
    )
    lw = model.logic_weights
    cls_w = model.clf_linear.weight
    cls_b = model.clf_linear.bias
    bias = model.ltt_bias

    lw_mask = (lw.data != 0).float()
    cls_w_mask = (cls_w.data != 0).float()
    bias_mask = ((lw.data != 0).any(dim=0) | (bias.data != 0)).float()

    masked_params: list[tuple[nn.Parameter, Tensor]] = [
        (lw, lw_mask),
        (cls_w, cls_w_mask),
        (bias, bias_mask),
    ]

    param_list = [lw, cls_w, bias, cls_b]
    opt = torch.optim.Adam(param_list, lr=lr)

    model.train()
    n = X_ltt.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=X_ltt.device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = X_ltt[idx]
            by = y[idx]
            bx_skip = X_skip[idx] if X_skip is not None else None

            opt.zero_grad()
            out = model(bx, bx_skip)
            if task_type in ("binary", "regression"):
                loss = crit(out.view(-1), by.view(-1))
            else:
                loss = crit(out, by)
            loss.backward()

            with torch.no_grad():
                for param, mask in masked_params:
                    if param.grad is not None:
                        param.grad.mul_(mask)

            opt.step()

            with torch.no_grad():
                for param, mask in masked_params:
                    param.data.mul_(mask)


def _add_frontier_point(
    frontier: list[_FrontierPoint], candidate: _FrontierPoint
) -> list[_FrontierPoint]:
    """Insert point and remove dominated solutions."""
    frontier.append(candidate)
    pruned: list[_FrontierPoint] = []
    for i, p in enumerate(frontier):
        dominated = False
        for j, q in enumerate(frontier):
            if i == j:
                continue
            if q.metric >= p.metric and q.complexity.total_cost <= p.complexity.total_cost:
                if q.metric > p.metric or q.complexity.total_cost < p.complexity.total_cost:
                    dominated = True
                    break
        if not dominated:
            pruned.append(p)
    pruned.sort(key=lambda p: (p.complexity.total_cost, -p.metric, p.iteration))
    return pruned


def _select_frontier_model(
    frontier: list[_FrontierPoint], max_drop_pct: float
) -> _FrontierPoint:
    best_metric = max(p.metric for p in frontier)
    min_allowed = best_metric * (1.0 - max_drop_pct / 100.0)
    feasible = [p for p in frontier if p.metric >= min_allowed]
    if not feasible:
        feasible = list(frontier)
    return min(feasible, key=lambda p: (p.complexity.total_cost, -p.metric, p.iteration))


@torch.no_grad()
def _eval_metric(model: TTSparseModel, X: Tensor, y: Tensor, Xs: Tensor | None, task: str) -> float:
    model.eval()
    out = model(X, Xs)
    if task == "binary":
        p = torch.sigmoid(out.view(-1)).cpu().numpy()
        try:
            return float(roc_auc_score(y.cpu().numpy(), p))
        except ValueError:
            return 0.5
    elif task == "regression":
        try:
            return float(r2_score(y.cpu().numpy(), out.view(-1).cpu().numpy()))
        except ValueError:
            return 0.0
    else:
        p = torch.softmax(out, 1).cpu().numpy()
        try:
            return float(roc_auc_score(y.cpu().numpy(), p, multi_class="ovr", average="macro"))
        except ValueError:
            return 0.5


def _snapshot(model: TTSparseModel) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _restore(model: TTSparseModel, state: dict[str, Tensor]) -> None:
    model.load_state_dict(state)


def prune(
    model: TTSparseModel,
    data: dict[str, np.ndarray | None],
    *,
    max_drop_pct: float = 0.0,
    collapse_drop_pct: float = 15.0,
    finetune_epochs: int = 30,
    finetune_batch_size: int = 2048,
    finetune_lr: float = 0.02,
    initial_prune_fraction: float = 0.10,
    min_prune_fraction: float = 0.005,
    max_iterations: int = 80,
    max_fanin: int = 16,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Complexity-aware iterative pruning with saliency scoring and Pareto frontier.

    Uses a two-zone search strategy:
    - Selection floor: metric must stay within max_drop_pct of best seen
    - Collapse floor: if metric drops below collapse_drop_pct, reject the step

    Edges are scored by saliency (|w * grad|) normalized by complexity gain,
    enabling the pruner to preferentially remove low-impact edges from
    high-complexity nodes.

    Args:
        model: Trained model (modified in-place; classifier is folded).
        data: Output of TabularEncoder.transform().
        max_drop_pct: Maximum allowed metric drop (%) from best observed.
        collapse_drop_pct: Metric drop (%) that triggers immediate rejection.
        finetune_epochs: Epochs of masked finetuning after each prune step.
        finetune_batch_size: Batch size for saliency computation and finetuning.
        finetune_lr: Learning rate for finetuning.
        initial_prune_fraction: Starting fraction of edges to prune per step.
        min_prune_fraction: Stop when prune fraction shrinks below this.
        max_iterations: Maximum pruning iterations.
        max_fanin: Hard cap on inputs per node (for extractable rules).
        device: Device string.
        seed: Random seed.
        verbose: Print pruning progress.

    Returns:
        Dict with pruning statistics including metrics, sparsity, and Pareto frontier.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    task = model.task_type

    # Prepare data
    X, Xs, y = data["X_ltt"], data["X_skip"], data["y"]
    if model.skip_size == 0:
        Xs = None
    y_dtype = torch.long if task == "multiclass" else torch.float32
    strat = y if task != "regression" else None
    idx = np.arange(len(y))
    ti, vi = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)

    Xt = torch.tensor(X[ti], dtype=torch.float32, device=device)
    yt = torch.tensor(y[ti], dtype=y_dtype, device=device)
    Xv = torch.tensor(X[vi], dtype=torch.float32, device=device)
    yv = torch.tensor(y[vi], dtype=y_dtype, device=device)
    Xst = torch.tensor(Xs[ti], dtype=torch.float32, device=device) if Xs is not None else None
    Xsv = torch.tensor(Xs[vi], dtype=torch.float32, device=device) if Xs is not None else None

    # Canonicalize: fold classifier, freeze connections, simplify
    if not model.is_frozen:
        model.freeze_connections()
    if not model.is_classifier_folded:
        model.fold_classifier()

    _enforce_max_fanin(model, max_fanin)
    _structural_simplify(model)

    baseline_metric = _eval_metric(model, Xv, yv, Xsv, task)
    baseline_complexity = _compute_complexity(model)

    if verbose:
        print(
            f"Baseline: metric={baseline_metric:.4f} "
            f"complexity={baseline_complexity.total_cost:.2f} "
            f"(lut={baseline_complexity.lut_cost:.1f} "
            f"bool={baseline_complexity.boolean_cost:.1f} "
            f"cls={baseline_complexity.classifier_cost:.1f})"
        )

    best_metric_seen = baseline_metric
    frontier: list[_FrontierPoint] = [
        _FrontierPoint(
            iteration=0, metric=baseline_metric,
            complexity=baseline_complexity, state_dict=_snapshot(model),
        )
    ]

    prune_fraction = initial_prune_fraction
    rejected_streak = 0
    accepted_steps = 0
    cached_node_costs: np.ndarray | None = None

    for iteration in range(1, max_iterations + 1):
        # Saliency maps
        lw_sal, cls_sal = _compute_saliency(
            model, Xt, yt, Xst, task, finetune_batch_size
        )

        # Node costs (recompute after accepted steps)
        if cached_node_costs is None:
            cached_node_costs = _compute_node_costs(model)

        # Select and apply pruning
        candidates = _select_edges(
            model, lw_sal, cls_sal, cached_node_costs, prune_fraction
        )
        if not candidates:
            if verbose:
                print("No removable edges left; stopping.")
            break

        snapshot = _snapshot(model)

        removed_score = _apply_pruning(model, candidates)
        removed_cap = _enforce_max_fanin(model, max_fanin)
        removed_struct = _structural_simplify(model)
        total_removed = removed_score + removed_cap + removed_struct

        if total_removed == 0:
            _restore(model, snapshot)
            if verbose:
                print("No effective edges removed; stopping.")
            break

        # Two-zone evaluation
        selection_floor = best_metric_seen * (1.0 - max_drop_pct / 100.0)
        collapse_floor = best_metric_seen * (1.0 - collapse_drop_pct / 100.0)

        pre_ft_metric = _eval_metric(model, Xv, yv, Xsv, task)
        collapsed = pre_ft_metric < collapse_floor

        if collapsed:
            if verbose:
                print(
                    f"Iter {iteration:03d} | removed={total_removed} | "
                    f"pre-ft={pre_ft_metric:.4f} floor={selection_floor:.4f}/{collapse_floor:.4f} | "
                    f"COLLAPSE (skip finetune)"
                )
        else:
            ft_epochs = finetune_epochs
            if pre_ft_metric < selection_floor:
                ft_epochs = max(finetune_epochs // 3, 1)

            _masked_finetune(
                model, Xt, yt, Xst, task, ft_epochs, finetune_lr, finetune_batch_size
            )

            trial_metric = _eval_metric(model, Xv, yv, Xsv, task)
            trial_complexity = _compute_complexity(model)
            collapsed = trial_metric < collapse_floor

            if verbose:
                status = "COLLAPSE" if collapsed else ("ACCEPT" if trial_metric >= selection_floor else "EXPLORE")
                print(
                    f"Iter {iteration:03d} | removed={total_removed} | "
                    f"metric={trial_metric:.4f} (pre-ft={pre_ft_metric:.4f}) "
                    f"floor={selection_floor:.4f}/{collapse_floor:.4f} | "
                    f"complexity={trial_complexity.total_cost:.2f} | "
                    f"epochs={ft_epochs} | {status}"
                )

            if not collapsed:
                if trial_metric > best_metric_seen:
                    best_metric_seen = trial_metric

                frontier = _add_frontier_point(
                    frontier,
                    _FrontierPoint(
                        iteration=iteration, metric=trial_metric,
                        complexity=trial_complexity, state_dict=_snapshot(model),
                    ),
                )
                accepted_steps += 1
                rejected_streak = 0
                cached_node_costs = None

                if trial_metric >= selection_floor:
                    prune_fraction = min(prune_fraction * 1.15, 0.25)
                continue

        # Collapse: revert and reduce prune fraction
        _restore(model, snapshot)
        rejected_streak += 1
        prune_fraction *= 0.5
        if prune_fraction < min_prune_fraction:
            if verbose:
                print("Prune fraction below minimum; stopping.")
            break
        if rejected_streak >= 6:
            if verbose:
                print("Too many consecutive collapses; stopping.")
            break

    # Select best point from frontier
    selected = _select_frontier_model(frontier, max_drop_pct)
    _restore(model, selected.state_dict)

    if verbose:
        print(
            f"Selected: iter={selected.iteration} "
            f"metric={selected.metric:.4f} "
            f"complexity={selected.complexity.total_cost:.2f}"
        )

    # Final finetune + cleanup
    _masked_finetune(model, Xt, yt, Xst, task, finetune_epochs, finetune_lr, finetune_batch_size)
    _enforce_max_fanin(model, max_fanin)
    _structural_simplify(model)

    final_metric = _eval_metric(model, Xv, yv, Xsv, task)
    final_complexity = _compute_complexity(model)

    ltt_sparsity = (
        (1.0 - final_complexity.ltt_nonzero / final_complexity.ltt_total) * 100.0
        if final_complexity.ltt_total > 0 else 0.0
    )
    cls_sparsity = (
        (1.0 - final_complexity.classifier_nonzero / final_complexity.classifier_total) * 100.0
        if final_complexity.classifier_total > 0 else 0.0
    )

    if verbose:
        print(
            f"Final: metric={final_metric:.4f} "
            f"complexity={final_complexity.total_cost:.2f} "
            f"ltt_sparsity={ltt_sparsity:.1f}% cls_sparsity={cls_sparsity:.1f}%"
        )

    model.eval()
    return {
        "baseline_metric": baseline_metric,
        "final_metric": final_metric,
        "ltt_sparsity_pct": ltt_sparsity,
        "classifier_sparsity_pct": cls_sparsity,
        "edges_remaining": final_complexity.ltt_nonzero,
        "complexity": final_complexity.total_cost,
        "lut_cost": final_complexity.lut_cost,
        "boolean_cost": final_complexity.boolean_cost,
        "accepted_steps": accepted_steps,
        "iterations": iteration,
        "pareto_frontier": [
            {
                "iteration": p.iteration,
                "metric": p.metric,
                "total_complexity": p.complexity.total_cost,
                "lut_cost": p.complexity.lut_cost,
                "boolean_cost": p.complexity.boolean_cost,
                "ltt_nonzero": p.complexity.ltt_nonzero,
            }
            for p in frontier
        ],
    }
