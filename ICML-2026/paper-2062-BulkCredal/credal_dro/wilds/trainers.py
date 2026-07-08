from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoothmax(losses: torch.Tensor, T: float) -> torch.Tensor:
    m = torch.max(losses)
    return m + T * torch.log(torch.mean(torch.exp((losses - m) / T)) + 1e-12)


def cvar(losses: torch.Tensor, frac: float) -> torch.Tensor:
    """
    Empirical CVaR at level ε=frac under the convention:
        CVaR^ε = inf_η { η + 1/(1-ε) E[(X-η)_+] }.

    For a minibatch of n losses (uniform empirical distribution), this equals
    (up to finite-sample quantile interpolation) the mean of the largest
        k = ceil((1 - ε) * n)
    losses.

      ε = 0   -> mean (ERM)
      ε -> 1  -> max
    """
    eps = float(frac)
    if eps >= 1.0:
        return losses.mean()
    if eps <= 0.0:
        return losses.max()
    n = int(losses.numel())
    k = max(1, int(np.ceil(eps * n)))
    return torch.topk(losses, k=k, largest=True, sorted=False).values.mean()


def trimmed(losses: torch.Tensor, frac: float) -> torch.Tensor:
    if frac <= 0.0:
        return losses.mean()
    n = losses.numel()
    k_keep = max(1, n - int(np.floor(float(frac) * n)))
    return torch.topk(losses, k=k_keep, largest=False, sorted=False).values.mean()


def chi2_proxy(losses: torch.Tensor, rho: float) -> torch.Tensor:
    if rho <= 0.0:
        return losses.mean()
    m = losses.mean()
    v = (losses - m).pow(2).mean()
    robust = m + torch.sqrt(2.0 * float(rho) * v + 1e-12)
    return torch.minimum(robust, losses.max())


@dataclass
class GroupDRO:
    w: torch.Tensor
    step: float

    def update(self, group_losses: torch.Tensor) -> None:
        self.w = self.w * torch.exp(self.step * group_losses.detach())
        self.w = self.w / (self.w.sum() + 1e-12)


def _group_means(losses: torch.Tensor, g: torch.Tensor, G: int) -> torch.Tensor:
    sums = torch.zeros(G, device=losses.device, dtype=losses.dtype)
    cnts = torch.zeros(G, device=losses.device, dtype=losses.dtype)
    sums.scatter_add_(0, g, losses)
    cnts.scatter_add_(0, g, torch.ones_like(losses))
    return sums / (cnts + 1e-12)


def make_head(d_in: int, n_classes: int, head: str, mlp_hidden: int) -> nn.Module:
    head = head.lower()
    if head == "linear":
        return nn.Linear(d_in, n_classes)
    if head == "mlp":
        return nn.Sequential(nn.Linear(d_in, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, n_classes))
    raise ValueError("head must be 'linear' or 'mlp'")

# --- LV-BAS (multiclass) helpers -------------------------------------------------

@torch.no_grad()
def _project_linear_fro_norm_(model: nn.Linear, max_norm: float) -> None:
    """
    Project the weight matrix onto {W : ||W||_F <= max_norm}.
    Bias is left unconstrained.
    """
    if max_norm is None:
        return
    max_norm = float(max_norm)
    if max_norm <= 0:
        return

    w = model.weight
    fro = torch.linalg.vector_norm(w)  # Frobenius norm for a matrix
    if fro > max_norm:
        w.mul_(max_norm / (fro + 1e-12))


def _project_linear_weight_l2_(linear: nn.Linear, W: float) -> None:
    """Project only the weight (not bias) onto the L2 ball of radius W."""
    if W <= 0.0:
        return
    with torch.no_grad():
        w = linear.weight
        n = torch.linalg.vector_norm(w)
        if n > W:
            w.mul_(W / (n + 1e-12))


def _lv_bas_bin_sup_loss_binary_linear(
    linear: nn.Linear,
    *,
    mu: torch.Tensor,      # (2, d)
    sigma2: torch.Tensor,  # (2, d) already includes ridge
    tau: torch.Tensor,     # (2,)
    pi: torch.Tensor,      # (2,)
) -> torch.Tensor:
    """
    Compute: sum_y pi_y sup_{x in Xi_{0,y}} BCEWithLogitsLoss(f(x), y)
    for a linear logit f(x)=w^T x + b and ellipsoid Xi_{0,y} defined by
      (x-mu_y)^T diag(1/sigma2_y) (x-mu_y) <= tau_y^2
    """
    w = linear.weight.view(-1)  # (d,)
    b = linear.bias.view(())    # scalar
    out = w.new_tensor(0.0)
    for y in (0, 1):
        sign = 1.0 - 2.0 * float(y)  # +1 if y=0 else -1
        margin = sign * (torch.dot(w, mu[y]) + b)
        rad = tau[y] * torch.sqrt(torch.sum((w * w) * sigma2[y]) + 1e-12)
        out = out + pi[y] * F.softplus(margin + rad)
    return out


def train_head(
    train_loader: DataLoader,
    d_in: int,
    n_classes: int,
    *,
    algorithm: str,
    epsilon: float,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    head: str = "linear",
    mlp_hidden: int = 256,
    smoothmax_T: float = 0.1,
    groupdro_step: float = 0.01,
    max_grad_norm: Optional[float] = None,
    train_stats: Optional[Dict[str, Any]] = None,
    known_num_groups: Optional[int] = None,      # pass splits.n_groups when using groupdro

    # --- Optional checkpoint selection (epoch-wise) ---
    checkpoint_loader: Optional[DataLoader] = None,
    checkpoint_metric: str = "worst_group_acc",
    checkpoint_mode: Optional[str] = None,  # "max" or "min"; default inferred from metric name
    checkpoint_verbose: bool = False,
    checkpoint_force_civilcomments_16_slices: bool = False,
) -> nn.Module:
    """
    Train an embedding head with multiple objectives.

    train_stats (optional):
      If provided, will be populated with:
        - train_steps, train_examples
        - first-batch objective values (ERM vs algorithm vs epsilon=0)
        - epsilon_used flag
        - objective_name

    known_num_groups:
      If provided, fixes GroupDRO initialisation.
    """
    from .group_dro import GroupDROState, group_dro_loss
    from .chi2_dro import chi2_dro_loss
    alg = algorithm.lower()

    # ---------- basic counters ----------
    steps = 0
    examples = 0
    first_batch_logged = False

    model = make_head(d_in, n_classes, head=head, mlp_hidden=int(mlp_hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    gdro_state: Optional[GroupDROState] = None
    # Prefer an explicitly supplied group count (avoids degenerate first-batch inference)
    num_groups: Optional[int] = int(known_num_groups) if known_num_groups is not None else None

    objective_name = alg
    epsilon_used = alg in (
        "rw_lv_empirical", "lv_empirical",
        "rw_lv_empirical_fair", "lv_empirical_fair",
        "rw_cvar", "cvar", "rw_cvar_b", "cvar_b",
        "rw_trimmed", "trimmed",
        "rw_chi2_dro", "chi2_dro", "chi2", "rw_chi2_dro_b", "chi2_dro_b",
    )


    # ---------- training ----------
    model.train()

    # --- Optional checkpoint selection (epoch-wise) ---
    ckpt_best_state: Optional[Dict[str, torch.Tensor]] = None
    ckpt_best_val: Optional[float] = None
    ckpt_best_epoch: Optional[int] = None
    ckpt_metric = str(checkpoint_metric)

    if checkpoint_mode is None:
        ckpt_mode = "min" if ckpt_metric.startswith("loss") else "max"
    else:
        ckpt_mode = str(checkpoint_mode).lower()
    if ckpt_mode not in ("min", "max"):
        raise ValueError(f"checkpoint_mode must be 'min' or 'max', got {checkpoint_mode!r}")

    for _ep in range(int(epochs)):
        for xb, yb, gb, _idx in train_loader:
            steps += 1
            # xb can be dense float tensor from collate or default collate
            examples += int(xb.shape[0])

            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)

            # ---- common logits/losses for non-binary-LV-BAS ----
            logits = model(xb)
            losses = loss_fn(logits, yb)

            # ---- objectives ----
            m = float(losses.mean().detach().cpu())
            if alg in ("rw_erm", "erm", "rw_erm_b", "erm_b"):
                obj = losses.mean()

            elif alg in ("rw_lv_empirical", "lv_empirical"):
                m = losses.mean()
                obj = m if float(epsilon) <= 0.0 else (1.0 - float(epsilon)) * m + float(epsilon) * smoothmax(losses, smoothmax_T)

            elif alg in ("rw_lv_empirical_fair", "lv_empirical_fair"):
                # GroupDRO inside the bulk + LV-style smoothmax tail
                # (group_dro_loss returns (robust_loss, updated_state))
                if num_groups is None:
                    # fallback if caller didn't provide it (less safe)
                    num_groups = int(gb.max().item()) + 1
                if gdro_state is None:
                    gdro_state = GroupDROState(num_groups=int(num_groups), eta=float(groupdro_step))

                bulk_obj, gdro_state = group_dro_loss(losses, gb, gdro_state)
                tail_obj = smoothmax(losses, smoothmax_T)
                obj = (1.0 - float(epsilon)) * bulk_obj + float(epsilon) * tail_obj

            elif alg in ("rw_cvar", "cvar", "rw_cvar_b", "cvar_b"):
                obj = cvar(losses, frac=float(epsilon))

            elif alg in ("rw_trimmed", "trimmed"):
                obj = trimmed(losses, frac=float(epsilon))

            elif alg in ("rw_groupdro", "groupdro", "rw_group_dro"):
                # Correctness: do NOT infer num_groups from first batch.
                if num_groups is None:
                    # fallback if caller didn't provide it (less safe)
                    num_groups = int(gb.max().item()) + 1
                if gdro_state is None:
                    gdro_state = GroupDROState(num_groups=int(num_groups), eta=float(groupdro_step))
                obj, gdro_state = group_dro_loss(losses, gb, gdro_state)

            elif alg in ("rw_chi2_dro", "chi2_dro", "chi2", "rw_chi2_dro_b", "chi2_dro_b"):
                obj, _w = chi2_dro_loss(losses, epsilon=float(epsilon), normalisation="max")

            else:
                raise ValueError(f"Unknown algorithm '{algorithm}'")

            # ---- first-batch checks (general) ----
            if alg in ("rw_groupdro", "groupdro", "rw_group_dro"):
                obj_eps0 = float(obj.detach().cpu())

            elif alg in ("rw_lv_empirical_fair", "lv_empirical_fair"):
                # For lv_empirical_fair, epsilon=0 => obj == bulk_obj (GroupDRO robust loss).
                # Use the already-computed bulk_obj from the objective branch (do NOT call group_dro_loss again).
                obj_eps0 = float(bulk_obj.detach().cpu())

            else:
                # for cvar/trimmed/lv_empirical/chi2_dro this is “objective with eps=0”
                # using the same branch logic
                eps_saved = float(epsilon)
                epsilon = 0.0
                try:
                    if alg in ("rw_lv_empirical", "lv_empirical"):
                        obj_eps0_t = losses.mean()
                    elif alg in ("rw_cvar", "cvar", "rw_cvar_b", "cvar_b"):
                        obj_eps0_t = cvar(losses, frac=0.0)
                    elif alg in ("rw_trimmed", "trimmed"):
                        obj_eps0_t = trimmed(losses, frac=0.0)
                    elif alg in ("rw_chi2_dro", "chi2_dro", "chi2", "rw_chi2_dro_b", "chi2_dro_b"):
                        obj_eps0_t, _ = chi2_dro_loss(losses, epsilon=0.0, normalisation="max")
                    else:
                        obj_eps0_t = losses.mean()
                    obj_eps0 = float(obj_eps0_t.detach().cpu())
                finally:
                    epsilon = eps_saved  # restore

                train_stats.update(
                    dict(
                        train_objective=str(objective_name),
                        train_epsilon=float(epsilon),
                        train_epsilon_used=bool(epsilon_used),
                        train_firstbatch_erm=float(losses.mean().detach().cpu()),
                        train_firstbatch_obj=float(obj.detach().cpu()),
                        train_firstbatch_obj_eps0=float(obj_eps0),
                        train_firstbatch_obj_minus_eps0=float(float(obj.detach().cpu()) - obj_eps0),
                    )
                )
                first_batch_logged = True

            # ---- optimiser step ----
            opt.zero_grad(set_to_none=True)
            obj.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            opt.step()


        # --- end of epoch: optional checkpoint selection ---
        if checkpoint_loader is not None:
            eval_n_groups = int(known_num_groups) if known_num_groups is not None else 1

            metrics = evaluate(
                model,
                checkpoint_loader,
                device=device,
                n_groups=eval_n_groups,
                force_civilcomments_16_slices=bool(checkpoint_force_civilcomments_16_slices),
                verbose=bool(checkpoint_verbose),
            )
            if ckpt_metric not in metrics:
                raise KeyError(
                    f"checkpoint_metric={ckpt_metric!r} not found in evaluate() outputs: {sorted(metrics.keys())}"
                )
            val = float(metrics[ckpt_metric])

            better = False
            if ckpt_best_val is None:
                better = True
            elif ckpt_mode == "max" and val > ckpt_best_val:
                better = True
            elif ckpt_mode == "min" and val < ckpt_best_val:
                better = True

            if better:
                ckpt_best_val = val
                ckpt_best_epoch = int(_ep) + 1  # 1-based for logging
                ckpt_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if checkpoint_verbose:
                print(
                    f"[train_head] ckpt epoch={int(_ep)+1}/{int(epochs)} "
                    f"{ckpt_metric}={val:.6f} best={ckpt_best_val:.6f} (mode={ckpt_mode})"
                )

            model.train()

    # ---------- load best checkpoint (if enabled) ----------
    if ckpt_best_state is not None:
        model.load_state_dict(ckpt_best_state)
        if train_stats is not None:
            train_stats["checkpoint_metric"] = str(ckpt_metric)
            train_stats["checkpoint_mode"] = str(ckpt_mode)
            train_stats["checkpoint_best_epoch"] = int(ckpt_best_epoch) if ckpt_best_epoch is not None else -1
            train_stats["checkpoint_best_value"] = float(ckpt_best_val) if ckpt_best_val is not None else float("nan")

    # ---------- final stats ----------
    if train_stats is not None:
        train_stats["train_steps"] = int(steps)
        train_stats["train_examples"] = int(examples)
        train_stats["train_epochs"] = int(epochs)
        train_stats["train_examples_per_step"] = float(examples / max(1, steps))

    return model

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_groups: int,
    *,
    force_civilcomments_16_slices: bool = False,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate mean loss/accuracy and worst-group accuracy.

    CivilComments (WILDS-comparable):
        worst_group_acc = min accuracy over the 16 *overlapping* slices:
            identities = {male, female, LGBTQ, christian, muslim, other_religions, black, white} (8)
            labels     = {0, 1} (2)
            slice      = (identity_i > 0) AND (y == label)

    All other datasets:
        worst_group_acc = min accuracy over the `n_groups` *disjoint* integer groups `g`.

    Notes:
      - We always compute the disjoint worst-group acc and expose it as `worst_group_acc_disjoint`.
      - If CivilComments metadata is available, we also compute the 16-slice metric and:
          * write it to `worst_group_acc_16slice`
          * overwrite `worst_group_acc` with the 16-slice value
      - If force_civilcomments_16_slices=True and the 16-slice metric cannot be computed, we raise.
    """
    model.eval()

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    losses: List[float] = []
    correct: List[int] = []
    groups: List[int] = []

    ys: List[int] = []
    idxs: List[int] = []

    for xb, yb, gb, idxb in loader:
        xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
        logits = model(xb)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            logit = logits.view(-1)
            l = F.binary_cross_entropy_with_logits(logit, yb.float(), reduction="none")
            p = (logit > 0).to(torch.long)
        else:
            l = ce_loss(logits, yb)
            p = torch.argmax(logits, dim=1)

        losses.extend(l.detach().cpu().tolist())
        correct.extend((p == yb).to(torch.int64).detach().cpu().tolist())
        groups.extend(gb.detach().cpu().tolist())

        ys.extend(yb.detach().cpu().tolist())
        idxs.extend(idxb.detach().cpu().tolist())

    lnp = np.asarray(losses, dtype=np.float64)
    cnp = np.asarray(correct, dtype=np.float64)
    gnp = np.asarray(groups, dtype=np.int64)
    ynp = np.asarray(ys, dtype=np.int64)
    idxnp = np.asarray(idxs, dtype=np.int64)

    out: Dict[str, float] = {
        "loss_mean": float(lnp.mean()),
        "acc_mean": float(cnp.mean()),
        "loss_p95": float(np.quantile(lnp, 0.95)),
        "loss_max": float(lnp.max()),
    }

    # -------------------- Disjoint worst-group acc (always computed) --------------------
    wg = []
    for g in range(int(n_groups)):
        m = gnp == g
        if np.any(m):
            wg.append(float(cnp[m].mean()))
    out["worst_group_acc_disjoint"] = float(np.min(wg)) if wg else float("nan")
    out["worst_group_acc"] = float(out["worst_group_acc_disjoint"])

    # ---------------- CivilComments: 16 overlapping identity × label slices ----------------
    base_ds = loader.dataset
    while isinstance(base_ds, Subset):
        base_ds = base_ds.dataset

    meta = None
    fields = None

    if hasattr(base_ds, "base_subset") and hasattr(base_ds.base_subset, "metadata_array"):
        meta = base_ds.base_subset.metadata_array
        fields = getattr(base_ds.base_subset, "metadata_fields", None)
    elif hasattr(base_ds, "metadata_array"):
        meta = base_ds.metadata_array
        fields = getattr(base_ds, "metadata_fields", None)

    if meta is not None:
        if torch.is_tensor(meta):
            meta = meta.detach().cpu().numpy()
        meta = np.asarray(meta)

    used_16slice = False
    if meta is not None and meta.ndim == 2 and meta.shape[1] >= 8 and idxnp.size > 0:
        # Prefer column lookup by name when possible (more robust than positional assumptions).
        id_names = ["male", "female", "lgbtq", "christian", "muslim", "other_religions", "black", "white"]
        id_cols = None

        if fields is not None:
            try:
                fl = [str(x).lower() for x in list(fields)]
                cols = []
                for nm in id_names:
                    if nm in fl:
                        cols.append(fl.index(nm))
                if len(cols) == 8:
                    id_cols = cols
            except Exception:
                id_cols = None

        # Fallback: assume first 8 columns are the identity indicators.
        # This is unsafe if metadata_fields exists but the identity columns cannot be resolved by name.
        if id_cols is None:
            msg = (
                "[evaluate] CivilComments: could not locate all 8 identity columns by name in metadata_fields. "
                "Positional fallback to the first 8 metadata columns may corrupt the 16-slice metric."
            )
            if force_civilcomments_16_slices:
                raise RuntimeError(msg)
            import warnings
            warnings.warn(msg + " Proceeding with positional fallback (first 8 columns).")
            id_cols = list(range(8))

        # Make the meta rows align with the evaluated examples.
        # If meta is already aligned to this loader (common after we attach split-local meta),
        # use it directly; otherwise try meta[idx].
        if meta.shape[0] == idxnp.size:
            meta_used = meta
        elif int(idxnp.max()) < int(meta.shape[0]):
            meta_used = meta[idxnp]
        else:
            meta_used = None

        if meta_used is not None and meta_used.ndim == 2 and meta_used.shape[0] == idxnp.size:
            id_mat = (meta_used[:, id_cols] > 0)  # (N, 8) bool

            slice_acc = []
            for j in range(8):
                for lab in (0, 1):
                    m = id_mat[:, j] & (ynp == lab)
                    if np.any(m):
                        slice_acc.append(float(cnp[m].mean()))

            if slice_acc:
                used_16slice = True
                out["worst_group_acc_idany4"] = float(out["worst_group_acc_disjoint"])
                out["worst_group_acc_16slice"] = float(np.min(slice_acc))
                out["worst_group_acc"] = float(out["worst_group_acc_16slice"])

                if verbose:
                    print(
                        "[evaluate] CivilComments detected: using 16 overlapping identity×label slices "
                        f"for worst_group_acc (min over {len(slice_acc)} non-empty slices). "
                        f"worst_group_acc_16slice={out['worst_group_acc_16slice']:.6f} "
                        f"(disjoint={out['worst_group_acc_disjoint']:.6f}) id_cols={id_cols}"
                    )

    if force_civilcomments_16_slices and (not used_16slice):
        raise RuntimeError(
            "[evaluate] force_civilcomments_16_slices=True but could not compute the 16-slice metric. "
            "This usually means the evaluation dataset does not expose WILDS metadata_array "
            "(identity indicators) in a way evaluate() can access."
        )

    return out
