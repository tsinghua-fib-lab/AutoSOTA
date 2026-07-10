import torch
import torch.nn.functional as F
from pdb import set_trace as bb
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import CalibrationError
from sklearn.metrics import average_precision_score, roc_auc_score
import math



def rbf_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigma=None):
    """
    Compute RBF (Gaussian) kernel matrix between tensors x and y.
    Supports multidimensional inputs (N, D).
    Includes Median Heuristic for sigma if not provided.

    Args:
        x: [N, D]
        y: [M, D]
        sigma: Bandwidth parameter. If None, computed via median heuristic.
    Returns:
        [N, M] Kernel matrix
    """
    dist_sq = torch.cdist(x, y, p=2) ** 2  # [N, M]

    if sigma is None:
        # Median Heuristic: sigma = sqrt(median_dist / 2)
        if dist_sq.numel() > 1:
            median_dist = torch.median(dist_sq.detach()[dist_sq > 0])
            sigma = torch.sqrt(median_dist / 2)
        else:
            sigma = torch.tensor(1.0, device=x.device)

        if torch.isnan(sigma) or sigma == 0:
            sigma = torch.tensor(1.0, device=x.device)

    gamma = 1.0 / (2 * sigma**2 + 1e-10)
    return torch.exp(-gamma * dist_sq)


def compute_kce(probs: torch.Tensor,
                labels: torch.Tensor,
                conditioning: torch.Tensor = None):
    """
    Compute Kernel Calibration Error (KCE) / Calibration MMD as in
    NeurIPS 2023 "Calibration by Distribution Matching".

    Canonical (classification) setting from Table 2:
        conditioning variable Z = q_{Y|X}(\cdot) = predicted probability vector probs.

    We compute:
        MMD^2( (Y, Z), (q_{Y|X}, Z) )
    with a product kernel k((y,z),(y',z')) = k_Y(y,y') * k_Z(z,z').

    Args:
        probs: [N, C] Predicted class probabilities.
        labels: [N] Ground-truth labels (0..C-1).
        conditioning: [N, D] Conditioning variable Z.
            - canonical choice: conditioning = probs
            - alternative: logits or x_test_torch if you want variants

    Returns:
        float: KCE = sqrt(max(0, MMD^2))
    """
    # --- 1) Conditioning Z (canonical = probs) ---
    if conditioning is None:
        conditioning = probs

    num_classes = probs.shape[1]
    labels_one_hot = F.one_hot(labels.view(-1), num_classes=num_classes).float()

    # --- 2) Kernel on Z-space ---
    K_zz = rbf_kernel_matrix(conditioning, conditioning, sigma=None)

    # --- 3) Kernel on Y-space with shared bandwidth sigma_y ---
    # Use a shared sigma_y across one-hot labels and probability vectors (important!)
    y_all = torch.cat([labels_one_hot, probs], dim=0)
    y_dist_sq = torch.cdist(y_all, y_all, p=2) ** 2
    median_dist_y = torch.median(y_dist_sq[y_dist_sq > 0])
    sigma_y = torch.sqrt(median_dist_y / 2)
    if torch.isnan(sigma_y) or sigma_y == 0:
        sigma_y = torch.tensor(1.0, device=probs.device)

    gamma_y = 1.0 / (2 * sigma_y**2 + 1e-10)

    def k_y(a, b):
        d = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-gamma_y * d)

    K_yy_true = k_y(labels_one_hot, labels_one_hot)   # one-hot vs one-hot
    K_yy_pred = k_y(probs, probs)                     # probs vs probs
    K_yy_cross = k_y(labels_one_hot, probs)           # one-hot vs probs

    # --- 4) Product-kernel MMD^2 ---
    term_true = (K_yy_true * K_zz).mean()
    term_pred = (K_yy_pred * K_zz).mean()
    term_cross = (K_yy_cross * K_zz).mean()

    kce_sq = term_true + term_pred - 2 * term_cross

    # Numerical stability: MMD^2 >= 0
    return math.sqrt(max(0.0, kce_sq.item()))



def _compute_ece_torchmetrics(probs: torch.Tensor,
                              labels: torch.Tensor,
                              num_bins: int = 20) -> float:
    """
    Compute Top-label ECE with torchmetrics.CalibrationError.
    """
    num_classes = probs.shape[1]
    task_type = 'binary' if num_classes == 2 else 'multiclass'

    if task_type == 'binary':
        preds_for_ece = probs[:, 1]  # prob of positive class
    else:
        preds_for_ece = probs        # full distribution [N, C]

    metric = CalibrationError(
        n_bins=num_bins,
        norm='l1',
        task=task_type,
        num_classes=num_classes
    )
    metric.update(preds=preds_for_ece, target=labels.view(-1))
    return float(metric.compute().item())


def _group_masks_by_X(x: torch.Tensor,
                      lower_frac: float,
                      upper_frac: float):
    """
    Build group masks by feature pairs with median thresholding.
    For each pair (i,j), create 4 groups:
      (x_i > med_i, x_j > med_j),
      (x_i > med_i, x_j <= med_j),
      (x_i <= med_i, x_j > med_j),
      (x_i <= med_i, x_j <= med_j).
    Keep only groups whose size is within [lower_frac*N, upper_frac*N].
    Return: list[mask: BoolTensor]
    """
    N, n = x.shape
    masks = []

    med = torch.median(x, dim=0).values  # [n]

    lo = int(max(1, lower_frac * N))
    hi = int(min(N, upper_frac * N))

    for i in range(n):
        for j in range(i + 1, n):
            xi = x[:, i]
            xj = x[:, j]
            mi = med[i]
            mj = med[j]

            g1 = (xi > mi) & (xj > mj)
            g2 = (xi > mi) & (xj <= mj)
            g3 = (xi <= mi) & (xj > mj)
            g4 = (xi <= mi) & (xj <= mj)

            for g in (g1, g2, g3, g4):
                sz = int(g.sum().item())
                if lo <= sz <= hi:
                    masks.append(g)

    return masks


def _group_masks_by_Y(labels: torch.Tensor):
    """
    One mask per class label (per-class grouping).
    """
    labels = labels.view(-1)
    unique = torch.unique(labels)
    masks = [(labels == c) for c in unique]
    return masks


def _group_ece_max_over_masks(probs: torch.Tensor,
                              labels: torch.Tensor,
                              masks,
                              num_bins: int = 20):
    """
    Compute ECE for each mask using torchmetrics (same settings as global ECE),
    and return the maximum ECE over all masks. If no valid mask, return None.
    """
    if len(masks) == 0:
        return None

    eces = []
    for m in masks:
        if int(m.sum().item()) == 0:
            continue
        ece_g = _compute_ece_torchmetrics(probs[m], labels[m], num_bins=num_bins)
        eces.append(ece_g)

    if len(eces) == 0:
        return None
    return max(eces)


def evaluate(model_str, model, x_test_torch, y_test_torch,
             t=0, num_bins=20, mc_samples=2000):
    """
    Evaluate all classifier variants with Accuracy, Entropy (NLL), ECE, KCE, AP, and AUC.
    Also compute:
      - GroupX_ECE: feature-based groups via median thresholding on all feature pairs,
                    keep groups whose size in [1/4*N, 3/4*N], and take the worst (max) ECE.
      - GroupY_ECE: per-class grouping, take the worst (max) ECE across classes.
    """
    # -------- 1) normalize model name: map MMD prefixes back to base --------
    base_model_str = model_str

    mmd_prefix_map = {
        "ALDMMD_": "ALD_",
        "NormMMD_": "Norm_",
        "ExpMMD_": "Exp_",
        "LogNormMMD_": "LogNorm_",
        "WeibullMMD_": "Weibull_",
        "LogisticMMD_": "Logistic_",
    }
    for mmd_prefix, base_prefix in mmd_prefix_map.items():
        if base_model_str.startswith(mmd_prefix):
            base_model_str = base_model_str.replace(mmd_prefix, base_prefix, 1)
            break

    model.eval()
    with torch.no_grad():
        # -------- 2) compute probs based on model structure --------
        N = x_test_torch.size(0)

        if base_model_str == "Softmax_Classifier":
            probs = model(x_test_torch, return_probs=True)

        elif base_model_str.endswith("_Classifier") and not base_model_str.startswith("IC"):
            probs = model(x_test_torch, return_probs=True, t=float(t))

        elif base_model_str.startswith("IC") and base_model_str.endswith("_Classifier"):
            probs_list = []
            for _ in range(mc_samples):
                q = torch.rand(N, 1, device=x_test_torch.device)
                p = model(x_test_torch, q=q, return_probs=True, t=float(t))
                probs_list.append(p)
            probs = torch.stack(probs_list, dim=0).mean(dim=0)

        elif base_model_str.endswith("ScalarStickClassifier") and not base_model_str.startswith("IC"):
            probs = model(x_test_torch, return_probs=True)

        elif base_model_str.startswith("IC") and base_model_str.endswith("ScalarStickClassifier"):
            probs_list = []
            for _ in range(mc_samples):
                q = torch.rand(N, 1, device=x_test_torch.device)
                p = model(x_test_torch, q=q, return_probs=True)
                probs_list.append(p)
            probs = torch.stack(probs_list, dim=0).mean(dim=0)

        else:
            raise ValueError(f"Unsupported model type: {model_str} (base name: {base_model_str})")

        # -------- 3) metrics --------
        pred_classes = probs.argmax(dim=1)
        acc = (pred_classes == y_test_torch).float().mean().item()
        nll = F.nll_loss(torch.log(probs + 1e-8), y_test_torch.view(-1)).item()

        # Global ECE
        num_classes = probs.shape[1]
        task_type = 'binary' if num_classes == 2 else 'multiclass'
        preds_for_ece = probs[:, 1] if task_type == 'binary' else probs

        ece_metric = CalibrationError(
            n_bins=num_bins,
            norm='l1',
            task=task_type,
            num_classes=num_classes
        )
        ece_metric.update(preds=preds_for_ece, target=y_test_torch)
        ece = ece_metric.compute().item()


        # Conditioning variable Z = q_{Y|X}(\cdot) = probs (Table 2 in Marx et al.)
        kce = compute_kce(probs=probs, labels=y_test_torch, conditioning=probs)

        # === GroupX ECE ===
        masks_X = _group_masks_by_X(
            x_test_torch,
            lower_frac=0.25,
            upper_frac=0.75
        )
        groupX_ece = _group_ece_max_over_masks(
            probs, y_test_torch, masks_X, num_bins=num_bins
        )

        # === GroupY ECE ===
        masks_Y = _group_masks_by_Y(y_test_torch)
        groupY_ece = _group_ece_max_over_masks(
            probs, y_test_torch, masks_Y, num_bins=num_bins
        )

        # AP & AUC
        y_true_np = y_test_torch.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        num_classes_np = probs_np.shape[1]
        ap, auc = None, None

        try:
            if num_classes_np == 2:
                prob_pos = probs_np[:, 1]
                ap = average_precision_score(y_true_np, prob_pos)
                auc = roc_auc_score(y_true_np, prob_pos)
            else:
                y_true_bin = label_binarize(y_true_np, classes=range(num_classes_np))
                ap = average_precision_score(y_true_bin, probs_np, average="macro")
                auc = roc_auc_score(y_true_bin, probs_np, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[Warning] Failed to compute AP/AUC: {e}")

    return {
        "Accuracy": acc,
        "Entropy": nll,
        "ECE": ece,
        "KCE": kce,               
        "GroupX_ECE": groupX_ece,
        "GroupY_ECE": groupY_ece,
        "AP": ap,
        "AUC": auc
    }