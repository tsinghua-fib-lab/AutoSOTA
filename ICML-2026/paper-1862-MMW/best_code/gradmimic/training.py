import torch
from torch.func import grad, vmap, functional_call


# ---------------------------------------------------------------------------
# Per-sample gradient computation
# ---------------------------------------------------------------------------

def _cost_model(input_, target, params, buffers, model):
    out = functional_call(model, (params, buffers), input_.unsqueeze(0))
    return torch.nn.CrossEntropyLoss()(out, target.unsqueeze(0))


def get_per_sample_gradients(model, inputs, targets):
    """Return a dict mapping parameter name → per-sample gradient tensor (shape: [B, *param_shape])."""
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad}
    ft_grad = grad(_cost_model, argnums=2)
    ft_vmap = vmap(ft_grad, in_dims=(0, 0, None, None, None))
    return ft_vmap(inputs, targets, params, buffers, model)


# ---------------------------------------------------------------------------
# Task vector
# ---------------------------------------------------------------------------

def compute_task_vector(current_model, reference_model, mimic_layer_name):
    """Compute weight delta: reference_model[layer] - current_model[layer]."""
    cur = current_model.state_dict()
    ref = reference_model.state_dict()
    return ref[mimic_layer_name] - cur[mimic_layer_name]


# ---------------------------------------------------------------------------
# Similarity / mimic scores
# ---------------------------------------------------------------------------

def compute_similarity(task_vector, sample_grads, method="normed_proj", temperature=1.0):
    """Compute per-sample mimic scores between task_vector and each gradient.

    Methods
    -------
    cos / normed_cos : cosine similarity
    proj / normed_proj : projection length onto unit task vector
    Prefix ``normed_`` applies softmax normalisation with ``temperature``.
    """
    tv_flat = task_vector.flatten()
    normed_tv = torch.nn.functional.normalize(tv_flat, p=2, dim=0)

    scores = torch.empty(len(sample_grads), device=tv_flat.device)
    for i, g in enumerate(sample_grads):
        g_flat = g.flatten()
        if method.endswith("cos"):
            scores[i] = torch.dot(torch.nn.functional.normalize(g_flat, p=2, dim=0), normed_tv)
        else:  # proj
            scores[i] = torch.dot(g_flat, normed_tv)

    if method.startswith("normed"):
        return torch.nn.Softmax(dim=0)(scores / temperature)
    return scores


# ---------------------------------------------------------------------------
# Gradient calibration
# ---------------------------------------------------------------------------

def gradient_calibration(model, per_sample_weights, per_sample_grads, mimic_layer_name, calibrate_mimic_layer_only=True):
    """Overwrite .grad on model parameters using per-sample weights."""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if calibrate_mimic_layer_only and name != mimic_layer_name:
            param.grad = torch.mean(per_sample_grads[name], dim=0)
        else:
            reshape = [1] * (len(param.shape) + 1)
            reshape[0] = len(per_sample_weights)
            w = per_sample_weights.reshape(reshape)
            param.grad = torch.sum(w * per_sample_grads[name], dim=0)


# ---------------------------------------------------------------------------
# Optimisation-based subset selection (requires cvxpy + MOSEK)
# ---------------------------------------------------------------------------

def _objective_fn_vector(neg_grad, task_vector, w, lambd, norm_way):
    import cvxpy as cp
    reg = cp.norm2(w) if norm_way == "l2" else cp.norm1(w)
    return cp.norm2((w @ neg_grad) - task_vector) + lambd * reg


def _objective_fn_matrix(neg_grad, task_vector, w, lambd, norm_way):
    import cvxpy as cp
    slice_norms = [cp.norm(neg_grad[i] * w[i] - task_vector, "fro") for i in range(neg_grad.shape[0])]
    reg = cp.norm2(w) if norm_way == "l2" else cp.norm1(w)
    return cp.sum(cp.hstack(slice_norms)) + lambd * reg


def solve_subset_selection(ref, inputs, num_datapoint, lambd_value, norm_way, device):
    """Solve the convex subset-selection problem with MOSEK via cvxpy."""
    import cvxpy as cp

    w = cp.Variable(num_datapoint)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lambd_value

    if ref.ndim == 1:
        obj = _objective_fn_vector(inputs, ref, w, lambd, norm_way)
    else:
        obj = _objective_fn_matrix(inputs, ref, w, lambd, norm_way)

    cp.Problem(cp.Minimize(obj)).solve(solver=cp.MOSEK, verbose=False)
    return torch.tensor(w.value, dtype=torch.float32, device=device)
