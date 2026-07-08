import torch


def raps_scores(probs, h_params, targets=None):
    kreg = h_params["kreg"]
    lamda = h_params["lamda"]
    
    # If targets provide (calibration), return scores only for the correct class.
    if targets is not None:
        r_scores = get_tau(probs, targets, kreg, lamda)
    # If no targets provided (test), return scores for each class.
    else:
        r_scores = get_taus(probs, kreg, lamda)

    return r_scores.numpy()


def sort_sum(probs):
    ordered, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(ordered, dim=-1)
    return indices, ordered, cumsum


# Get the 'p-value' for the correct class
def get_tau(probs, target, kreg, lamda):
    indices, ordered, cumsum = sort_sum(probs)
    U = torch.rand(indices.shape[0], device=probs.device)
    idx = torch.where(indices == target.view(-1, 1))
    reg = torch.maximum(lamda * (idx[1] + 1 - kreg), torch.tensor(0).to(probs.device))
    scores_first_rank = U * ordered[idx] + reg
    idx_minus_one = (idx[0], idx[1] - 1)
    scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg

    return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


# Get the 'p-values' for all classes
def get_taus(probs, kreg, lamda):
    indices, ordered, cumsum = sort_sum(probs)
    U = torch.rand(probs.shape, device=probs.device)
    reg = torch.maximum(lamda * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - kreg),
                        torch.tensor(0, device=probs.device))
    ordered_scores = cumsum - ordered * U + reg
    _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
    scores = ordered_scores.gather(dim=-1, index=sorted_indices)

    return scores
