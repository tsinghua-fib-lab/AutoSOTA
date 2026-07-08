import torch


def saps_scores(probs, h_params, targets=None):
    """
    Compute the SAPS non-conformity scores for the given probabilities.
    :param h_params: the hyperparam dict with weight of label ranking.
    :param probs: tensor of probabilities.
    :param label: target label (optional).
    :return: scores
    """
    weight = h_params["lamda"]

    # If targets provide (calibration), return scores only for the correct class.
    if targets is None:
        r_scores = calculate_all_label(probs, weight)
    # If no targets provided (test), return scores for each class.
    else:
        r_scores = calculate_single_label(probs, targets, weight)

    return r_scores.numpy()


def sort_sum(probs):
    ordered, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(ordered, dim=-1)
    return indices, ordered, cumsum


def calculate_single_label(probs, label, weight):
    indices, ordered, cumsum = sort_sum(probs)
    U = torch.rand(indices.shape[0], device=probs.device)
    idx = torch.where(indices == label.view(-1, 1))
    scores_first_rank = U * cumsum[idx]
    scores_usual = weight * (idx[1] - U) + ordered[:, 0]
    return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


def calculate_all_label(probs, weight):
    indices, ordered, cumsum = sort_sum(probs)
    ordered[:, 1:] = weight
    cumsum = torch.cumsum(ordered, dim=-1)
    U = torch.rand(probs.shape, device=probs.device)
    ordered_scores = cumsum - ordered * U
    _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
    scores = ordered_scores.gather(dim=-1, index=sorted_indices)
    return scores
