import torch

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quota_loss(
    T: torch.Tensor, S_X: torch.Tensor, S_Y: torch.Tensor, F: torch.Tensor
) -> torch.Tensor:
    """
    Quota loss between two distributions X and Y matched through the optimal
    transport plan T with respect to their respective sensitive attributes S_X
    and S_Y.

    Parameters:
    ----------
    T (torch.Tensor): The optimal transport plan.
    S_X (torch.Tensor): The sensitive attribute for the first distribution.
    S_Y (torch.Tensor): The sensitive attribute for the second distribution.
    F (torch.Tensor): Target fairness matrix.
    """
    # Ensure T is a torch tensor with consistent dtype
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, dtype=torch.float64)
    if T.dtype != torch.float64:
        T = T.to(torch.float64)

    n_s_x, n_s_y = F.shape

    # Create one-hot encodings for sensitive attributes
    S_X_onehot = torch.nn.functional.one_hot(
        S_X.long(), num_classes=n_s_x
    ).to(torch.float64)  # shape: (n_x, n_s_x)
    S_Y_onehot = torch.nn.functional.one_hot(
        S_Y.long(), num_classes=n_s_y
    ).to(torch.float64)  # shape: (n_y, n_s_y)

    # Compute joint distribution using matrix multiplication
    joint_distribution = S_X_onehot.T @ T @ S_Y_onehot  # shape: (n_s_x, n_s_y)

    # Ensure F is on the same device as joint_distribution
    if F.device != joint_distribution.device:
        F = F.to(joint_distribution.device)
    F = F.to(torch.float64)

    return torch.sum(torch.square(joint_distribution - F))


def weighted_quota_loss(
    T: torch.Tensor,
    C: torch.Tensor,
    S_X: torch.Tensor,
    S_Y: torch.Tensor,
    F: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted quota loss between two distributions X and Y matched through the
    optimal transport plan T with respect to their respective sensitive
    attributes S_X and S_Y.

    Parameters:
    ----------
    T (torch.Tensor): The optimal transport plan.
    C (torch.Tensor): The weighting matrix matrix.
    S_X (torch.Tensor): The sensitive attribute for the first distribution.
    S_Y (torch.Tensor): The sensitive attribute for the second distribution.
    F (torch.Tensor): Target fairness matrix.
    return_all (bool): Whether to return all intermediate values.

    Returns:
    -------
    torch.Tensor: The cost per group loss.
    """
    # Ensure T and C are torch tensors with consistent dtype
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, dtype=torch.float64)
    if T.dtype != torch.float64:
        T = T.to(torch.float64)
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=torch.float64)
    if C.dtype != torch.float64:
        C = C.to(torch.float64)

    n_s_x, n_s_y = F.shape

    # Create one-hot encodings for sensitive attributes
    S_X_onehot = torch.nn.functional.one_hot(
        S_X.long(), num_classes=n_s_x
    ).to(torch.float64)  # shape: (n_x, n_s_x)
    S_Y_onehot = torch.nn.functional.one_hot(
        S_Y.long(), num_classes=n_s_y
    ).to(torch.float64)  # shape: (n_y, n_s_y)

    # Compute groupwise cost using matrix multiplication
    weighted_cost = C * T  # element-wise multiplication
    groupwise_cost = (
        S_X_onehot.T @ weighted_cost @ S_Y_onehot
    )  # shape: (n_s_x, n_s_y)

    # Ensure F is on the same device as groupwise_cost
    if F.device != groupwise_cost.device:
        F = F.to(groupwise_cost.device)
    F = F.to(torch.float64)

    return torch.sum(torch.square(groupwise_cost - F))
