import torch


def pad_x(X: torch.Tensor, num_features=100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros((*X.shape[:-1], num_features - n_features), device=X.device)
    return torch.cat([X, zero_feature_padding], dim=-1)
