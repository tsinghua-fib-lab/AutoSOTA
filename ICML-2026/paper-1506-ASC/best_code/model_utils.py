import torch

class BasicStrategicHingeLoss(torch.nn.Module):
    def __init__(self, scale_loss=1.0):
        super().__init__()
        self.scale_loss = scale_loss

    def forward(self, model, X, Y):
        hinge = torch.ones_like(Y) - Y * (model.product(X) + (2 / self.scale_loss) * model.w_norm()) 
        return torch.clamp_min(hinge, 0).mean()

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(model, X, Y):
        hinge = torch.ones_like(Y) - Y * model.product(X) 
        return torch.clamp_min(hinge, 0).mean()

class AmbiguousStrategicHingeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(model, X, Y, values):
        hinge = torch.ones_like(Y) - Y * (model.product(X) + values)
        hinge = torch.clamp_min(hinge, 0)

        return hinge.mean()

def calculate_utility(preds, norms):
    """
    Calculate the utility of all points.
    Utility is defined as the difference between the classification sign and the norms.
    """
    utility = preds - norms
    return sum(utility), utility.mean()

def calculate_burden_to_AOP(model, X_moved, y_true, logits, norms):
    y_is_pos = (y_true == 1)
    is_positive_classified = (logits >= 0)

    burden = torch.zeros_like(norms)
    w_chosen = model.get_w_chosen()
    b_chosen = model.get_b_chosen()
    
    burden[y_is_pos & is_positive_classified] = (
        norms[y_is_pos & is_positive_classified] * model.cost_scaling
    )

    raw_margins = (X_moved @ w_chosen + b_chosen).abs() / model.w_norm()
    burden[y_is_pos & ~is_positive_classified] = (
        raw_margins[y_is_pos & ~is_positive_classified] * model.cost_scaling
    )

    total_burden = burden[y_is_pos].sum()
    avg_burden = burden[y_is_pos].mean() if y_is_pos.sum() > 0 else torch.tensor(0.0)
    
    return total_burden, avg_burden

def calculate_burden_to_w_chosen(model, X, y_true):
    """
    Calculates the cost (burden) for positive users to cross the decision boundary.
    Burden = Geometric Distance * Cost Scaling
    """
    y_is_pos = (y_true == 1)
    
    if y_is_pos.sum() == 0:
        return torch.tensor(0.0), torch.tensor(0.0)

    X_pos = X[y_is_pos]
    
    w_chosen = model.get_w_chosen()
    b_chosen = model.get_b_chosen()
    w_norm = model.w_norm()
    
    cost_scaling = model.cost_scaling 

    logits = (X_pos @ w_chosen) + b_chosen

    geometric_dists = torch.relu(-logits) / w_norm
    
    real_burdens = geometric_dists * cost_scaling
    
    total_burden = real_burdens.sum()
    avg_burden = real_burdens.mean()

    return total_burden, avg_burden


def calculate_recall(model, X_moved, y_true, logits):
        y_is_pos = (y_true == 1)
        y_is_neg = (y_true == -1)
        is_positive_classified = (logits >= 0)
        is_negative_classified = (logits < 0)

        true_positives = (y_is_pos & is_positive_classified).sum().item()
        true_negatives = (y_is_neg & is_negative_classified).sum().item()

        pos_recall = true_positives / y_is_pos.sum() if (y_is_pos.sum() > 0) else torch.tensor(0.0)
        neg_recall = true_negatives / y_is_neg.sum() if (y_is_neg.sum() > 0) else torch.tensor(0.0)

        return pos_recall, neg_recall

def calculate_moving_ratio(X, X_proj, y_true):

    dists = torch.norm(X - X_proj, dim=1)
    move_mask = dists > 1e-5
    
    pos_indices = (y_true == 1)
    neg_indices = (y_true == -1)
    
    num_pos = pos_indices.sum().item()
    if num_pos > 0:
        num_moved_pos = (move_mask & pos_indices).sum().item()
        percent_moved_pos = (num_moved_pos / num_pos) * 100.0
    else:
        percent_moved_pos = 0.0

    num_neg = neg_indices.sum().item()
    if num_neg > 0:
        num_moved_neg = (move_mask & neg_indices).sum().item()
        percent_moved_neg = (num_moved_neg / num_neg) * 100.0
    else:
        percent_moved_neg = 0.0

    return percent_moved_pos, percent_moved_neg