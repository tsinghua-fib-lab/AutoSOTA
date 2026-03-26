import torch

def mask_np(array, null_val):
    return torch.not_equal(array, null_val).float()

def masked_mape_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mape = torch.abs((y_pred - y_true) / y_true)
    mape = mask * mape
    return mape * 100


def masked_rmse_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return torch.sqrt(mask * mse)


def masked_mse_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return mask * mse


def masked_mae_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = torch.abs(y_true - y_pred)
    return mask * mae
