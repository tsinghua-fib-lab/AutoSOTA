import numpy as np
import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        corr = 0
    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
    else:
        raise TypeError
    return mae, rmse, mape, rrse, corr

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr


