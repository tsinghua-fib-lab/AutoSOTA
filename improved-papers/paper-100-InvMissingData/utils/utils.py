import torch, random, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperimpute.plugins.utils.simulate import simulate_nan
import pandas as pd

def enable_reproducible_results(seed: int = 0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)

def kip_ampute(x, mechanism, p_miss):
    
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)
    
    
    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]
    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)
def partition_ampute(x, mechanism, p_miss,feature_miss):
    x_miss=x[:,:int(0.2*x.shape[1])]
    x_whole=x[:,int(0.2*x.shape[1]):int((0.2+feature_miss)*x.shape[1])]
    whole_mask=np.zeros_like(x_whole)
    x_simulated = simulate_nan(np.asarray(x_miss), p_miss, mechanism)
    
    
    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]
    mask=np.concatenate([mask,np.asarray(whole_mask)],axis=-1)
    x_miss=np.concatenate([x_miss,np.asarray(x_whole)],axis=-1)
    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)
def partition_number_ampute(x, mechanism, p_miss,feature_miss):
    x_miss=x[:,:int(2)]
    x_whole=x[:,int(2):int(2+feature_miss)]
    whole_mask=np.zeros_like(x_whole)
    x_simulated = simulate_nan(np.asarray(x_miss), p_miss, mechanism)
    
    
    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]
    mask=np.concatenate([mask,np.asarray(whole_mask)],axis=-1)
    x_miss=np.concatenate([x_miss,np.asarray(x_whole)],axis=-1)
    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


def scale_data(X):
    X = np.asarray(X)
    # preproc = MinMaxScaler()
    preproc = StandardScaler()
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def diff_scale_data(X):
    X = np.asarray(X)
    # preproc = MinMaxScaler()
    preproc = MinMaxScaler()
    print(f"we are using minmax scaler for diffusion models!")
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def simulate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False):
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            datasets[ampute_mechanism][p_miss] = ampute(X, ampute_mechanism, p_miss)

    return datasets
def partition_number_simluate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False,feature_drop=[1,2,3,4]):
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        datasets[ampute_mechanism] = {}
        for p_miss in percentages:
            datasets[ampute_mechanism][p_miss] = {}
            for feature_miss in feature_drop:

                datasets[ampute_mechanism][p_miss][feature_miss] = partition_number_ampute(X, ampute_mechanism, p_miss,feature_miss)

    return datasets
def partition_simluate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False,feature_drop=[0.1, 0.3, 0.5, 0.7]):
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        datasets[ampute_mechanism] = {}
        for p_miss in percentages:
            datasets[ampute_mechanism][p_miss] = {}
            for feature_miss in feature_drop:

                datasets[ampute_mechanism][p_miss][feature_miss] = partition_ampute(X, ampute_mechanism, p_miss,feature_miss)

    return datasets
def kip_simulate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False,feature_drop=[0.1, 0.3, 0.5, 0.7]):
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        datasets[ampute_mechanism] = {}
        for p_miss in percentages:
            
                datasets[ampute_mechanism][p_miss] = kip_ampute(X, ampute_mechanism, p_miss)

    return datasets


# third party
import numpy as np


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = np.absolute(X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]])
                _output = _output.sum() / mask_[:, i].sum()
                output += [str(_output.round(5))]

        return output

def MSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return ((X[mask_] - X_true[mask_])**2).sum() / mask_.sum()
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = (X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]])**2
                _output = _output.sum() / mask_[:, i].sum()
                output += [str(_output.round(5))]

        return output

def RMSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        RMSE : np.ndarray

    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = (X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]]) ** 2
                _output = _output.sum() / mask_[:, i].sum()
                _output = np.sqrt(_output)
                output += [str(_output.round(5))]

        return output


__all__ = ["MAE", "RMSE","MSE"]
