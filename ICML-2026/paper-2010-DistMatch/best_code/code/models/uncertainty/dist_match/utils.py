import torch
import os
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.special import kl_div
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score


def match_ks_p_val(sample1: np.ndarray, sample2: np.ndarray) -> float:
    return ks_2samp(sample1, sample2).pvalue

def match_ks_stat(sample1: np.ndarray, sample2: np.ndarray) -> float:
    return ks_2samp(sample1, sample2).statistic

def match_rand(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    score = adjusted_rand_score(sample1, sample2)
    score = (score + 0.5) / 1.5
    return score

def match_mi(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    score = adjusted_mutual_info_score(sample2, sample1)
    score = (score + 0.5) / 1.5
    return score

def match_kl(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    score = kl_div(sample1, sample2)
    score = np.nan_to_num(score, nan=1, posinf=1, neginf=1).mean()
    return 1 - score

def match_wd(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    score = wasserstein_distance(sample1, sample2)
    return 1 - score

class EarlyStopping:
    def __init__(self, filepath: str, patience: int = 3):
        self.filepath = filepath
        self.patience = patience
        self.min_value = None
        self.cur_stat = self.patience

    def step(self, model: torch.nn.Module, value: float | int) -> bool:
        if self.min_value is None or value < self.min_value:
            self.cur_stat = self.patience
            self.min_value = value
            self.save(model)
            return True

        self.cur_stat -= 1
        print(f"[Early Stopping] Counter: {self.cur_stat + 1}")
        return self.cur_stat > 0

    def save(self, model: torch.nn.Module):
        dir_name = os.path.dirname(self.filepath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        torch.save(model.state_dict(), self.filepath)

    def checkpoint_exists(self, path: str | None = None):
        path = path if path is not None else self.filepath
        
        return os.path.exists(path)

    def load(self, model: torch.nn.Module, path: str | None = None) -> torch.nn.Module | None:
        path = path if path is not None else self.filepath
        if not self.checkpoint_exists(path):
            return None
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return model
