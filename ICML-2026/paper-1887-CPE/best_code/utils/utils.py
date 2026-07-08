import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def entropy_categorical(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def ess(weights: np.ndarray) -> float:
    w = weights / np.sum(weights)
    return 1.0 / np.sum(w**2)


def normalize(weights: np.ndarray) -> np.ndarray:
    s = float(np.sum(weights))
    if s <= 0:
        return np.full_like(weights, 1.0 / len(weights), dtype=float)
    return weights / s


