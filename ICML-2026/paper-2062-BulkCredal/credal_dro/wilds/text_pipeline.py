from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader
import os
import uuid

def make_vectorizer(n_features: int, ngram_range: Tuple[int, int]) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=int(n_features),
        ngram_range=ngram_range,
        alternate_sign=False,
        norm="l2",
        binary=False,
        lowercase=True,
    )


def collate_hashing(vectorizer: HashingVectorizer):
    def _collate(batch):
        texts = [str(x[0]) for x in batch]
        X = vectorizer.transform(texts)
        X = torch.as_tensor(X.toarray(), dtype=torch.float32)
        y = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
        g = torch.tensor([int(x[2]) for x in batch], dtype=torch.long)
        idx = torch.tensor([int(x[3]) for x in batch], dtype=torch.long)
        return X, y, g, idx

    return _collate


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load(path: Path) -> np.ndarray:
    return np.load(path)


def _collate_text_only(batch):
    texts = [str(x[0]) for x in batch]
    y = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
    g = torch.tensor([int(x[2]) for x in batch], dtype=torch.long)
    idx = torch.tensor([int(x[3]) for x in batch], dtype=torch.long)
    return texts, y, g, idx


def fit_classcond_diag_gaussian_text(
    dataset,
    vectorizer: HashingVectorizer,
    *,
    batch_size: int,
    num_workers: int,
    verbose: bool,
    desc: str,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit per-class diagonal Gaussian moments on hashed n-gram features.
    Returns (mu, std) arrays of shape (C, D).
    """
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_text_only,
        pin_memory=True,
    )

    D = int(vectorizer.n_features)
    C = int(n_classes)

    sum_x = np.zeros((C, D), dtype=np.float64)
    sum_x2 = np.zeros((C, D), dtype=np.float64)
    cnt = np.zeros((C,), dtype=np.int64)

    it = loader
    if verbose:
        from tqdm import tqdm  # type: ignore

        it = tqdm(loader, desc=desc, leave=False)

    for texts, y, _g, _idx in it:
        X = vectorizer.transform(texts)  # scipy sparse CSR
        y_np = y.numpy()

        for c in range(C):
            m = y_np == c
            if not np.any(m):
                continue
            Xc = X[m]
            sum_x[c] += np.asarray(Xc.sum(axis=0)).reshape(-1)
            sum_x2[c] += np.asarray(Xc.multiply(Xc).sum(axis=0)).reshape(-1)
            cnt[c] += int(np.sum(m))

    mu = (sum_x / np.maximum(cnt[:, None], 1)).astype(np.float32)
    ex2 = (sum_x2 / np.maximum(cnt[:, None], 1)).astype(np.float32)
    var = ex2 - mu * mu
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mu, std


def scores_classcond_diag_mahalanobis_text(
    dataset,
    vectorizer: HashingVectorizer,
    mu: np.ndarray,
    std: np.ndarray,
    *,
    ridge: float,
    batch_size: int,
    num_workers: int,
    verbose: bool,
    desc: str,
) -> np.ndarray:
    """
    Compute *minimum over classes* diagonal Mahalanobis distance:
        s(x) = min_y sqrt( sum_j (x_j - mu_{y,j})^2 / (std_{y,j}^2 + ridge) )
    """
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_text_only,
        pin_memory=True,
    )

    C, D = mu.shape
    denom = (std * std + float(ridge)).astype(np.float32)
    inv_std2 = (1.0 / denom).astype(np.float32)
    mu_over_std2 = (mu * inv_std2).astype(np.float32)
    const = np.sum(mu * mu * inv_std2, axis=1).astype(np.float32)

    s_list = []
    it = loader
    if verbose:
        from tqdm import tqdm  # type: ignore

        it = tqdm(loader, desc=desc, leave=False)

    for texts, _y, _g, _idx in it:
        X = vectorizer.transform(texts)  # sparse CSR
        dist2_all = np.empty((X.shape[0], C), dtype=np.float32)
        X2 = X.multiply(X)
        for c in range(C):
            term1 = np.asarray(X2.dot(inv_std2[c])).reshape(-1)
            term2 = np.asarray(X.dot(mu_over_std2[c])).reshape(-1)
            dist2 = const[c] + term1 - 2.0 * term2
            dist2_all[:, c] = np.maximum(dist2, 0.0)
        s_list.append(np.sqrt(np.min(dist2_all, axis=1)).astype(np.float32))

    return np.concatenate(s_list, axis=0)


def scores_trueclass_diag_mahalanobis_text(
    dataset,
    vectorizer: HashingVectorizer,
    mu: np.ndarray,
    std: np.ndarray,
    *,
    ridge: float,
    batch_size: int,
    num_workers: int,
    verbose: bool,
    desc: str,
) -> np.ndarray:
    """
    Compute *true-class* diagonal Mahalanobis distance:
        s_y(x) = sqrt( sum_j (x_j - mu_{y,j})^2 / (std_{y,j}^2 + ridge) )
    """
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_text_only,
        pin_memory=True,
    )

    C, D = mu.shape
    denom = (std * std + float(ridge)).astype(np.float32)
    inv_std2 = (1.0 / denom).astype(np.float32)
    mu_over_std2 = (mu * inv_std2).astype(np.float32)
    const = np.sum(mu * mu * inv_std2, axis=1).astype(np.float32)

    s_list = []
    it = loader
    if verbose:
        from tqdm import tqdm  # type: ignore

        it = tqdm(loader, desc=desc, leave=False)

    for texts, y, _g, _idx in it:
        X = vectorizer.transform(texts)
        y_np = y.numpy()
        X2 = X.multiply(X)

        dist2 = np.zeros((X.shape[0],), dtype=np.float32)
        for c in range(C):
            m = y_np == c
            if not np.any(m):
                continue
            Xm = X[m]
            Xm2 = X2[m]
            term1 = np.asarray(Xm2.dot(inv_std2[c])).reshape(-1)
            term2 = np.asarray(Xm.dot(mu_over_std2[c])).reshape(-1)
            dist2_m = const[c] + term1 - 2.0 * term2
            dist2[m] = np.maximum(dist2_m, 0.0)

        s_list.append(np.sqrt(dist2).astype(np.float32))

    return np.concatenate(s_list, axis=0)


def _ridge_suffix(ridge: float) -> str:
    if abs(float(ridge)) < 1e-12:
        return ""
    return f"__ridge{int(round(float(ridge) * 1e4))}"


@dataclass(frozen=True)
class TextScoreCache:
    cache_dir: Path
    mu_path: Path
    std_path: Path
    train_s: Path
    cal_s: Path
    val_s: Path
    test_s: Path
    # Optional true-class score paths (needed for rw_lv_bas_bin)
    train_s_true: Optional[Path] = None
    cal_s_true: Optional[Path] = None
    val_s_true: Optional[Path] = None
    test_s_true: Optional[Path] = None

    def load_scores(self) -> Dict[str, np.ndarray]:
        return dict(
            train=_load(self.train_s),
            cal=_load(self.cal_s),
            val=_load(self.val_s),
            test=_load(self.test_s),
        )

    def load_true_scores(self) -> Dict[str, np.ndarray]:
        if self.train_s_true is None or self.cal_s_true is None or self.val_s_true is None or self.test_s_true is None:
            raise RuntimeError(
                "True-class scores are not available in this cache. "
                "Call ensure_text_score_cache(..., need_trueclass_scores=True)."
            )
        return dict(
            train=_load(self.train_s_true),
            cal=_load(self.cal_s_true),
            val=_load(self.val_s_true),
            test=_load(self.test_s_true),
        )


def ensure_text_score_cache(
    *,
    dataset_tag: str,
    splits: Dict[str, object],
    cache_root: Path,
    n_features: int,
    ngram_range: Tuple[int, int],
    split_seed: int,
    cal_fraction: float,
    batch_size: int,
    num_workers: int,
    verbose: bool,
    n_classes: int,
    ridge: float = 0.0,
    need_trueclass_scores: bool = False,
) -> TextScoreCache:
    """
    Cache class-conditional diagonal Gaussian moments and Mahalanobis scores on hashed n-gram features.

    - The min-over-class score s(x) is always cached (with ridge-specific filenames).
    - If need_trueclass_scores=True, also cache true-class scores s_y(x) (needed by rw_lv_bas_bin).
    """
    cal_pct = int(round(100 * float(cal_fraction)))
    cache_dir = (
        Path(cache_root)
        / f"{dataset_tag}__hash{int(n_features)}__ng{int(ngram_range[0])}-{int(ngram_range[1])}__split{int(split_seed)}__cal{cal_pct}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    mu_path = cache_dir / "mu.npy"
    std_path = cache_dir / "std.npy"

    suffix = _ridge_suffix(float(ridge))
    train_s = cache_dir / f"train.s{suffix}.npy"
    cal_s = cache_dir / f"cal.s{suffix}.npy"
    val_s = cache_dir / f"val.s{suffix}.npy"
    test_s = cache_dir / f"test.s{suffix}.npy"

    vec = make_vectorizer(n_features=n_features, ngram_range=ngram_range)
    # ---------------------------------------------------------------------
    # Concurrency safety: make cache writes atomic.
    # This prevents corrupted .npy files when multiple jobs write the same
    # cache_dir concurrently (e.g. parallel eps/replication sweeps).
    #
    # Implementation: write to a unique temp file in the same directory and
    # then os.replace() to the final path (atomic on POSIX filesystems).
    # ---------------------------------------------------------------------
    _save_raw = globals()["_save"]

    def _save_atomic(path, arr):
        tmp = path.parent / f"{path.stem}.tmp_{uuid.uuid4().hex}{path.suffix}"
        _save_raw(tmp, arr)
        os.replace(tmp, path)

    _save = _save_atomic  # local override: all _save(...) calls below become atomic

    if (not mu_path.exists()) or (not std_path.exists()):
        if verbose:
            print(f"[text_pipeline] Fitting diag-Gaussian moments -> {mu_path.parent}")
        mu, std = fit_classcond_diag_gaussian_text(
            splits["train"],
            vec,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            desc="fit mu/std (train)",
            n_classes=n_classes,
        )
        _save(mu_path, mu)
        _save(std_path, std)

    mu = _load(mu_path).astype(np.float32)
    std = _load(std_path).astype(np.float32)

    # Min-over-class scores
    if not train_s.exists():
        _save(
            train_s,
            scores_classcond_diag_mahalanobis_text(
                splits["train"],
                vec,
                mu,
                std,
                ridge=float(ridge),
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
                desc=f"scores train{suffix}",
            ),
        )
    if not cal_s.exists():
        _save(
            cal_s,
            scores_classcond_diag_mahalanobis_text(
                splits["cal"],
                vec,
                mu,
                std,
                ridge=float(ridge),
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
                desc=f"scores cal{suffix}",
            ),
        )
    if not val_s.exists():
        _save(
            val_s,
            scores_classcond_diag_mahalanobis_text(
                splits["val"],
                vec,
                mu,
                std,
                ridge=float(ridge),
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
                desc=f"scores val{suffix}",
            ),
        )
    if not test_s.exists():
        _save(
            test_s,
            scores_classcond_diag_mahalanobis_text(
                splits["test"],
                vec,
                mu,
                std,
                ridge=float(ridge),
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
                desc=f"scores test{suffix}",
            ),
        )

    train_s_true: Optional[Path] = None
    cal_s_true: Optional[Path] = None
    val_s_true: Optional[Path] = None
    test_s_true: Optional[Path] = None

    if need_trueclass_scores:
        train_s_true = cache_dir / f"train.s_true{suffix}.npy"
        cal_s_true = cache_dir / f"cal.s_true{suffix}.npy"
        val_s_true = cache_dir / f"val.s_true{suffix}.npy"
        test_s_true = cache_dir / f"test.s_true{suffix}.npy"

        if not train_s_true.exists():
            _save(
                train_s_true,
                scores_trueclass_diag_mahalanobis_text(
                    splits["train"],
                    vec,
                    mu,
                    std,
                    ridge=float(ridge),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    verbose=verbose,
                    desc=f"scores train true{suffix}",
                ),
            )
        if not cal_s_true.exists():
            _save(
                cal_s_true,
                scores_trueclass_diag_mahalanobis_text(
                    splits["cal"],
                    vec,
                    mu,
                    std,
                    ridge=float(ridge),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    verbose=verbose,
                    desc=f"scores cal true{suffix}",
                ),
            )
        if not val_s_true.exists():
            _save(
                val_s_true,
                scores_trueclass_diag_mahalanobis_text(
                    splits["val"],
                    vec,
                    mu,
                    std,
                    ridge=float(ridge),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    verbose=verbose,
                    desc=f"scores val true{suffix}",
                ),
            )
        if not test_s_true.exists():
            _save(
                test_s_true,
                scores_trueclass_diag_mahalanobis_text(
                    splits["test"],
                    vec,
                    mu,
                    std,
                    ridge=float(ridge),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    verbose=verbose,
                    desc=f"scores test true{suffix}",
                ),
            )

    return TextScoreCache(
        cache_dir=cache_dir,
        mu_path=mu_path,
        std_path=std_path,
        train_s=train_s,
        cal_s=cal_s,
        val_s=val_s,
        test_s=test_s,
        train_s_true=train_s_true,
        cal_s_true=cal_s_true,
        val_s_true=val_s_true,
        test_s_true=test_s_true,
    )


def _open_memmap_f32(path: Path, shape: Tuple[int, int]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=shape)


def cache_dense_hashed_features(
    dataset,
    vectorizer: HashingVectorizer,
    *,
    batch_size: int,
    num_workers: int,
    X_path: Path,
    y_path: Path,
    g_path: Path,
    verbose: bool,
    desc: str,
) -> None:
    """
    Materialise HashingVectorizer features to a dense float32 memmap (N, D),
    plus y/g int64 arrays, in dataset order (shuffle=False).
    """
    n = len(dataset)
    D = int(vectorizer.n_features)

    X = _open_memmap_f32(X_path, shape=(n, D))
    y = np.empty((n,), dtype=np.int64)
    g = np.empty((n,), dtype=np.int64)

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_text_only,
        pin_memory=False,
    )

    it = loader
    if verbose:
        from tqdm import tqdm  # type: ignore
        it = tqdm(loader, desc=desc, leave=False)

    cur = 0
    for texts, yb, gb, _idx in it:
        Xb = vectorizer.transform(texts).toarray().astype(np.float32, copy=False)
        b = int(Xb.shape[0])
        X[cur : cur + b] = Xb
        y[cur : cur + b] = yb.numpy().astype(np.int64, copy=False)
        g[cur : cur + b] = gb.numpy().astype(np.int64, copy=False)
        cur += b

    np.save(y_path, y)
    np.save(g_path, g)


@dataclass(frozen=True)
class CachedTextFeatures:
    X_path: Path
    y_path: Path
    g_path: Path

    def load(self):
        return (
            np.load(self.X_path, mmap_mode="r"),
            np.load(self.y_path),
            np.load(self.g_path),
        )


@dataclass(frozen=True)
class TextFeatureCache:
    cache_dir: Path
    train: CachedTextFeatures
    cal: CachedTextFeatures
    val: CachedTextFeatures
    test: CachedTextFeatures


def ensure_text_feature_cache(
    *,
    dataset_tag: str,
    splits: Dict[str, object],   # train/cal/val/test
    cache_root: Path,
    n_features: int,
    ngram_range: Tuple[int, int],
    split_seed: int,
    cal_fraction: float,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> TextFeatureCache:
    """
    Cache dense hashed features for train/cal/val/test so all algorithms can train/eval
    without paying per-batch HashingVectorizer cost.
    """
    cal_pct = int(round(100 * float(cal_fraction)))
    cache_dir = (
        Path(cache_root)
        / "text_features"
        / f"{dataset_tag}__hash{int(n_features)}__ng{int(ngram_range[0])}-{int(ngram_range[1])}__split{int(split_seed)}__cal{cal_pct}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    vec = make_vectorizer(n_features=int(n_features), ngram_range=ngram_range)

    out: Dict[str, CachedTextFeatures] = {}
    for split_name, ds in splits.items():
        Xp = cache_dir / f"{split_name}.X.npy"
        yp = cache_dir / f"{split_name}.y.npy"
        gp = cache_dir / f"{split_name}.g.npy"
        out[split_name] = CachedTextFeatures(Xp, yp, gp)

        if not (Xp.exists() and yp.exists() and gp.exists()):
            cache_dense_hashed_features(
                ds,
                vec,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                X_path=Xp,
                y_path=yp,
                g_path=gp,
                verbose=bool(verbose),
                desc=f"cache dense features: {dataset_tag}/{split_name}",
            )

    return TextFeatureCache(
        cache_dir=cache_dir,
        train=out["train"],
        cal=out["cal"],
        val=out["val"],
        test=out["test"],
    )
