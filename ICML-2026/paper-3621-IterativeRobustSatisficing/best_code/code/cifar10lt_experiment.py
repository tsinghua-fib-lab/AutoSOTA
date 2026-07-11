from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

try:
    from .methods.chi2_dro import Chi2DROConfig, Chi2DROTrainer
    from .methods.cvar_dro import CVaRDROConfig, CVaRDROTrainer
    from .methods.erm import ERMConfig, ERMTrainer
    from .methods.erm_adam import ERMAdamConfig, ERMAdamTrainer
    from .methods.groupdro import GroupDROConfig, GroupDROTrainer
    from .methods.irm import IRMConfig, IRMTrainer
    from .methods.irs import IRSConfig, IRSTrainer
    from .methods.klrs import KLRSConfig, KLRSTrainer
    from .methods.mmrex import MMREXConfig, MMREXTrainer
    from .methods.sam import SAMConfig, SAMTrainer
    from .methods.vrex import VREXConfig, VREXTrainer
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from methods.chi2_dro import Chi2DROConfig, Chi2DROTrainer
    from methods.cvar_dro import CVaRDROConfig, CVaRDROTrainer
    from methods.erm import ERMConfig, ERMTrainer
    from methods.erm_adam import ERMAdamConfig, ERMAdamTrainer
    from methods.groupdro import GroupDROConfig, GroupDROTrainer
    from methods.irm import IRMConfig, IRMTrainer
    from methods.irs import IRSConfig, IRSTrainer
    from methods.klrs import KLRSConfig, KLRSTrainer
    from methods.mmrex import MMREXConfig, MMREXTrainer
    from methods.sam import SAMConfig, SAMTrainer
    from methods.vrex import VREXConfig, VREXTrainer


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CIFAR10LTConfig:
    # Dataset.
    data_dir: str = "./data"
    # Paper CIFAR-10-LT imbalance factor, max_count / min_count.
    # Values in (0, 1] are also accepted for backward compatibility as
    # min_count / max_count ratios, so 0.01 and 100 define the same LT split.
    imbalance_factor: float = 100.0 # 1, 20, 50, 100

    # Shared training.
    seeds: Tuple[int, ...] = (42,)
    batch_size: int = 512
    val_fraction: float = 0.1
    weight_decay: float = 0.0
    warmup_epochs: int = 4
    model_arch: str = "wrn28_10"  # "resnet18" or "wrn28_10"
    use_if_hyperparams: bool = True

    # IRS.
    irs_epochs: int = 100
    irs_lr: float = 1e-3
    tau_multiplier: float = 1.01
    target_tau: float = 0.1
    h_min: float = 1e-3
    h_max: float = 50.0
    h_grid_points: int = 64
    refine_rounds: int = 3

    # KL-RS.
    klrs_lr: float = 1e-3
    klrs_inner_epochs: int = 20
    klrs_max_bisect_steps: int = 5
    klrs_lambda_init: float = 1.0
    klrs_lambda_bisect_tol: float = 0.0

    # Old-code baselines and robust methods.
    baseline_epochs: int = 100
    baseline_weight_decay: float = 0.0
    erm_lr: float = 0.01
    erm_adam_lr: float = 1e-3
    sam_lr: float = 1e-3
    sam_rho: float = 0.05
    vrex_lr: float = 1e-3
    vrex_beta: float = 0.5
    mmrex_lr: float = 1e-3
    mmrex_lambda: float = 1.5
    rex_penalty_anneal_epochs: int = 10
    irm_lr: float = 1e-3
    irm_penalty_weight: float = 10.0
    irm_penalty_anneal_epochs: int = 20
    groupdro_lr: float = 1e-3
    groupdro_eta_q: float = 0.1
    groupdro_gamma: float = 0.1
    chi2_lr: float = 1e-3
    chi2_rho: float = 0.01
    cvar_lr: float = 1e-3
    cvar_alpha: float = 0.5
    amp: bool = True

    # Output.
    output_dir: str = "./cifar10lt_results"
    device: str = field(default_factory=_default_device)

    # Method keys registered below. Add future algorithms there, then list them here.
    methods: Tuple[str, ...] = (
        "irs",
        "klrs",
        "erm",
        "erm_adam",
        "sam",
        "vrex",
        "mmrex",
        "irm",
        "groupdro",
        "chi2_dro",
        "cvar_dro",
    )


IF_HYPERPARAMS: Dict[int, Dict[str, float]] = {
    1: {
        "erm_lr": 1e-3,
        "erm_adam_lr": 1e-3,
        "sam_lr": 1e-3,
        "irs_lr": 1e-3,
        "klrs_lr": 1e-3,
        "vrex_lr": 1e-3,
        "mmrex_lr": 1e-3,
        "irm_lr": 1e-4,
        "groupdro_lr": 1e-3,
        "chi2_lr": 1e-3,
        "cvar_lr": 1e-3,
        "sam_rho": 0.05,
        "klrs_max_bisect_steps": 5,
        "vrex_beta": 1.0,
        "mmrex_lambda": 2.0,
        "irm_penalty_weight": 100.0,
        "groupdro_eta_q": 0.1,
        "chi2_rho": 1.0,
        "cvar_alpha": 0.3,
    },
    20: {
        "erm_lr": 1e-3,
        "erm_adam_lr": 1e-3,
        "sam_lr": 1e-3,
        "irs_lr": 1e-3,
        "klrs_lr": 1e-3,
        "vrex_lr": 1e-3,
        "mmrex_lr": 1e-3,
        "irm_lr": 1e-4,
        "groupdro_lr": 1e-3,
        "chi2_lr": 1e-3,
        "cvar_lr": 1e-3,
        "sam_rho": 0.05,
        "klrs_max_bisect_steps": 10,
        "vrex_beta": 1.0,
        "mmrex_lambda": 1.0,
        "irm_penalty_weight": 100.0,
        "groupdro_eta_q": 0.5,
        "chi2_rho": 0.1,
        "cvar_alpha": 0.5,
    },
    50: {
        "erm_lr": 1e-3,
        "erm_adam_lr": 1e-3,
        "sam_lr": 1e-3,
        "irs_lr": 1e-3,
        "klrs_lr": 1e-3,
        "vrex_lr": 1e-3,
        "mmrex_lr": 1e-3,
        "irm_lr": 1e-4,
        "groupdro_lr": 1e-3,
        "chi2_lr": 1e-3,
        "cvar_lr": 1e-3,
        "sam_rho": 0.05,
        "klrs_max_bisect_steps": 5,
        "vrex_beta": 0.5,
        "mmrex_lambda": 1.5,
        "irm_penalty_weight": 100.0,
        "groupdro_eta_q": 0.1,
        "chi2_rho": 0.01,
        "cvar_alpha": 0.5,
    },
    100: {
        "erm_lr": 1e-3,
        "erm_adam_lr": 1e-3,
        "sam_lr": 1e-3,
        "irs_lr": 1e-3,
        "klrs_lr": 1e-4,
        "vrex_lr": 1e-3,
        "mmrex_lr": 1e-3,
        "irm_lr": 1e-4,
        "groupdro_lr": 1e-3,
        "chi2_lr": 1e-3,
        "cvar_lr": 1e-3,
        "sam_rho": 0.05,
        "klrs_max_bisect_steps": 15,
        "vrex_beta": 0.5,
        "mmrex_lambda": 1.5,
        "irm_penalty_weight": 100.0,
        "groupdro_eta_q": 0.2,
        "chi2_rho": 0.5,
        "cvar_alpha": 0.5,
    },
}


def paper_if_from_imbalance_factor(imbalance_factor: float) -> int:
    """
    Return paper IF = max_count / min_count.

    The current camera-ready config uses the paper convention by default
    (e.g. 100). Older code used the inverse ratio (e.g. 0.01), so values in
    (0, 1] are converted for compatibility.
    """
    value = float(imbalance_factor)
    if value <= 0:
        raise ValueError("imbalance_factor must be positive.")
    paper_if = value if value >= 1.0 else 1.0 / value
    return int(round(paper_if))


def lt_ratio_from_imbalance_factor(imbalance_factor: float) -> float:
    """Return min_count / max_count ratio used by the LT sampling formula."""
    value = float(imbalance_factor)
    if value <= 0:
        raise ValueError("imbalance_factor must be positive.")
    return 1.0 / value if value >= 1.0 else value


def apply_if_hyperparams(cfg: CIFAR10LTConfig) -> int:
    paper_if = paper_if_from_imbalance_factor(cfg.imbalance_factor)
    if not cfg.use_if_hyperparams:
        return paper_if
    if paper_if not in IF_HYPERPARAMS:
        available = ", ".join(str(k) for k in sorted(IF_HYPERPARAMS))
        raise ValueError(
            f"No CIFAR-10-LT hyperparameter preset for IF={paper_if}. "
            f"Available presets: {available}."
        )
    for name, value in IF_HYPERPARAMS[paper_if].items():
        setattr(cfg, name, value)
    return paper_if


def make_irs_config(cfg: CIFAR10LTConfig) -> IRSConfig:
    return IRSConfig(
        epochs=cfg.irs_epochs,
        lr=cfg.irs_lr,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        tau_multiplier=cfg.tau_multiplier,
        target_tau=cfg.target_tau,
        h_min=cfg.h_min,
        h_max=cfg.h_max,
        h_grid_points=cfg.h_grid_points,
        refine_rounds=cfg.refine_rounds,
    )


def make_klrs_config(cfg: CIFAR10LTConfig) -> KLRSConfig:
    kwargs = {
        "lr": cfg.klrs_lr,
        "weight_decay": cfg.weight_decay,
        "warmup_epochs": cfg.warmup_epochs,
        "target_tau": cfg.target_tau,
        "inner_epochs": cfg.klrs_inner_epochs,
        "max_bisect_steps": cfg.klrs_max_bisect_steps,
        "lambda_init": cfg.klrs_lambda_init,
        "lambda_bisect_tol": cfg.klrs_lambda_bisect_tol,
    }
    if "total_epochs" in getattr(KLRSConfig, "__dataclass_fields__", {}):
        kwargs["total_epochs"] = cfg.baseline_epochs
    return KLRSConfig(**kwargs)


def make_erm_config(cfg: CIFAR10LTConfig) -> ERMConfig:
    return ERMConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.erm_lr,
        weight_decay=cfg.baseline_weight_decay,
        amp=cfg.amp,
    )


def make_erm_adam_config(cfg: CIFAR10LTConfig) -> ERMAdamConfig:
    return ERMAdamConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.erm_adam_lr,
        weight_decay=cfg.baseline_weight_decay,
        amp=cfg.amp,
    )


def make_sam_config(cfg: CIFAR10LTConfig) -> SAMConfig:
    return SAMConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.sam_lr,
        rho=cfg.sam_rho,
        weight_decay=cfg.baseline_weight_decay,
        amp=cfg.amp,
    )


def make_vrex_config(cfg: CIFAR10LTConfig) -> VREXConfig:
    return VREXConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.vrex_lr,
        weight_decay=cfg.baseline_weight_decay,
        rex_beta=cfg.vrex_beta,
        penalty_anneal_epochs=cfg.rex_penalty_anneal_epochs,
        amp=cfg.amp,
    )


def make_mmrex_config(cfg: CIFAR10LTConfig) -> MMREXConfig:
    return MMREXConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.mmrex_lr,
        weight_decay=cfg.baseline_weight_decay,
        rex_lambda=cfg.mmrex_lambda,
        penalty_anneal_epochs=cfg.rex_penalty_anneal_epochs,
        amp=cfg.amp,
    )


def make_irm_config(cfg: CIFAR10LTConfig) -> IRMConfig:
    return IRMConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.irm_lr,
        weight_decay=cfg.baseline_weight_decay,
        penalty_weight=cfg.irm_penalty_weight,
        penalty_anneal_epochs=cfg.irm_penalty_anneal_epochs,
        amp=cfg.amp,
    )


def make_groupdro_config(cfg: CIFAR10LTConfig) -> GroupDROConfig:
    return GroupDROConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.groupdro_lr,
        weight_decay=cfg.baseline_weight_decay,
        eta_q=cfg.groupdro_eta_q,
        gamma=cfg.groupdro_gamma,
        amp=cfg.amp,
    )


def make_chi2_dro_config(cfg: CIFAR10LTConfig) -> Chi2DROConfig:
    return Chi2DROConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.chi2_lr,
        rho=cfg.chi2_rho,
        weight_decay=cfg.baseline_weight_decay,
        amp=cfg.amp,
    )


def make_cvar_dro_config(cfg: CIFAR10LTConfig) -> CVaRDROConfig:
    return CVaRDROConfig(
        epochs=cfg.baseline_epochs,
        lr=cfg.cvar_lr,
        alpha=cfg.cvar_alpha,
        weight_decay=cfg.baseline_weight_decay,
        amp=cfg.amp,
    )


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    output_stem: str
    trainer_cls: Callable
    config_factory: Callable[[CIFAR10LTConfig], object]
    needs_env_loaders: bool = False
    needs_group_counts: bool = False


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "irs": MethodSpec(
        key="irs",
        display_name="IRS",
        output_stem="irs",
        trainer_cls=IRSTrainer,
        config_factory=make_irs_config,
    ),
    "klrs": MethodSpec(
        key="klrs",
        display_name="KL-RS",
        output_stem="klrs",
        trainer_cls=KLRSTrainer,
        config_factory=make_klrs_config,
    ),
    "erm": MethodSpec(
        key="erm",
        display_name="ERM",
        output_stem="erm",
        trainer_cls=ERMTrainer,
        config_factory=make_erm_config,
    ),
    "erm_adam": MethodSpec(
        key="erm_adam",
        display_name="ERM-Adam",
        output_stem="erm_adam",
        trainer_cls=ERMAdamTrainer,
        config_factory=make_erm_adam_config,
    ),
    "sam": MethodSpec(
        key="sam",
        display_name="SAM",
        output_stem="sam",
        trainer_cls=SAMTrainer,
        config_factory=make_sam_config,
    ),
    "vrex": MethodSpec(
        key="vrex",
        display_name="V-REx",
        output_stem="vrex",
        trainer_cls=VREXTrainer,
        config_factory=make_vrex_config,
        needs_env_loaders=True,
    ),
    "mmrex": MethodSpec(
        key="mmrex",
        display_name="MM-REx",
        output_stem="mmrex",
        trainer_cls=MMREXTrainer,
        config_factory=make_mmrex_config,
        needs_env_loaders=True,
    ),
    "irm": MethodSpec(
        key="irm",
        display_name="IRMv1",
        output_stem="irm",
        trainer_cls=IRMTrainer,
        config_factory=make_irm_config,
        needs_env_loaders=True,
    ),
    "groupdro": MethodSpec(
        key="groupdro",
        display_name="GroupDRO",
        output_stem="groupdro",
        trainer_cls=GroupDROTrainer,
        config_factory=make_groupdro_config,
        needs_group_counts=True,
    ),
    "chi2_dro": MethodSpec(
        key="chi2_dro",
        display_name="Chi2-DRO",
        output_stem="chi2_dro",
        trainer_cls=Chi2DROTrainer,
        config_factory=make_chi2_dro_config,
    ),
    "cvar_dro": MethodSpec(
        key="cvar_dro",
        display_name="CVaR-DRO",
        output_stem="cvar_dro",
        trainer_cls=CVaRDROTrainer,
        config_factory=make_cvar_dro_config,
    ),
}


def selected_method_specs(method_keys: Tuple[str, ...]) -> Tuple[MethodSpec, ...]:
    unknown = [key for key in method_keys if key not in METHOD_REGISTRY]
    if unknown:
        known = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown method key(s): {unknown}. Available: {known}")
    return tuple(METHOD_REGISTRY[key] for key in method_keys)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=7),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

TEST_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)


class LabelledSubset(Dataset):
    """
    Wraps CIFAR-10 and returns (x, y, g, env) tuples.

    Here g = y = class label, and env = 0 because CIFAR-10-LT uses one
    training environment.
    """

    def __init__(self, cifar_dataset, indices: np.ndarray):
        self.dataset = cifar_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        g = torch.tensor(y, dtype=torch.long)
        env = torch.tensor(0, dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.long), g, env


class EnvironmentSubset(Dataset):
    """
    Dataset wrapper for the old REx/IRM environment loaders.
    """

    def __init__(self, cifar_dataset, indices: np.ndarray, env_id: int):
        self.dataset = cifar_dataset
        self.indices = indices
        self.env_id = env_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        g = torch.tensor(y, dtype=torch.long)
        env = torch.tensor(self.env_id, dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.long), g, env


def make_cifar10lt(
    data_dir: str,
    imbalance_factor: float,
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Tuple[LabelledSubset, LabelledSubset, LabelledSubset, np.ndarray]:
    """
    Build CIFAR-10-LT training/validation sets and balanced CIFAR-10 test set.

    Training uses geometric long-tail down-sampling:
        n_c = n_max * ratio ** (c / (C - 1)),
    where ratio = min_count / max_count. For convenience this function accepts
    either the paper IF convention (100) or the inverse ratio convention (0.01).
    """
    rng = np.random.default_rng(seed)
    lt_ratio = lt_ratio_from_imbalance_factor(imbalance_factor)

    base_train = torchvision.datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=TRAIN_TRANSFORM,
    )
    base_test = torchvision.datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=TEST_TRANSFORM,
    )

    targets = np.array(base_train.targets)
    n_classes = 10
    class_counts_orig = np.bincount(targets, minlength=n_classes)
    n_max = int(class_counts_orig.max())

    lt_indices: List[int] = []

    for c in range(n_classes):
        n_c = max(int(n_max * (lt_ratio ** (c / (n_classes - 1)))), 1)
        idx_c = np.where(targets == c)[0]
        idx_c = rng.permutation(idx_c)[:n_c]
        lt_indices.extend(idx_c.tolist())

    lt_indices = np.array(lt_indices, dtype=np.int64)

    split_rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for c in range(n_classes):
        class_indices = lt_indices[targets[lt_indices] == c]
        split_rng.shuffle(class_indices)
        split = int((1.0 - val_fraction) * len(class_indices))
        train_indices.extend(class_indices[:split].tolist())
        val_indices.extend(class_indices[split:].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    class_counts = np.bincount(targets[train_indices], minlength=n_classes).astype(np.int64)

    train_dataset = LabelledSubset(base_train, train_indices)
    val_dataset = LabelledSubset(base_train, val_indices)
    test_dataset = LabelledSubset(base_test, np.arange(len(base_test)))

    return train_dataset, val_dataset, test_dataset, class_counts


def build_irm_environments_probabilistic(
    train_dataset: LabelledSubset,
    seed: int,
) -> Tuple[EnvironmentSubset, EnvironmentSubset]:
    """
    Split the LT training subset into the two probabilistic environments used
    by the old CIFAR-10-LT code.
    """
    rng = np.random.default_rng(seed)
    base_dataset = train_dataset.dataset
    lt_indices = train_dataset.indices
    targets_full = np.array(base_dataset.targets, dtype=np.int64)
    targets_lt = targets_full[lt_indices]

    n_classes = int(targets_lt.max()) + 1
    class_indices = [lt_indices[targets_lt == c] for c in range(n_classes)]
    class_counts = np.array([len(indices) for indices in class_indices], dtype=np.int64)

    def geometric_sample(indices, p, size):
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        sampled = []
        for _ in range(size):
            idx = rng.integers(0, len(indices))
            while rng.random() < p and len(indices) > 1:
                idx = rng.integers(0, len(indices))
            sampled.append(indices[idx])
        return np.array(sampled, dtype=np.int64)

    env1_indices = []
    for c in range(n_classes):
        env1_indices.extend(geometric_sample(class_indices[c], p=0.7, size=class_counts[c]))

    env2_indices = []
    for c in range(n_classes):
        env2_indices.extend(geometric_sample(class_indices[c], p=0.3, size=class_counts[c]))

    env1_ds = EnvironmentSubset(base_dataset, np.array(env1_indices, dtype=np.int64), env_id=1)
    env2_ds = EnvironmentSubset(base_dataset, np.array(env2_indices, dtype=np.int64), env_id=2)
    return env1_ds, env2_ds


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate
        self.shortcut = (
            None
            if self.equal_in_out
            else nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        residual = x if self.equal_in_out else self.shortcut(x)
        return out + residual


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            block_stride = stride if i == 0 else 1
            layers.append(block(in_planes, out_planes, block_stride, drop_rate))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WRN depth should be 6n+4.")
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_stages[0], n_stages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, n_stages[1], n_stages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, n_stages[2], n_stages[3], BasicBlock, 2, drop_rate)
        self.bn = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_stages[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_model(n_classes: int = 10, arch: str = "resnet18") -> nn.Module:
    """
    Build the CIFAR-10-LT backbone.

    Supported architectures:
      - "resnet18": rebuttal code backbone, adapted for 32x32 inputs.
      - "wrn28_10": original CIFAR-10-LT notebook backbone.
    """
    arch_key = arch.lower().replace("-", "_")
    if arch_key in {"wrn28_10", "wide_resnet_28_10"}:
        return WideResNet(depth=28, widen_factor=10, num_classes=n_classes, drop_rate=0.0)
    if arch_key not in {"resnet18", "resnet_18"}:
        raise ValueError(f"Unknown model_arch={arch!r}. Use 'resnet18' or 'wrn28_10'.")

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, n_classes)
    return model


def build_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, train_eval_loader, val_loader, test_loader


def build_environment_loaders(
    train_dataset: LabelledSubset,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    env1_ds, env2_ds = build_irm_environments_probabilistic(train_dataset, seed=seed)
    env_batch_size = max(batch_size // 2, 1)
    env_loader1 = DataLoader(
        env1_ds,
        batch_size=env_batch_size,
        shuffle=True,
        drop_last=False,
    )
    env_loader2 = DataLoader(
        env2_ds,
        batch_size=env_batch_size,
        shuffle=True,
        drop_last=False,
    )
    return env_loader1, env_loader2


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 10,
) -> Dict[str, float]:
    """
    Compute loss and accuracy metrics on a DataLoader.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_n = 0
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)
    loss_per_class = np.zeros(n_classes, dtype=np.float64)

    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_vec = criterion(logits, y)
        total_loss += loss_vec.sum().item()
        total_n += len(y)
        preds = logits.argmax(dim=1)
        for c in range(n_classes):
            mask = y == c
            correct_per_class[c] += (preds[mask] == c).sum().item()
            total_per_class[c] += mask.sum().item()
            if mask.any():
                loss_per_class[c] += loss_vec[mask].sum().item()

    per_class_acc = np.where(
        total_per_class > 0,
        correct_per_class / total_per_class,
        np.nan,
    )
    per_class_loss = np.where(
        total_per_class > 0,
        loss_per_class / total_per_class,
        np.nan,
    )
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": correct_per_class.sum() / max(total_n, 1),
        "balanced_acc": float(np.nanmean(per_class_acc)),
        "worst_acc": float(np.nanmin(per_class_acc)),
        "per_class_acc": per_class_acc,
        "per_class_loss": per_class_loss,
        "classwise_acc": " ".join(
            f"{c}:{acc:.3f}" for c, acc in enumerate(per_class_acc)
        ),
    }


def _get_class_names_from_dataset(dataset: LabelledSubset, n_classes: int) -> List[str]:
    """
    Get class names from torchvision CIFAR-10 if available.
    """
    return list(getattr(dataset.dataset, "classes", [f"class_{i}" for i in range(n_classes)]))


def make_class_count_table(
    class_counts: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    total = max(int(class_counts.sum()), 1)

    return pd.DataFrame(
        {
            "class_id": np.arange(len(class_counts)),
            "class_name": class_names,
            "train_count": class_counts.astype(int),
            "train_fraction": class_counts / total,
        }
    )


@torch.no_grad()
def make_classwise_results_table(
    model: nn.Module,
    loader: DataLoader,
    class_counts: np.ndarray,
    class_names: List[str],
    prefix: str,
    device: torch.device,
    n_classes: int = 10,
) -> pd.DataFrame:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)
    loss_sum_per_class = np.zeros(n_classes, dtype=np.float64)

    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_vec = criterion(logits, y)
        preds = logits.argmax(dim=1)

        for c in range(n_classes):
            mask = y == c
            n_c = mask.sum().item()
            if n_c > 0:
                total_per_class[c] += n_c
                correct_per_class[c] += (preds[mask] == c).sum().item()
                loss_sum_per_class[c] += loss_vec[mask].sum().item()

    per_class_acc = np.where(
        total_per_class > 0,
        correct_per_class / total_per_class,
        np.nan,
    )
    per_class_loss = np.where(
        total_per_class > 0,
        loss_sum_per_class / total_per_class,
        np.nan,
    )

    return pd.DataFrame(
        {
            "class_id": np.arange(n_classes),
            "class_name": class_names,
            "train_count": class_counts.astype(int),
            "test_count": total_per_class.astype(int),
            f"{prefix}_correct": correct_per_class.astype(int),
            f"{prefix}_acc": per_class_acc,
            f"{prefix}_loss": per_class_loss,
        }
    )


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n  {title}")
    print(
        df.to_string(
            index=False,
            formatters={
                "train_fraction": lambda x: f"{x:.4f}",
                "IRS_acc": lambda x: f"{x:.4f}",
                "IRS_loss": lambda x: f"{x:.4f}",
                "KLRS_acc": lambda x: f"{x:.4f}",
                "KLRS_loss": lambda x: f"{x:.4f}",
            },
        )
    )


TAIL_CLASS_NAMES = ("horse", "ship", "truck")
TAIL_CLASS_IDS = (7, 8, 9)


def tail_accuracy_from_classwise(df: pd.DataFrame, acc_col: str) -> float:
    names = df["class_name"].astype(str).str.lower()
    tail_mask = names.isin(TAIL_CLASS_NAMES)
    if not tail_mask.any():
        tail_mask = df["class_id"].isin(TAIL_CLASS_IDS)
    return float(np.nanmean(df.loc[tail_mask, acc_col]))


def print_classwise_results(df: pd.DataFrame, prefix: str, title: str) -> None:
    acc_col = f"{prefix}_acc"
    loss_col = f"{prefix}_loss"
    shown = df[
        [
            "class_id",
            "class_name",
            "train_count",
            "test_count",
            acc_col,
            loss_col,
        ]
    ]
    print(f"\n  {title}")
    print(
        shown.to_string(
            index=False,
            formatters={
                acc_col: lambda x: f"{x:.4f}",
                loss_col: lambda x: f"{x:.4f}",
            },
        )
    )
    tail_acc = tail_accuracy_from_classwise(df, acc_col)
    print(f"  Tail acc ({', '.join(TAIL_CLASS_NAMES)}): {tail_acc:.4f}")


def _format_final_metrics(method_key: str, final_row: pd.Series) -> str:
    fragility = ""
    if method_key == "irs" and "kappa" in final_row:
        fragility = f"  kappa={final_row['kappa']:.4f}"
    elif method_key == "klrs" and "lam" in final_row:
        fragility = f"  lambda={final_row['lam']:.4f}"

    return (
        f"test_acc={final_row['test_acc']:.3f}  "
        f"bal_acc={final_row['test_balanced_acc']:.3f}"
        f"{fragility}"
    )


def _format_active_hyperparams(cfg: CIFAR10LTConfig) -> str:
    return (
        "LRs: "
        f"IRS={cfg.irs_lr:g}, KLRS={cfg.klrs_lr:g}, ERM={cfg.erm_lr:g}, "
        f"SAM={cfg.sam_lr:g}, VREX={cfg.vrex_lr:g}, MMREX={cfg.mmrex_lr:g}, "
        f"IRM={cfg.irm_lr:g}, GroupDRO={cfg.groupdro_lr:g}, "
        f"Chi2={cfg.chi2_lr:g}, CVaR={cfg.cvar_lr:g} | "
        "params: "
        f"SAM rho={cfg.sam_rho:g}, KLRS steps={cfg.klrs_max_bisect_steps}, "
        f"KLRS epochs={cfg.baseline_epochs}, "
        f"VREX beta={cfg.vrex_beta:g}, MMREX lambda={cfg.mmrex_lambda:g}, "
        f"IRM pen={cfg.irm_penalty_weight:g}, GroupDRO eta_q={cfg.groupdro_eta_q:g}, "
        f"Chi2 rho={cfg.chi2_rho:g}, CVaR alpha={cfg.cvar_alpha:g}"
    )


class CIFAR10LTRunner:
    def __init__(self, cfg: CIFAR10LTConfig):
        self.cfg = cfg
        self.method_specs = selected_method_specs(cfg.methods)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(cfg.output_dir) / ts
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.histories: Dict[str, List[pd.DataFrame]] = {
            spec.key: [] for spec in self.method_specs
        }
        self.irs_histories = self.histories.get("irs", [])
        self.klrs_histories = self.histories.get("klrs", [])

    def run(self) -> None:
        cfg = self.cfg
        device = torch.device(cfg.device)
        paper_if = apply_if_hyperparams(cfg)
        lt_ratio = lt_ratio_from_imbalance_factor(cfg.imbalance_factor)

        print("=" * 80)
        print(
            f"CIFAR-10-LT  |  methods={cfg.methods}  |  "
            f"IF={paper_if}  |  LT ratio={lt_ratio:g}"
        )
        print(f"Seeds: {cfg.seeds}   Device: {device}   Model: {cfg.model_arch}")
        if cfg.use_if_hyperparams:
            print(f"Hyperparameters: preset for IF={paper_if}")
        else:
            print("Hyperparameters: explicit config values")
        print(_format_active_hyperparams(cfg))
        print(f"Output: {self.output_dir}")
        print("=" * 80)

        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        for seed in cfg.seeds:
            print(f"\n{'-' * 80}")
            print(f"Seed {seed}")
            print(f"{'-' * 80}")
            set_seed(seed)

            train_ds, val_ds, test_ds, class_counts = make_cifar10lt(
                cfg.data_dir,
                cfg.imbalance_factor,
                seed=seed,
                val_fraction=cfg.val_fraction,
            )
            n_classes = len(class_counts)
            class_names = _get_class_names_from_dataset(train_ds, n_classes=n_classes)
            class_count_df = make_class_count_table(class_counts, class_names)
            print_table(class_count_df, "Train class counts by class")
            class_count_df.to_csv(
                self.output_dir / f"class_counts_seed{seed}.csv",
                index=False,
            )
            print(f"  Val size: {len(val_ds)}   Test size: {len(test_ds)}")

            train_loader, train_eval_loader, val_loader, test_loader = build_loaders(
                train_ds,
                val_ds,
                test_ds,
                cfg.batch_size,
            )
            env_loaders = None
            if any(spec.needs_env_loaders for spec in self.method_specs):
                env_loaders = build_environment_loaders(train_ds, cfg.batch_size, seed=seed)
                print(
                    f"  Env loaders: env1={len(env_loaders[0].dataset)}  "
                    f"env2={len(env_loaders[1].dataset)}"
                )
            reference_prior = torch.tensor(
                class_counts / class_counts.sum(),
                dtype=torch.float32,
            )

            def eval_fn(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
                return evaluate(model, loader, device=device, n_classes=n_classes)

            for spec in self.method_specs:
                print(f"\n  [{spec.display_name}] training...")
                set_seed(seed)
                model = build_model(n_classes=n_classes, arch=cfg.model_arch)
                trainer_kwargs = {
                    "cfg": spec.config_factory(cfg),
                    "model": model,
                    "train_loader": train_loader,
                    "train_eval_loader": train_eval_loader,
                    "test_loader": test_loader,
                    "reference_prior": reference_prior,
                    "device": device,
                    "n_classes": n_classes,
                    "evaluate_fn": eval_fn,
                }
                if spec.needs_env_loaders:
                    trainer_kwargs["env_loaders"] = env_loaders
                if spec.needs_group_counts:
                    trainer_kwargs["group_counts"] = class_counts
                trainer = spec.trainer_cls(**trainer_kwargs)
                history = trainer.train()
                history["seed"] = seed
                self.histories[spec.key].append(history)
                history.to_csv(
                    self.output_dir / f"{spec.output_stem}_seed{seed}.csv",
                    index=False,
                )

                final_row = history.iloc[-1]
                print(
                    f"\n  {spec.display_name} final - "
                    f"{_format_final_metrics(spec.key, final_row)}"
                )
                classwise_df = make_classwise_results_table(
                    trainer.model,
                    test_loader,
                    class_counts,
                    class_names,
                    prefix=spec.output_stem,
                    device=device,
                    n_classes=n_classes,
                )
                classwise_df.to_csv(
                    self.output_dir / f"{spec.output_stem}_classwise_seed{seed}.csv",
                    index=False,
                )
                print_classwise_results(
                    classwise_df,
                    prefix=spec.output_stem,
                    title=f"{spec.display_name} class-wise test metrics",
                )

        for spec in self.method_specs:
            histories = self.histories[spec.key]
            if histories:
                pd.concat(histories, ignore_index=True).to_csv(
                    self.output_dir / f"{spec.output_stem}_all.csv",
                    index=False,
                )

        print(f"\nResults saved to: {self.output_dir}")


def _mean_std(
    histories: List[pd.DataFrame],
    col: str,
    epoch_col: str = "epoch",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = pd.concat(histories, ignore_index=True)
    grp = combined.groupby(epoch_col)[col]
    epochs = np.array(sorted(combined[epoch_col].unique()))
    mean = grp.mean().reindex(epochs).values
    std = grp.std(ddof=1).fillna(0.0).reindex(epochs).values
    return epochs, mean, std


def _shaded(
    ax,
    epochs,
    mean,
    std,
    color,
    label,
    linestyle="-",
) -> None:
    ax.plot(epochs, mean, color=color, linewidth=1.8, linestyle=linestyle, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)


def plot_inline(runner: CIFAR10LTRunner) -> None:
    """
    Produce the same IRS vs KL-RS comparison figure as the rebuttal script.
    """
    if "irs" not in runner.histories or "klrs" not in runner.histories:
        print("[plot_inline] IRS and KL-RS histories are both required.")
        return

    irs_h = runner.histories["irs"]
    klrs_h = runner.histories["klrs"]
    if not irs_h or not klrs_h:
        print("[plot_inline] No histories found. Run runner.run() first.")
        return

    irs_color = "#1f77b4"
    klrs_color = "#d62728"
    cfg = runner.cfg
    n_seeds = len(irs_h)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"IRS vs KL-RS  |  CIFAR-10-LT imbalance={cfg.imbalance_factor}  "
        f"({n_seeds} seed{'s' if n_seeds > 1 else ''})",
        fontsize=13,
    )

    panels = [
        (0, 0, "test_loss", "Cross-entropy", "Test Loss"),
        (0, 1, "test_acc", "Accuracy", "Test Accuracy"),
        (0, 2, "test_balanced_acc", "Balanced Acc", "Test Balanced Accuracy"),
        (1, 0, "test_worst_acc", "Worst-class Acc", "Worst-class Accuracy"),
    ]

    for row, col, metric_col, ylabel, title in panels:
        ax = axes[row, col]
        ep_i, mu_i, sd_i = _mean_std(irs_h, metric_col)
        ep_k, mu_k, sd_k = _mean_std(klrs_h, metric_col)
        _shaded(ax, ep_i, mu_i, sd_i, irs_color, "IRS")
        _shaded(ax, ep_k, mu_k, sd_k, klrs_color, "KL-RS")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    ax = axes[1, 1]
    ep_i, mu_i, sd_i = _mean_std(irs_h, "kappa")
    _shaded(ax, ep_i, mu_i, sd_i, irs_color, "IRS kappa")
    ax.set_title("IRS Fragility (kappa, lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("kappa")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax = axes[1, 2]
    klrs_h_lam = [df.dropna(subset=["lam"]) for df in klrs_h]
    if any(len(d) > 0 for d in klrs_h_lam):
        ep_k, mu_k, sd_k = _mean_std(klrs_h_lam, "lam")
        _shaded(ax, ep_k, mu_k, sd_k, klrs_color, "KL-RS lambda")
    ax.set_title("KL-RS Fragility (lambda, lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lambda")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    plt.show()

    out = runner.output_dir / "comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Figure saved to: {out}")


def main() -> None:
    runner = CIFAR10LTRunner(CONFIG)
    runner.run()
    plot_inline(runner)


CONFIG = CIFAR10LTConfig()


if __name__ == "__main__":
    main()
