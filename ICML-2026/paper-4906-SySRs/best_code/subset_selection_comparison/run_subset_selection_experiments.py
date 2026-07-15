"""
Subset Selection Comparison: Best Arm Identification (No Train/Test Split)

Methods compared:
  - AnchorPointsWeighted : k-medoids on inter-question Pearson correlation
  - Correctness Coresets  : KMeans on score-vector space (question × model matrix)
  - IRT Coresets (D=2,5,10,15): KMeans on pre-trained IRT feature space
  - SySRs                : Smart Successive Rejects bandit algorithm
  - MetaBench            : 2PL IRT + Fisher-info coreset selection + GAM calibration

IRT models are pre-trained ONCE per (dataset, dimension) and cached to disk at
  codebase/datasets/{dataset}/irt_params_no_split_D{D}.pkl
Randomness comes only from the clustering seed → BAI error-rate estimates.

Attribution:
  - IRT coreset methods (binarize_data, load_or_train_irt, select_coreset_irt)
    are adapted from the tinyBenchmarks repository:
    https://github.com/felipemaiapolo/tinyBenchmarks/tree/9c7e20302301ad531bfdfd9a7288e6e916bf22e9/tutorials
  - AnchorPoints method (select_coreset_correct) is adapted from the
    AnchorPoints repository:
    https://github.com/rvivek3/AnchorPoints/blob/64b6087d11176cc707ebeabfd6a5b13f8a1cfaf2/optimal_valset_validation.py
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import abc
import kmedoids
from ordered_set import OrderedSet
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from typing import Any, Dict, List, Optional, Set, Union

import torch
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO
from rich.console import Console
from rich.live import Live
from rich.table import Table

# ─── paths ────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).parent
codebase_dir = script_dir.parent

# ─── inlined from benchmark-prediction-release/benchpred/py_irt ──────────────
# MIT License — Copyright (c) 2019 John Lalor <john.lalor@nd.edu>
#                              and Pedro Rodriguez <me@pedro.ai>
# Source: https://github.com/nd-ball/py-irt
#   adapted by Felipe Maiapolo: https://github.com/felipemaiapolo/py-irt
#   further adapted in: https://github.com/guanhuazhang/benchmark-prediction

_irt_console = Console()


# -- Dataset (from py_irt/dataset.py) -----------------------------------------
class Dataset(BaseModel):
    item_ids: Union[Set[str], OrderedSet]
    subject_ids: Union[Set[str], OrderedSet]
    item_id_to_ix: Dict[str, int]
    ix_to_item_id: Dict[int, str]
    subject_id_to_ix: Dict[str, int]
    ix_to_subject_id: Dict[int, str]
    observation_subjects: List[int]
    observation_items: List
    observations: List[float]
    training_example: List[bool]

    class Config:
        arbitrary_types_allowed = True

    def get_item_accuracies(self) -> Dict[str, Dict[str, int]]:
        item_accuracies: Dict[str, Dict[str, int]] = {}
        for ix, response in enumerate(self.observations):
            item_id = self.ix_to_item_id[self.observation_items[ix]]
            if item_id not in item_accuracies:
                item_accuracies[item_id] = {"correct": 0, "total": 0}
            item_accuracies[item_id]["correct"] += int(response)
            item_accuracies[item_id]["total"]   += 1
        return item_accuracies


# -- IrtModel abstract + registry (from py_irt/models/abstract_model.py) ------
_IRT_REGISTRY: Dict[str, Any] = {}


class IrtModel(abc.ABC):
    def __init__(self, *, num_items: int, num_subjects: int,
                 verbose: bool = False, device: str = "cpu") -> None:
        super().__init__()
        self.device      = device
        self.num_items   = num_items
        self.num_subjects = num_subjects
        self.verbose     = verbose

    @classmethod
    def register(cls, name: str):
        def add_to_registry(class_):
            _IRT_REGISTRY[name] = class_
            return class_
        return add_to_registry

    @classmethod
    def from_name(cls, name: str):
        if name not in _IRT_REGISTRY:
            raise ValueError(f"Unknown IRT model: {name}. Registry: {list(_IRT_REGISTRY)}")
        return _IRT_REGISTRY[name]

    @classmethod
    def validate_name(cls, name: str):
        if name not in _IRT_REGISTRY:
            raise ValueError(f"Unknown IRT model: {name}. Registry: {list(_IRT_REGISTRY)}")

    @abc.abstractmethod
    def get_model(self): pass

    @abc.abstractmethod
    def get_guide(self): pass

    @abc.abstractmethod
    def export(self) -> Dict[str, Any]: pass


# -- Multidim2PL (from py_irt/models/multidim_2pl.py) -------------------------
@IrtModel.register("multidim_2pl")
class Multidim2PL(IrtModel):
    def __init__(self, *, num_items: int, num_subjects: int, dims: int = 2,
                 verbose: bool = False, device: str = "cpu", **kwargs):
        super().__init__(device=device, num_items=num_items,
                         num_subjects=num_subjects, verbose=verbose)
        self.dims = dims

    def export(self) -> Dict[str, Any]:
        return {
            "ability":        pyro.param("loc_ability").data.tolist(),
            "scale_ability":  pyro.param("scale_ability").data.tolist(),
            "diff":           pyro.param("loc_diff").data.tolist(),
            "disc":           pyro.param("loc_disc").data.tolist(),
            "loc_mu_theta":   pyro.param("loc_mu_theta").data.tolist(),
            "scale_mu_theta": pyro.param("scale_mu_theta").data.tolist(),
            "alpha_theta":    pyro.param("alpha_theta").data.tolist(),
            "beta_theta":     pyro.param("beta_theta").data.tolist(),
        }

    def get_model(self): return self.model_hierarchical
    def get_guide(self): return self.guide_hierarchical

    def model_hierarchical(self, subjects, items, obs):
        d = self.device
        with pyro.plate("mu_b_plate", 1):
            mu_b = pyro.sample("mu_b", dist.Normal(torch.tensor(0.0, device=d), torch.tensor(1e1, device=d)))
        with pyro.plate("u_b_plate", 1):
            u_b = pyro.sample("u_b", dist.Gamma(torch.tensor(1.0, device=d), torch.tensor(1.0, device=d)))
        with pyro.plate("mu_theta_plate", self.dims):
            mu_theta = pyro.sample("mu_theta", dist.Normal(torch.tensor(0.0, device=d), torch.tensor(1e1, device=d)))
        with pyro.plate("u_theta_plate", self.dims):
            u_theta = pyro.sample("u_theta", dist.Gamma(torch.tensor(1.0, device=d), torch.tensor(1.0, device=d)))
        with pyro.plate("mu_gamma_plate", self.dims):
            mu_gamma = pyro.sample("mu_gamma", dist.Normal(torch.tensor(0.0, device=d), torch.tensor(1e1, device=d)))
        with pyro.plate("u_gamma_plate", self.dims):
            u_gamma = pyro.sample("u_gamma", dist.Gamma(torch.tensor(1.0, device=d), torch.tensor(1.0, device=d)))
        with pyro.plate("thetas", self.num_subjects, dim=-2, device=d):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
        with pyro.plate("bs", self.num_items, dim=-2, device=d):
            with pyro.plate("bs_dims", 1, dim=-1):
                diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
        with pyro.plate("gammas", self.num_items, dim=-2, device=d):
            with pyro.plate("gamma_dims", self.dims, dim=-1):
                disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))
        with pyro.plate("observe_data", obs.size(0)):
            logits = (disc[items] * ability[subjects] - diff[items]).sum(axis=-1)
            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)

    def guide_hierarchical(self, subjects, items, obs):
        d = self.device
        loc_mu_b    = pyro.param("loc_mu_b",    torch.zeros(1, device=d))
        scale_mu_b  = pyro.param("scale_mu_b",  torch.ones(1,  device=d), constraint=constraints.positive)
        loc_mu_th   = pyro.param("loc_mu_theta",   torch.zeros(self.dims, device=d))
        scale_mu_th = pyro.param("scale_mu_theta", torch.ones(self.dims,  device=d), constraint=constraints.positive)
        loc_mu_ga   = pyro.param("loc_mu_gamma",   torch.zeros(self.dims, device=d))
        scale_mu_ga = pyro.param("scale_mu_gamma", torch.ones(self.dims,  device=d), constraint=constraints.positive)
        alpha_b  = pyro.param("alpha_b",  torch.ones(1,          device=d), constraint=constraints.positive)
        beta_b   = pyro.param("beta_b",   torch.ones(1,          device=d), constraint=constraints.positive)
        alpha_th = pyro.param("alpha_theta", torch.ones(self.dims, device=d), constraint=constraints.positive)
        beta_th  = pyro.param("beta_theta",  torch.ones(self.dims, device=d), constraint=constraints.positive)
        alpha_ga = pyro.param("alpha_gamma", torch.ones(self.dims, device=d), constraint=constraints.positive)
        beta_ga  = pyro.param("beta_gamma",  torch.ones(self.dims, device=d), constraint=constraints.positive)
        m_th = pyro.param("loc_ability",   torch.zeros([self.num_subjects, self.dims], device=d))
        s_th = pyro.param("scale_ability", torch.ones( [self.num_subjects, self.dims], device=d), constraint=constraints.positive)
        m_b  = pyro.param("loc_diff",   torch.zeros([self.num_items, 1],          device=d))
        s_b  = pyro.param("scale_diff", torch.ones( [self.num_items, 1],          device=d), constraint=constraints.positive)
        m_ga = pyro.param("loc_disc",   torch.zeros([self.num_items, self.dims],  device=d))
        s_ga = pyro.param("scale_disc", torch.ones( [self.num_items, self.dims],  device=d), constraint=constraints.positive)
        with pyro.plate("mu_b_plate", 1):
            pyro.sample("mu_b", dist.Normal(loc_mu_b, scale_mu_b))
        with pyro.plate("u_b_plate", 1):
            pyro.sample("u_b", dist.Gamma(alpha_b, beta_b))
        with pyro.plate("mu_theta_plate", self.dims):
            pyro.sample("mu_theta", dist.Normal(loc_mu_th, scale_mu_th))
        with pyro.plate("u_theta_plate", self.dims):
            pyro.sample("u_theta", dist.Gamma(alpha_th, beta_th))
        with pyro.plate("mu_gamma_plate", self.dims):
            pyro.sample("mu_gamma", dist.Normal(loc_mu_ga, scale_mu_ga))
        with pyro.plate("u_gamma_plate", self.dims):
            pyro.sample("u_gamma", dist.Gamma(alpha_ga, beta_ga))
        with pyro.plate("thetas", self.num_subjects, dim=-2, device=d):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                pyro.sample("theta", dist.Normal(m_th, s_th))
        with pyro.plate("bs", self.num_items, dim=-2, device=d):
            with pyro.plate("bs_dims", 1, dim=-1):
                pyro.sample("b", dist.Normal(m_b, s_b))
        with pyro.plate("gammas", self.num_items, dim=-2, device=d):
            with pyro.plate("gamma_dims", self.dims, dim=-1, device=d):
                pyro.sample("gamma", dist.Normal(m_ga, s_ga))


# -- Initializers (from py_irt/initializers.py) --------------------------------
INITIALIZERS: Dict[str, Any] = {}


class IrtInitializer(abc.ABC):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def initialize(self) -> None:
        pass


class DifficultySignInitializer(IrtInitializer):
    def __init__(self, dataset: Dataset, magnitude: float = 3.0, n_to_init: int = 4):
        super().__init__(dataset)
        self._magnitude = magnitude
        self._n_to_init = n_to_init

    def initialize(self) -> None:
        item_acc: Dict[int, Dict[str, int]] = {}
        for item_ix, response in zip(self._dataset.observation_items, self._dataset.observations):
            if item_ix not in item_acc:
                item_acc[item_ix] = {"correct": 0, "total": 0}
            item_acc[item_ix]["correct"] += int(response)
            item_acc[item_ix]["total"]   += 1
        sorted_items = sorted(item_acc.items(),
                              key=lambda kv: kv[1]["correct"] / max(1, kv[1]["total"]))
        diff = pyro.param("loc_diff")
        for item_ix, _ in sorted_items[: self._n_to_init]:
            diff.data[item_ix] = torch.tensor(self._magnitude,
                                              dtype=diff.data.dtype, device=diff.data.device)
        for item_ix, _ in sorted_items[-self._n_to_init :]:
            diff.data[item_ix] = torch.tensor(-self._magnitude,
                                              dtype=diff.data.dtype, device=diff.data.device)


INITIALIZERS["difficulty_sign"] = DifficultySignInitializer


# -- IrtConfig (from py_irt/config.py) ----------------------------------------
class IrtConfig(BaseModel):
    model_type: str
    epochs: int = 2000
    priors: Optional[str] = None
    initializers: Optional[List[Union[str, Dict]]] = None
    dims: Optional[int] = None
    lr: float = 0.1
    lr_decay: float = 0.9999
    dropout: float = 0.5
    hidden: int = 100
    vocab_size: Optional[int] = None
    log_every: int = 100
    seed: Optional[int] = None
    deterministic: bool = False


# -- IrtModelTrainer (from py_irt/training.py) --------------------------------
class IrtModelTrainer:
    def __init__(self, *, config: IrtConfig, data_path=None,
                 dataset: Optional[Dataset] = None, verbose: bool = True) -> None:
        self._config = config
        IrtModel.validate_name(config.model_type)
        self._device      = None
        self.irt_model    = None
        self._pyro_model  = None
        self._pyro_guide  = None
        self._verbose     = verbose
        self.best_params  = None
        self.last_params  = None
        self.amortized    = "amortized" in self._config.model_type
        if dataset is None:
            raise ValueError("dataset must be provided")
        self._dataset = dataset
        _irt_console.log(f'Vocab size: {self._config.vocab_size}')
        # filter to training examples only
        idx = [i for i, t in enumerate(self._dataset.training_example) if t]
        self._dataset.observation_subjects = [self._dataset.observation_subjects[i] for i in idx]
        self._dataset.observation_items    = [self._dataset.observation_items[i]    for i in idx]
        self._dataset.observations         = [self._dataset.observations[i]         for i in idx]
        self._dataset.training_example     = [self._dataset.training_example[i]     for i in idx]
        inits = config.initializers or []
        self._initializers = []
        for init in inits:
            if isinstance(init, IrtInitializer):
                self._initializers.append(init)
            elif isinstance(init, str):
                self._initializers.append(INITIALIZERS[init](self._dataset))
            elif isinstance(init, dict):
                name = init.pop("name")
                self._initializers.append(INITIALIZERS[name](self._dataset, **init))
            else:
                raise TypeError(f"invalid initializer type: {type(init)}")

    def train(self, *, epochs: Optional[int] = None, device: str = "cpu") -> None:
        if epochs is None:
            epochs = self._config.epochs
        self._device = device
        if self._config.seed is not None:
            torch.manual_seed(self._config.seed)
            pyro.set_rng_seed(self._config.seed)
        args: Dict[str, Any] = {
            "device":       device,
            "num_items":    len(self._dataset.ix_to_item_id),
            "num_subjects": len(self._dataset.ix_to_subject_id),
        }
        _irt_console.log(f'args: {args}')
        args["priors"] = self._config.priors if self._config.priors is not None else "vague"
        if self._config.dims is not None:
            args["dims"] = self._config.dims
        args["dropout"]    = self._config.dropout
        args["hidden"]     = self._config.hidden
        args["vocab_size"] = self._config.vocab_size
        _irt_console.log(f"Parsed Model Args: {args}")
        self.irt_model   = IrtModel.from_name(self._config.model_type)(**args)
        pyro.clear_param_store()
        self._pyro_model = self.irt_model.get_model()
        self._pyro_guide = self.irt_model.get_guide()
        device_obj = torch.device(device)
        scheduler = pyro.optim.ExponentialLR({
            "optimizer":  torch.optim.Adam,
            "optim_args": {"lr": self._config.lr},
            "gamma":      self._config.lr_decay,
        })
        svi       = SVI(self._pyro_model, self._pyro_guide, scheduler, loss=Trace_ELBO())
        subjects  = torch.tensor(self._dataset.observation_subjects, dtype=torch.long,  device=device_obj)
        items     = torch.tensor(self._dataset.observation_items,    dtype=torch.long,  device=device_obj)
        responses = torch.tensor(self._dataset.observations,         dtype=torch.float, device=device_obj)
        # initialise params before running initializers
        _ = self._pyro_model(subjects, items, responses)
        _ = self._pyro_guide(subjects, items, responses)
        for init in self._initializers:
            init.initialize()
        table = Table()
        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column("Best Loss")
        table.add_column("New LR")
        loss = float("inf")
        best_loss  = loss
        current_lr = self._config.lr
        with Live(table) as live:
            live.console.print(f"Training Pyro IRT Model for {epochs} epochs")
            for epoch in range(epochs):
                loss = svi.step(subjects, items, responses)
                if loss < best_loss:
                    best_loss        = loss
                    self.best_params = self._export()
                scheduler.step()
                current_lr *= self._config.lr_decay
                if epoch % self._config.log_every == 0:
                    table.add_row(f"{epoch+1}", f"{loss:.4f}", f"{best_loss:.4f}", f"{current_lr:.4f}")
            table.add_row(f"{epoch+1}", f"{loss:.4f}", f"{best_loss:.4f}", f"{current_lr:.4f}")
        self.last_params = self._export()

    def _export(self) -> Dict[str, Any]:
        results = self.irt_model.export()
        results["irt_model"]   = self._config.model_type
        results["item_ids"]    = self._dataset.ix_to_item_id
        results["subject_ids"] = self._dataset.ix_to_subject_id
        return results

# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(codebase_dir))
from bai_algs import smart_successive_rejects_wo_replacement_no_budget_limit

# ─── available datasets ───────────────────────────────────────────────────────
ALL_DATASETS = [
    'arc_challenge', 'bbh', 'commonsense', 'gpqa', 'gsm',
    'ifeval', 'legalbench', 'math', 'med_qa', 'mmlu',
    'mmlu_pro', 'musr', 'narrative_qa', 'natural_qa', 'wmt_14',
]

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Run subset selection comparison experiments.')
    parser.add_argument(
        'dataset', nargs='?', default='commonsense',
        help='Dataset name (default: commonsense). Ignored when --all is set.'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all datasets sequentially.'
    )
    parser.add_argument(
        '--k', type=int, default=100,
        help='Number of independent runs per method (default: 100).'
    )
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory for JSON results (default: <script_dir>/results).'
    )
    parser.add_argument(
        '--irt-epochs', type=int, default=2000,
        help='IRT training epochs (default: 2000).'
    )
    parser.add_argument(
        '--irt-device', type=str, default='cpu',
        help='IRT training device (default: cpu).'
    )
    parser.add_argument(
        '--n-cpus', type=int,
        default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
        help='Parallel workers for clustering (default: SLURM_CPUS_PER_TASK or 1).'
    )
    return parser.parse_args()


# ─── inlined from benchpred/tiny_bench.py ────────────────────────────────────
# Adapted from the tinyBenchmarks repository:
# https://github.com/felipemaiapolo/tinyBenchmarks/tree/9c7e20302301ad531bfdfd9a7288e6e916bf22e9/tutorials

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def item_curve(theta, a, b):
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)


def estimate_ability_parameters(
    responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"
):
    D = A.shape[1]

    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A, B).squeeze()
        log_likelihood = np.sum(
            responses_test * np.log(P + eps)
            + (1 - responses_test) * np.log(1 - P + eps)
        )
        return -log_likelihood

    if isinstance(theta_init, np.ndarray):
        theta_init = theta_init.reshape(-1)
        assert theta_init.shape[0] == D
    else:
        theta_init = np.zeros(D)

    return minimize(neg_log_like, theta_init, method=optimizer).x[None, :, None]


class NewDataset(Dataset):
    @classmethod
    def from_list(cls, data_list, train_items=None, amortized=False):
        item_ids       = OrderedSet()
        subject_ids    = OrderedSet()
        item_id_to_ix  = {}
        ix_to_item_id  = {}
        subject_id_to_ix  = {}
        ix_to_subject_id  = {}

        for line in data_list:
            subject_ids.add(line["subject_id"])
            for item_id in line["responses"].keys():
                item_ids.add(item_id)

        for idx, item_id in enumerate(item_ids):
            item_id_to_ix[item_id]  = idx
            ix_to_item_id[idx]      = item_id
        for idx, subject_id in enumerate(subject_ids):
            subject_id_to_ix[subject_id] = idx
            ix_to_subject_id[idx]        = subject_id

        if amortized:
            vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words="english")
            vectorizer.fit(item_ids)

        observation_subjects = []
        observation_items    = []
        observations         = []
        training_example     = []
        for line in data_list:
            subject_id = line["subject_id"]
            for item_id, response in line["responses"].items():
                observations.append(response)
                observation_subjects.append(subject_id_to_ix[subject_id])
                if not amortized:
                    observation_items.append(item_id_to_ix[item_id])
                else:
                    observation_items.append(
                        vectorizer.transform([item_id]).todense().tolist()[0]
                    )
                if train_items is not None:
                    training_example.append(train_items[subject_id][item_id])
                else:
                    training_example.append(True)

        return cls(
            item_ids=item_ids,
            subject_ids=subject_ids,
            item_id_to_ix=item_id_to_ix,
            ix_to_item_id=ix_to_item_id,
            subject_id_to_ix=subject_id_to_ix,
            ix_to_subject_id=ix_to_subject_id,
            observation_subjects=observation_subjects,
            observation_items=observation_items,
            observations=observations,
            training_example=training_example,
        )


def create_irt_dataset(responses):
    dataset = []
    for i in range(responses.shape[0]):
        aux_q = {"q" + str(j): int(responses[i, j]) for j in range(responses.shape[1])}
        dataset.append({"subject_id": str(i), "responses": aux_q})
    return NewDataset.from_list(dataset)

# ──────────────────────────────────────────────────────────────────────────────

# ─── IRT helpers ──────────────────────────────────────────────────────────────
def binarize_data(Y_data):
    """Find the threshold minimising |binarised_mean - original_mean| per model."""
    cs     = np.linspace(0.01, 0.99, 100)
    best_c = cs[np.argmin([
        np.mean(np.abs((Y_data > c).mean(axis=1) - Y_data.mean(axis=1)))
        for c in cs
    ])]
    return (Y_data > best_c).astype(int)


def load_or_train_irt(D_total, Y_data, dataset_name, irt_epochs, irt_device):
    """
    Train multidim_2pl with alpha_dims = D_total-1, beta_dims = 1.
    Saves / loads raw (A, B) arrays to
      codebase/datasets/{dataset}/irt_params_no_split_D{D_total}.pkl
    Returns (A, B, X_feat) where:
      X_feat  shape (n_questions, D_total)
    """
    cache_path = codebase_dir / 'datasets' / dataset_name / f'irt_params_no_split_D{D_total}.pkl'
    alpha_dims = D_total - 1

    if cache_path.exists():
        print(f"  [D={D_total}] Loading cached IRT params from {cache_path.name}")
        with open(cache_path, 'rb') as fh:
            A, B = pickle.load(fh)
    else:
        print(f"  [D={D_total}] Training IRT (alpha_dims={alpha_dims}, "
              f"epochs={irt_epochs}, device={irt_device}) …")
        Y_bin   = binarize_data(Y_data)
        dataset = create_irt_dataset(Y_bin)
        config  = IrtConfig(
            model_type    = "multidim_2pl",
            priors        = "hierarchical",
            dims          = alpha_dims,
            lr            = 0.1,
            epochs        = irt_epochs,
            log_every     = 200,
            deterministic = True,
            seed          = 42,
        )
        trainer = IrtModelTrainer(config=config, dataset=dataset, verbose=False)
        trainer.train(device=irt_device)
        params = trainer.best_params
        A = np.array(params["disc"]).T[None, :, :]
        B = np.array(params["diff"]).T[None, :, :]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump((A, B), fh)
        print(f"  [D={D_total}] Saved to {cache_path.name}")

    X_feat = np.vstack((A.squeeze(), B.squeeze().reshape(1, -1))).T
    print(f"  [D={D_total}] Feature matrix: {X_feat.shape}")
    return A, B, X_feat


# ─── coreset selection helpers ────────────────────────────────────────────────
def select_coreset_correct(dist_matrix, corrs, num_anchors, seed):
    """k-medoids on correlation distance → (medoid_indices, weights)."""
    result  = kmedoids.fasterpam(dist_matrix, num_anchors,
                                 init="random", random_state=seed)
    medoids = list(set(result.medoids))
    cluster_members = np.argmax(corrs[medoids, :], axis=0)
    sizes   = np.array([np.sum(cluster_members == i) for i in range(len(medoids))],
                       dtype=float)
    weights = sizes / sizes.sum()
    return np.array(medoids), weights


def select_coreset_sv(sv_feat, num_anchors, seed):
    """KMeans on score-vector space → (anchor_indices, weights)."""
    kmeans = KMeans(n_clusters=num_anchors, n_init=10, random_state=seed)
    kmeans.fit(sv_feat)
    dists   = pairwise_distances(kmeans.cluster_centers_, sv_feat, metric="euclidean")
    anchors = dists.argmin(axis=1)
    sizes   = np.array([np.sum(kmeans.labels_ == c) for c in range(num_anchors)], dtype=float)
    weights = sizes / sizes.sum()
    return anchors, weights


def select_coreset_irt(X_feat, num_anchors, seed):
    """KMeans on IRT feature space → (anchor_indices, weights)."""
    kmeans = KMeans(n_clusters=num_anchors, n_init=10, random_state=seed)
    kmeans.fit(X_feat)
    dists   = pairwise_distances(kmeans.cluster_centers_, X_feat, metric="euclidean")
    anchors = dists.argmin(axis=1)
    sizes   = np.array([np.sum(kmeans.labels_ == c) for c in range(num_anchors)], dtype=float)
    weights = sizes / sizes.sum()
    return anchors, weights


# ─── MetaBench helpers (adapted from https://github.com/socialfoundations/benchmark-prediction/blob/release/benchpred/metabench.py) ────
def _mb_fit_irt(X, n_iters=2):
    """
    Fit 2PL IRT via alternating EM.
    X: (n_subjects, n_items), binary. Returns dict with 'a' (disc) and 'b' (diff).
    """
    n, m = X.shape
    a     = np.ones(m)
    b     = np.zeros(m)
    theta = np.zeros(n)
    eps   = 1e-8

    def _all_items_nll(ab_flat):
        ai = ab_flat[:m]
        bi = ab_flat[m:]
        P  = expit(ai[None, :] * (theta[:, None] - bi[None, :]))
        return -np.sum(X * np.log(P + eps) + (1 - X) * np.log(1 - P + eps))

    def _theta_nll(t_i, idx):
        p  = expit(a * (t_i - b))
        xi = X[idx]
        return -np.sum(xi * np.log(p + eps) + (1 - xi) * np.log(1 - p + eps))

    for _ in range(n_iters):
        # M-step: jointly optimise all item parameters
        res = minimize(_all_items_nll, np.concatenate([a, b]), method="L-BFGS-B",
                       bounds=[(0.01, 5.0)] * m + [(-4.0, 4.0)] * m)
        ab  = res.x
        a   = np.clip(ab[:m], 0.01, 5.0)
        b   = ab[m:]
        # E-step: optimise each ability independently
        for i in range(n):
            res_i = minimize(lambda t, _i=i: _theta_nll(t, _i),
                             x0=theta[i], bounds=[(-4.0, 4.0)], method="L-BFGS-B")
            theta[i] = res_i.x[0]

    return {"a": a, "b": b}


def _mb_estimate_abilities(params, X_sub):
    """MAP ability estimation for a fitted 2PL model. Returns theta_hat shape (n,)."""
    a, b = params["a"], params["b"]
    n    = X_sub.shape[0]
    eps  = 1e-8
    theta_hat = np.zeros(n)
    for i in range(n):
        xi  = X_sub[i]
        res = minimize(
            lambda t, _xi=xi: -np.sum(_xi * np.log(expit(a * (t - b)) + eps)
                                      + (1 - _xi) * np.log(1 - expit(a * (t - b)) + eps)),
            x0=0.0, bounds=[(-4.0, 4.0)], method="L-BFGS-B",
        )
        theta_hat[i] = res.x[0]
    return theta_hat


def _load_or_fit_mb_irt(Y_full, dataset_name):
    """Fit (or load cached) MetaBench 2PL IRT params on the full score matrix."""
    cache = codebase_dir / 'datasets' / dataset_name / 'metabench_irt.pkl'
    if cache.exists():
        print(f"  [MetaBench IRT] Loading cache from {cache.name}")
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    print(f"  [MetaBench IRT] Fitting 2PL IRT on {Y_full.shape} …")
    params = _mb_fit_irt(Y_full, n_iters=2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(params, fh)
    print(f"  [MetaBench IRT] Saved to {cache.name}")
    return params


def load_or_fit_metabench(Y_full, num_anchors, dataset_name, mb_irt_params):
    """
    Select a MetaBench coreset of size num_anchors via Fisher-info quantile selection,
    then fit the GAM calibration model.  Results cached per (dataset, coreset_size).
    Returns (coreset_items, co_params, gam).

    Adapted from https://github.com/socialfoundations/benchmark-prediction/blob/release/benchpred/metabench.py (MetaBench.fit).
    """
    from pygam import LinearGAM, s as gam_s

    cache = codebase_dir / 'datasets' / dataset_name / f'metabench_n{num_anchors}.pkl'
    if cache.exists():
        print(f"  [MetaBench n={num_anchors}] Loading cache from {cache.name}")
        with open(cache, 'rb') as fh:
            return pickle.load(fh)

    # Estimate training abilities on the full bank
    print(f"  [MetaBench n={num_anchors}] Estimating abilities …")
    theta_train = _mb_estimate_abilities(mb_irt_params, Y_full)

    # Fisher information per item over training abilities
    a_full = mb_irt_params["a"]
    b_full = mb_irt_params["b"]
    logits = a_full[:, None] * (theta_train[None, :] - b_full[:, None])
    p_mat  = expit(logits)                          # (n_items, n_models)
    fis    = (a_full[:, None] ** 2) * p_mat * (1 - p_mat)

    # Select most informative item per quantile bin
    quantiles = np.quantile(theta_train, np.linspace(0, 1, num_anchors + 1))
    selected  = set()
    for i in range(num_anchors):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask   = (theta_train >= lo) & (theta_train < hi)
        if not np.any(mask):
            continue
        avg_info = fis[:, mask].mean(axis=1)
        for idx in np.argsort(avg_info)[::-1]:
            if idx not in selected:
                selected.add(idx)
                break
        if len(selected) >= num_anchors:
            break

    # Fallback: if num_anchors > n_models, the quantile bins cannot yield
    # enough distinct items.  Fill remaining slots with the globally most
    # informative items not yet selected.
    if len(selected) < num_anchors:
        global_avg_info = fis.mean(axis=1)
        for idx in np.argsort(global_avg_info)[::-1]:
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= num_anchors:
                break

    coreset_items = np.array(sorted(selected))

    # Fit GAM: predicted_full_mean ~ s(theta_coreset) + s(mean_coreset_score)
    co_params  = {"a": mb_irt_params["a"][coreset_items],
                  "b": mb_irt_params["b"][coreset_items]}
    sub_scores = Y_full[:, coreset_items]
    theta_co   = _mb_estimate_abilities(co_params, sub_scores)
    y_full_mean = Y_full.mean(axis=1)
    X_gam = np.vstack([theta_co, sub_scores.mean(axis=1)]).T
    gam   = LinearGAM(gam_s(0) + gam_s(1), fit_intercept=True).fit(X_gam, y_full_mean)

    result = (coreset_items, co_params, gam)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(result, fh)
    print(f"  [MetaBench n={num_anchors}] Saved to {cache.name}")
    return result


# ─── per-run trial functions ──────────────────────────────────────────────────
def run_anchor(run_idx, num_anchors, dist_matrix, corrs, Y_full, best_model_idx):
    medoids, weights = select_coreset_correct(dist_matrix, corrs, num_anchors, seed=run_idx)
    pred = (Y_full[:, medoids] * weights).sum(axis=1)
    return int(np.argmax(pred) == best_model_idx)


def run_correct_sv(run_idx, num_anchors, sv_feat, Y_full, best_model_idx):
    anchors, weights = select_coreset_sv(sv_feat, num_anchors, seed=run_idx)
    pred = (Y_full[:, anchors] * weights).sum(axis=1)
    return int(np.argmax(pred) == best_model_idx)


def run_irt(run_idx, num_anchors, X_feat, Y_full, best_model_idx):
    anchors, weights = select_coreset_irt(X_feat, num_anchors, seed=run_idx)
    pred = (Y_full[:, anchors] * weights).sum(axis=1)
    return int(np.argmax(pred) == best_model_idx)


def run_metabench_once(Y_full, coreset_items, co_params, gam, best_model_idx):
    """
    Single deterministic MetaBench BAI trial.
    MetaBench is fully deterministic given fixed input data, so run_idx is unused.
    Returns 1 if the predicted best model matches the ground-truth best model.
    """
    sub_scores = Y_full[:, coreset_items]
    theta_new  = _mb_estimate_abilities(co_params, sub_scores)
    X_new = np.vstack([theta_new, sub_scores.mean(axis=1)]).T
    pred  = gam.predict(X_new)
    return int(np.argmax(pred) == best_model_idx)


# ─── main experiment for one dataset ─────────────────────────────────────────
def run_dataset(dataset_name, k_runs, out_dir, irt_epochs, irt_device, n_cpus):
    np.random.seed(42)

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # load data
    pickle_path = codebase_dir / 'datasets' / dataset_name / 'model_accuracies_filtered.pkl'
    with open(pickle_path, 'rb') as f:
        Y_full = pickle.load(f)

    n_models, n_questions = Y_full.shape
    print(f"Loaded {dataset_name}: {n_models} models × {n_questions} questions")

    ground_truth_scores       = Y_full.mean(axis=1)
    best_model_idx            = int(np.argmax(ground_truth_scores))
    print(f"Best model: idx={best_model_idx}, acc={ground_truth_scores[best_model_idx]:.4f}")

    # budget table
    total_budget     = n_models * n_questions
    budget_fractions = [0.05 * (i + 1) for i in range(7)]   # 5% … 35%
    budget_configs   = {}
    print(f"\nBudget config (total = {n_models} × {n_questions} = {total_budget:,}):")
    print(f"{'Frac':<8} {'Bandit budget':>15} {'Num anchors':>13}")
    print("-" * 40)
    for frac in budget_fractions:
        bandit_budget = int(np.round(total_budget * frac))
        num_anchors   = int(np.round(bandit_budget / n_models))
        num_anchors   = min(max(num_anchors, 1), n_questions)
        budget_configs[frac] = num_anchors
        print(f"{frac*100:5.1f}%   {bandit_budget:>15,} {num_anchors:>13,}")

    # methods
    IRT_DIMS        = [2, 5, 10, 15]
    irt_method_keys = [f"irt_d{D}" for D in IRT_DIMS]
    all_methods     = ["anchor", "correct"] + irt_method_keys + ["smart_sr", "metabench"]

    print(f"\nUsing {n_cpus} parallel workers.  Methods: {all_methods}")

    # pre-compute correlation matrix (for anchor method)
    print("\nPre-computing inter-question Pearson correlation …")
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = np.corrcoef(Y_full, rowvar=False)
        corrs[np.isnan(corrs)] = 0.0
    dist_matrix = (1.0 - corrs).astype(np.float64)
    print("Correlation matrix ready.")

    # score-vector feature matrix (for correct method)
    sv_feat = Y_full.T   # (n_questions, n_models)

    # pre-train / load IRT models
    print(f"\n{'='*60}\nPre-training / loading IRT models\n{'='*60}")
    irt_models = {}
    for D_total in IRT_DIMS:
        irt_models[D_total] = load_or_train_irt(D_total, Y_full, dataset_name,
                                                 irt_epochs, irt_device)

    # pre-train / load MetaBench models (one per unique budget level)
    # MetaBench is deterministic: IRT params cached once, GAM cached per coreset_size.
    print(f"\n{'='*60}\nPre-training / loading MetaBench models\n{'='*60}")
    mb_irt_params = _load_or_fit_mb_irt(Y_full, dataset_name)
    mb_models: dict = {}
    for frac in budget_fractions:
        num_a = budget_configs[frac]
        if num_a not in mb_models:
            mb_models[num_a] = load_or_fit_metabench(
                Y_full, num_a, dataset_name, mb_irt_params
            )

    # bandit setup
    import bai_algs as _bai
    _bai.model_accuracies  = Y_full
    _bai.total_n_arms      = n_models
    _bai.total_n_tasks     = n_questions
    _bai.USE_ACCURACY_MODE = True

    # experiment loop
    print(f"\n{'='*70}\nRUNNING EXPERIMENTS\n{'='*70}")
    print(f"k_runs={k_runs}, ground-truth best arm={best_model_idx}\n")

    raw_results = {m: {frac: [] for frac in budget_fractions} for m in all_methods}

    for frac in budget_fractions:
        num_anchors   = budget_configs[frac]
        bandit_budget = int(np.round(total_budget * frac))
        print(f"Budget {frac*100:.0f}%  (num_anchors={num_anchors}, bandit_budget={bandit_budget:,})")

        # anchor (k-medoids on correlation)
        print(f"  anchor …")
        raw_results["anchor"][frac] = Parallel(n_jobs=n_cpus, prefer="threads")(
            delayed(run_anchor)(i, num_anchors, dist_matrix, corrs, Y_full, best_model_idx)
            for i in range(k_runs)
        )

        # correct (SV-KMeans + weighted avg)
        print(f"  correct …")
        raw_results["correct"][frac] = Parallel(n_jobs=n_cpus, prefer="threads")(
            delayed(run_correct_sv)(i, num_anchors, sv_feat, Y_full, best_model_idx)
            for i in range(k_runs)
        )

        # IRT KMeans + weighted avg
        for D_total in IRT_DIMS:
            _, _, X_feat = irt_models[D_total]
            mkey = f"irt_d{D_total}"
            print(f"  {mkey} …")
            raw_results[mkey][frac] = Parallel(n_jobs=n_cpus, prefer="threads")(
                delayed(run_irt)(i, num_anchors, X_feat, Y_full, best_model_idx)
                for i in range(k_runs)
            )

        # SySRs
        print(f"  smart_sr (budget={bandit_budget:,}, k={k_runs}) …")
        sr_arms = smart_successive_rejects_wo_replacement_no_budget_limit(
            n_items=bandit_budget, k=k_runs, verbose=False
        )
        raw_results["smart_sr"][frac] = [
            int(arm == best_model_idx) for arm in sr_arms
        ]

        # MetaBench (deterministic – evaluate once, replicate across k_runs)
        print(f"  metabench …")
        coreset_items, co_params, gam = mb_models[num_anchors]
        mb_correct = run_metabench_once(Y_full, coreset_items, co_params, gam, best_model_idx)
        raw_results["metabench"][frac] = [mb_correct] * k_runs

    print("\nAll runs complete!")

    # summary stats
    summary_stats = {}
    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    for method in all_methods:
        summary_stats[method] = {}
        print(f"\n{method.upper()}")
        print(f"  {'Budget':>8}  {'Mean':>8}  {'Std':>8}")
        for frac in budget_fractions:
            rates = raw_results[method][frac]
            summary_stats[method][frac] = {
                'mean':  float(np.mean(rates)),
                'std':   float(np.std(rates)),
                'min':   float(np.min(rates)),
                'max':   float(np.max(rates)),
                'rates': rates,
            }
            print(f"  {frac*100:7.1f}%  {np.mean(rates):8.4f}  {np.std(rates):8.4f}")

    # comparison dataframe
    comparison_data = []
    for method in all_methods:
        for frac in budget_fractions:
            s = summary_stats[method][frac]
            comparison_data.append({
                'Method':        method,
                'Budget%':       frac * 100,
                'Mean BAI Rate': s['mean'],
                'Std Dev':       s['std'],
                'Num Questions': budget_configs[frac],
            })
    df_comparison = pd.DataFrame(comparison_data)
    print(f"\n{df_comparison.to_string(index=False)}")

    # save JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f'subset_selection_{dataset_name}_k{k_runs}.json'
    results_to_save = {
        'metadata': {
            'dataset_name':              dataset_name,
            'k_runs':                    k_runs,
            'n_models':                  n_models,
            'n_questions':               n_questions,
            'irt_dims':                  IRT_DIMS,
            'irt_alpha_dims':            [D - 1 for D in IRT_DIMS],
            'budget_fractions':          budget_fractions,
            'all_methods':               all_methods,
            'best_model_idx_ground_truth': best_model_idx,
        },
        'summary_stats': {
            method: {
                str(frac): {
                    'mean':  summary_stats[method][frac]['mean'],
                    'std':   summary_stats[method][frac]['std'],
                    'min':   summary_stats[method][frac]['min'],
                    'max':   summary_stats[method][frac]['max'],
                    'rates': [int(r) for r in summary_stats[method][frac]['rates']],
                }
                for frac in budget_fractions
            }
            for method in all_methods
        },
        'comparison_df': df_comparison.to_dict('records'),
    }
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved: {output_file}")


# ─── entry point ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / 'results'

    datasets = ALL_DATASETS if args.all else [args.dataset]

    for ds in datasets:
        run_dataset(
            dataset_name = ds,
            k_runs       = args.k,
            out_dir      = out_dir,
            irt_epochs   = args.irt_epochs,
            irt_device   = args.irt_device,
            n_cpus       = args.n_cpus,
        )

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results in: {out_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
