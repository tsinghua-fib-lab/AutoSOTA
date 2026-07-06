"""
NAS-Bench-Tabular Proxy Evaluator Bridge

Connects NASBenchTabularSpace + DNNModel with all zero-cost proxy evaluators
and pTProxy. This is the key integration module that wires together:
  - search_space.nas_bench_mlp (DNNModel, NASBenchTabularSpace)
  - proxies (all 13 baseline evaluators + ptproxy_score)
  - search_space.query_api (GTMLP ground truth)

Usage:
    evaluator = NASBenchEvaluator("frappe", device="cpu")

    # Evaluate a single proxy on a single architecture
    score = evaluator.evaluate_proxy("synflow", "8-16-32-64")

    # Evaluate all proxies on a single architecture
    scores = evaluator.evaluate_all_proxies("8-16-32-64")

    # Compute pTProxy score for an architecture
    score = evaluator.evaluate_ptproxy("8-16-32-64")

    # Prepare batch data for external evaluators
    batch_data, batch_labels = evaluator.prepare_batch(batch_size=32)
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.constant import Config
from search_space.nas_bench_mlp import NASBenchTabularSpace, DNNModel, DATASET_CONFIGS
from proxies import PROXY_EVALUATORS, get_evaluator
from proxies.ptproxy import ptproxy_score
from proxies.alg_base import Evaluator


# Path to LibSVM-format data for live evaluation (only needed for label-dependent proxies)
_LIBSVM_DATA_BASE = "/Users/kevin/project_python/relational_data/VLDB_code/exp_data2/result_base"


class NASBenchEvaluator:
    """
    Bridge between NAS-Bench-Tabular search space and zero-cost proxy evaluators.

    Handles:
    1. Creating DNNModel instances from architecture IDs
    2. Preparing batch data (all-ones embedding for data-agnostic proxies,
       or real data for label-dependent proxies)
    3. Evaluating any proxy on any architecture
    4. Evaluating pTProxy (our proposed proxy)
    """

    def __init__(self, dataset: str, device: str = "cpu",
                 batch_size: int = 32, use_bn: bool = False):
        """
        Args:
            dataset: "frappe", "uci_diabetes", or "criteo"
            device: "cpu" or "cuda:X"
            batch_size: batch size for proxy evaluation
            use_bn: whether to use BatchNorm in DNNModel
        """
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.space = NASBenchTabularSpace(dataset)

        # Cached embedding (shared across all models for speed)
        self._cached_embedding = None

    def _create_model(self, arch_id: str) -> DNNModel:
        """Create a DNNModel from architecture ID string."""
        model = self.space.new_architecture(arch_id, use_bn=self.use_bn)
        return model

    def prepare_batch_wo_embedding(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Prepare all-ones batch data bypassing embedding layer.
        Used by data-agnostic proxies (SynFlow, pTProxy, WeightNorm).

        Returns:
            Tensor of shape [batch_size, nfield * nemb]
        """
        bs = batch_size or self.batch_size
        # Create a temporary model to get the right input dimension
        temp_arch_id = self.space.random_architecture_id()
        temp_model = self._create_model(temp_arch_id)
        batch_data = temp_model.generate_all_ones_embedding(bs).float()
        del temp_model
        return batch_data

    def prepare_batch_with_labels(self, batch_size: Optional[int] = None) -> Tuple[dict, torch.Tensor]:
        """
        Prepare batch data WITH labels for label-dependent proxies
        (GradNorm, GradPlain, Fisher, GraSP, SNIP, NTK-*, KNAS).

        Uses synthetic random data matching the expected format.

        Returns:
            (batch_data_dict, batch_labels)
            batch_data_dict: {'id': LongTensor [B, F], 'value': FloatTensor [B, F]}
            batch_labels: LongTensor [B]
        """
        bs = batch_size or self.batch_size
        cfg = DATASET_CONFIGS[self.dataset]
        nfield = cfg["nfield"]
        nfeat = cfg["nfeat"]
        num_labels = cfg["num_labels"]

        # Generate synthetic batch data
        batch_data = {
            'id': torch.randint(0, nfeat, (bs, nfield)),
            'value': torch.ones(bs, nfield, dtype=torch.float32),
        }
        # Binary classification labels
        batch_labels = torch.randint(0, max(2, num_labels), (bs,))

        return batch_data, batch_labels

    # ----------------------------------------------------------
    # Data-agnostic proxies (no labels needed)
    # ----------------------------------------------------------

    # These proxies only need the model weights, no real data:
    _DATA_AGNOSTIC_PROXIES = {Config.SYNFLOW, Config.WEIGHT_NORM}

    # These proxies need batch_data (as tensor, no labels):
    _TENSOR_INPUT_PROXIES = {Config.NAS_WOT, Config.JACOB_COV}

    # These proxies need batch_data as dict + labels:
    _LABEL_DEPENDENT_PROXIES = {
        Config.GRAD_NORM, Config.GRAD_PLAIN,
        Config.PRUNE_FISHER, Config.PRUNE_GRASP, Config.PRUNE_SNIP,
        Config.NTK_COND_NUM, Config.NTK_TRACE, Config.NTK_TRACE_APPROX,
        Config.KNAS,
    }

    def evaluate_proxy(self, proxy_name: str, arch_id: str) -> float:
        """
        Evaluate a single zero-cost proxy on an architecture.

        Args:
            proxy_name: one of Config.GRAD_NORM, Config.SYNFLOW, etc.
            arch_id: e.g. "8-16-32-64"

        Returns:
            Proxy score (float)
        """
        evaluator = get_evaluator(proxy_name)
        model = self._create_model(arch_id)

        # Initialize embedding for proxies that need full forward pass
        if proxy_name in self._LABEL_DEPENDENT_PROXIES or proxy_name in self._TENSOR_INPUT_PROXIES:
            if self._cached_embedding is None:
                model.init_embedding(requires_grad=False)
                self._cached_embedding = model.embedding
            else:
                model.init_embedding(cached_embedding=self._cached_embedding, requires_grad=False)

        model = model.to(self.device)
        model.train()
        model.zero_grad()

        space_name = Config.MLPSP

        if proxy_name in self._DATA_AGNOSTIC_PROXIES:
            # SynFlow / WeightNorm: use all-ones embedding, bypass embedding layer
            batch_data = model.generate_all_ones_embedding(self.batch_size).float().to(self.device)
            batch_labels = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
            score = evaluator.evaluate(model, self.device, batch_data, batch_labels, space_name)

        elif proxy_name in self._TENSOR_INPUT_PROXIES:
            # NAS-WOT, JacobCov: use dict batch_data (with embedding)
            batch_data, batch_labels = self.prepare_batch_with_labels()
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)
            score = evaluator.evaluate(model, self.device, batch_data, batch_labels, space_name)

        else:
            # Label-dependent: need dict batch_data + labels
            batch_data, batch_labels = self.prepare_batch_with_labels()
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)
            score = evaluator.evaluate(model, self.device, batch_data, batch_labels, space_name)

        # Cleanup
        del model
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        return float(score) if math.isfinite(float(score)) else 0.0

    def evaluate_ptproxy(self, arch_id: str, weight_mode: str = "traj_width",
                         epsilon: float = 1e-5) -> Tuple[float, float]:
        """
        Evaluate pTProxy (our proposed proxy) on an architecture.

        Args:
            arch_id: e.g. "8-16-32-64"
            weight_mode: "traj_width", "traj", or "width"
            epsilon: perturbation magnitude

        Returns:
            (score, elapsed_time)
        """
        model = self._create_model(arch_id)
        batch_data = model.generate_all_ones_embedding(self.batch_size).float().to(self.device)
        model = model.to(self.device)

        score, elapsed = ptproxy_score(
            arch=model.mlp,  # evaluate the MLP head only
            batch_data=batch_data,
            device=self.device,
            use_wo_embedding=False,
            linearize_target=None,
            epsilon=epsilon,
            weight_mode=weight_mode,
            use_fp64=False,
        )

        del model
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        return score, elapsed

    def evaluate_all_proxies(self, arch_id: str,
                             include_ptproxy: bool = True) -> Dict[str, float]:
        """
        Evaluate all available proxies on an architecture.

        Args:
            arch_id: e.g. "8-16-32-64"
            include_ptproxy: whether to also evaluate pTProxy

        Returns:
            Dict mapping proxy_name -> score
        """
        results = {}

        for proxy_name in PROXY_EVALUATORS:
            try:
                score = self.evaluate_proxy(proxy_name, arch_id)
                results[proxy_name] = score
            except Exception as e:
                results[proxy_name] = float('nan')

        if include_ptproxy:
            try:
                score, _ = self.evaluate_ptproxy(arch_id)
                results[Config.PTPROXY] = score
            except Exception as e:
                results[Config.PTPROXY] = float('nan')

        return results
