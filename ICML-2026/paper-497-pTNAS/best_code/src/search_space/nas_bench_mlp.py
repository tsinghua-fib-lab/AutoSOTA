"""
NAS-Bench-Tabular MLP Model

Ported from VLDB_code/TRAILS MLP search space.
This is the DNNModel used for the NAS-Bench-Tabular experiments
(Frappe, UCI Diabetes, Criteo) with sparse feature input format.

Architecture encoding: hyphen-separated hidden layer sizes, e.g. "8-16-32-64"
Search space: 20^4 = 160,000 architectures (Frappe/UCI) or 10^4 = 10,000 (Criteo)
"""

import itertools
import random
from copy import deepcopy
from typing import List, Tuple, Generator, Optional

import torch
import torch.nn as nn


# ============================================================
# Layer choices (from TRAILS)
# ============================================================

DEFAULT_LAYER_CHOICES_20 = [
    8, 16, 24, 32,
    48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    384, 512,
]

DEFAULT_LAYER_CHOICES_10 = [
    8, 16, 32,
    48, 96, 112, 144, 176, 240,
    384,
]

# ============================================================
# Dataset configurations
# ============================================================

DATASET_CONFIGS = {
    "frappe": {
        "nfield": 10,
        "nfeat": 5500,
        "nemb": 10,
        "num_layers": 4,
        "num_labels": 1,
        "layer_choices": DEFAULT_LAYER_CHOICES_20,
    },
    "uci_diabetes": {
        "nfield": 43,
        "nfeat": 369,
        "nemb": 10,
        "num_layers": 4,
        "num_labels": 1,
        "layer_choices": DEFAULT_LAYER_CHOICES_20,
    },
    "criteo": {
        "nfield": 39,
        "nfeat": 2100000,
        "nemb": 10,
        "num_layers": 4,
        "num_labels": 1,
        "layer_choices": DEFAULT_LAYER_CHOICES_10,
    },
}


# ============================================================
# Model components
# ============================================================

class Embedding(nn.Module):
    """Sparse feature embedding layer."""

    def __init__(self, nfeat: int, nemb: int):
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: {'id': LongTensor [B, F], 'value': FloatTensor [B, F]}
        Returns:
            Tensor [B, F, E]
        """
        emb = self.embedding(x['id'])       # [B, F, E]
        return emb * x['value'].unsqueeze(2) # [B, F, E]


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""

    def __init__(self, ninput: int, hidden_layer_list: list,
                 dropout_rate: float, noutput: int, use_bn: bool):
        super().__init__()
        layers = []
        for layer_size in hidden_layer_list:
            layers.append(nn.Linear(ninput, layer_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            ninput = layer_size

        last_dim = hidden_layer_list[-1] if hidden_layer_list else ninput
        layers.append(nn.Linear(last_dim, noutput))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


class DNNModel(nn.Module):
    """
    Deep Neural Network for tabular data with sparse feature input.
    Used in NAS-Bench-Tabular experiments (Frappe, UCI Diabetes, Criteo).
    """

    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 hidden_layer_list: list, dropout_rate: float = 0.0,
                 noutput: int = 1, use_bn: bool = True):
        super().__init__()
        self.nfeat = nfeat
        self.nemb = nemb
        self.nfield = nfield
        self.embedding = None
        self.mlp_ninput = nfield * nemb
        self.mlp = MLP(self.mlp_ninput, hidden_layer_list,
                       dropout_rate, noutput, use_bn)
        self.hidden_layer_list = hidden_layer_list

    def init_embedding(self, cached_embedding=None, requires_grad=False):
        """Initialize the embedding layer (can be cached for speed)."""
        if self.embedding is None:
            if cached_embedding is None:
                self.embedding = Embedding(self.nfeat, self.nemb)
            else:
                self.embedding = cached_embedding
        if not requires_grad:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def generate_all_ones_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """Generate all-ones input for proxy evaluation (bypasses embedding)."""
        return torch.ones(batch_size, self.mlp_ninput).double()

    def forward_wo_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without embedding (for proxy evaluation)."""
        y = self.mlp(x)
        return y.squeeze(1)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Full forward pass with embedding.
        Args:
            x: {'id': LongTensor [B, F], 'value': FloatTensor [B, F]}
        """
        x_emb = self.embedding(x)                    # [B, F, E]
        y = self.mlp(x_emb.view(-1, self.mlp_ninput)) # [B, noutput]
        return y.squeeze(1)

    def estimate_capacity(self) -> int:
        """Count total parameters in the MLP (exclude embedding)."""
        return sum(p.numel() for p in self.mlp.parameters())


# ============================================================
# NAS-Bench-Tabular Search Space
# ============================================================

class NASBenchTabularSpace:
    """
    Search space for NAS-Bench-Tabular experiments.
    Wraps DNNModel creation and architecture enumeration.
    """

    def __init__(self, dataset: str):
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}. "
                             f"Choose from {list(DATASET_CONFIGS.keys())}")
        self.dataset = dataset
        cfg = DATASET_CONFIGS[dataset]
        self.nfield = cfg["nfield"]
        self.nfeat = cfg["nfeat"]
        self.nemb = cfg["nemb"]
        self.num_layers = cfg["num_layers"]
        self.num_labels = cfg["num_labels"]
        self.layer_choices = cfg["layer_choices"]

    def __len__(self) -> int:
        return len(self.layer_choices) ** self.num_layers

    def new_architecture(self, arch_id: str, use_bn: bool = True) -> DNNModel:
        """Create a DNNModel from architecture encoding string."""
        hidden_layer_list = [int(x) for x in arch_id.split("-")]
        return DNNModel(
            nfield=self.nfield,
            nfeat=self.nfeat,
            nemb=self.nemb,
            hidden_layer_list=hidden_layer_list,
            dropout_rate=0,
            noutput=self.num_labels,
            use_bn=use_bn,
        )

    def random_architecture_id(self) -> str:
        """Sample a random architecture."""
        arch = [random.choice(self.layer_choices) for _ in range(self.num_layers)]
        return "-".join(str(x) for x in arch)

    def sample_all_models(self) -> Generator[Tuple[str, List[int]], None, None]:
        """Enumerate all architectures in the search space (shuffled)."""
        space = [self.layer_choices] * self.num_layers
        combinations = list(itertools.product(*space))
        random.shuffle(combinations)
        for combo in combinations:
            hidden_list = list(combo)
            arch_id = "-".join(str(x) for x in hidden_list)
            yield arch_id, hidden_list

    def mutate_architecture(self, arch_id: str) -> str:
        """Mutate one random layer of the architecture."""
        hidden_list = [int(x) for x in arch_id.split("-")]
        idx = random.randint(0, len(hidden_list) - 1)
        while True:
            new_val = random.choice(self.layer_choices)
            if new_val != hidden_list[idx]:
                hidden_list[idx] = new_val
                return "-".join(str(x) for x in hidden_list)

    @staticmethod
    def arch_id_to_list(arch_id: str) -> List[int]:
        return [int(x) for x in arch_id.split("-")]

    @staticmethod
    def list_to_arch_id(hidden_list: List[int]) -> str:
        return "-".join(str(x) for x in hidden_list)
