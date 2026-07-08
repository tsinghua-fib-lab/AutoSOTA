from torch import nn, Tensor
import torch


class Memory(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vectors: int,
    ):
        self.memory = nn.Parameter(torch.empty((n_vectors, d_model)))
        nn.init.kaiming_normal_(self.memory)

    def forward(self, x: Tensor) -> Tensor:
        x @ self.memory


class MemMatch(nn.Module):
    def __init__(
        self,
    ):
        self.memory = Memory()

    def forward(self, x: Tensor) -> Tensor:
        pass
