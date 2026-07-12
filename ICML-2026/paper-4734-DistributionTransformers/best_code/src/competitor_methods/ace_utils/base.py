# Taken from https://github.com/acerbilab/amortized-conditioning-engine/blob/main/src/model/base.py
from competitor_methods.ace_utils.utils import AttrDict
import torch
import torch.nn as nn
from typing import Union
from .embedder import NonLinearEmbedder, EmbedderMarker, EmbedderMarkerPrior
from .encoder import TNPDEncoder
from .target_head import GaussianHead, MixtureGaussian

Embedder = Union[NonLinearEmbedder, EmbedderMarker, EmbedderMarkerPrior]
TargetHead = Union[GaussianHead, MixtureGaussian]


class ACEBaseTransformer(nn.Module):
    """
    Base transformer model that uses an embedder, encoder, and head to process
    input data.

    Attributes:
        embedder (Embedder): An embedder module used to convert input data into
            embeddings.
        encoder (TNPDEncoder): An encoder module that performs the attention mechanism.
        head (TargetHead): A head module that outputs predictions or computes
            log-likelihoods.
    """

    def __init__(
        self, embedder: Embedder, encoder: TNPDEncoder, head: TargetHead
    ) -> None:
        """
        Initializes the BaseTransformer with the specified embedder, encoder,
        and head modules.

        Args:
            embedder (Embedder): An embedder instance to embed the input data.
            encoder (TNPDEncoder): An encoder instance that performs the attention
                mechanism.
            head (TargetHead): A head instance that outputs predictions or compute
                log-likelihoods.
        """
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: AttrDict,
        predict: bool = False,
        reduce_ll: bool = True,
        num_samples: int = 1000,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model, processing the input batch
        of data.

        Args:
            batch (AttrDict): A batch of input data to be processed.
            predict (bool): Whether to output predictions (True) or not (False).
            reduce_ll (bool): Whether to reduce the log-likelihood values (if
                applicable).
            num_samples (int): Number of samples from the head distribution (if
                applicable).

        Returns:
            torch.Tensor: The output tensor from the head module, which could be
                predictions or log-likelihoods.
        """
        embedding = self.embedder(batch)
        encoding = self.encoder(batch, embedding)

        if predict:
            return self.head(batch, encoding, predict=True, num_samples=num_samples)

        else:
            return self.head(batch, encoding, predict=False, reduce_ll=reduce_ll)