"""
CITE-seq single-cell data from NeurIPS 2022 Multimodal Single-cell Integration challenge.

Components:
- data: Data loading and preprocessing utilities
- trainer: CiteSeqTrainer class for training
"""

from experiments.citeseq.trainer import CiteSeqTrainer

__all__ = ["CiteSeqTrainer"]
