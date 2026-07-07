"""Embedding modules for QuITE-equipped MTS backbones.

Available modes (selected via args.mode + args.irr_emb):
    irr_emb=True
      - 'quite' : QuITEEmbedding         (paper main method, Eq. 5-13)
      - 'mean'  : MeanPoolEmbedding      (Table 5 baseline)
      - 'mtand' : MTANDEmbedding         (Table 5 baseline)
    irr_emb=False
      - 'add'    : *BaselineEmbedding(mode='add')    (Table 5 baseline)
      - 'concat' : *BaselineEmbedding(mode='concat') (Table 5 baseline)
      - other    : *BaselineEmbedding(mode='vanilla') / PatchMixerLinearEmbedding

`get_embedding(args, family)` returns a single nn.Module appropriate for the
selected (mode, family) combination.
"""
from models.embeddings._base import LearnableTE
from models.embeddings.quite import QuITEEmbedding
from models.embeddings.mean_pool import MeanPoolEmbedding
from models.embeddings.mtand import MTANDEmbedding
from models.embeddings.baseline import (
    PatchBaselineEmbedding, VariateBaselineEmbedding, PatchMixerLinearEmbedding,
)


def get_embedding(args, family):
    """Build the embedding module for (mode, family).

    family: 'variate' | 'patch'
    """
    if args.irr_emb:
        if args.mode == 'quite':
            return QuITEEmbedding(args, family=family)
        if args.mode == 'mean':
            return MeanPoolEmbedding(args, family=family)
        if args.mode == 'mtand':
            return MTANDEmbedding(args, family=family)
        raise ValueError(f"Unknown irr_emb mode: {args.mode}")

    # Non-irregular baselines
    mode = args.mode if args.mode in ('add', 'concat') else 'vanilla'
    if family == 'patch':
        if args.model == 'patchmixer' and mode == 'vanilla':
            return PatchMixerLinearEmbedding(args)
        return PatchBaselineEmbedding(args, mode=mode)
    return VariateBaselineEmbedding(args, mode=mode)


__all__ = [
    "LearnableTE", "QuITEEmbedding", "MeanPoolEmbedding", "MTANDEmbedding",
    "PatchBaselineEmbedding", "VariateBaselineEmbedding", "PatchMixerLinearEmbedding",
    "get_embedding",
]
