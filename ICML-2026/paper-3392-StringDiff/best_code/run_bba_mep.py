#!/usr/bin/env python3
"""Reproduce BBA MEP energy profile from ScoreMD.
Paper: Probing the Geometry of Diffusion Models with the String Method
Metric: Peak Energy Along Pathway (lower_better)
Target: MEP ~10 kbT vs Initial String ~50 kbT
"""

import os, sys, argparse, json
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# ---------------------------------------------------------------------------
# Reparametrization (pure JAX, mirrors diffusion_strings/reparametrization.py)
# ---------------------------------------------------------------------------
def _segment_lengths_rn(segments):
    return jnp.linalg.norm(segments.reshape(segments.shape[0], -1), axis=1)

def uniform_string_repametrize_rn_linear(string, n_new):
    segments = jnp.diff(string, axis=0)
    seg_lengths = _segment_lengths_rn(segments)
    total = jnp.sum(seg_lengths)
    if string.shape[0] <= 1 or total < 1e-12:
        return jnp.tile(string[0:1], (n_new,) + (1,) * (string.ndim - 1))

    cumulative = jnp.concatenate([jnp.zeros(1, dtype=seg_lengths.dtype), jnp.cumsum(seg_lengths)])
    new_cum = jnp.linspace(0, total, n_new)
    indices = jnp.clip(jnp.searchsorted(cumulative, new_cum) - 1, 0, cumulative.shape[0] - 2)
    seg_len = seg_lengths[indices]
    rel_prog = jnp.where(seg_len > 1e-12, (new_cum - cumulative[indices]) / seg_len, 0.0)
    extra_dims = (1,) * (segments.ndim - 1)
    return string[indices] + segments[indices] * rel_prog.reshape(-1, *extra_dims)

# ---------------------------------------------------------------------------
# Load ScoreMD model and build adapter callables
# ---------------------------------------------------------------------------
def load_scoremd_model(model_dir, eval_t=1e-5):
    """Load ScoreMD BBA model and return energy/score callables."""
    import yaml
    from flax import serialization
    from orbax import checkpoint as orbax_ckpt

    # Load config
    with open(os.path.join(model_dir, '.hydra', 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    # Build model architecture matching config
    from scoremd.models.graph_transformer import GraphTransformer

    # Model parameters from config
    model_cfg = cfg['model']['transformer_large_score'] if 'transformer_large_score' in cfg['model'] else cfg['model']['transformer_large_potential']
    hidden_nf = model_cfg.get('hidden_nf', 128)
    feature_embedding_dim = model_cfg.get('feature_embedding_dim', 16)
    n_layers = model_cfg.get('n_layers', 3)
    dropout = model_cfg.get('dropout', 0.0)
    potential = model_cfg.get('potential', False)

    # Build the three ranged models
    from scoremd.models.base import RangedModel
    from scoremd.models.graph_transformer import GraphTransformerModelInfo

    # We need to build all three models and load them separately
    # The training config has:
    #   ranged_models[0]: transformer_large_score, [1.0, 0.6]
    #   ranged_models[1]: transformer_large_score, [0.6, 0.1]
    #   ranged_models[2]: transformer_large_potential, [0.1, 0.0]

    # For MEP at t ≈ 0, we primarily need the potential model (range [0.1, 0.0])
    # But for pure transport, we need score from all models

    from scoremd.data.dataset.protein import SingleProteinDataset

    # Load dataset to get normalization
    dataset = SingleProteinDataset(
        paths=['./storage/deshaw/bba-0_ca.h5', './storage/deshaw/bba-1_ca.h5'],
        tica_path='./storage/deshaw/bba_tica.pic',
        topology_path='./storage/deshaw/bba.pdb',
        train_split=0.8,
    )

    # Build the three sub-models
    from flax import linen as nn

    # Model 1: score model [1.0, 0.6]
    model1_info = GraphTransformerModelInfo(
        hidden_nf=hidden_nf,
        feature_embedding_dim=feature_embedding_dim,
        n_layers=n_layers,
        potential=False,
        dropout=dropout,
    )
    model1 = model1_info.build(dataset, t0=0.6, t1=1.0, rescale_time=True, clip_time=True,
                               norm_factor=jnp.array(1.0))

    # Model 2: score model [0.6, 0.1]
    model2_info = GraphTransformerModelInfo(
        hidden_nf=hidden_nf,
        feature_embedding_dim=feature_embedding_dim,
        n_layers=n_layers,
        potential=False,
        dropout=dropout,
    )
    model2 = model2_info.build(dataset, t0=0.1, t1=0.6, rescale_time=True, clip_time=True,
                               norm_factor=jnp.array(1.0))

    # Model 3: potential model [0.1, 0.0]
    model3_info = GraphTransformerModelInfo(
        hidden_nf=hidden_nf,
        feature_embedding_dim=feature_embedding_dim,
        n_layers=n_layers,
        potential=True,
        dropout=dropout,
    )
    model3 = model3_info.build(dataset, t0=0.0, t1=0.1, rescale_time=True, clip_time=True,
                               norm_factor=jnp.array(1.0))

    # Load checkpoints
    import orbax.checkpoint as ocp
    ckpt_mgr = ocp.CheckpointManager(os.path.join(model_dir, 'model'), ocp.Checkpointer(ocp.PyTreeCheckpointHandler()))

    # Load all params
    params1 = ckpt_mgr.restore(1800, items={'params': None})  # model_1 params
    # Actually the checkpoint structure is different. Let me try a simpler approach.
    # The checkpoint stores params directly

    # Let me try loading the full checkpoint structure
    import orbax
    ckpt_mgr = ocp.CheckpointManager(
        os.path.join(model_dir, 'model'),
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    )

    # The checkpoint at step 1800 should have params for all three models
    # Structure: step/ema_params/
    restored = ckpt_mgr.restore(1800)

    print("Checkpoint keys:", list(restored.keys()) if isinstance(restored, dict) else type(restored))
    print("Checkpoint structure:", jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), restored))

    return dataset, model1, model2, model3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/models/scoremd_models/models/bba/both')
    parser.add_argument('--n-images', type=int, default=51, help='Number of string images')
    parser.add_argument('--n-mep-iters', type=int, default=1000, help='MEP convergence iterations')
    parser.add_argument('--mep-step-size', type=float, default=0.01, help='MEP score step size')
    parser.add_argument('--eval-t', type=float, default=1e-5, help='Evaluation time (ScoreMD convention)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reparam-every', type=int, default=1)
    parser.add_argument('--output', default='bba_mep_results.json')
    args = parser.parse_args()

    # Set up
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    key = jax.random.PRNGKey(args.seed)

    print("Loading ScoreMD BBA model...")
    # For now, just test the loading path
    load_scoremd_model(args.model_dir, args.eval_t)

if __name__ == '__main__':
    main()
