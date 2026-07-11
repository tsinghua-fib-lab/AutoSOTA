from __future__ import annotations
import os
import sys
import json
import time
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import numpy as np
import pandas as pd

os.environ["JAX_PLUGINS"] = "cuda12"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# Get the directory one level above the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# ----------------------------
# Helpers
# ----------------------------

def set_cuda_env(cuda_mask: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_mask)


def add_sys_path(parent: str) -> None:
    sys.path.append(os.path.abspath(parent))


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_train_args(train_args_json, ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    """Attempt to load training args dict.
    Priority: explicit --train-args-json > train_args.json next to checkpoint > old log*.pkl fallback.
    Returns dict or None.
    """
    # 1) explicit json
    if train_args_json:
        p = Path(train_args_json)
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
        else:
            print(f"[WARN] --train-args-json not found at {p}")

    # 2) train_args.json next to checkpoint
    p_json = ckpt_dir / "train_args.json"
    if p_json.exists():
        try:
            with open(p_json, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {p_json}: {e}")

    # 3) legacy pkl pattern in parent dir
    parent = ckpt_dir.parent
    try:
        # Heuristic: pick newest file that starts with 'log' and endswith '.pkl'
        candidates = sorted(parent.glob("log*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            with open(cand, "rb") as fh:
                payload = pickle.load(fh)
                if isinstance(payload, dict) and "settings" in payload:
                    return payload["settings"]
    except Exception as e:
        print(f"[WARN] Could not load legacy pkl train args: {e}")

    # Nothing
    return None

def sample(args,df):
    # Environment first
    set_cuda_env(args["cuda_visible_devices"])
    add_sys_path(args["sys_path_parent"])

    # Imports that depend on CUDA/JAX visibility
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_bridge
    from flax import nnx
    import orbax.checkpoint as ocp

    from maskedit.datasets.dataset import dataset, datasetSet
    from maskedit.training.train_loop import initialize
    from maskedit.architectures.simformer.simformer_mask import Simformer
    from maskedit.training.diffusion.gaussian_prob_path import GaussianConditionalPath
    from maskedit.training.diffusion.diffusion_mask import DiffusionModel

    print(f"JAX backend: {jax.default_backend()}")

    # Resolve paths
    project_root = Path(args["project_root"]).resolve()
    indices_pkl_path = (project_root / args["indices_pkl"]) if not Path(args["indices_pkl"]).is_absolute() else Path(args["indices_pkl"])
    ckpt_dir = Path(args["checkpoint_dir"]).resolve()

    # Load dataset
    data = np.expand_dims(df.to_numpy(), axis=-1)  # (n, nodes, 1)
    lablist = list(df.columns)

    with open(indices_pkl_path, "rb") as f:
        fullindices: Dict[str, Any] = pickle.load(f)

    # Wrap & compute masks
    Nsamples = data.shape[0]
    full_data = dataset(data, labels=lablist)
    indices = full_data.blank_mod(fullindices["indices"])  # apply blanking
    topo_mask_index = indices[-1]
    indices = indices[:-1]  # remove mask row from indices (topo mask lives in components tail)

    nvals: int = int(full_data.iblank)  # number of value labels

    # Training args discovery / fallback
    train_args = load_train_args(args["train_args_json"], ckpt_dir) or {}
    
    # Model shape params (prefer train_args, else CLI)
    dim_id = train_args.get("dim_id", args["dim_id"])
    dim_value = train_args.get("dim_value", args["dim_value"])
    dim_condition = train_args.get("dim_condition", args["dim_condition"])
    nlayers = train_args.get("nlayers", args["nlayers"])
    num_heads = train_args.get("num_heads", args["num_heads"])
    time_dim = train_args.get("time_dim", args["time_dim"])
    Ndata = int(train_args.get("split", args["split"]) * data.shape[0])
    datastore = full_data.split(Ndata)
    num_components = sum([len(comps) for comps in train_args["indices"]]) # OR full_data.data[:, full_data.iblank:, :].shape[1] # number of components in the topomask
    num_component_types = len(train_args["indices"])
    

    # Diffusion sigma
    sigma = (train_args.get("diffusion", {}) or {}).get("sigma", args["sigma"]) if "diffusion" in train_args else args["sigma"]

    # Build SDE and DiffusionModel scaffold to restore
    sde = GaussianConditionalPath(sigma)

    # Conditionally import the model and prepare its specific arguments
    model_kwargs = {}

    print("Using Simformer model architecture.")
    from maskedit.architectures.simformer.simformer_mask import Simformer
    ModelClass = Simformer

    model_kwargs = {
        "nlayers": args["nlayers"], # single int
        "edge2d": jnp.ones((nvals, nvals)), # single matrix
    }

    # We need optx and graphdef to build DiffusionModel; reuse initialize for shapes
    # Note: optimizer and training loaders are unused here; we only need optx and a model to derive graphdef
    dummy_train_args = dict(train_args)
    if not dummy_train_args:
        # minimal required fields for initialize
        dummy_train_args = {
            "n_devices": jax.local_device_count(),
            "rng": jax.random.PRNGKey(args["seed"]),
            "rng_nnx": nnx.Rngs(args["seed"]),
            "ids": jnp.arange(nvals),
            "num_components": num_components,
            "nvals": nvals,
            "num_heads": num_heads,
            "time_dim": time_dim,
        }


    # Call initialize with the correct model class and unpacked kwargs
    optimizer, optx, model = initialize(
        dummy_train_args,
        ModelClass,
        sde,
        num_nodes=nvals,
        dim_id=dim_id,
        dim_value=dim_value,
        dim_condition=dim_condition,
        rngs=dummy_train_args.get("rng_nnx", nnx.Rngs(args["seed"])),
        num_heads=num_heads,
        time_dim=time_dim,
        dropout_rate = args["dropout_rate"],
        **model_kwargs, # Unpack model-specific args here
    )

    graphdef, _ = nnx.split(model)
    diff = DiffusionModel(sde, dummy_train_args, graphdef, optx)

    # Restore trained state from Orbax
    checkpointer = ocp.StandardCheckpointer()
    restored_pure_dict = checkpointer.restore(ckpt_dir / "train_state")["state"]
    checkpointer.wait_until_finished()

    abstract_model = nnx.eval_shape(lambda: ModelClass(
        sde,
        num_nodes=nvals,
        dim_id=dim_id,
        dim_value=dim_value,
        dim_condition=dim_condition,
        rngs=nnx.Rngs(args["seed"]),
        num_heads=num_heads,
        time_dim=time_dim,
        dropout_rate = args["dropout_rate"],
        **model_kwargs, # Unpack model-specific args here
    ))
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    trained_model = nnx.merge(graphdef, abstract_state)

    # Convenience common tensors
    rng = train_args.get("rng", jax.random.PRNGKey(args["seed"])) if train_args else jax.random.PRNGKey(args["seed"])
    rng_nnx = train_args.get("rng_nnx", nnx.Rngs(args["seed"])) if train_args else nnx.Rngs(args["seed"])
    node_ids = (train_args.get("ids") if train_args else jnp.arange(nvals)).astype(jnp.int32)
    
    condition_value = jnp.zeros((1, nvals, 1))
    condition_mask = jnp.zeros((1, nvals, 1))
    topo_mask = full_data.data[:,topo_mask_index[0],:]

    condition = (
        jnp.tile(condition_mask.astype(jnp.bool_), (Nsamples, 1, 1)),
        jnp.tile(condition_value, (Nsamples, 1, 1)),
        topo_mask,
    )

    # Sample with batching to avoid GPU OOM on large topologies
    # Process samples in batches to limit peak GPU memory usage
    sample_batch_size = 200  # samples per batch, safe for ~10GB free GPU memory
    all_sample_batches = []
    for batch_start in range(0, Nsamples, sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, Nsamples)
        batch_n = batch_end - batch_start
        batch_shape = (batch_n, nvals, 1)
        batch_condition = (
            condition[0][batch_start:batch_end],
            condition[1][batch_start:batch_end],
            condition[2][batch_start:batch_end],
        )
        batch_samples = diff.sample(
            trained_model,
            jax.random.fold_in(rng, batch_start),
            rng_nnx,
            shape=batch_shape,
            condition=batch_condition,
            node_ids=node_ids,
            steps=args["sample_steps"],
        )
        all_sample_batches.append(batch_samples)
        # Cleanup between batches to prevent memory accumulation
        jax.clear_caches()
    sampledata = jnp.concatenate(all_sample_batches, axis=0)
    
    sample_concatenated = jnp.concatenate((sampledata, condition[2]), axis=1)

    datastore = datasetSet(full_data)
    datastore.addDset(dataset(sample_concatenated, norm=True), "samples", norm=True)

    return datastore


def main():
    ns = build_parser().parse_args()
    run(ns)


if __name__ == "__main__":
    main()
