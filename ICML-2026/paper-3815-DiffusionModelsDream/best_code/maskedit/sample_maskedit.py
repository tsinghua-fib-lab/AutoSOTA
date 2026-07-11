#!/usr/bin/env python3
"""
Run investigations: load a saved Simformer checkpoint, apply a list of
conditioning/topology masks ("investigations"), sample, and save
artifacts.

Example usage: IT IS RECOMMENDED TO USE THE SOURCE FILE run_maskedit_example.sh INSTEAD, WHICH CALLS THIS FUNCTION
  python -m maskedit.sample_maskedit \
    --cuda-visible-devices 0 \
    --project-root ./training_data/maskedit/data \
    --data-csv ./training_data/maskedit/data/train_set.csv \
    --indices-pkl ./training_data/maskedit/data/train_indices.pkl \
    --checkpoint-dir ./training_data/maskedit/model/MaskeditModel/ \
    --investigations example_settings_maskedit.json \
    --output-root ./output \
    --Nsamples 1000 --sample-steps 500

Notes
- This script **does not train**. It restores a model from `--checkpoint-dir`.
- It *tries* to discover training args from:
    1) --train-args-json (recommended), else
    2) <checkpoint-dir>/train_args.json, else
    3) a "log{name}_{sigma}.pkl" in the parent checkpoint root.
  If none are found, you must supply model shape flags via CLI (see below).

Folder layout written to `--output-root`:
  <output-root>/<invset>/<run_stamp>/
    meta.json                # summary with checkpoint path, data sources, seeds, etc.
    investigations.json      # exact investigations used
    train_args.json          # if discovered/provided
    inv_<investigation_name>/
      samples.npy            # (Nsamples, nvals, 1)
      samples.csv            # columns=labels
      sampling_args.json     # steps, Nsamples, rng seed, etc.
      condition.json         # condition & topo mask used (label-based)
"""

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

# ----------------------------
# Argparse
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample Simformer from a saved checkpoint for a set of investigations")

    # Environment / paths
    p.add_argument("--cuda-visible-devices", default="0", type=str)
    p.add_argument("--sys-path-parent", default="..", type=str, help="Path to add to sys.path (for local maskedit imports)")

    # Data
    p.add_argument("--project-root", default="./", type=str, help="Folder containing data CSV and indices PKL")
    p.add_argument("--data-csv", default="dataset.csv", type=str, help="CSV filename relative to project-root or absolute path")
    p.add_argument("--indices-pkl", default="indices.pkl", type=str, help="PKL filename with fullindices dict (or absolute path)")

    # Model restore
    p.add_argument("--checkpoint-dir", required=True, type=str, help="Directory containing Orbax 'train_state' for the run")
    p.add_argument("--train-args-json", default=None, type=str, help="Optional: explicit JSON of training args to avoid guessing")
    p.add_argument("--multicomponent", action="store_true", help="Use the Multicomponent model architecture instead of Simformer")

    # Investigations & output
    p.add_argument("--investigations", required=True, type=str, help="Path to investigations JSON (see example in prompt)")
    p.add_argument("--output-root", default="./investigation_runs", type=str)

    # Sampling (can override what was in training args)
    p.add_argument("--Nsamples", default=1000, type=int)
    p.add_argument("--sample-steps", default=500, type=int)
    p.add_argument("--seed", default=0, type=int)

    # Fallback model-shape flags (used if we fail to load from train_args)
    p.add_argument("--nlayers", default=8, type=int) #16
    p.add_argument("--dim-value", dest="dim_value", default=64, type=int)
    p.add_argument("--dim-id", dest="dim_id", default=64, type=int)
    p.add_argument("--dim-condition", dest="dim_condition", default=32, type=int)
    p.add_argument("--num-heads", dest="num_heads", default=2, type=int)
    p.add_argument("--time-dim", dest="time_dim", default=64, type=int)
    p.add_argument("--split", default=0.8, type=float)
    p.add_argument("--dropout-rate", default=0.1) # None

    # Diffusion params (sigma is needed to construct SDE)
    p.add_argument("--sigma", default=2.5, type=float)

    return p


# ----------------------------
# Helpers
# ----------------------------

def set_cuda_env(cuda_mask: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_mask)


def add_sys_path(parent: str) -> None:
    sys.path.append(os.path.abspath(parent))


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_train_args(args: argparse.Namespace, ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    """Attempt to load training args dict.
    Priority: explicit --train-args-json > train_args.json next to checkpoint > old log*.pkl fallback.
    Returns dict or None.
    """
    # 1) explicit json
    if args.train_args_json:
        p = Path(args.train_args_json)
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def make_json_safe(obj):
    """
    Recursively convert obj into something json.dump can handle.
    - numpy/jax arrays -> lists
    - numpy scalars -> python scalars
    - dict/list/tuple -> recurse
    - nnx.Rngs and other odd types -> string (or minimal dict)
    """
    import numpy as np

    # Basic JSON types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    try:
        if hasattr(obj, "__array__"):
            return np.asarray(obj).tolist()
    except Exception:
        pass

    # Containers
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # Special known types we don't need in full fidelity
    try:
        # Try to render something stable-ish
        return str(obj)
    except Exception:
        return "<non-serializable>"

# ----------------------------
# Core
# ----------------------------

def run(ns: argparse.Namespace) -> None:
    # Environment first
    set_cuda_env(ns.cuda_visible_devices)
    add_sys_path(ns.sys_path_parent)

    # Imports that depend on CUDA/JAX visibility
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_bridge
    from flax import nnx
    import orbax.checkpoint as ocp

    from .datasets.dataset import dataset
    from .training.train_loop import initialize
    from .architectures.simformer.simformer_mask import Simformer
    from .training.diffusion.gaussian_prob_path import GaussianConditionalPath
    from .training.diffusion.diffusion_mask import DiffusionModel

    print(f"JAX backend: {jax.default_backend()}")

    # Resolve paths
    project_root = Path(ns.project_root).resolve()
    data_csv_path = (project_root / ns.data_csv) if not Path(ns.data_csv).is_absolute() else Path(ns.data_csv)
    indices_pkl_path = (project_root / ns.indices_pkl) if not Path(ns.indices_pkl).is_absolute() else Path(ns.indices_pkl)
    ckpt_dir = Path(ns.checkpoint_dir).resolve()
    inv_path = Path(ns.investigations).resolve()
    out_root = Path(ns.output_root).resolve()

    # Load investigations JSON (must be valid JSON)
    with open(inv_path, "r") as f:
        inv_spec = json.load(f)
    invset_name: str = inv_spec.get("name", f"invset_{now_stamp()}")
    inv_list: List[Dict[str, Any]] = inv_spec.get("investigations", [])
    if not inv_list:
        raise ValueError("Investigations JSON has no 'investigations' entries.")

    # Load dataset
    df = pd.read_csv(data_csv_path)
    data = np.expand_dims(df.to_numpy(), axis=-1)  # (n, nodes, 1)
    lablist = list(df.columns)

    with open(indices_pkl_path, "rb") as f:
        fullindices: Dict[str, Any] = pickle.load(f)

    # Wrap & compute masks
    full_data = dataset(data, labels=lablist)
    indices = full_data.blank_mod(fullindices["indices"])  # apply blanking
    indices = indices[:-1]  # remove mask row from indices (topo mask lives in components tail)

    nvals: int = int(full_data.iblank)  # number of value labels
   

    # Training args discovery / fallback
    train_args = load_train_args(ns, ckpt_dir) or {}
    
    # Model shape params (prefer train_args, else CLI)
    dim_id = train_args.get("dim_id", ns.dim_id)
    dim_value = train_args.get("dim_value", ns.dim_value)
    dim_condition = train_args.get("dim_condition", ns.dim_condition)
    nlayers = train_args.get("nlayers", ns.nlayers)
    num_heads = train_args.get("num_heads", ns.num_heads)
    time_dim = train_args.get("time_dim", ns.time_dim)
    Ndata = int(train_args.get("split", ns.split) * data.shape[0])
    datastore = full_data.split(Ndata)
    num_components = sum([len(comps) for comps in train_args["indices"]])
    num_component_types = len(train_args["indices"])

    # Diffusion sigma
    sigma = (train_args.get("diffusion", {}) or {}).get("sigma", ns.sigma) if "diffusion" in train_args else ns.sigma

    # Build SDE and DiffusionModel scaffold to restore
    sde = GaussianConditionalPath(sigma)

    # Conditionally import the model and prepare its specific arguments
    model_kwargs = {}
    print("Using Simformer model architecture.")
    from .architectures.simformer.simformer_mask import Simformer
    ModelClass = Simformer

    model_kwargs = {
        "nlayers": nlayers,  # <-- MUST match training
        "edge2d": jnp.ones((nvals, nvals)), # single matrix
    }

    # We need optx and graphdef to build DiffusionModel; reuse initialize for shapes
    # Note: optimizer and training loaders are unused here; we only need optx and a model to derive graphdef
    dummy_train_args = dict(train_args)
    if not dummy_train_args:
        # minimal required fields for initialize
        dummy_train_args = {
            "n_devices": jax.local_device_count(),
            "rng": jax.random.PRNGKey(ns.seed),
            "rng_nnx": nnx.Rngs(ns.seed),
            "ids": jnp.arange(nvals),
            "num_components": num_components,
            "nvals": nvals,
            "num_heads": num_heads,
            "time_dim": time_dim,
        }


    # Call initialize
    optimizer, optx, model = initialize(
        dummy_train_args,
        ModelClass,
        sde,
        num_nodes=nvals,
        dim_id=dim_id,
        dim_value=dim_value,
        dim_condition=dim_condition,
        rngs=dummy_train_args.get("rng_nnx", nnx.Rngs(ns.seed)),
        num_heads=num_heads,
        time_dim=time_dim,
        dropout_rate = ns.dropout_rate,
        **model_kwargs,
    )

    graphdef, _ = nnx.split(model)
    diff = DiffusionModel(sde, dummy_train_args, graphdef, optx)

    # Restore trained state from Orbax
    checkpointer = ocp.StandardCheckpointer()
    restored_pure_dict = checkpointer.restore(ckpt_dir / "train_state")["state"]
    checkpointer.wait_until_finished()

    # Recreate model with the same constructor signature and replace by pure dict
    abstract_model = nnx.eval_shape(lambda: ModelClass(
        sde,
        num_nodes=nvals,
        dim_id=dim_id,
        dim_value=dim_value,
        dim_condition=dim_condition,
        rngs=nnx.Rngs(ns.seed),
        num_heads=num_heads,
        time_dim=time_dim,
        dropout_rate = ns.dropout_rate,
        **model_kwargs, # Unpack model-specific args here
    ))

    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    trained_model = nnx.merge(graphdef, abstract_state)

    # Prepare output directory
    run_dir = out_root / invset_name / now_stamp()
    ensure_dir(run_dir)

    with open(run_dir / "investigations.json", "w") as f:
        json.dump(inv_spec, f, indent=2)

    if train_args:
        with open(run_dir / "train_args.json", "w") as f:
            json.dump(make_json_safe(train_args), f, indent=2)

    # Save meta info
    meta = {
        "model_type": "Multicomponent" if ns.multicomponent else "Simformer",
        "checkpoint_dir": str(ckpt_dir), "project_root": str(project_root),
        "data_csv": str(data_csv_path), "indices_pkl": str(indices_pkl_path),
        "nvals": nvals, "num_components": num_components, "labels": lablist, "sigma": float(sigma),
        "rng_seed": int(ns.seed), "Nsamples": int(ns.Nsamples),
        "sample_steps": int(ns.sample_steps), "backend": str(xla_bridge.get_backend().platform),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rng = train_args.get("rng", jax.random.PRNGKey(ns.seed)) if train_args else jax.random.PRNGKey(ns.seed)
    rng_nnx = train_args.get("rng_nnx", nnx.Rngs(ns.seed)) if train_args else nnx.Rngs(ns.seed)
    node_ids = (train_args.get("ids") if train_args else jnp.arange(nvals)).astype(jnp.int32)


    # --- Investigation loop ---
    for inv in inv_list:
        inv_name = inv.get("name", "investigation")
        inv_dir = run_dir / f"inv_{inv_name}"
        ensure_dir(inv_dir)

        # Merge local and global properties
        for key, fixed_val in inv_spec["fixed"].items():
            local_val = inv.get(key)

            # If both are dicts: merge
            if isinstance(fixed_val, dict) and isinstance(local_val, dict):
                inv[key] = { **fixed_val, **local_val }

            # Otherwise: fixed overrides local
            else:
                inv[key] = fixed_val

        print(inv)

        # Build topo mask from fullindices groups
        topo_mask = jnp.zeros((1, num_components, 1))
        i = 0
        for component_type, value in fullindices.items():
            if component_type in ("indices", "results"):
                continue
            num_components_type = len(value)
            if component_type in inv.get("topo_mask", {}):
                how_many = int(inv["topo_mask"][component_type])
                for j in range(min(how_many, num_components_type)):
                    topo_mask = topo_mask.at[0, i + j, 0].set(1)
            i += num_components_type

        # Build condition (mask & values) from labels
        condition_value = jnp.zeros((1, nvals, 1))
        condition_mask = jnp.zeros((1, nvals, 1))

        i = 0
        for component_lab in lablist:
            if component_lab in inv["condition"].keys():
                condition_mask = condition_mask.at[0,i,0].set(1)
                condition_value = condition_value.at[0,i,0].set(inv["condition"][component_lab])
            i += 1
    
        condition = (
            jnp.tile(condition_mask.astype(jnp.bool_), (ns.Nsamples, 1, 1)),
            jnp.tile(condition_value, (ns.Nsamples, 1, 1)),
            jnp.tile(topo_mask, (ns.Nsamples, 1, 1)),
        )

        # Sample
        sampledata = diff.sample(
            trained_model,
            rng,
            rng_nnx,
            shape=(ns.Nsamples, nvals, 1),
            condition=condition,
            node_ids=node_ids,
            steps=ns.sample_steps,
        )

        # Concatenate topo mask
        sample_concatenated = jnp.concatenate((sampledata, condition[2]), axis=1)
        datastore.addDset(dataset(sample_concatenated, norm=True), "samples", norm=True)
        sampledata = datastore.sets["samples"].data[:, :sampledata.shape[1]]

        # Add the values created by lambda functions
        lablist_full = lablist[:]
        for d in inv_spec.get("fixed", {}).get("lambdas", []) + inv_spec.get("lambdas", []):

            name = d["name"]
            func_str = d["function"]
            local_vars = {k: sampledata[:,lablist.index(v)] for k, v in d.items() if k not in ["name", "function"]}
            
            # Evaluate the lambda in this local scope
            lablist_full = lablist_full + ["/lambdas/"+name]
            lambda_fn = eval(func_str, {}, {})
            new_column = lambda_fn(**local_vars)
            sampledata = np.append(sampledata, new_column[:, None, :], axis=1)

        # Save arrays
        np_samples = np.asarray(sampledata)
        np.save(inv_dir / "samples.npy", np_samples)
        
        df_samples = pd.DataFrame(np_samples[:, :, 0], columns=lablist_full)
        df_samples.to_csv(inv_dir / "samples.csv", index=False)

        # Save sampling args and the exact condition used
        with open(inv_dir / "sampling_args.json", "w") as f:
            json.dump({
                "Nsamples": ns.Nsamples,
                "sample_steps": ns.sample_steps,
                "seed": ns.seed,
                "sigma": float(sigma),
            }, f, indent=2)

        with open(inv_dir / "condition.json", "w") as f:
            json.dump({
                "condition": inv,
                "topo_mask": inv.get("topo_mask", {}),
            }, f, indent=2)

        print(f"[OK] Saved {inv_name} → {inv_dir}")

    print(f"\nAll done. Outputs in: {run_dir}")


def main():
    ns = build_parser().parse_args()
    run(ns)


if __name__ == "__main__":
    main()
