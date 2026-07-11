#!/usr/bin/env python3
"""
Train Simformer w/ diffusion masking, sample, plot, and evaluate.

Usage example:
python -m maskedit.train_maskedit --name MaskeditModel_v2 --dropout-rate 0.
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Tuple, Dict, Any, Sequence

# ----------------------------
# Argparse
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Simformer (mask) with diffusion + evaluation")
    # Environment / paths
    p.add_argument("--cuda-visible-devices", default="0", type=str, help="CUDA device mask (e.g., '0' or '0,1')")
    p.add_argument("--project-root", default="./training_data/maskedit/data", type=str, help="Root path for dataset/labels/indices PKLs")
    p.add_argument("--checkpoint-root", default="./training_data/maskedit/model", type=str,
                   help="Root path to store checkpoints and logs")
    p.add_argument("--sys-path-parent", default=".", type=str, help="Path to add to sys.path (for local imports)")
    p.add_argument("--results-index", nargs="+", type=int, default=[8, 9])

    # Training hyperparameters (defaults match original script)
    p.add_argument("--name", default="MaskeditModel", type=str, help="Run name (used in checkpoint subdir and logs)")
    p.add_argument("--epochs", default=500, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--test-batch", default=256, type=int)
    p.add_argument("--window", default=500, type=int)
    p.add_argument("--split", default=0.8, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--learning-rate", default=1e-4, type=float)
    p.add_argument("--iterations", default=10, type=int, help="Training iterations per epoch (original default=10)")

    # Model sizes
    p.add_argument("--nlayers", default=16, type=int)
    p.add_argument("--dim-value", default=64, type=int)
    p.add_argument("--dim-id", default=64, type=int)
    p.add_argument("--dim-condition", default=32, type=int)
    p.add_argument("--num-heads", default=2, type=int)
    p.add_argument("--time-dim", default=64, type=int)
    p.add_argument("--dropout-rate", default=0.1)

    # Diffusion params
    p.add_argument("--T", default=1.0, type=float)
    p.add_argument("--T-min", dest="T_min", default=1e-5, type=float)
    p.add_argument("--sigma-min", default=1e-4, type=float)
    p.add_argument("--sigma-max", default=15.0, type=float)
    p.add_argument("--sigma", default=2.5)
    

    # Checkpoint manager
    p.add_argument("--max-to-keep", default=3, type=int)
    p.add_argument("--save-every", default=20, type=int, help="Save checkpoint every N epochs (tuple[0] in original)")

    # Sampling / evaluation
    p.add_argument("--sample-steps", default=500, type=int, help="Diffusion sampling steps")
    p.add_argument("--Nsamples", default=1000, type=int, help="Diffusion sampling steps")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting pairplot")
    p.add_argument("--metrics-prefix", default="metrics", type=str, help="Prefix for metrics file (default: 'metrics')")
    p.add_argument(
    "--sample",
    action="store_true",
    help="Set this flag to enable sample only"
)
    return p


# ----------------------------
# Helpers
# ----------------------------
def set_cuda_env(cuda_mask: str) -> None:
    # Must be set before importing JAX
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_mask)


def add_sys_path(parent: str) -> None:
    sys.path.append(os.path.abspath(parent))


def gpu_info() -> None:
    # Non-fatal; mirrors original behavior
    _ = os.system("nvidia-smi --query-gpu=name --format=csv,noheader")


# ----------------------------
# Main pipeline
# ----------------------------
def run(args: argparse.Namespace) -> None:
    # 1) Environment first
    set_cuda_env(args.cuda_visible_devices)
    add_sys_path(args.sys_path_parent)
    gpu_info()

    # 2) Imports that depend on CUDA visibility
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_bridge
    import numpy as np
    import pickle
    from flax import nnx
    import orbax.checkpoint as ocp

    print(xla_bridge.get_backend().platform)

    # 3) Load data
    pth = Path(args.project_root)
    import pandas as pd
    df = pd.read_csv(pth / "train_set.csv")
    lablist = list(df.keys())
    data = np.expand_dims(df.to_numpy(), axis=-1)
    
    with open(pth / "train_indices.pkl", "rb") as f:
        indices = pickle.load(f)["indices"]

    # 4) Dataset wrappers & masking
    from .datasets.dataset import dataset
    full_data = dataset(data, labels=lablist)
    indices = full_data.blank_mod(indices)
    indices = indices[:-1] # removes the masking from the indices (removes topmask from data)

    nvals = int(full_data.iblank)
    print(nvals)
    Ndata = int(args.split * data.shape[0])
    print("Train data size:", Ndata)

    # 5) Train args dict
    train_args: Dict[str, Any] = {}
    train_args["epochs"] = args.epochs
    train_args["batch_size"] = args.batch_size
    train_args["test_batch"] = args.test_batch
    train_args["window"] = args.window
    train_args["name"] = args.name
    train_args["split"] = args.split
    train_args["n_devices"] = len(args.cuda_visible_devices)
    train_args["rng"] = jax.random.PRNGKey(args.seed)
    train_args["rng_nnx"] = nnx.Rngs(args.seed)
    train_args["learning_rate"] = args.learning_rate
    train_args["iterations"] = args.iterations
    train_args["nvals"] = nvals
    train_args["indices"] = indices
    ncomp = full_data.data[:, full_data.iblank:, :].shape[1]
    train_args["num_components"] = ncomp
    train_args["ids"] = jnp.arange(nvals)
    train_args["num_heads"] = args.num_heads
    train_args["time_dim"] = args.time_dim
    train_args["diffusion"] = {
        "T": args.T,
        "T_min": args.T_min,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
    }
    train_args["topo"] = True

    
    train_args["results_index"]= args.results_index #Masses and result args.datasplit
    train_args["restore"] = (False,"best")

    # 6) Split data for loaders
    datastore = full_data.split(Ndata)

    # 7) Model + SDE + optimizer init
    from .training.train_loop import initialize
    from .architectures.simformer.simformer_mask import Simformer
    from .training.diffusion.gaussian_prob_path import GaussianConditionalPath

    sde = GaussianConditionalPath(args.sigma) # could set sigma

    optimizer, optx, model = initialize(
        train_args,
        Simformer,
        sde,
        num_nodes=nvals,
        dim_id=args.dim_id,
        dim_value=args.dim_value,
        dim_condition=args.dim_condition,
        nlayers=args.nlayers,
        edge2d=jnp.ones((nvals, nvals)),
        rngs=train_args["rng_nnx"],
        num_heads = args.num_heads,
        time_dim = args.time_dim,
        dropout_rate = args.dropout_rate,
    )

    # 8) Training/validation fns
    from .training.train_loop import train, load_ckpt
    from .training.diffusion.diffusion_mask import DiffusionModel
    from .datasets.dataloader import dataloader

    batched_data = dataloader(train_args, datastore)
    graphdef, _ = nnx.split(model)
    diff = DiffusionModel(sde, train_args, graphdef, optx)

    training = (batched_data.train.data, batched_data.load_train_split, diff.pstep)
    validation = (batched_data.test.data, batched_data.load_test_split, diff.pval)

    # 9) Checkpoint manager (step-numbered)
    ckpt_dir = (Path(args.checkpoint_root) / args.name).resolve()
    chkpt = (args.save_every, ckpt_dir)

    # 10) Train

    if not args.sample:
        trained_model, losslist, veriflist, bepoch = train(
            train_args, model, optx, training, validation, chkpt)

    else:
        checkpointer = ocp.StandardCheckpointer()
        restored_pure_dict = checkpointer.restore(ckpt_dir / "train_state")["state"]
        checkpointer.wait_until_finished()
        abstract_model = nnx.eval_shape(lambda: Simformer(sde, 
                                            num_nodes=nvals,
                                            dim_id=args.dim_id,
                                            dim_value=args.dim_value,
                                            dim_condition=args.dim_condition,
                                            nlayers=args.nlayers,
                                            edge2d=jnp.ones((nvals, nvals)),
                                            rngs=nnx.Rngs(args.seed),
                                            num_heads = args.num_heads,
                                            time_dim = args.time_dim,
                                            dropout_rate = args.dropout_rate,
                                            ))
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
        trained_model = nnx.merge(graphdef, abstract_state)

    # 11) Sampling once for checking (condition = zeros, zeros, ones on topo mask)
    condition = (
        jnp.zeros((args.Nsamples, nvals, 1)),
        jnp.zeros((args.Nsamples, nvals, 1)),
        jnp.ones((args.Nsamples, ncomp, 1)),
    )
    sampledata = diff.sample(
        trained_model,
        train_args["rng"],
        train_args["rng_nnx"],
        shape=(args.Nsamples, nvals, 1),
        condition=condition,
        node_ids=train_args["ids"],
        steps=args.sample_steps,
    )

    # Concat topo mask and store
    sample_concatenated = jnp.concatenate((sampledata, condition[2]), axis=1)
    samples = datastore.addDset(dataset(sample_concatenated, norm=True), "samples", norm=True)

    # 12) Save run log (unchanged location pattern)
    train_args["nlayers"] = args.nlayers
    train_args["dim_value"] = args.dim_value
    train_args["dim_id"] = args.dim_id
    train_args["dim_condition"] = args.dim_condition
    train_args["num_heads"] = args.num_heads
    train_args["time_dim"] = args.time_dim
    train_args["dropout_rate"] = args.dropout_rate
    output = {
        "samples": datastore.sets["samples"].data,
        "settings": train_args
    }
    log_path = Path(args.checkpoint_root) / f"log{args.name}_{args.sigma}.pkl"
    with open(log_path, "wb") as fh:
        import pickle as _p
        _p.dump(output, fh)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
