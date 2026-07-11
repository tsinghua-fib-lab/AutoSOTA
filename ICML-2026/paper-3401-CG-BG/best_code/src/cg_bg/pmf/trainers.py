import logging
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import optax
from chemtrain import trainers
from chemtrain.data import preprocessing
from chemutils.models import mace
from jax_md import partition, space
from torch.utils.data import DataLoader

from cg_bg.models.rbf_mlp import RBFMLP
from cg_bg.pmf.train_utils import MBForceMatching

logger = logging.getLogger(__name__)


def get_ala_trainer(
    dataset: dict,
    batch_size: int,
    epochs: int,
    init_lr: float,
    decay_rate: float,
    r_cutoff: float,
    hidden_irreps: str,
    readout_irreps: str,
    output_irreps: str,
    max_ell: int,
    num_interactions: int,
    correlation: int,
    rng: Any,
):
    warnings.filterwarnings("ignore", message="Explicitly requested dtype <class 'jax.numpy.float64'>")
    dataset = jax.tree_util.tree_map(jnp.asarray, dataset["training"])
    box = dataset["box"][0]
    R = dataset["R"][0]
    num_samples = dataset["R"].shape[0]
    species = dataset["species"][0]
    mask = dataset["mask"][0]
    displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

    neighbor_fn, stats = preprocessing.allocate_neighborlist(
        dataset,
        displacement_fn,
        box,
        r_cutoff=r_cutoff,
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=batch_size,
    )

    init_fn, gnn_energy_fn = mace.mace_neighborlist_pp(
        displacement_fn,
        r_cutoff=r_cutoff,
        n_species=len(species),
        max_edges=stats[1],
        per_particle=False,
        avg_num_neighbors=stats[2],
        mode="energy",
        hidden_irreps=hidden_irreps,
        max_ell=max_ell,
        num_interactions=num_interactions,
        correlation=correlation,
        readout_mlp_irreps=readout_irreps,
        output_irreps=output_irreps,
    )

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            assert "species" in dynamic_kwargs.keys(), "species not in dynamic_kwargs"

            if "mask" not in dynamic_kwargs:
                logger.warning("No mask provided; using all-positive mask.")
                dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)

            if "box" in dynamic_kwargs:
                logger.debug("Found box in energy kwargs")

            return gnn_energy_fn(energy_params, pos, neighbor, **dynamic_kwargs)

        return energy_fn

    init_params = init_fn(rng, R, neighbor_fn, species=species, mask=mask)

    total_steps = (epochs * num_samples) // batch_size
    scheduler = optax.exponential_decay(init_value=init_lr, transition_steps=total_steps, decay_rate=decay_rate)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1.0)
    )

    trainer = trainers.ForceMatching(
        init_params,
        optimizer,
        energy_fn_template,
        neighbor_fn,
        log_file="force_matching.log",
        batch_per_device=batch_size,
    )
    return trainer


def get_mb_trainer(
    kT: float,
    dataloader: DataLoader,
    hidden_dim: int,
    n_layers: int,
    learning_rate: float,
    rng: Any,
):
    model = RBFMLP(hidden_dim=hidden_dim, n_layers=n_layers, num_rbf_centers=100, sigma=5.0)
    trainer = MBForceMatching(dataloader=dataloader, model=model, learning_rate=learning_rate, kT=kT, rng=rng)
    return trainer
