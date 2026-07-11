import argparse
import functools
import os
import pathlib
import sys

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import numpy as onp

import jax
import jax.numpy as jnp
from jax import random

from jax_md_mod import custom_quantity, io
from jax_md import (
    partition, space, simulate
)

import optax

from matplotlib import pyplot as plt
from cycler import cycler

from collections import OrderedDict

from chemtrain.data import preprocessing
from chemtrain.deploy import exporter, graphs
from chemtrain.compose import mace_jax as mace_jax_compose

from chemtrain import trainers


from chemtrain.ensemble import sampling
from chemtrain import quantity, trainers
from chemtrain.quantity import observables


from mace_jax.modules.wrapper_ops import CuEquivarianceConfig


def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--disable_cue", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Run on device {args.device} in mode {'TEST' if args.test else 'PRODUCTION'}")
    return OrderedDict(
        seed=11,
        confs=[
            ("data/liquid_T_1979.pdb", 1979.0),
        ],
        simulator_type="nose_hoover",
        simulator_settings=OrderedDict(
            kT=1942.0 * quantity.kb, 
            thermostat_kwargs=OrderedDict(
                chain_steps=1, tau=2., chain_length=3,
            ),
        ),
        targets=OrderedDict(
            rdf = ["data/rdf_liquid.csv"],
        ),
        timings=OrderedDict(
            dt=4e-3,
            print_every=0.2 if not args.test else 0.1,
            t_equilib=10. if not args.test else 1.0,
            total_time=50. if not args.test else 2.0,
            # t_equilib=50.,  # 1000.0,
            # total_time=350.,  # 2000.0,
        ),
        optimizer=OrderedDict(
            init_lr=args.lr,
            lr_decay=1e-1,
            epochs=args.epochs,
            weight_decay=0e-2,
            batch=1, # Note: Batch size -1 possible -> All statepoints
            optimizer_kwargs=OrderedDict(
                b1=0.5,
                b2=0.9,
                eps=1e-8,
                eps_root=1e-16,
                nesterov=False,
            )
        ),
        gammas=OrderedDict(
            rdf = [1.0],
        ),
        reweighting_ratio=0.5,
        disable_cue=args.disable_cue,
    )

def main():

    config = get_default_config()
    out_dir = pathlib.Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(config["seed"])

    confs = load_confs(config, fractional=True)
    displacement_fn, shift_fn = space.periodic_general(
        confs["box"][0], fractional_coordinates=True)

    # Load the model
    torch_model, model_config = mace_jax_compose.load_foundational_model(
        family="mp", version="medium-0b3"
    )

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    nbrs_init, _ = preprocessing.allocate_neighborlist(
        confs, displacement_fn, confs["box"][0],
        model_config["r_max"] / 10., mask_key="mask",
        box_key="box", format=partition.Sparse, fractional_coordinates=True,
        capacity_multiplier=2.0
    )


    # Currently, models with cuequivariance are not supported in
    # chemtrain-deploy
    if config["disable_cue"]:
        cueq_config = None
    else:
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout=(
                'mul_ir'
            ),
            group=(
                'O3'
            ),
            optimize_all=True,
            conv_fusion=True,
        )

    print("Loaded model with config:", model_config)
    
    variables, apply_fn = mace_jax_compose.mace_jax_neighborlist(
        model_config, torch_model, displacement_fn, max_edge_multiplier=1.25,
        per_particle=False,
        scale_pos=0.1,  # Convert from Angstrom to nm
        scale_pot=96.185,  # Convert from eV to kJ/mol
        species_mapping=mace_jax_compose.AtomicNumberMapping(max_number=90),
        cueq_config=cueq_config
    )

    
    # init_params = jax.tree.map(jnp.zeros_like, variables["params"])
    init_params = variables["params"]

    def energy_fn_template(params):
        vars = {**variables}
        # vars["params"] = jax.tree.map(jnp.add, variables["params"], params)
        vars["params"] = params

        def energy_fn(position, neighbor, **kwargs):
            pot = apply_fn(vars, position, neighbor, **kwargs)
            return pot

        return energy_fn

    sim_template, timings = init_simulator(
        config, shift_fn, simulator=simulate.nvt_nose_hoover,
    )

    @jax.vmap
    def init_sim_state(key, sample):
        assert "mass" in sample.keys(), "Masses are required for the simulation."

        pos = sample.pop("R")
        init_fn, _ = sim_template(energy_fn_template(init_params))
        nbrs = nbrs_init.update(pos, **sample)
        sim_state = init_fn(key, pos, neighbor=nbrs, **sample)
        return sampling.SimulatorState(sim_state=sim_state, nbrs=nbrs)

    key, split = random.split(key)
    init_states = init_sim_state(random.split(split, confs["R"].shape[0]), confs)

    rdf_discretization = custom_quantity.rdf_discretization(
        rdf_start=0.0, rdf_cut=0.9, nbins=150
    )
    rdf_params = custom_quantity.RDFParams(jnp.zeros(150), *rdf_discretization)

    quantities = {
        "pressure": custom_quantity.init_pressure(energy_fn_template),
        "rdf":  custom_quantity.init_rdf(displacement_fn, rdf_params)
    }

    obs = {
        "pressure": quantity.observables.init_traj_mean_fn("pressure"),
        "rdf": quantity.observables.init_traj_mean_fn("rdf")
    }


    optimizer = init_optimizer(config)

    state_kwargs = {
            key: val for key, val in confs.items()
            if key not in ["R", "mass"]
    }

    rdf_targets = []
    for file in config["targets"]["rdf"]:
        r, rdf = onp.loadtxt(file, unpack=True, delimiter=",")
        rdf_targets.append(
            onp.interp(
                jnp.linspace(0.0, 0.9, 150), r / 10., rdf, left=0.0, right=1.0
            )
        )
    rdf_targets = jnp.stack(rdf_targets, axis=0)

    print(f"Loaded RDF targets {rdf_targets}")

    key, split = random.split(key)
    trainer_difftre = trainers.DifftreParallel(
        split, init_params, optimizer,
        log_dir=out_dir / "training.log", checkpoint_path=out_dir / "checkpoints",
        sim_batch_size=config["optimizer"]["batch"],
        targets={
            "rdf": {
                "target": rdf_targets,
                "gamma": jnp.asarray(config["gammas"]["rdf"] * rdf_targets.shape[0])
            }
        },
        observables=obs,
        state_kwargs=state_kwargs,
        quantities=quantities,
        reference_states=init_states,
        neighbor_fn=nbrs_init,
        energy_fn_template=energy_fn_template,
        simulator_template=sim_template,
        timings=timings,
        vmap_batch=5,
        reweight_ratio=config.get("reweighting_ratio", 0.9),
        allowed_reduction=None,
        num_runs_init=1,
    )

    # Train and save the results to a new folder
    trainer_difftre.train(config["optimizer"]["epochs"], checkpoint_freq=10)

    trainer_difftre.save_trainer(
        out_dir / "trainer.pkl")
    trainer_difftre.save_energy_params(
        out_dir / "final_params.pkl", best=False, save_format=".pkl"
    )


def init_optimizer(config):
    transition_steps = int(
        config["optimizer"]["epochs"] * len(config["confs"]) / config["optimizer"]["batch"]
    )

    lr_schedule_fm = optax.exponential_decay(
        config["optimizer"]["init_lr"], transition_steps, decay_rate=config["optimizer"]["lr_decay"])
    optimizer_fm = optax.chain(
        optax.scale_by_adam(**config["optimizer"]["optimizer_kwargs"]),
        optax.add_decayed_weights(config["optimizer"]["weight_decay"]),
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


def load_confs(config, fractional=True):

    _data = []
    for conf in config["confs"]:
        extra_args = {}
        conf, temp = conf

        (box, coords, mass, species) = io.load_box(conf)
        extra_args["kT"] = temp * quantity.kb

        if fractional:
            coords = jnp.einsum("ij,nj->ni", jnp.linalg.inv(box), coords)

        extra_args.update({
            "box": box,
            "R": coords,
            "species": species,
            "mass": mass,
        })
        _data.append(extra_args)

    # Padd the data to have the same number of atoms
    max_atoms = max([d["R"].shape[0] for d in _data])
    n_confs = len(_data)
    data = {
        "box": onp.zeros((n_confs, 3, 3)),
        "R": onp.zeros((n_confs, max_atoms, 3)),
        "species": onp.zeros((n_confs, max_atoms), dtype=jnp.int32),
        "mass": onp.zeros((n_confs, max_atoms)),
        "mask": onp.zeros((n_confs, max_atoms), dtype=jnp.bool),
        "kT": onp.zeros((n_confs,)),
        "pressure": onp.zeros((n_confs,)),
    }
    for idx, d in enumerate(_data):
        data["box"][idx, :] = d["box"]
        data["R"][idx, :d["R"].shape[0], :] = d["R"]
        data["species"][idx, :d["species"].shape[0]] = d["species"]
        data["mass"][idx, :d["mass"].shape[0]] = d["mass"]
        data["mask"][idx, :d["mass"].shape[0]] = True
        if "kT" in d:
            data["kT"][idx] = d["kT"]
        else:
            data.pop("kT", None)
        if "pressure" in d:
            data["pressure"][idx] = d["pressure"]
        else:
            data.pop("pressure", None)

    print(f"Loaded {n_confs} configurations with max. number of atoms {max_atoms}: {jax.tree.map(jnp.shape, data)}")

    return jax.tree.map(jnp.asarray, data)


def init_simulator(config, shift_fn, simulator=None):
    """Initializes simulator"""
    simulator_template = functools.partial(
        simulator, shift_fn=shift_fn,
        dt=config["timings"]["dt"], **config["simulator_settings"]
    )

    timings = sampling.process_printouts(
        config["timings"]["dt"], config["timings"]["total_time"],
        config["timings"]["t_equilib"], config["timings"]["print_every"]
    )

    return simulator_template, timings






def plot_predictions(predictions, reference_data, out_dir, name):
    # Simplifies comparison to reported values
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    cmap = plt.get_cmap('tab20')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")

    fig.suptitle("Predictions")
    pred_u_per_a = predictions['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy
    ref_u_per_a = reference_data['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy

    mae = onp.mean(onp.abs(pred_u_per_a - ref_u_per_a))
    ax1.set_title(f"Energy (MAE: {mae * 1000:.1f} meV/atom)")
    ax1.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    
    ax1.scatter(ref_u_per_a , pred_u_per_a, c=reference_data["total_charge"])
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    if "F" in predictions:
        # Select only the atoms that are not masked
        pred_F = predictions['F'].reshape((-1, 3))[
                 reference_data['mask'].ravel(), :] / scale_energy * scale_pos
        ref_F = reference_data['F'].reshape((-1, 3))[
                reference_data['mask'].ravel(), :] / scale_energy * scale_pos

        mae = onp.mean(onp.abs(pred_F - ref_F))
        ax2.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")
        ax2.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
        ax2.plot(ref_F.ravel(), pred_F.ravel(), ".")
        ax2.set_xlabel("Ref. F [eV/A]")
        ax2.set_ylabel("Pred. F [eV/A]")

    fig.savefig( out_dir / f"{name}.tiff", bbox_inches="tight")


def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
