import argparse
import os
import pathlib
import sys


parser = argparse.ArgumentParser()
parser.add_argument("device", type=str, nargs="?", default=None)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--async_dataloading", type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--numpy_loader", type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--disable_cue", type=bool, default=False)
parser.add_argument("--outdir", type=str, default="./output")
args = parser.parse_args()

if args.device is not None and args.device != "-1":
    # If the caller (e.g. mpirun) already set CUDA_VISIBLE_DEVICES,
    # do not override it.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import h5py
import jax_sgmc
import numpy as onp

import jax
import jax.numpy as jnp

from jax_md_mod import custom_quantity
from jax_md import partition, space

import optax

from matplotlib import pyplot as plt
from cycler import cycler

from collections import OrderedDict

import chemtrain
chemtrain.config.update(async_dataloading=args.async_dataloading)

from chemtrain.data import preprocessing, data_loaders
from chemtrain.deploy import exporter, graphs
from chemtrain.compose import mace_jax as mace_jax_compose

from chemtrain import trainers

from chemutils.datasets import spice

from mace_jax.modules.wrapper_ops import CuEquivarianceConfig


def get_default_config():
    print(f"Run on device {args.device}")

    return OrderedDict(
        optimizer=OrderedDict(
            init_lr=1e-5,
            lr_decay=1e-1,
            epochs=args.epochs,
            batch=args.batch,
            cache=100,
            power="exponential",
            weight_decay=1e-3,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.995,
                eps=1e-8,
            )
        ),
        dataset=OrderedDict(
            subsets=[
                "Dipeptides",
                "Amino Acid",
                "PubChem",
                "Water",
                "DES",
            ],
            total_charge='total_charge', # Use all samples if commented out
            max_samples=10000 # Use all samples if commented out
        ),
        gammas=OrderedDict(
            U=1e-3,
            # U=0.0, # There seems to be a difference in predicted energies
            F=1e-2,
        ),
        disable_cue=args.disable_cue,
        out_dir=args.outdir,
    )

def main():

    config = get_default_config()
    out_dir = pathlib.Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, info = spice.download_spice(
        "/ds/project/paul/Datasets/",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False, neutral=True
    )
    dataset = spice.process_dataset(dataset)

    displacement_fn, _ = space.free()

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

    # Load the model
    torch_model, model_config = mace_jax_compose.load_foundational_model(
        family="off", version="medium"
    )

    print("Loaded model with config:", model_config)
    
    variables, apply_fn = mace_jax_compose.mace_jax_neighborlist(
        model_config, torch_model, displacement_fn, max_edge_multiplier=1.25,
        per_particle=True,
        scale_pos=0.1,  # Convert from Angstrom to nm
        scale_pot=96.185,  # Convert from eV to kJ/mol
        # Map species to atomic numbers for MACE-JAX compatibility
        species_mapping=mace_jax_compose.AtomicNumberMapping(max_number=60),
        cueq_config=cueq_config
    )

    # Update the loaded subset information
    config["dataset"]["subsets"] = list(info["subsets"].values())

    # Infer the number of neighbors within the model cutoff
    nbrs_init, _ = preprocessing.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0,
        model_config["r_max"] / 10., mask_key="mask",
        format=partition.Sparse, count_triplets=False
    )

    nbrs_init = nbrs_init.update(dataset['training']['R'][0], mask=dataset['training']['mask'][0])
    species_init = dataset['training']['species'][0]

    print(f"Compute initial energy as a test: "
          f"{apply_fn(variables, dataset['training']['R'][0], nbrs_init, species=species_init, mask=dataset['training']['mask'][0])}")

    init_params = variables["params"]

    def energy_fn_template(params):
        vars = {**variables}
        vars["params"] = params

        def energy_fn(position, neighbor, species, **kwargs):
            
            pots = apply_fn(
                vars, position, neighbor, species=species, **kwargs
            )

            # Substract the prodived atomic energies
            atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
            mapped_species = jnp.argmax(species[:, None] == atomic_numbers[None, :], axis=-1)
            pots -= jnp.asarray(
                model_config['atomic_energies'] * 96.185, dtype=jnp.float32
            )[mapped_species] * kwargs.get('mask', 1.0)

            return jnp.sum(pots)

        return energy_fn


    optimizer = init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch=config["optimizer"]["batch"],
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        weights_keys={
            "F": "F_weight",
        },
    )

    if args.numpy_loader:
        with h5py.File("../../../../datasets/spice/training.h5", "r") as f:
            trainer_fm.set_dataset(
                {k: onp.asarray(f[k]) for k in f.keys()},
                stage='training'
            )

        with h5py.File("../../../../datasets/spice/validation.h5", "r") as f:
            trainer_fm.set_dataset(
                {k: onp.asarray(f[k]) for k in f.keys()},
                stage='validation', include_all=True
            )

        with h5py.File("../../../../datasets/spice/testing.h5", "r") as f:
            trainer_fm.set_dataset(
                {k: onp.asarray(f[k]) for k in f.keys()},
                stage='testing', include_all=True
            )

    else:
        trainer_fm.set_loader(
            data_loaders.HDF5ParallelDataLoader("../../../../datasets/spice/training.h5"),
            stage='training'
        )
        trainer_fm.set_loader(
            data_loaders.HDF5ParallelDataLoader("../../../../datasets/spice/validation.h5"),
            stage='validation', include_all=True
        )
        trainer_fm.set_loader(
            data_loaders.HDF5ParallelDataLoader("../../../../datasets/spice/testing.h5"),
            stage='testing', include_all=True
        )

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"], checkpoint_freq=10)

    plot_convergence(trainer_fm, out_dir)


    test_predictions = trainer_fm.predict(
        dataset['testing'],
        batch_size=config["optimizer"]["batch"],
    )
    train_predictions = trainer_fm.predict(
        dataset['training'],
        batch_size=config["optimizer"]["batch"],
    )
    validation_predictions = trainer_fm.predict(
        dataset["validation"], trainer_fm.best_params,
        batch_size=config["optimizer"]["batch"],
    )


    plot_predictions(validation_predictions, dataset["validation"],
                                 out_dir,
                                 f"preds_validation")
    plot_predictions(test_predictions, dataset["testing"], out_dir,
                                 f"preds_testing")
    plot_predictions(train_predictions, dataset["training"],
                                 out_dir, f"preds_training")

    # Do not export if cuequivariance is enabled
    if not config["disable_cue"]: return

    def export_energy_fn_template(params):
        vars = {**variables}
        vars["params"] = params

        def energy_fn(position, neighbor, species, **kwargs):
            
            # Need to map from our atomic numbers to the species provided by the
            # model (0-based).
            species += 1 # From 0-based to 1-based            

            pots = apply_fn(
                vars, position, neighbor, species=species, **kwargs
            )

            return pots

        return energy_fn

    # Test exporting the MACE-JAX model
    class ExportedModel(exporter.Exporter):

        graph_type = graphs.SimpleSparseNeighborList
        r_cutoff = model_config["r_max"]
        unit_style = "real"
        nbr_order = [
            model_config["num_interactions"],
            2*model_config["num_interactions"]
        ]

        def energy_fn(self, position, species, graph):

            neighbors = partition.NeighborList(
                jax.numpy.stack((graph.senders, graph.receivers)),
                position, None, None, graph.senders.size, partition.Sparse,
                None, None, None
            )

            assert neighbors.idx.shape[0] == 2, "Wrong shape"

            position /= 10.0 # From A to nm
            pots = export_energy_fn_template(trainer_fm.best_params)(
                position, neighbors, species=species
            )
            pots /= 4.184 # From kJ/mol to kcal/mol

            return pots

    print("Exporting the trained model...")

    model =  ExportedModel()

    model.export()
    model.save(out_dir / "out.ptb")

    print("Exported the trained model to", out_dir / "out.ptb")



def init_optimizer(config, dataset, key="optimizer"):

    num_samples = 1
    if 'U' in dataset['training']:
        num_samples = dataset['training']['U'].shape[0]
    elif 'dF' in dataset['training']:
        num_samples = dataset['training']['dF'].shape[0]
    else:
        exit()

    global_batch = int(config[key]["batch"])
    transition_steps = int(config[key]["epochs"] * num_samples) // global_batch

    if config[key].get("power") == "exponential":
        lr_schedule_fm = optax.exponential_decay(
            config[key]["init_lr"],
            transition_steps,
            config[key]["lr_decay"],
        )
    else:
        lr_schedule_fm = optax.polynomial_schedule(
            config[key]["init_lr"],
            config[key]["lr_decay"] * config[key]["init_lr"],
            config[key].get("power", 2.0),
            transition_steps,
        )

    print(f"Decay LR with power {config[key].get('power', 2.0)}")

    transforms = []

    if config[key].get("normalize"):
        transforms.append(optax.scale_by_param_block_norm())

    if config[key]["type"] == "ADAM":
        transforms.append(optax.scale_by_adam(
            b1=config[key]["optimizer_kwargs"]["b1"],
            b2=config[key]["optimizer_kwargs"]["b2"],
            eps=config[key]["optimizer_kwargs"]["eps"],
            eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
            nesterov=True,
        ))
    elif config[key]["type"] == "AdaBelief":
        transforms.append(optax.scale_by_belief(
            b1=config[key]["optimizer_kwargs"]["b1"],
            b2=config[key]["optimizer_kwargs"]["b2"],
            eps=config[key]["optimizer_kwargs"]["eps"],
            eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
        ))
    else:
        raise NotImplementedError(f"Optimizer {config[key]['type']} not implemented.")

    weight_decay = config[key].get("weight_decay")
    if weight_decay is not None:
        transforms.append(optax.transforms.add_decayed_weights(weight_decay))

    optimizer_fm = optax.chain(
        *transforms,
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


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
