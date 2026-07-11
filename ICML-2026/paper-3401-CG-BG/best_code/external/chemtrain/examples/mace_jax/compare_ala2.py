"""
Comparing MACE-JAX and Chemutils MACE on the ALA2 dataset.
Tested with:
- Chemutils commit: 87e1313eabc3a45980f0e16bf24334dfa49f9661
- MACE-JAX commit: 7e9d467d1701290b6606a20ff2c625c27e973254
"""

import os
import sys
import argparse
import copy
import numpy as onp

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


import jax
import jax.numpy as jnp
from jax import random, tree_util
import optax
import matplotlib.pyplot as plt
from cycler import cycler
from collections import OrderedDict

# Add paths to include src/chemutils, src/mace-jax, src/chemtrain
current_dir = os.path.dirname(os.path.abspath(__file__))
# root is ../../../../
root_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, os.path.join(root_dir, "src/chemutils"))
sys.path.insert(0, os.path.join(root_dir, "src/mace-jax"))
sys.path.insert(0, os.path.join(root_dir, "src/chemtrain"))

from chemtrain import trainers, quantity
from chemtrain.data import preprocessing
from jax_md import partition, space

# Import implementations
from chemtrain.compose import mace_jax as mace_jax_compose
from chemutils.models.mace.e3nn import mace_neighborlist_pp

SEED = 11


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")

    if fractional:
        box = dataset["box"][0, 0, 0]
        dataset["R"] = dataset["R"] / box
    else:
        dataset["R"] = dataset["R"] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset["box"] = scale_R * dataset["box"]
    dataset["F"] *= scale_F

    return dataset


class BaseDataset:
    """Base class for molecular dynamics datasets."""

    def __init__(self, dataset_path, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        """
        Initialize dataset with train/val/test splits.

        Args:
            dataset_path: Path to dataset file
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            shuffle: Whether to shuffle data during split
        """
        print(f"Loading dataset from:", dataset_path)
        dataset = onp.load(dataset_path, allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=shuffle,
            shuffle_seed=SEED,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

        # Ensure all required fields are present
        for split in dataset_.keys():
            dataset_[split]["R"] = dataset_[split]["R"]
            dataset_[split]["F"] = dataset_[split]["F"]
            dataset_[split]["box"] = dataset_[split]["box"]
            dataset_[split]["species"] = dataset_[split]["species"]
            dataset_[split]["mask"] = dataset_[split]["mask"]

        self.dataset_X = copy.deepcopy(dataset_)

        # Create fractional coordinate versions
        dataset_frac = {}
        self.splits = dataset_.keys()
        for split in self.splits:
            out = scale_dataset(dataset_[split], scale_R=1, scale_U=1, fractional=True)
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        # Set up displacement and shift functions for periodic boundary conditions
        self._setup_displacement_functions()

    def _setup_displacement_functions(self):
        """Set up displacement and shift functions for both coordinate systems."""
        displacement_fn_U, shift_fn_U = space.periodic_general(
            box=self.box, fractional_coordinates=True
        )
        self.displacement_fn_U = displacement_fn_U
        self.shift_fn_U = shift_fn_U

        displacement_fn_X, shift_fn_X = space.periodic_general(
            box=self.box, fractional_coordinates=False
        )
        self.displacement_fn_X = displacement_fn_X
        self.shift_fn_X = shift_fn_X


class Ala2_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        super().__init__(
            "/ds/project/franz/Datasets/l-ala2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
        )


def get_default_train_config(epochs=20, batch_size=32):
    return OrderedDict(
        optimizer=OrderedDict(
            init_lr=1e-4,
            lr_decay=1e-2,
            epochs=epochs,
            batch=batch_size,
            cache=100,
            power="exponential",
            weight_decay=1e-3,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.995,
                eps=1e-8,
            ),
        ),
        gammas=OrderedDict(
            U=1e-3,
            F=1e-2,
        ),
    )


def init_optimizer(config, dataset, key="optimizer"):
    num_samples = 1
    if "U" in dataset["training"]:
        num_samples = dataset["training"]["U"].shape[0]
    elif "F" in dataset["training"]:
        num_samples = dataset["training"]["F"].shape[0]
    else:
        print("No energy or force data available")
        exit()

    transition_steps = int(config[key]["epochs"] * num_samples) // config[key]["batch"]

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
        transforms.append(
            optax.scale_by_adam(
                b1=config[key]["optimizer_kwargs"]["b1"],
                b2=config[key]["optimizer_kwargs"]["b2"],
                eps=config[key]["optimizer_kwargs"]["eps"],
                eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
                nesterov=True,
            )
        )
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


def setup_mace_jax_training(dataset_obj, mace_config, train_config, r_cutoff=0.5):
    """Setup training with MACE-JAX implementation (from scratch)."""
    print("\n--- Setting up MACE-JAX training ---")

    dataset = dataset_obj.dataset_U  # Use fractional coordinates
    box = dataset_obj.box
    displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

    # Allocate neighborlist
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
        preprocessing.allocate_neighborlist(
            dataset["training"],
            displacement_fn,
            box,
            r_cutoff=r_cutoff,
            mask_key="mask",
            box_key="box",
            format=partition.Sparse,
            batch_size=100,
        )
    )
    print(
        f"MACE-JAX stats: Max neighbors: {max_neighbors}, Avg neighbors: {avg_num_neighbors}"
    )

    # Setup MACE config
    mace_cfg = {
        "r_cutoff": r_cutoff,
        "hidden_irreps": "32x0e + 32x1o",
        "MLP_irreps": "32x0e",
        "num_interactions": 2,
        "max_ell": 3,
        "correlation": 3,
        "n_radial_basis": 8,
        "output_irreps": "1x0e",
    }

    # Initialize MACE-JAX model (from scratch, not foundational)
    template_vars, gnn_energy_fn, model_config = (
        mace_jax_compose.mace_jax_neighborlist_pp(
            displacement=displacement_fn,
            r_cutoff=r_cutoff,
            n_species=100,
            per_particle=False,
            avg_num_neighbors=avg_num_neighbors,
            mode="energy",
            use_custom_batch_fn=True,  # Required for batched training
            mace_config=mace_cfg,
        )
    )

    init_params = template_vars["params"]
    variables = template_vars

    species_init = dataset["training"]["species"][0]

    def energy_fn_template(params):
        vars = {**variables}
        vars["params"] = params

        def energy_fn(position, neighbor, species=species_init, **kwargs):
            pots = gnn_energy_fn(vars, position, neighbor, species=species, **kwargs)

            # Subtract the provided atomic energies
            atomic_numbers = jnp.asarray(
                model_config["atomic_numbers"], dtype=jnp.int32
            )
            mapped_species = jnp.argmax(
                species[:, None] == atomic_numbers[None, :], axis=-1
            )
            pots -= jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)[
                mapped_species
            ] * kwargs.get("mask", 1.0)

            return jnp.sum(pots)

        return energy_fn

    # Update neighborlist with initial positions
    r_init = jnp.asarray(dataset["training"]["R"][0])
    mask_init = jnp.asarray(dataset["training"]["mask"][0])
    nbrs_init = nbrs_init.update(r_init, mask=mask_init)

    # Setup optimizer and trainer
    optimizer = init_optimizer(train_config, dataset)

    trainer = trainers.ForceMatching(
        init_params,
        optimizer,
        energy_fn_template,
        nbrs_init,
        batch_per_device=train_config["optimizer"]["batch"],
        batch_cache=train_config["optimizer"]["cache"],
        gammas=train_config["gammas"],
    )

    trainer.set_dataset(dataset["training"], stage="training")
    trainer.set_dataset(dataset["validation"], stage="validation", include_all=True)
    trainer.set_dataset(dataset["testing"], stage="testing", include_all=True)

    return trainer


def setup_chemutils_training(dataset_obj, config, train_config, r_cutoff=0.5):
    """Setup training with chemutils MACE implementation."""
    print("\n--- Setting up Chemutils MACE training ---")

    dataset = dataset_obj.dataset_U  # Use fractional coordinates (same as MACE-JAX)
    box = dataset_obj.box
    displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

    # Allocate neighborlist
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
        preprocessing.allocate_neighborlist(
            dataset["training"],
            displacement_fn,
            box,
            r_cutoff=r_cutoff,
            mask_key="mask",
            box_key="box",
            format=partition.Sparse,
            batch_size=100,
        )
    )

    print(
        f"Chemutils stats: Max neighbors: {max_neighbors}, Avg neighbors: {avg_num_neighbors}"
    )

    # Initialize chemutils MACE model
    init_fn, gnn_energy_fn = mace_neighborlist_pp(
        displacement_fn,
        r_cutoff,
        n_species=100,
        per_particle=False,
        avg_num_neighbors=avg_num_neighbors,
        mode="energy",
        hidden_irreps="32x0e + 32x1o",
        max_ell=3,
        num_interactions=2,
        correlation=3,
        readout_mlp_irreps="32x0e",
        output_irreps="1x0e",
        n_radial_basis=8,
        positive_species=True,
    )

    key = random.PRNGKey(SEED)
    r_init = jnp.asarray(dataset["training"]["R"][0])
    species_init = jnp.asarray(dataset["training"]["species"][0])
    mask_init = jnp.asarray(dataset["training"]["mask"][0])

    nbrs_init = nbrs_init.update(r_init, mask=mask_init)
    init_params = init_fn(key, r_init, nbrs_init, species=species_init, mask=mask_init)

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, species=species_init, **kwargs):
            return gnn_energy_fn(
                energy_params, pos, neighbor, species=species, **kwargs
            )

        return energy_fn

    # Setup optimizer and trainer
    optimizer = init_optimizer(train_config, dataset)

    trainer = trainers.ForceMatching(
        init_params,
        optimizer,
        energy_fn_template,
        nbrs_init,
        batch_per_device=train_config["optimizer"]["batch"],
        batch_cache=train_config["optimizer"]["cache"],
        gammas=train_config["gammas"],
    )

    trainer.set_dataset(dataset["training"], stage="training")
    trainer.set_dataset(dataset["validation"], stage="validation", include_all=True)
    trainer.set_dataset(dataset["testing"], stage="testing", include_all=True)

    return trainer


# =============================================================================
# Plotting
# =============================================================================


def plot_convergence(trainer_mace_jax, trainer_chemutils, out_dir):
    """Plot convergence comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")

    ax.set_title("Convergence Comparison")
    ax.semilogy(trainer_mace_jax.train_losses, label="MACE-JAX Train", color="tab:blue")
    ax.semilogy(
        trainer_mace_jax.val_losses,
        label="MACE-JAX Val",
        color="tab:blue",
        linestyle="--",
    )
    ax.semilogy(
        trainer_chemutils.train_losses, label="Chemutils Train", color="tab:orange"
    )
    ax.semilogy(
        trainer_chemutils.val_losses,
        label="Chemutils Val",
        color="tab:orange",
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(out_dir / "convergence_comparison.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "convergence_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved to {out_dir}")


def plot_force_comparison(
    trainer_mace_jax, trainer_chemutils, dataset, out_dir, config
):
    """Plot force predictions comparison."""
    # Unit conversions
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    fig.suptitle("Force Predictions Comparison")

    models = [("MACE-JAX", trainer_mace_jax), ("Chemutils", trainer_chemutils)]
    datasets_split = [
        ("Validation", dataset["validation"]),
        ("Testing", dataset["testing"]),
    ]

    for col, (model_name, trainer) in enumerate(models):
        for row, (split_name, data) in enumerate(datasets_split):
            ax = axes[row, col]

            # Get predictions
            predictions = trainer.predict(
                data, trainer.best_params, batch_size=config["optimizer"]["batch"]
            )
            predictions = tree_util.tree_map(onp.asarray, predictions)

            if "F" in predictions and "F" in data:
                pred_F = (
                    predictions["F"].reshape((-1, 3))[data["mask"].ravel(), :]
                    / scale_energy
                    * scale_pos
                )
                ref_F = (
                    data["F"].reshape((-1, 3))[data["mask"].ravel(), :]
                    / scale_energy
                    * scale_pos
                )

                mae = onp.mean(onp.abs(pred_F - ref_F))

                ax.plot(ref_F.ravel(), pred_F.ravel(), ".", alpha=0.1, markersize=1)

                # Diagonal line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

                ax.set_xlabel("Reference Force [eV/Å]")
                ax.set_ylabel("Predicted Force [eV/Å]")
                ax.set_title(
                    f"{model_name} - {split_name}\nMAE: {mae * 1000:.1f} meV/Å"
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Force Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    fig.savefig(out_dir / "force_comparison.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "force_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Force comparison plot saved to {out_dir}")


def print_metrics(trainer_mace_jax, trainer_chemutils, dataset, config):
    """Print comparison metrics."""
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    print("\n" + "=" * 60)
    print("COMPARISON METRICS")
    print("=" * 60)

    for split_name in ["validation", "testing"]:
        data = dataset[split_name]
        print(f"\n--- {split_name.upper()} SET ---")

        for model_name, trainer in [
            ("MACE-JAX", trainer_mace_jax),
            ("Chemutils", trainer_chemutils),
        ]:
            predictions = trainer.predict(
                data, trainer.best_params, batch_size=config["optimizer"]["batch"]
            )
            predictions = tree_util.tree_map(onp.asarray, predictions)

            if "F" in predictions and "F" in data:
                pred_F = (
                    predictions["F"].reshape((-1, 3))[data["mask"].ravel(), :]
                    / scale_energy
                    * scale_pos
                )
                ref_F = (
                    data["F"].reshape((-1, 3))[data["mask"].ravel(), :]
                    / scale_energy
                    * scale_pos
                )

                mae = onp.mean(onp.abs(pred_F - ref_F)) * 1000  # meV/Å
                rmse = onp.sqrt(onp.mean((pred_F - ref_F) ** 2)) * 1000  # meV/Å

                print(f"  {model_name}:")
                print(f"    Force MAE:  {mae:.2f} meV/Å")
                print(f"    Force RMSE: {rmse:.2f} meV/Å")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare MACE implementations on Ala2")
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--rcut", type=float, default=0.5, help="Cutoff radius (nm)")
    args = parser.parse_args()

    import pathlib

    out_dir = pathlib.Path("./output_comparison_ala2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (same as train_ala2_example.py)
    print("Loading Ala2 dataset...")
    dataset_obj = Ala2_Dataset()

    # Config
    train_config = get_default_train_config(epochs=args.epochs, batch_size=args.batch)

    # Setup and train MACE-JAX
    trainer_mace_jax = setup_mace_jax_training(
        dataset_obj, {}, train_config, r_cutoff=args.rcut
    )
    print(f"\nTraining MACE-JAX for {args.epochs} epochs...")
    trainer_mace_jax.train(args.epochs)

    # Setup and train Chemutils MACE
    trainer_chemutils = setup_chemutils_training(
        dataset_obj, {}, train_config, r_cutoff=args.rcut
    )
    print(f"\nTraining Chemutils MACE for {args.epochs} epochs...")
    trainer_chemutils.train(args.epochs)

    # Compare and plot results
    print("\nGenerating comparison plots...")
    plot_convergence(trainer_mace_jax, trainer_chemutils, out_dir)
    plot_force_comparison(
        trainer_mace_jax,
        trainer_chemutils,
        dataset_obj.dataset_U,
        out_dir,
        train_config,
    )
    print_metrics(
        trainer_mace_jax, trainer_chemutils, dataset_obj.dataset_U, train_config
    )

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
