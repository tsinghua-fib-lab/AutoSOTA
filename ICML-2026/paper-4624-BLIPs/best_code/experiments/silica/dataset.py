from ase.io import read
from orb_models.forcefield import atomic_system
from orb_models.forcefield.atomic_system import SystemConfig
import torch
from torch.utils.data import Dataset


class SilicaDataset(Dataset):
    def __init__(self, partition="train", train_size=None, system_config=None):
        max_train_size = 1024
        if train_size is None:
            train_size = max_train_size
        if partition == "test":
            atoms_list = read("data/silica/silica_test.xyz", index=":")
        elif partition in ["train", "val"]:
            atoms_list = read("data/silica/silica_train.xyz", index=":")
            if train_size > max_train_size:
                raise ValueError('set 0 < train_size <= 1024.')
            atoms_list = (
                atoms_list[:train_size] if partition == "train" else atoms_list[1024:]
            )
        else:
            raise RuntimeError("Wrong partition, insert train, val, test.")
        self.atoms_list = atoms_list
        self.system_config = system_config or SystemConfig(
            radius=6.0, max_num_neighbors=20
        )

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        # extract atoms
        atoms = self.atoms_list[idx]
        # attach targets (this does not work!)
        atoms.info["node_targets"] = {
            "forces": torch.tensor(atoms.get_forces()) * 0.0433641 # from kcal/mol/A to eV/A
            }
        atoms.info["graph_targets"] = {
            "energy": (torch.tensor([atoms.get_potential_energy()]) - 125667.96875)
            * 0.0433641  # from kcal/mol to eV
        }
        # convert to AtomsGraph
        graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms, system_config=self.system_config
        )
        return graph
