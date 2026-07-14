import torch
import numpy as np
from nff.data import Dataset, collate_dicts
from ase.io import read

class AtomsDict(dict):
    def to(self, *args, **kwargs):
        return AtomsDict({
            k: v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v
            for k, v in self.items()
        })

def collate_fn(*args, **kwargs):
    return AtomsDict(collate_dicts(*args, **kwargs))

def get_neighbor_list(xyz, cutoff=5, undirected=True):
    if not torch.is_tensor(xyz):
        xyz = torch.Tensor(xyz)
    n = xyz.size(0)
    dist = (
        (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1))
        .pow(2)
        .sum(dim=2)
        .sqrt()
    )
    mask = dist <= cutoff
    mask[np.diag_indices(n)] = 0
    nbr_list = mask.nonzero(as_tuple=False)
    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]
    return nbr_list

def atoms_to_data(list_atoms, cutoff):
    _default_dtype = torch.float32
    nxyz = []
    num_atoms = []
    energy = []
    neg_force = []
    for atoms in list_atoms:
        pos = torch.tensor(atoms.positions, dtype=_default_dtype)
        elems = torch.tensor(atoms.numbers, dtype=torch.long).reshape(-1, 1)
        nxyz.append(torch.cat([elems, pos], dim=1))
        num_atoms.append(torch.LongTensor([len(elems)]))
        try:
            energy.append(
                torch.tensor(atoms.get_potential_energy(), dtype=_default_dtype)
            )
        except RuntimeError:
            pass
        try:
            neg_force.append(-torch.tensor(atoms.get_forces(), dtype=_default_dtype))
        except RuntimeError:
            pass

    return AtomsDict({
        "nxyz": nxyz,
        "num_atoms": num_atoms,
        "energy": energy,
        "energy_grad": neg_force,
        "nbr_list": [get_neighbor_list(d[:, 1:4], cutoff, True) for d in nxyz],
    })

class AmmoniaDataset:
    def __new__(cls, cutoff=5.0, partition="train"):
        props = cls._load(partition, cutoff)
        return Dataset(props, units = "kcal/mol")

    @staticmethod
    def _load(partition, cutoff):
        if partition == "test":
            atoms = read("data/ammonia/ammonia_test.xyz", index=":")
        elif partition in ["train", "val"]:
            atoms = read("data/ammonia/ammonia_train.xyz", index=":")
            atoms = atoms[:64] if partition == "train" else atoms[64:]
        else:
            raise RuntimeError("Wrong partition, insert train, val, test.")
        return atoms_to_data(atoms, cutoff)
