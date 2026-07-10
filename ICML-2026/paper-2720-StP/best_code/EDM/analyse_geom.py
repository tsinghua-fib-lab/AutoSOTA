from rdkit import Chem
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
import collections
import pickle
import os
import json
from tqdm import tqdm
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
from qm9.analyze import check_stability
from qm9.rdkit_functions import BasicMolecularMetrics
import configs.datasets_config
atomic_number_list = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
inverse = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 13: 6, 14: 7, 15: 8, 16: 9, 17: 10, 33: 11, 35: 12, 53: 13,
           80: 14, 83: 15}
atom_name = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
n_atom_types = len(atomic_number_list)
n_bond_types = 4


def extract_conformers(args):
    Counter = collections.Counter
    bond_length_dict = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}

    summary_file = os.path.join(args.data_dir, args.data_file)
    with open(summary_file, "r") as f:
        drugs_summ = json.load(f)

    all_paths = []
    for smiles, sub_dic in drugs_summ.items():
        if 'pickle_path' in sub_dic:
            pickle_path = os.path.join(args.data_dir, "rdkit_folder", sub_dic["pickle_path"])
            all_paths.append(pickle_path)

    for i, mol_path in tqdm(enumerate(all_paths)):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)

        # Get the energy of each conformer. Keep only the lowest values
        conformers = dic['conformers']
        all_energies = []
        for conformer in conformers:
            all_energies.append(conformer['totalenergy'])
        all_energies = np.array(all_energies)
        argsort = np.argsort(all_energies)
        lowest_energies = argsort[:args.conformations]

        for id in lowest_energies:
            conformer = conformers[id]
            rd_mol = conformer["rd_mol"]

            atoms = rd_mol.GetAtoms()
            atom_nums = []
            for atom in atoms:
                atom_nums.append(atom.GetAtomicNum())

            rd_conf = rd_mol.GetConformers()[0]
            coords = rd_conf.GetPositions()      # list of elts of size 3?

            bonds = [bond for bond in rd_mol.GetBonds()]
            for bond in bonds:
                atom1, atom2 = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                # bond length
                c1 = coords[atom1]
                c2 = coords[atom2]
                dist = np.linalg.norm(c1 - c2)
                # Bin the distance
                dist = int(dist * 100)

                # atom types
                at1_type = atom_nums[atom1]
                at2_type = atom_nums[atom2]
                if at1_type > at2_type:     # Sort the pairs to avoid redundancy
                    temp = at2_type
                    at2_type = at1_type
                    at1_type = temp

                bond_type = bond.GetBondType().name.lower()
                if bond_type == 'single':
                    type = 0
                elif bond_type == 'double':
                    type = 1
                elif bond_type == 'triple':
                    type = 2
                elif bond_type == 'aromatic':
                    type = 3
                else:
                    raise ValueError("Unknown bond type", bond_type)

                bond_length_dict[type][(at1_type, at2_type, dist)] += 1

        if i % 5000 == 0:
            print("Current state of the bond length dictionary", bond_length_dict)
            if os.path.exists('bond_length_dict.pkl'):
                os.remove('bond_length_dict.pkl')
            with open('bond_length_dict', 'wb') as bond_dictionary_file:
                pickle.dump(bond_length_dict, bond_dictionary_file)


def create_matrix(args):
    with open('bond_length_dict', 'rb') as bond_dictionary_file:
        all_bond_types = pickle.load(bond_dictionary_file)
    x = np.zeros((n_atom_types, n_atom_types, n_bond_types, 350))
    for bond_type, d in all_bond_types.items():
        for key, count in d.items():
            at1, at2, bond_len = key
            x[inverse[at1], inverse[at2], bond_type, bond_len - 50] = count

    np.save('bond_length_matrix', x)

def create_histograms(args):
    x = np.load('../processed_dataset/geom/bond_length_matrix.npy')
    x = x[:, :, :, :307]
    label_list = ['single', 'double', 'triple', 'aromatic']
    for j in range(n_atom_types):
        for i in range(j + 1):
            if np.sum(x[i, j]) == 0:    # Bond does not exist
                continue

            # Remove outliers
            y = x[i, j]
            y[y < 0.02 * np.sum(y, axis=0)] = 0


            plt.figure()
            existing_bond_lengths = np.array(np.nonzero(y))[1]
            mini, maxi = existing_bond_lengths.min(), existing_bond_lengths.max()
            y = y[:, mini: maxi + 1]
            x_range = np.arange(mini, maxi + 1)
            for k in range(n_bond_types):
                if np.sum(y[k]) > 0:
                    plt.plot(x_range, y[k], label=label_list[k])
            plt.xlabel("Bond length")
            plt.ylabel("Count")
            plt.title(f'{atom_name[i]} - {atom_name[j]} bonds')
            plt.legend()
            plt.savefig(f'./figures/{atom_name[i]}-{atom_name[j]}-hist.png')
            plt.close()


def analyse_geom_stability():
    data_file = '../processed_dataset/geom/geom_drugs_30.npy'
    dataset_info = configs.datasets_config.get_dataset_info('geom', remove_h=False)
    atom_dict = dataset_info['atom_decoder']
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

    x = np.load(data_file)
    mol_id = x[:, 0].astype(int)
    all_atom_types = x[:, 1].astype(int)
    all_positions = x[:, 2:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    all_atom_types_split = np.split(all_atom_types, split_indices)
    all_positions_split = np.split(all_positions, split_indices)

    atomic_nb_list = torch.Tensor(dataset_info['atomic_nb'])[None, :].long()
    num_stable_mols = 0
    num_mols = 0
    num_stable_atoms_total = 0
    num_atoms_total = 0
    formatted_data = []
    for i, (p, at_types) in tqdm(enumerate(zip(all_positions_split, all_atom_types_split))):
        p = torch.from_numpy(p)
        at_types = torch.from_numpy(at_types)[:, None]
        one_hot = torch.eq(at_types, atomic_nb_list).int()
        at_types = torch.argmax(one_hot, dim=1)     # Between 0 and 15
        formatted_data.append([p, at_types])

        mol_is_stable, num_stable_atoms, num_atoms = check_stability(p, at_types, dataset_info)
        num_mols += 1
        num_stable_mols += mol_is_stable
        num_stable_atoms_total += num_stable_atoms
        num_atoms_total += num_atoms
        if i % 5000 == 0:
            print(f"IN PROGRESS -- Stable molecules: {num_stable_mols} / {num_mols}"
                  f" = {num_stable_mols / num_mols * 100} %")
            print(
                f"IN PROGRESS -- Stable atoms: {num_stable_atoms_total} / {num_atoms_total}"
                f" = {num_stable_atoms_total / num_atoms_total * 100} %")

    print(f"Stable molecules: {num_stable_mols} / {num_mols} = {num_stable_mols / num_mols * 100} %")
    print(f"Stable atoms: {num_stable_atoms_total} / {num_atoms_total} = {num_stable_atoms_total / num_atoms_total * 100} %")

    metrics = BasicMolecularMetrics(dataset_info)
    metrics.evaluate(formatted_data)


def debug_geom_stability(num_atoms=100000):
    data_file = '../processed_dataset/geom/geom_drugs_30.npy'
    dataset_info = configs.datasets_config.get_dataset_info('geom', remove_h=False)
    atom_dict = dataset_info['atom_decoder']
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

    x = np.load(data_file)
    x = x[:num_atoms]

    # Print non hydrogen atoms
    x = x[x[:, 1] != 1.0, :]

    mol_id = x[:, 0].astype(int)
    max_mol_id = mol_id.max()
    may_be_incomplete = mol_id == max_mol_id
    x = x[~may_be_incomplete]
    mol_id = mol_id[~may_be_incomplete]
    all_atom_types = x[:, 1].astype(int)
    all_positions = x[:, 2:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    all_atom_types_split = np.split(all_atom_types, split_indices)
    all_positions_split = np.split(all_positions, split_indices)

    atomic_nb_list = torch.Tensor(dataset_info['atomic_nb'])[None, :].long()

    formatted_data = []
    for p, at_types in zip(all_positions_split, all_atom_types_split):
        p = torch.from_numpy(p)
        at_types = torch.from_numpy(at_types)[:, None]
        one_hot = torch.eq(at_types, atomic_nb_list).int()
        at_types = torch.argmax(one_hot, dim=1)  # Between 0 and 15
        formatted_data.append([p, at_types])

    metrics = BasicMolecularMetrics(atom_dict, bond_dict, dataset_info)
    m, smiles_list = metrics.evaluate(formatted_data)
    print(smiles_list)


def compute_n_nodes_dict(file='../processed_dataset/geom/geom_drugs_30.npy', remove_hydrogens=True):
    all_data = np.load(file)
    atom_types = all_data[:, 1]
    if remove_hydrogens:
        hydrogens = np.equal(atom_types, 1.0)
        all_data = all_data[~hydrogens]

    # Get the size of each molecule
    mol_id = all_data[:, 0].astype(int)
    max_id = mol_id.max()
    mol_id_counter = np.zeros(max_id + 1, dtype=int)
    for id in mol_id:
        mol_id_counter[id] += 1

    unique_sizes, size_count = np.unique(mol_id_counter, return_counts=True)

    sizes_dict = {}
    for size, count in zip(unique_sizes, size_count):
        sizes_dict[size] = count

    print(sizes_dict)
    return sizes_dict




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--conformations", type=int, default=30,
    #                     help="Max number of conformations kept for each molecule.")
    # parser.add_argument("--data_dir", type=str, default='/home/vignac/diffusion/data/geom/')
    # parser.add_argument("--data_file", type=str, default="rdkit_folder/summary_drugs.json")
    # args = parser.parse_args()
    # # extract_conformers(args)
    # # create_histograms(args)
    #
    analyse_geom_stability()
    # compute_n_nodes_dict()