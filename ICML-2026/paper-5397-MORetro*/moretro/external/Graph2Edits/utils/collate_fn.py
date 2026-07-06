from typing import Any, List, Tuple

import numpy as np
from rdkit import Chem

import torch
from Graph2Edits.utils.mol_features import ATOM_FDIM, BOND_FDIM
from Graph2Edits.utils.rxn_graphs import MolGraph


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist])
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)


def prepare_edit_labels(graph_batch: List[MolGraph], edits: List[Any], edit_atoms: List[Any], bond_vocab: List, atom_vocab: List) -> torch.tensor:
    """ 
    Prepare edit label including atom edits and bond edits.
    """
    bond_vocab_size = bond_vocab.size()
    atom_vocab_size = atom_vocab.size()
    edit_labels = []

    for prod_graph, edit, edit_atom in zip(graph_batch, edits, edit_atoms):
        bond_label = np.zeros((prod_graph.num_bonds, bond_vocab_size))
        atom_label = np.zeros((prod_graph.num_atoms, atom_vocab_size))
        stop_label = np.zeros((1,))

        if edit == 'Terminate':
            stop_label[0] = 1.0

        elif edit[0] == 'Change Atom' or edit[0] == 'Attaching LG':
            a_map = edit_atom
            a_idx = prod_graph.amap_to_idx[a_map]
            edit_idx = atom_vocab.get_index(edit)
            atom_label[a_idx][edit_idx] = 1

        else:
            a1, a2 = edit_atom[0], edit_atom[1]
            a_start, a_end = prod_graph.amap_to_idx[a1], prod_graph.amap_to_idx[a2]
            b_idx = prod_graph.mol.GetBondBetweenAtoms(a_start, a_end).GetIdx()
            edit_idx = bond_vocab.get_index(edit)
            bond_label[b_idx][edit_idx] = 1

        edit_label = np.concatenate(
            (bond_label.flatten(), atom_label.flatten(), stop_label.flatten()))
        edit_label = torch.from_numpy(edit_label)
        edit_labels.append(edit_label)

    return edit_labels


def get_batch_graphs(graph_batch: List[MolGraph], use_rxn_class: bool = False) -> Tuple[torch.Tensor, List[Tuple[int]]]:
    """
    Featurization of a batch of molecules.
    """
    # Start n_atoms and n_bonds at 1 b/c zero padding
    n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
    n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
    a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
    b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

    # All start with zero padding so that indexing with zero padding returns zeros
    if use_rxn_class:
        atom_fdim = ATOM_FDIM + 10
    else:
        atom_fdim = ATOM_FDIM
    bond_fdim = atom_fdim + BOND_FDIM

    f_atoms = [[0] * atom_fdim]  # atom features
    f_bonds = [[0] * bond_fdim]  # combined atom/bond features
    a2b = [[]]  # mapping from atom index to incoming bond indices
    b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = [0]  # mapping from bond index to the index of the reverse bond
    undirected_b2a = [[]] # mapping from the undirected bond index to the beginindex and endindex of the atoms
    
    for mol_graph in graph_batch:
        f_atoms.extend(mol_graph.f_atoms)
        f_bonds.extend(mol_graph.f_bonds)

        for a in range(mol_graph.n_atoms):
            a2b.append([b + n_bonds for b in mol_graph.a2b[a]])

        for b in range(mol_graph.n_bonds):
            b2a.append(n_atoms + mol_graph.b2a[b])
            b2revb.append(n_bonds + mol_graph.b2revb[b])

        n_undirected_bonds = len(undirected_b2a)
        for bond in mol_graph.mol.GetBonds():
            undirected_b2a.append(sorted([bond.GetBeginAtomIdx() + n_atoms, bond.GetEndAtomIdx() + n_atoms]))

        a_scope.append((n_atoms, mol_graph.n_atoms))
        b_scope.append((n_undirected_bonds, mol_graph.num_bonds))
        n_atoms += mol_graph.n_atoms
        n_bonds += mol_graph.n_bonds

    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    a2b = create_pad_tensor(a2b)
    b2a = torch.LongTensor(b2a)
    b2revb = torch.LongTensor(b2revb)
    undirected_b2a = create_pad_tensor(undirected_b2a)

    graph_tensors = (f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a)
    scopes = (a_scope, b_scope)

    return graph_tensors, scopes
    