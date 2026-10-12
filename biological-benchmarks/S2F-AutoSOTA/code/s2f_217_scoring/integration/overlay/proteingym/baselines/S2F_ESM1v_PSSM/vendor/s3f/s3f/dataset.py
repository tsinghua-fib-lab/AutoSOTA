import os
import csv
import math
import pickle

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import normalize
from Bio.PDB import PDBParser

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils, core
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from s3f import residue_constants


# protein gym datasets

def bio_load_pdb(pdb):
    # Load raw pdb file with biopython
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id.get(residue.get_resname(), 0) for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name, 
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor,
                residue_number=residue_number, node_position=node_position, residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])


@R.register("datasets.ProteinGym")
class ProteinGym(core.Configurable):

    def __init__(self, path, csv_file):
        path = os.path.expanduser(path)
        self.path = path
        csv_file = os.path.join(path, csv_file)

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            assay_list = [row for row in reader]
        self.ids = [assay["DMS_id"] for assay in assay_list]
        self.assay_dict = {assay["DMS_id"]: assay for assay in assay_list}


class MutantDataset(torch_data.Dataset):

    def __init__(self, mutated_sequences, wild_type, surf_graph=None, transform=None, wt_sequences=None):
        self.mutated_sequences = mutated_sequences
        self.wild_type = wild_type
        self.surf_graph = surf_graph
        self.transform = transform
        # IDEA-404: unmasked wild-type clones of each windowed sequence, used
        # only by the wt-ESM third expert (task.wt_esm_weight > 0).
        self.wt_sequences = wt_sequences

    def __len__(self):
        return len(self.mutated_sequences)

    def truncate(self, sequence_graph, structure_graph, surface_graph=None):
        num_residue = structure_graph.num_residue
        start = sequence_graph.start - structure_graph.start
        end = sequence_graph.end - structure_graph.start
        residue_mask = torch.zeros((num_residue, ), dtype=torch.bool)
        residue_mask[start:end] = 1
        structure_graph = structure_graph.subresidue(residue_mask)

        if surface_graph:
            surf_idx = structure_graph.res2surf
            surf_mask = torch.zeros((surface_graph.num_node, ), dtype=torch.bool)
            surf_mask[surf_idx.flatten()] = 1
            surface_graph = surface_graph.subgraph(surf_mask)

            _, res2surf = torch.unique(surf_idx, return_inverse=True)
            with structure_graph.residue():
                structure_graph.res2surf = res2surf.view(*surf_idx.shape)

        return sequence_graph, structure_graph, surface_graph

    def assign_structure(self, sequence_graph, structure_graph):
        graph = structure_graph.clone()
        # Assume the backbone structure won't change
        assert graph.num_residue == sequence_graph.num_residue
        with graph.residue():
            graph.residue_type = sequence_graph.residue_type
        return graph

    def __getitem__(self, index):
        sequence_graph = self.mutated_sequences[index]
        structure_graph = self.wild_type
        surface_graph = self.surf_graph

        # we need to truncate structure if structure is longer than sequence
        if structure_graph.start <= sequence_graph.start and structure_graph.end >= sequence_graph.end:
            sequence_graph, structure_graph, surface_graph = self.truncate(sequence_graph, structure_graph, surface_graph)
        elif structure_graph.start != sequence_graph.start or structure_graph.end != sequence_graph.end:
            raise ValueError("the structure range (%d, %d) doesn't match the sequence range (%d, %d)" % 
                             (structure_graph.start, structure_graph.end, sequence_graph.start, sequence_graph.end))
        
        graph = self.assign_structure(sequence_graph, structure_graph)
        item = {"graph": graph}
        if self.wt_sequences is not None:
            item["wt_graph"] = self.wt_sequences[index]
        if surface_graph:
            item["surf_graph"] = surface_graph
        if self.transform:
            item = self.transform(item)
        return item
    

# pre-training dataset

atom_type_mapping = torch.tensor([data.feature.atom_vocab[n[0]] for n in residue_constants.atom_order])     # (37, )
atom_name_mapping = torch.tensor([data.Protein.atom_name2id[n] for n in residue_constants.atom_order])      # (37, )
inv_atom_name_mapping = torch.zeros((len(data.Protein.atom_name2id)), dtype=torch.long)
inv_atom_name_mapping[atom_name_mapping] = torch.arange(residue_constants.atom_type_num, dtype=torch.long)      # (37, )
residue_type_mapping = torch.tensor([data.Protein.residue_symbol2id.get(n, 0) for n in residue_constants.restypes_with_x])    # (21, )


def load_protein(data_dict):
    # Load pickled protein structure
    atom_mask = torch.tensor(data_dict['atom_mask']).bool()
    atom_type = atom_type_mapping[None, :]
    atom_type = atom_type.expand_as(atom_mask)[atom_mask]
    atom_name = atom_name_mapping[None, :]
    atom_name = atom_name.expand_as(atom_mask)[atom_mask]
    node_position = torch.tensor(data_dict['atom_positions'])[atom_mask]
    residue_type = torch.tensor(data_dict['aatype'])
    residue_type = residue_type_mapping[residue_type]
    residue_number = torch.tensor(data_dict['residue_index'])
    b_factor = torch.tensor(data_dict['b_factors'])[atom_mask]
    chain_id = torch.tensor(data_dict['chain_index'])
    num_residue = residue_type.shape[0]
    num_atom = atom_name.shape[0]
    
    atom2residue = torch.arange(num_residue)[:, None]
    atom2residue = atom2residue.expand_as(atom_mask)[atom_mask]

    edge_list = torch.zeros((1, 3), dtype=torch.long)
    bond_type = torch.zeros((1,), dtype=torch.long)

    residue_feature = F.one_hot(residue_type, len(residue_constants.restypes_with_x))
    atom_feature = torch.cat([
        F.one_hot(atom_name, residue_constants.atom_type_num),
        residue_feature[atom2residue]
    ], dim=-1)
    
    protein = data.Protein(edge_list=edge_list, atom_type=atom_type, bond_type=bond_type, 
                        residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue, 
                        residue_feature=residue_feature, atom_feature=atom_feature, bond_feature=None,
                        residue_number=residue_number, b_factor=b_factor, chain_id=chain_id,
                        node_position=node_position, num_node=num_atom, num_residue=num_residue,
    )
    return protein


def load_surface(surf_dict):
    # Load pickled protein surface
    surf_points = torch.tensor(surf_dict["surf_points"]).float()
    normals = torch.tensor(surf_dict["surf_normals"]).float()
    hks = torch.tensor(surf_dict["surf_hks"]).float()
    curvatures = torch.tensor(surf_dict["surf_curvatures"]).float()
    
    num_surf_points = surf_points.shape[0]
    edge_list = torch.zeros((1, 3), dtype=torch.long)
    node_feature = torch.cat([hks, curvatures], dim=-1)
    
    surf_graph = data.Graph(edge_list=edge_list, node_feature=node_feature, bond_feature=None, 
                            num_node=num_surf_points, num_relation=1)
    with surf_graph.node():
        surf_graph.normals = normals
        surf_graph.node_position = surf_points

    return surf_graph
    

@R.register("datasets.CATH")
class CATH(data.ProteinDataset):

    def __init__(self, path, max_length=None, surf_path=None, transform=None):
        path = os.path.expanduser(path)
        self.path = path
        self.max_length = max_length

        self.pkl_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pkl")])
        self.transform = transform
        if surf_path:
            surf_path = os.path.expanduser(surf_path)
        self.surf_path = surf_path

    def truncate(self, data_dict, surf_dict=None):
        length = data_dict["aatype"].shape[0]
        if length <= self.max_length:
            return data_dict, surf_dict
        start = np.random.randint(length - self.max_length, size=(1,))[0]
        end = start + self.max_length
        for k in data_dict.keys():
            data_dict[k] = data_dict[k][start:end]

        if surf_dict is not None:
            # Remove surfaces of the truncated part
            surf_idx = surf_dict["res2surf"][start:end]
            surf_mask = np.zeros(surf_dict["surf_points"].shape[0], dtype=bool)
            surf_mask[surf_idx.flatten()] = 1
            for k in surf_dict.keys():
                if not k.startswith("surf_"): continue
                surf_dict[k] = surf_dict[k][surf_mask]
            # Re-index surface graph points
            _, surf_dict["res2surf"] = np.unique(surf_idx, return_inverse=True)
            surf_dict["res2surf"] = surf_dict["res2surf"].reshape(*surf_idx.shape)

        return data_dict, surf_dict

    def get_item(self, idx):
        with open(self.pkl_files[idx], "rb") as fin:
            data_dict = pickle.load(fin)
        if self.surf_path:
            surf_file = os.path.join(self.surf_path, os.path.basename(self.pkl_files[idx]))
            with open(surf_file, "rb") as fin:
                surf_dict = pickle.load(fin)
        else:
            surf_dict = None
        if self.max_length:
            data_dict, surf_dict = self.truncate(data_dict, surf_dict=surf_dict)
        protein = load_protein(data_dict)

        item = {"graph": protein}
        if surf_dict is not None:
            surf_graph = load_surface(surf_dict)
            item.update({"surf_graph": surf_graph})
            with protein.residue():
                protein.res2surf = torch.as_tensor(surf_dict["res2surf"])    # Need to transform local index to global index after batching
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.pkl_files)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))