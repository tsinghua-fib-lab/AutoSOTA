"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import sys
try:
    from rdkit import Chem
    from rdkit.Chem import MolFromSmiles
except:
    sys.exit('rdkit is not installed. Install with:\nconda install -c rdkit rdkit')

import logging
import numpy as np
import torch
import sys
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def load_data_from_smiles(x_smiles, add_dummy_node=True):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to True.

    Returns:
        A tuple of lists of graph descriptors (node features, adjacency matrices)
    """

    x_all = []
    for smiles in x_smiles:
        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                continue  # Skip invalid SMILES
            afm, adj = featurize_mol(mol, add_dummy_node)
            x_all.append([afm, adj])
        except Exception as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\\nREASON: {}'.format(smiles,e))

    return x_all

def featurize_mol(mol, add_dummy_node):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix).
    """
    node_features = np.array([get_atom_features(atom)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

    return node_features, adj_matrix


#lookup table for one-hot elements
anummap = {5:0, 6: 1, 7:2, 8: 3, 9: 4,  15:5, 16:6, 17:7, 35:8, 53:9 }
anumtable = np.full(128,10)
for i,val in anummap.items():
    anumtable[i] = val
    
def get_atom_features(atom):
    """Calculate atom features.

            Identity            -- [B,C,N,O,F,P,S,Cl,Br,I,Dummy,Other]
            #Heavy Neighbors    -- [0,1,2,3,4,5]
            #H atoms            -- [0,1,2,3,4]
            Formal Charge       -- [-1,0,1]
            Is in a Ring        -- [0,1]
            Is Aromatic         -- [0,1]
        Dummy and Other types, have the same one-hot encoding, but the dummy node is unconnected.
    Args:
        atom (rdchem.Atom): An RDKit Atom object.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = np.zeros(27)    
    anum = atom.GetAtomicNum()        
    attributes[anumtable[anum]] = 1.0
        
    ncnt = min(len(atom.GetNeighbors()),5)
    attributes[11+ncnt] = 1.0

    hcnt = min(atom.GetTotalNumHs(),4)
    attributes[17+hcnt] = 1.0
    
    charge = atom.GetFormalCharge()
    if charge == 0:
        attributes[23] = 1.0
    elif charge < 0:
        attributes[22] = 1.0
    else:
        attributes[24] = 1.0
        
    attributes[25] = atom.IsInRing()
    attributes[26] = atom.GetIsAromatic()
    return attributes


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, index, smile=''):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.index = index
        self.smile = smile


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features: (adjacency matrices, node features).
    """
    adjacency_list, features_list, smiles_list, index_list = [], [], [], []

    max_size = 0
    for molecule in batch:
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        smiles_list.append(molecule.smile)
        index_list.append(molecule.index)

    #do not use cuda memory during data loading
    return [torch.FloatTensor(adjacency_list), torch.FloatTensor(features_list), smiles_list, index_list]

def construct_dataset(x_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data, i)
              for i, data in enumerate(x_all)]
    return MolDataset(output)

def construct_loader(x, batch_size=32, shuffle=False):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        batch_size (int): The batch size. Defaults to 32
        shuffle (bool): If True the data will be loaded in a random order. Defaults to False.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x)
    loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,collate_fn=mol_collate_func,shuffle=shuffle)
    return loader


class MolIterableDataset(torch.utils.data.IterableDataset):
    '''An iterable dataset over Molecule objects.'''
    
    def __init__(self, x_smiles, add_dummy_node=True):
        '''Initialize iterable dataset with list of smiles'''
        super(MolIterableDataset).__init__()
        self.x_smiles = x_smiles
        self.add_dummy_node=add_dummy_node
                
    def __iter__(self):
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            index = worker_info.id
            nworkers = worker_info.num_workers
        else:
            index = 0
            nworkers = 1
                    
        for i,smiles in enumerate(self.x_smiles):
            try:
                if i%nworkers == index:
                    mol = MolFromSmiles(smiles.split()[0])
                    afm, adj = featurize_mol(mol, self.add_dummy_node)
                    yield Molecule([afm, adj],i,smiles)
            except ValueError as e:
                logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles,e))


def construct_loader_from_smiles(smiles, batch_size=32, num_workers=1, shuffle=False):
    """Construct a data loader for the provided data.

    Args:
        smiles (list): A list of smiles.
        batch_size (int): The batch size. Defaults to 32
        shuffle (bool): If True the data will be loaded in a random order. Defaults to False.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = MolIterableDataset(smiles)
    loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,collate_fn=mol_collate_func,shuffle=shuffle,num_workers=num_workers)
    return loader
