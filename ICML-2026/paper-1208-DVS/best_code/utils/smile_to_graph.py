# Code adapted from chainer_chemistry\dataset\preprocessors\common
import numpy

from rdkit import Chem
from rdkit.Chem import rdmolops


class GGNNPreprocessor(object):
    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False, kekulize=True):
        super(GGNNPreprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize

        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array

    def prepare_smiles_and_mol(self, mol):
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    def get_label(self, mol, label_names=None):
        if label_names is None:
            return []

        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)
        return label_list


class MolFeatureExtractionError(Exception):
    pass


def type_check_num_atoms(mol, num_max_atoms=-1):
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


def construct_atomic_number_array(mol, out_size=-1):
    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if out_size < 0:
        return numpy.array(atom_list, dtype=numpy.int32)
    elif out_size >= n_atom:
        atom_array = numpy.zeros(out_size, dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list, dtype=numpy.int32)
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atom))


def construct_adj_matrix(mol, out_size=-1, self_connection=True):
    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connection:
        adj = adj + numpy.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(numpy.float32)
    elif out_size >= s0:
        adj_array = numpy.zeros((out_size, out_size),
                                dtype=numpy.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))
    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1):
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    return adjs
