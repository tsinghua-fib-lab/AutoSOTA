from typing import List, Tuple, Union

from rdkit import Chem

from Graph2Edits.utils.mol_features import get_atom_features, get_bond_features


class MolGraph:
    """
    'MolGraph' represents the graph structure and featurization of a single molecule.

     A MolGraph computes the following attributes:

    * n_atoms: The number of atoms in the molecule.
    * n_bonds: The number of bonds in the molecule.
    * f_atoms: A mapping from an atom index to a list of atom features.
    * f_bonds: A mapping from a bond index to a list of bond features.
    * a2b: A mapping from an atom index to a list of incoming bond indices.
    * b2a: A mapping from a bond index to the index of the atom the bond originates from.
    * b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self.use_rxn_class = use_rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key,
                            value in self.amap_to_idx.items()}

    def _build_graph(self):
        """Builds the graph attributes."""
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Get atom features
        self.f_atoms = [get_atom_features(
            atom, rxn_class=self.rxn_class, use_rxn_class=self.use_rxn_class) for atom in self.mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])
        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = get_bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class RxnGraph:
    """
    RxnGraph contains the information of a reaction, like reactants, products. The edits associated with the reaction are also captured in edit labels.
    """

    def __init__(self, prod_mol: Chem.Mol, edit_to_apply: Tuple, edit_atom: List = [], reac_mol: Chem.Mol = None, rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edit_to_apply: Tuple,
            Edits to apply to the product molecule
        edit_atom: List,
            Edit atom of product molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.prod_graph = MolGraph(
            mol=prod_mol, rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        if reac_mol is not None:
            self.reac_mol = reac_mol
        self.edit_to_apply = edit_to_apply
        self.edit_atom = edit_atom
        self.rxn_class = rxn_class

    def get_components(self, attrs: List = ['prod_graph', 'edit_to_apply', 'edit_atom']) -> Tuple:
        """ 
        Returns the components associated with the reaction graph. 
        """
        attr_tuple = ()
        for attr in attrs:
            if hasattr(self, attr):
                attr_tuple += (getattr(self, attr),)
            else:
                print(f"Does not have attr {attr}")

        return attr_tuple


class Vocab:
    """
    Vocab class to deal with vocabularies and other attributes.
    """

    def __init__(self, elem_list: List) -> None:
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list
        if isinstance(elem_list, dict):
            self.elem_list = list(elem_list.keys())
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}

    def __getitem__(self, a_type: Tuple) -> int:
        return self.elem_to_idx[a_type]

    def get(self, elem: Tuple, idx: int = None) -> int:
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)

    def get_elem(self, idx: int) -> Tuple:
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]

    def __len__(self) -> int:
        return len(self.elem_list)

    def get_index(self, elem: Tuple) -> int:
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]

    def size(self) -> int:
        """Returns length of Vocab."""
        return len(self.elem_list)
