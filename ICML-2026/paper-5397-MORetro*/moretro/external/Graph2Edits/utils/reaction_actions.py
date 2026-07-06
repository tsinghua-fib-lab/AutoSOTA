"""
Definitions of basic 'edits' (Actions) to transform a product into synthons and reactants
"""
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Mol, rdchem

from Graph2Edits.utils.chem import attach_lg, fix_Hs_Charge, get_atom_Chiral, get_bond_stereo

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}


class ReactionAction(metaclass=ABCMeta):
    def __init__(self, atom_map1: int, atom_map2: int, action_vocab: str):
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.action_vocab = action_vocab

    @abstractmethod
    def get_tuple(self) -> Tuple[str, ...]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def apply(self, mol: Mol) -> Mol:
        raise NotImplementedError('Abstract method')


class AtomEditAction(ReactionAction):
    def __init__(self, atom_map1: int, num_explicit_hs: int, chiral_tag: int, action_vocab: str):
        super(AtomEditAction, self).__init__(atom_map1, -1, action_vocab)
        self.num_explicit_hs = num_explicit_hs
        self.chiral_tag = chiral_tag

    @property
    def feat_vals(self) -> Tuple[int, int]:
        return self.num_explicit_hs, self.chiral_tag

    def get_tuple(self) -> Tuple[str, Tuple[int, int]]:
        return self.action_vocab, self.feat_vals

    def apply(self, mol: Mol) -> Mol:
        new_mol = Chem.RWMol(mol)
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()
                    if atom.GetAtomMapNum() != 0}
        atom_idx = amap_idx[self.atom_map1]
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetNumExplicitHs(self.num_explicit_hs)
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]
        atom.SetChiralTag(a_chiral)
        pred_mol = new_mol.GetMol()
        return pred_mol

    def __str__(self):
        return f'Edit Atom {self.atom_map1}: Num explicit Hs={self.num_explicit_hs}, Chiral_tag={self.chiral_tag}'


class BondEditAction(ReactionAction):
    def __init__(self, atom_map1: int, atom_map2: int,
                 bond_type: Optional[int], bond_stereo: Optional[int],
                 action_vocab: str):
        super(BondEditAction, self).__init__(
            atom_map1, atom_map2, action_vocab)
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo

    def get_tuple(self) -> Tuple[str, Tuple[Optional[int], Optional[int]]]:
        return self.action_vocab, (self.bond_type, self.bond_stereo)

    def apply(self, mol: Mol) -> Mol:
        new_mol = Chem.RWMol(mol)
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()
                    if atom.GetAtomMapNum() != 0}
        atom1 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map1])
        atom2 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map2])

        if self.bond_type is None:  # delete bond
            bond = new_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
            pred_mol = new_mol.GetMol()

        else:
            b_type = rdchem.BondType.values[self.bond_type]
            b_stereo = rdchem.BondStereo.values[self.bond_stereo]

            bond = new_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
            b1 = bond.GetBondTypeAsDouble()
            if bond is None:  # add new bond
                pass
            else:  # change an existing bond
                bond.SetBondType(b_type)
                bond.SetStereo(b_stereo)
                b2 = bond.GetBondTypeAsDouble()

                val = b1 - b2
                if val > 0:
                    atom1.SetNumExplicitHs(int(atom1.GetNumExplicitHs() + val))
                    atom2.SetNumExplicitHs(int(atom2.GetNumExplicitHs() + val))

                elif val < 0:
                    atom1.SetNumExplicitHs(
                        int(max(0, atom1.GetNumExplicitHs() + val)))
                    atom2.SetNumExplicitHs(
                        int(max(0, atom2.GetNumExplicitHs() + val)))

                if atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'O':
                    if b1 == 1.0 and b2 == 2.0 and atom2.GetFormalCharge() == -1:
                        atom2.SetFormalCharge(0)
                    if b1 == 2.0 and b2 == 1.0 and atom2.GetFormalCharge() == 0:
                        atom1.SetNumExplicitHs(0)

                elif atom2.GetSymbol() == 'S' and atom1.GetSymbol() == 'O':
                    if b1 == 1.0 and b2 == 2.0 and atom1.GetFormalCharge() == -1:
                        atom1.SetFormalCharge(0)
                    if b1 == 2.0 and b2 == 1.0 and atom1.GetFormalCharge() == 0:
                        atom2.SetNumExplicitHs(0)

                elif atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N':
                    if b1 == 3.0 and b2 == 1.0 and atom1.GetFormalCharge() == -1 and atom2.GetFormalCharge() == 1:
                        atom1.SetFormalCharge(0)
                        atom2.SetFormalCharge(0)
                        atom1.SetNumExplicitHs(1)
                        atom2.SetNumExplicitHs(1)

                elif atom2.GetSymbol() == 'C' and atom1.GetSymbol() == 'N':
                    if b1 == 3.0 and b2 == 1.0 and atom2.GetFormalCharge() == -1 and atom1.GetFormalCharge() == 1:
                        atom1.SetFormalCharge(0)
                        atom2.SetFormalCharge(0)
                        atom1.SetNumExplicitHs(1)
                        atom2.SetNumExplicitHs(1)

            pred_mol = new_mol.GetMol()
            # fix explicit Hs and charge
            pred_mol = fix_Hs_Charge(pred_mol)

        return pred_mol

    def __str__(self):
        if self.bond_type is None:
            return f'Delete bond {self.atom_map1, self.atom_map2}'
        bond_feat = f'Bond type={self.bond_type}, Bond Stereo={self.bond_stereo}'
        return f'{self.action_vocab} {self.atom_map1, self.atom_map2}: {bond_feat}'


class AddGroupAction(ReactionAction):
    def __init__(self, atom_map1: int, leaving_group: str, action_vocab: str):
        super(AddGroupAction, self).__init__(atom_map1, -1, action_vocab)
        self.leaving_group = leaving_group

    def get_tuple(self) -> Tuple[str, str]:
        return self.action_vocab, self.leaving_group

    def apply(self, mol: Mol) -> Mol:
        lg_mol = Chem.MolFromSmiles(self.leaving_group)
        Chem.Kekulize(lg_mol)
        try:
            pred_mol = attach_lg(main_mol=mol, lg_mol=lg_mol,
                                 attach_atom_map=self.atom_map1)
        except Exception as e:
            print('fail to attach lg')
            pred_mol = mol
        # fix explicit Hs and charge
        pred_mol = fix_Hs_Charge(pred_mol)
        return pred_mol

    def __str__(self):
        return f'Attaching {self.leaving_group} to atom {self.atom_map1}'


class Termination(ReactionAction):
    def __init__(self, action_vocab: str):
        super(Termination, self).__init__(-1, -1, action_vocab=action_vocab)

    def get_tuple(self) -> Tuple[str]:
        return self.action_vocab

    def apply(self, mol: Mol) -> Mol:

        atom_chiral = get_atom_Chiral(mol)
        bond_stereo = get_bond_stereo(mol)
        if all(int(bt) == 0 for bt in bond_stereo.values()) and all(int(chiral) == 0 for chiral in atom_chiral.values()):
            mol = Chem.MolFromSmiles(
                Chem.MolToSmiles(mol, isomericSmiles=False))
            return mol

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        # Dealing with the inconsistency between molecular mol chirality and atomic chiral tag
        for atom in mol.GetAtoms():
            amap_num = atom.GetAtomMapNum()
            atom.SetChiralTag(atom_chiral[amap_num])

        # Dealing with the inconsistency between molecular mol stereo and bond stereo
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                    for atom in mol.GetAtoms()}
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            atom1 = mol.GetAtomWithIdx(amap_idx[a1])
            atom2 = mol.GetAtomWithIdx(amap_idx[a2])
            bond_atoms = sorted([a1, a2])
            st = bond_stereo[tuple(bond_atoms)]

            a1_max_neigh = None
            a2_max_neigh = None
            bond.SetStereo(st)

            if int(st) != 0 and len(list(bond.GetStereoAtoms())) < 2:
                if len([a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]) == 0 or len(
                        [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()]) == 0:
                    continue
                else:
                    a1_max_neigh_num = max(
                        [a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()])
                    a2_max_neigh_num = max(
                        [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()])

                    for a in atom1.GetNeighbors():
                        if a.GetAtomicNum() == a1_max_neigh_num and a.GetIdx() != atom2.GetIdx():
                            a1_max_neigh = a.GetIdx()
                    for a in atom2.GetNeighbors():
                        if a.GetAtomicNum() == a2_max_neigh_num and a.GetIdx() != atom1.GetIdx():
                            a2_max_neigh = a.GetIdx()

                    if all(a.GetAtomicNum() == a1_max_neigh_num for a in atom1.GetNeighbors() if
                           a.GetIdx() != atom2.GetIdx()) and len(
                            [a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]) == 2:
                        a11_max_neigh_num = 0
                        for a in atom1.GetNeighbors():
                            if a.GetIdx() != atom2.GetIdx():
                                if len([a1.GetAtomicNum() for a1 in a.GetNeighbors() if
                                        a1.GetIdx() != atom1.GetIdx()]) == 0:
                                    continue
                                else:
                                    a11_max_neigh_num = max(a11_max_neigh_num,
                                                            max([a1.GetAtomicNum() for a1 in a.GetNeighbors() if
                                                                 a1.GetIdx() != atom1.GetIdx()]))
                        for a in atom1.GetNeighbors():
                            if a.GetIdx() != atom2.GetIdx():
                                for a1 in a.GetNeighbors():
                                    if a1.GetAtomicNum() == a11_max_neigh_num and a1.GetIdx() != atom1.GetIdx():
                                        a1_max_neigh = a.GetIdx()

                    if all(a.GetAtomicNum() == a2_max_neigh_num for a in atom2.GetNeighbors() if
                           a.GetIdx() != atom1.GetIdx()) and len(
                            [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()]) == 2:
                        a12_max_neigh_num = 0
                        for a in atom2.GetNeighbors():
                            if a.GetIdx() != atom1.GetIdx():
                                if len([a2.GetAtomicNum() for a2 in a.GetNeighbors() if
                                        a2.GetIdx() != atom2.GetIdx()]) == 0:
                                    continue
                                else:
                                    a12_max_neigh_num = max(a12_max_neigh_num,
                                                            max([a2.GetAtomicNum() for a2 in a.GetNeighbors() if
                                                                 a2.GetIdx() != atom2.GetIdx()]))
                        for a in atom2.GetNeighbors():
                            if a.GetIdx() != atom1.GetIdx():
                                for a2 in a.GetNeighbors():
                                    if a2.GetAtomicNum() == a12_max_neigh_num and a2.GetIdx() != atom2.GetIdx():
                                        a2_max_neigh = a.GetIdx()

                if a1_max_neigh is not None and a2_max_neigh is not None:
                    try:
                        bond.SetStereoAtoms(a1_max_neigh, a2_max_neigh)
                    except:
                        bond.SetStereoAtoms(a2_max_neigh, a1_max_neigh)

        return mol

    def __str__(self):
        return 'Terminate'
