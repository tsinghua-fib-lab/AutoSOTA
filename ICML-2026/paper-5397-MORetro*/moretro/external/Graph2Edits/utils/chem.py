from typing import Dict, Tuple
from rdkit import Chem
from rdkit.Chem import Mol, RWMol, rdchem

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1,
             'Cl': 1, 'F': 1, 'I': 1, 'Li': 1, 'Na': 1, 'K': 1}


def get_atom_info(mol: Mol) -> Dict:
    if mol is None:
        return {}

    atom_info = {}
    for atom in mol.GetAtoms():
        feat = [atom.GetNumExplicitHs(), int(atom.GetChiralTag())]
        amap_num = atom.GetAtomMapNum()
        atom_info[amap_num] = tuple(feat)
    return atom_info


def get_atom_Chiral(mol: Mol) -> Dict:
    if mol is None:
        return {}

    atom_Chiral = {}
    for atom in mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        atom_Chiral[amap_num] = atom.GetChiralTag()
    return atom_Chiral


def get_bond_info(mol: Mol) -> Dict:
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        bt = int(bond.GetBondType())
        st = int(bond.GetStereo())
        bond_atoms = sorted([a1, a2])
        bond_info[tuple(bond_atoms)] = [bt, st]
    return bond_info


def get_bond_stereo(mol: Mol) -> Dict:
    if mol is None:
        return {}

    bond_stereo = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        bond_atoms = sorted([a1, a2])
        bond_stereo[tuple(bond_atoms)] = bond.GetStereo()
    return bond_stereo


def align_kekulize_pairs(r_mol: Mol, p_mol: Mol) -> Tuple[Mol, Mol]:
    prod_old = get_bond_info(p_mol)
    Chem.Kekulize(p_mol)
    prod_new = get_bond_info(p_mol)

    react_old = get_bond_info(r_mol)
    Chem.Kekulize(r_mol)
    react_new = get_bond_info(r_mol)

    r_mol = Chem.RWMol(r_mol)
    r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                  for atom in r_mol.GetAtoms()}
    for bond in prod_new:
        if bond in react_new and (react_old[bond][0] == prod_old[bond][0]) and (react_new[bond][0] != prod_new[bond][0]):
            idx1, idx2 = r_amap_idx[bond[0]], r_amap_idx[bond[1]]
            bt = prod_new[bond][0]
            b_type = rdchem.BondType.values[bt]
            r_mol.GetBondBetweenAtoms(idx1, idx2).SetBondType(b_type)

    return r_mol.GetMol(), p_mol


def get_atom_idx(mol: RWMol or Mol, atom_map: int) -> int:
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomMapNum() == atom_map:
            return i
    raise ValueError(f'No atom with map number: {atom_map}')


def attach_lg(main_mol: Mol, lg_mol: Mol, attach_atom_map: int) -> Mol:
    combined_mol = Chem.CombineMols(main_mol, lg_mol)
    rw_mol = Chem.RWMol(Chem.Mol(combined_mol))

    lg_attach_num = 0
    for atom in rw_mol.GetAtoms():
        if atom.GetSymbol() == '*':
            lg_attach_num += 1

    if lg_attach_num == 1:
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                remove_idx = atom.GetIdx()
                lg_attach_atom = atom.GetNeighbors()
                lg_attach_atom[0].SetAtomMapNum(500)
                bond = atom.GetBonds()
                bt = bond[0].GetBondType()

        rw_mol.RemoveAtom(remove_idx)
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                    if atom.GetAtomMapNum() != 0}

        attach_atom_idx = amap_idx[attach_atom_map]
        lg_attach_idx = amap_idx[500]

        rw_mol.AddBond(attach_atom_idx, lg_attach_idx, bt)
        rw_mol.GetAtomWithIdx(amap_idx[500]).ClearProp('molAtomMapNumber')

    else:
        lg_attach_amap = 500
        remove_atommap = 1000
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                atom.SetAtomMapNum(remove_atommap)
                lg_attach_atom = atom.GetNeighbors()
                lg_attach_atom[0].SetAtomMapNum(lg_attach_amap)
                lg_attach_amap += 1
                remove_atommap += 1

        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                    if atom.GetAtomMapNum() != 0}

        for num in range(lg_attach_num):
            lg_attach_amap = 500
            remove_atommap = 1000
            remove_idx = amap_idx[remove_atommap + num]
            remove_atom = rw_mol.GetAtomWithIdx(remove_idx)
            bond = remove_atom.GetBonds()
            bt = bond[0].GetBondType()

            rw_mol.RemoveAtom(remove_idx)
            amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                        if atom.GetAtomMapNum() != 0}

            attach_atom_idx = amap_idx[attach_atom_map]
            lg_attach_idx = amap_idx[lg_attach_amap + num]

            rw_mol.AddBond(attach_atom_idx, lg_attach_idx, bt)
            rw_mol.GetAtomWithIdx(
                amap_idx[lg_attach_amap + num]).ClearProp('molAtomMapNumber')

    max_amap = max([atom.GetAtomMapNum() for atom in rw_mol.GetAtoms()])
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            atom.SetNoImplicit(True)
            max_amap += 1

    new_mol = rw_mol.GetMol()

    return new_mol


def fix_Hs_Charge(mol: Mol) -> Mol:
    # fix explicit Hs and charge
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                        for b in atom.GetBonds()]))

        if not atom.IsInRing():
            atom.SetIsAromatic(False)

        if charge == 0:
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals > MAX_BONDS[atom_symbol]:
                num = int(explicit_hs + bond_vals - MAX_BONDS[atom_symbol])
                for i in range(num):
                    if explicit_hs > 0:
                        explicit_hs -= 1
                        atom.SetNumExplicitHs(explicit_hs)
                    else:
                        atom.SetFormalCharge(1)

            elif atom_symbol in MAX_BONDS and explicit_hs + bond_vals < MAX_BONDS[atom_symbol]:
                num = int(MAX_BONDS[atom_symbol] - explicit_hs - bond_vals)
                for i in range(num):
                    explicit_hs += 1
                    atom.SetNumExplicitHs(explicit_hs)

            # "-N=N+=N-"
            if atom_symbol == 'N' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 2 and atom.GetNeighbors()[0].GetSymbol() == 'N':
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(-1)

            # "NC-"
            if atom_symbol == 'C' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 3 and atom.GetNeighbors()[0].GetSymbol() == 'N':
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(-1)

            if atom_symbol == 'S' and explicit_hs == 0 and bond_vals == 1:
                atom.SetNumExplicitHs(1)

            if atom_symbol == 'S' and explicit_hs == 1 and bond_vals in [2, 4, 6]:
                atom.SetNumExplicitHs(0)

            if atom_symbol == 'P':  # 'P(OCC)3'
                bond_vals = [bond.GetBondTypeAsDouble()
                             for bond in atom.GetBonds()]
                if sum(bond_vals) == 3 and len(bond_vals) == 3:
                    atom.SetNumExplicitHs(0)
                if sum(bond_vals) == 4 and len(bond_vals) == 4:
                    atom.SetFormalCharge(1)

            if atom_symbol == 'Sn':
                if explicit_hs == 0 and bond_vals == 3:
                    atom.SetNumExplicitHs(1)
                if explicit_hs == 1 and bond_vals == 4:
                    atom.SetNumExplicitHs(0)

        else:
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals == MAX_BONDS[atom_symbol]:
                atom.SetFormalCharge(0)

            if atom_symbol == 'O':
                bond_vals = bond_vals + explicit_hs
                if bond_vals == 1 and charge == -1 and atom.GetNeighbors()[0].GetSymbol() != 'N':
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(1)

            if atom_symbol == 'N':
                if bond_vals == 4 and explicit_hs == 0 and charge == -1:
                    atom.SetFormalCharge(1)
                if bond_vals == 3 and explicit_hs == 1 and charge == -1:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)
                if bond_vals == 3 and explicit_hs == 2 and charge == 1:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)

    for atom in mol.GetAtoms():  # Dealing with the problem 'C+'
        if atom.GetSymbol() == 'C' and atom.GetFormalCharge() == 1:
            atom.SetFormalCharge(0)

    return mol
