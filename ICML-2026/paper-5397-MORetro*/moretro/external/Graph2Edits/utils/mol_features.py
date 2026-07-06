from typing import Any, List, Set, Union
from rdkit import Chem

ATOM_SYMBOL_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'Si', 'P', 'B', 'I', 'Li', 'Na', 'K', 'Ca',
                    'Mg', 'Al', 'Cu', 'Zn', 'Sn', 'Se', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'As', 'Bi', 'Te', 'Sb',
                    'Ba', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Pb', 'Cs', 'Sm', 'Os', 'Ir', '*', 'unk']

DEGREES = list(range(10))
FORMAL_CHARGE = [-1, -2, 1, 2, 0]
VALENCE = [0, 1, 2, 3, 4, 5, 6]
NUM_Hs = [0, 1, 2, 3, 4]
CHIRALTAG = [0, 1, 2, 3]
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BONDSTEREO = list(range(6))

RXN_CLASSES = list(range(10))
ATOM_FDIM = len(ATOM_SYMBOL_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + \
    len(VALENCE) + len(NUM_Hs) + len(CHIRALTAG) + len(HYBRIDIZATION) + 1
BOND_FDIM = len(BOND_TYPES) + len(BONDSTEREO) + 2


_ONE_HOT_CACHE = {}
def one_of_k_encoding(x: Any, allowable_set: Union[List, Set]) -> List:
    key = (x, id(allowable_set))
    global _ONE_HOT_CACHE
    if key in _ONE_HOT_CACHE:
        return _ONE_HOT_CACHE[key]
        
    original_x = x
    if x not in allowable_set:
        x = list(allowable_set)[-1] if isinstance(allowable_set, set) else allowable_set[-1]
        
    res = [float(x == s) for s in allowable_set]
    _ONE_HOT_CACHE[key] = res
    return res


_ATOM_CACHE = {}
def get_atom_features(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[Union[bool, int, float]]:
    key = (atom.GetSymbol(), atom.GetDegree(), atom.GetFormalCharge(),
           atom.GetHybridization(), atom.GetTotalValence(), atom.GetTotalNumHs(),
           int(atom.GetChiralTag()), atom.GetIsAromatic(),
           rxn_class if use_rxn_class else None, use_rxn_class)
    global _ATOM_CACHE
    if key in _ATOM_CACHE:
        return _ATOM_CACHE[key]
        
    if use_rxn_class:
        atom_features = (one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL_LIST) +
            one_of_k_encoding(atom.GetDegree(), DEGREES) +
            one_of_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGE) +
            one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION) +
            one_of_k_encoding(atom.GetTotalValence(), VALENCE) +
            one_of_k_encoding(atom.GetTotalNumHs(), NUM_Hs) +
            one_of_k_encoding(int(atom.GetChiralTag()), CHIRALTAG) +
            [atom.GetIsAromatic()] + one_of_k_encoding(rxn_class, RXN_CLASSES))
    else:
        atom_features = (one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL_LIST) +
            one_of_k_encoding(atom.GetDegree(), DEGREES) +
            one_of_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGE) +
            one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION) +
            one_of_k_encoding(atom.GetTotalValence(), VALENCE) +
            one_of_k_encoding(atom.GetTotalNumHs(), NUM_Hs) +
            one_of_k_encoding(int(atom.GetChiralTag()), CHIRALTAG) +
            [atom.GetIsAromatic()])
            
    _ATOM_CACHE[key] = atom_features
    return atom_features


def get_bond_features(bond: Chem.Bond) -> List[Union[bool, int, float]]:
    """
    Get bond features.
    """
    # if bond is None:
    #     bond_features = [1] + [0] * (BOND_FDIM - 1)
    # else:
    bond_features = one_of_k_encoding(bond.GetBondType(), BOND_TYPES) + \
        one_of_k_encoding(int(bond.GetStereo()), BONDSTEREO) + \
        [bond.GetIsConjugated()] + [bond.IsInRing()]

    return bond_features
