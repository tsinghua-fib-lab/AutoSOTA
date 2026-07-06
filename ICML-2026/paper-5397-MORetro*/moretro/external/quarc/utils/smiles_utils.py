from __future__ import print_function

import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

_density_clean = None
_COMMON_SOLVENTS_CANONICAL = None


def _get_density_data():
    """Lazy loading of density data. Returns None if file is unavailable."""
    global _density_clean
    if _density_clean is None:
        try:
            # Use docker path
            _density_clean = pd.read_csv('/app/quarc/data/external/densities.tsv', sep="\t")

        # v1: Silent failure
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            _density_clean = False
            return None

        # # v2: Raise error
        # except Exception as e:
        #     raise FileNotFoundError(f"Could not load density file from /app/quarc/data/external/densities.tsv: {e}")


    return _density_clean if _density_clean is not False else None


def _get_common_solvents():
    """Lazy loading of common solvents list. Returns empty set if density data unavailable."""
    global _COMMON_SOLVENTS_CANONICAL
    if _COMMON_SOLVENTS_CANONICAL is None:
        density_clean = _get_density_data()
        if density_clean is None:
            _COMMON_SOLVENTS_CANONICAL = set()
        else:
            common_solvents_list = density_clean[density_clean["name"].isin(solvents_verified)][
                "can_smiles"
            ].tolist()
            _COMMON_SOLVENTS_CANONICAL = set(common_solvents_list)
    return _COMMON_SOLVENTS_CANONICAL


solvents_verified = [
    "pentane",
    "hexane",
    "heptane",
    "octane",
    "diethyl ether",
    "triethylamine",
    "decane",
    "methyl tert-butyl ether",
    "cyclopentane",
    "cyclohexane",
    "acetone",
    "acetonitrile",
    "2-propanol",
    "ethanol",
    "tert-butanol",
    "methanol",
    "isobutanol",
    "1-propanol",
    "2-butanone",
    "2-pentanone",
    "2-butanol",
    "1-butanol",
    "2-pentanol",
    "cyclohexene",
    "1-pentanol",
    "3-pentanone",
    "1-heptanol",
    "1-octanol",
    "1-nonanol",
    "1-decanol",
    "m-xylene",
    "p-xylene",
    "cumene",
    "ethylbenzene",
    "toluene",
    "1,2-dimethoxyethane",
    "benzene",
    "benzonitrile",
    "o-xylene",
    "THF",
    "hexamethylphosphorus triamide",
    "ethyl acetate",
    "ethyl formate",
    "methyl acetate",
    "diglyme",
    "cyclohexanone",
    "DMF",
    "cyclohexanol",
    "diethyl carbonate",
    "methyl formate",
    "pyridine",
    "anisole",
    "water",
    "aniline",
    "1,4-dioxane",
    "hexamethylphosphoramide",
    "N-methyl-2-pyrrolidone",
    "naphthalene",
    "benzaldehyde",
    "benzyl alcohol",
    "acetic acid",
    "dimethyl carbonate",
    "DMSO",
    "2-phenoxyethanol",
    "chlorobenzene",
    "heavy water",
    "ethylene glycol",
    "diethylene glycol",
    "1,2-dichloroethane",
    "formic acid",
    "1,2-dichloroethane",
    "glycerol",
    "carbon disulfide",
    "DCM",
    "nitromethane",
    "chloroform",
    "TFA",
    "carbon tetrachloride",
    "hydrazine hydrate",
    "isopropyl alcohol",
    # 16 newly added solv with densities
    "deuterochloroform",
    "tert-Amyl alcohol",
    "2-ethoxyethanol",
    "propionitrile",
    "mesitylene",
    "Isohexane",
    "dimethylether",
    "trifluoromethylbenzene",
    "Dowtherm",
    "butyronitrile",
    "tert-butyl acetate",
    "propylene glycol",
    "1,1,2,2-tetrachloroethane",
    "1,4-dichlorobenzene",
    "triethylene glycol",
    "p-cymene",
]


def get_common_solvents_canonical() -> set[str]:
    """Get the set of common solvent canonical SMILES. Returns empty set if density data unavailable."""
    return _get_common_solvents()


def parse_rxn_smiles(rxn_smiles: str) -> tuple[list[str], list[str], list[str]]:
    """Split rxn_smiles into reactants, agents, and products by dot, no combination needed"""
    try:
        reactants, products = rxn_smiles.split(">>")
        agents = ""
        if " |" in products:
            products, _ = products.split(" |")
    except (TypeError, ValueError):
        raise ValueError(f"\tError splitting rxn SMILES: {rxn_smiles}")

    try:
        reactants = [canonicalize_smiles(s) for s in reactants.split(".")]
        agents = [canonicalize_smiles(s) for s in agents.split(".")]
        products = [canonicalize_smiles(s) for s in products.split(".")]
    except:
        raise ValueError(f"\tError canonicalize SMILES for {rxn_smiles}")

    reactants = [s for s in reactants if s]
    agents = [s for s in agents if s]
    products = [s for s in products if s]

    return reactants, agents, products


def canonicalize_smiles(s, is_sanitize=True):
    """Remove atom mapping, canonicalize smiles"""
    if s is None:
        return None
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        if is_sanitize:
            Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def is_atom_map_mol(mol):
    res = False
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            res = True
            break
    return res


def is_atom_map_rxn_smiles(s):
    s_list = s.split(" ")
    r = AllChem.ReactionFromSmarts(str(s_list[0]))
    products = r.GetProducts()
    res = False
    for p in products:
        if is_atom_map_mol(p):
            res = True
            break
    return res


def remove_stereo(smiles) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def canonicalize_smiles_list(s_list):
    for i in range(len(s_list)):
        s_list[i] = canonicalize_smiles(s_list[i])
    return s_list


def prep_rxn_smi_input(rxn_smi):
    """remove the |f| part and agents from rxn_smiles"""
    clean_smi = rxn_smi.split(" |")[0]
    reactants, products = clean_smi.split(">>")
    return f"{reactants}>>{products}"
