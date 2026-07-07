
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, RDKFingerprint
from rdkit.Contrib.SA_Score import sascorer

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0,#-1!h0]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.GetSetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol
def get_fingerprint(mols):
    fps = []
    for mol in mols:
        fps.append(RDKFingerprint(mol))
    return fps
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def internal_diversity(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    fps = get_fingerprint(mols)
    n = len(mols)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = tanimoto_similarity(fps[i], fps[j])
            sims.append(sim)
    avg_sim = np.mean(sims)
    return 1 - avg_sim


def validate_and_standardize(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    standard_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    if "." in standard_smiles:
        fragments = standard_smiles.split('.')
        standard_smiles = max(fragments, key=len)
        mol = Chem.MolFromSmiles(standard_smiles)

    return Chem.MolToSmiles(mol)


def calculate_normalized_sa(mol):
    raw_sa = sascorer.calculateScore(mol)
    norm_sa = max(0.0, min(1.0, (10.0 - raw_sa) / 9.0))
    return norm_sa