"""
Optimized version of sascorer.py with performance improvements.
"""

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import math
import pickle
import os.path as op

# Global variables for caching
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
_fscores = None
_fscores_loaded = False

def readFragmentScores(name="fpscores.pkl.gz"):
    """Load fragment scores with better caching."""
    import gzip
    global _fscores, _fscores_loaded
    
    if _fscores_loaded and _fscores is not None:
        return _fscores
    
    # generate the full path filename:
    if name == "fpscores.pkl.gz":
        name = op.join(op.dirname(__file__), name)
    
    data = pickle.load(gzip.open(name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    
    _fscores = outDict
    _fscores_loaded = True
    return _fscores

# Load fragment scores once at module import
_fscores = readFragmentScores()

def numBridgeheadsAndSpiro(mol, ri=None):
    """Calculate bridgeheads and spiro atoms."""
    if ri is None:
        ri = mol.GetRingInfo()
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    """
    Optimized SA score calculation.
    """
    nAtoms = m.GetNumAtoms()
    if not nAtoms:
        return None

    # Fragment score - this is the most expensive part per molecule
    sfp = mfpgen.GetSparseCountFingerprint(m)
    nze = sfp.GetNonzeroElements()
    
    # Optimized fragment scoring
    score1 = 0.0
    nf = 0
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count
    
    if nf > 0:
        score1 /= nf
    else:
        score1 = 0.0

    # Features score - cache ring info for reuse
    ri = m.GetRingInfo()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    
    # Count macrocycles more efficiently
    nMacrocycles = sum(1 for ring in ri.AtomRings() if len(ring) > 8)

    # Calculate penalties
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # Fingerprint density correction
    score3 = 0.0
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * 0.5

    sascore = score1 + score2 + score3

    # Transform to 1-10 scale
    min_val = -4.0
    max_val = 2.5
    sascore = 11.0 - (sascore - min_val + 1) / (max_val - min_val) * 9.0

    # Smooth the high end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    
    # Clamp to [1, 10] range
    return max(1.0, min(10.0, sascore))

def calculateScoresBatch(molecules):
    """
    Calculate SA scores for multiple molecules efficiently.
    
    Args:
        molecules: List of RDKit molecule objects
        
    Returns:
        List of SA scores
    """
    return [calculateScore(mol) for mol in molecules if mol is not None]