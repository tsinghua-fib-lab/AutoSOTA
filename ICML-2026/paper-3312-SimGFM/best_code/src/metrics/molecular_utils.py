# --- Helper functions and cache class for MOSES metrics ---
from collections import Counter
import os, pickle, pathlib, warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def _scaffold_histogram(smiles, generic: bool = True) -> Counter:
    """
    Map SMILES sequences to frequency histogram of (generic) Murcko scaffolds.
    
    Note: This function is kept for backward compatibility but is no longer 
    used for MOSES Scaf metric. MOSES Scaf uses unique scaffold overlap percentage.
    """
    hist = Counter()
    for s in smiles or []:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None or scaf.GetNumAtoms() == 0:
            continue
        if generic:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        scaf_smi = Chem.MolToSmiles(scaf, isomericSmiles=False)
        if scaf_smi:
            hist[scaf_smi] += 1
    return hist

def _extract_unique_scaffolds(smiles, generic: bool = True) -> set:
    """
    Extract unique Murcko scaffolds from a list of SMILES.
    
    Args:
        smiles: List of SMILES strings
        generic: If True, use generic scaffolds (heteroatoms → C, bonds → single)
        
    Returns:
        Set of unique scaffold SMILES strings
    """
    scaffolds = set()
    for s in smiles or []:
        if not s:
            continue
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            if scaf is None or scaf.GetNumAtoms() == 0:
                continue
            if generic:
                scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
            scaf_smi = Chem.MolToSmiles(scaf, isomericSmiles=False)
            if scaf_smi:
                scaffolds.add(scaf_smi)
        except:
            continue
    return scaffolds

def _cosine_from_counters(a: Counter, b: Counter):
    """
    Cosine similarity between two count dictionaries; returns None if not computable.
    
    Note: This function is kept for backward compatibility but is no longer 
    used for MOSES Scaf metric. MOSES Scaf uses unique scaffold overlap percentage.
    """
    if not a or not b:
        return None
    keys = set(a) | set(b)
    if not keys:
        return None
    v1 = np.array([a.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return None
    return float(v1.dot(v2) / denom)

class MosesRefCache:
    """
    Cache for MOSES reference data: ref_fps (for SNN), ref_scaf_hist (for Scaf), and FCD reference embeddings.
    Generic for any dataset: uses reference_smiles['train'] as reference distribution.
    """
    def __init__(self, reference_smiles: dict, cfg, mm_module, fcd_metric=None, cache_name_prefix: str = "moses_ref"):
        self.reference_smiles = reference_smiles
        self.cfg = cfg
        self.mm = mm_module  # MM module passed in (remove_invalid / fingerprints / etc)
        self.fcd_metric = fcd_metric  # FCD metric instance for computing embeddings
        # Unified cache directory
        datadir = getattr(cfg.dataset, "datadir", "")
        root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.cache_dir = os.path.join(root_dir, datadir) if datadir else str(root_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_name_prefix = cache_name_prefix

        # In-memory cache for current runtime
        self._ref_fps = None
        self._ref_scaf_hist = None
        self._fcd_prefs = {}  # dict: split -> precomputed FCD embedding

    def _path(self, suffix: str) -> str:
        # Include remove_h / dataset name in filename to avoid mismatches
        remove_h = getattr(getattr(self.cfg, "dataset", None), "remove_h", None)
        dataset_name = getattr(getattr(self.cfg, "dataset", None), "name", "")
        
        # For QM9 dataset, distinguish between with/without H
        if dataset_name == "qm9":
            tag = f"{self.cache_name_prefix}_{dataset_name}{'_no_h' if remove_h else '_h'}"
        else:
            tag = f"{self.cache_name_prefix}{'_no_h' if remove_h else ''}"
        
        return os.path.join(self.cache_dir, f"{tag}_{suffix}.pkl")

    def ensure_ref_fps(self, split='test'):
        """
        Get fingerprints for reference set.
        
        Args:
            split: Which split to use as reference. Default 'test' (MOSES convention).
                   MOSES uses TEST set for SNN calculation, not TRAIN set.
        """
        cache_key = f'_ref_fps_{split}'
        if hasattr(self, cache_key) and getattr(self, cache_key) is not None:
            return getattr(self, cache_key)
        
        path = self._path(f"fps_{split}")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                fps = pickle.load(fh)
                setattr(self, cache_key, fps)
                return fps
        
        ref_smiles = [s for s in self.reference_smiles.get(split) if s]
        ref_valid = self.mm.remove_invalid(ref_smiles)
        fps = self.mm.fingerprints(ref_valid)
        setattr(self, cache_key, fps)
        with open(path, "wb") as fh:
            pickle.dump(fps, fh)
        return fps

    def ensure_ref_scaf_hist(self):
        """
        Legacy method: returns scaffold histogram (for backward compatibility).
        Note: MOSES Scaf metric now uses ensure_ref_unique_scaffolds() instead.
        """
        if self._ref_scaf_hist is not None:
            return self._ref_scaf_hist
        path = self._path("scaf_hist")
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    self._ref_scaf_hist = pickle.load(fh)
                    return self._ref_scaf_hist
            except Exception as e:
                warnings.warn(f"Load cached scaf hist failed, rebuilding. err={e}")
        ref_smiles = [s for s in self.reference_smiles.get("train") if s]
        ref_valid = self.mm.remove_invalid(ref_smiles)
        self._ref_scaf_hist = _scaffold_histogram(ref_valid, generic=True)
        with open(path, "wb") as fh:
            pickle.dump(self._ref_scaf_hist, fh)
        return self._ref_scaf_hist
    
    def ensure_ref_unique_scaffolds(self):
        """Get unique scaffolds from reference (training) set."""
        if hasattr(self, '_ref_unique_scaffolds') and self._ref_unique_scaffolds is not None:
            return self._ref_unique_scaffolds
        path = self._path("unique_scaffolds")
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    self._ref_unique_scaffolds = pickle.load(fh)
                    return self._ref_unique_scaffolds
            except Exception as e:
                warnings.warn(f"Load cached unique scaffolds failed, rebuilding. err={e}")
        ref_smiles = [s for s in self.reference_smiles.get("train") if s]
        ref_valid = self.mm.remove_invalid(ref_smiles)
        self._ref_unique_scaffolds = _extract_unique_scaffolds(ref_valid, generic=True)
        with open(path, "wb") as fh:
            pickle.dump(self._ref_unique_scaffolds, fh)
        return self._ref_unique_scaffolds

    def ensure_fcd_pref(self, split: str = "train"):
        """
        Load or compute FCD reference embedding (pref) for the specified split.
        Args:
            split: One of 'train', 'val', 'test'
        Returns:
            Precomputed FCD embedding object (can be passed to fcd_metric(gen=..., pref=...))
        """
        if not self.fcd_metric:
            warnings.warn("FCD metric not initialized, cannot compute FCD pref")
            return None
        
        # Check in-memory cache
        if split in self._fcd_prefs:
            return self._fcd_prefs[split]
        
        # Check disk cache
        path = self._path(f"fcd_pref_{split}")
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    self._fcd_prefs[split] = pickle.load(fh)
                    print(f"Loaded FCD pref for {split} from cache: {path}")
                    return self._fcd_prefs[split]
            except Exception as e:
                warnings.warn(f"Load cached FCD pref failed, rebuilding. err={e}")
        
        # Compute and cache
        ref_smiles = self.reference_smiles.get(split, [])
        
        # Handle numpy arrays properly
        if ref_smiles is None:
            warnings.warn(f"No reference smiles for split {split}")
            return None
        
        # Convert to list if it's a numpy array
        if hasattr(ref_smiles, 'tolist'):
            ref_smiles = ref_smiles.tolist()
        
        if len(ref_smiles) == 0:
            warnings.warn(f"No reference smiles for split {split}")
            return None
        
        # Filter out None values
        ref_smiles_clean = [s for s in ref_smiles if s is not None]
        if len(ref_smiles_clean) == 0:
            warnings.warn(f"No valid reference smiles for split {split}")
            return None
        
        print(f"Computing FCD pref for {split} ({len(ref_smiles_clean)} molecules)...")
        try:
            self._fcd_prefs[split] = self.fcd_metric.precalc(ref_smiles_clean)
            with open(path, "wb") as fh:
                pickle.dump(self._fcd_prefs[split], fh)
            print(f"Saved FCD pref for {split} to: {path}")
            return self._fcd_prefs[split]
        except Exception as e:
            warnings.warn(f"Failed to compute FCD pref for {split}: {e}")
            return None
