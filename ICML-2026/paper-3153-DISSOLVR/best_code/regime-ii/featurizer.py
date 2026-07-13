# featurizer.py
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

class MoleculeFeaturizer:
    def __init__(self):
        self.enumerator = rdMolStandardize.TautomerEnumerator()
        
        # Features to exclude (highly correlated with structural features)
        exclude_prefixes = ['BCUT2D', 'SMR_VSA', 'SlogP_VSA', 'VSA_EState', 'Chi', 'FpDensityMorgan']
        # Note: LabuteASA kept for regime-ii council compatibility
        # HeavyAtomMolWt, ExactMolWt removed as redundant with MolWt
        exclude_exact = {'Ipc', 'Kappa3', 'MolMR', 'HeavyAtomMolWt', 'ExactMolWt'}
        
        self.desc_map = {}
        for name, func in Descriptors.descList:
            # Skip excluded features
            if name in exclude_exact:
                continue
            if any(name.startswith(prefix) for prefix in exclude_prefixes):
                continue
            self.desc_map[name] = func
        
        # atom inventory (commented out - redundant with other features)
        # self.inventory = ['C', 'O', 'N', 'Cl', 'S', 'F', 'P', 'Br', 'Na', 'I', 'K', 'B', 'Se', 'Ca', 'Li']

    def transform(self, smiles_list):
        print(f"Computing features for {len(smiles_list)} molecules...")
        feats = [self._calc_feats(s) for s in tqdm(smiles_list, desc="Featurizing")]
        df = pd.DataFrame(feats)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.fillna(0)

    def _get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        try: return self.enumerator.Canonicalize(mol)
        except: return mol

    # def _atom_counts(self, mol):
    #     symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    #     counts = {f"num_{s}": symbols.count(s) for s in self.inventory}
    #     counts["total_atoms"] = len(symbols)
    #     return counts

    def _mose_features(self, mol):
        # 1. Get Adjacency Matrix (Float for precision)
        A = Chem.GetAdjacencyMatrix(mol).astype(float)
        degrees = np.sum(A, axis=0)
        
        # 2. Compute Matrix Powers (A^2 to A^8)
        # Used for counting Cycles (Trace) and Paths (Sum)
        A2 = np.linalg.matrix_power(A, 2)
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        A6 = A5 @ A
        A7 = A6 @ A
        A8 = A7 @ A

        # 3. Compile Homomorphism Counts
        # Note: Benzene and Fused kept as structural matches to preserve chemical specificity
        # while switching topological patterns to strict Homomorphism (Spectral) counts.
        return {
            # Cycles: Hom(C_k, G) = Trace(A^k)
            "mose_cyc3": float(np.trace(A3)),
            "mose_cyc4": float(np.trace(A4)),
            "mose_cyc5": float(np.trace(A5)),
            "mose_cyc6": float(np.trace(A6)),
            "mose_cyc7": float(np.trace(A7)),
            "mose_cyc8": float(np.trace(A8)),
            
            # Paths: Hom(P_k, G) = Sum(A^(k-1))
            # Path3 (3 nodes, 2 edges) -> Sum(A^2)
            "mose_path3": float(np.sum(A2)), 
            "mose_path4": float(np.sum(A3)),
            "mose_path5": float(np.sum(A4)),

            # Stars: Hom(S_k, G) = Sum(degree^k) where k is num_leaves
            # branched_4 = Center + 3 leaves -> 3rd moment
            "mose_branched_4": float(np.sum(np.power(degrees, 3))),
            # star_5 = Center + 4 leaves -> 4th moment
            "mose_star_5": float(np.sum(np.power(degrees, 4))),

            # Preserved Chemical Patterns (RDKit)
            "mose_benzene": len(mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1"))),
            "mose_fused": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]")))
        }

    def _thermo_proxies(self, mol):
        # thermodynamic proxies
        # Joback T_m Increments
        joback = {
            "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
            "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
            "ch2_r": ("[CH2;X4;R]", 8.25), "ch_r": ("[CH1;X4;R]", 20.15),
            "c_r": ("[CH0;X4;R]", 37.40), "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
            "c=c_r": ("[c,C;R]=[c,C;R]", 13.02), "F": ("[F]", 9.88),
            "Cl": ("[Cl]", 17.51), "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
            "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
            "ether_c": ("[OD2;!R]([#6])[#6]", 22.42), "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
            "co": ("[CX3]=[OX1]", 26.15), "ester": ("[CX3](=[OX1])[OX2H0]", 30.0),
            "nh2": ("[NH2]", 25.72), "nh_c": ("[NH1;!R]", 27.15), "nh_r": ("[NH1;R]", 30.12),
            "nitro": ("[NX3](=[OX1])=[OX1]", 45.0), "nitrile": ("[NX1]#[CX2]", 33.15)
        }
        tm_sum = 122.5 + sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(p))) * w for p, w in joback.values())
        
        pred_Tm = np.maximum(tm_sum, 150.0)

        # Abraham Proxies (A, B, S, E, V)
        rd_hbd, rd_hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
        acid_ref = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(s))) for s in ["[OH]c", "C(=O)[OH]"])
        base_ref = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(s))) for s in ["[NH2,NH1,NH0]", "n1ccccc1", "[CX3]=[OX1]"])
        
        proxy_E = Descriptors.MolMR(mol) / 10.0
        hetero_smarts = "[O,N,S,F,Cl,Br,I,P,Se,B,Na,K,Ca,Li]"
        hetero_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(hetero_smarts)))
        proxy_S = (hetero_count * 0.2) + (len(mol.GetSubstructMatches(Chem.MolFromSmarts("a1aaaaa1"))) * 0.3)

        atom_v_map = {
            'C': 16.35, 'N': 14.39, 'O': 12.43, 'F': 10.48, 'Cl': 20.95, 'Br': 26.21, 
            'I': 34.53, 'S': 22.91, 'P': 24.87, 'H': 8.71, 'Se': 25.10,
            'B': 18.32, 'Na': 18.00, 'K': 24.00, 'Ca': 21.00, 'Li': 14.00
        }
        v_total = sum(atom_v_map.get(a.GetSymbol(), 15.0) for a in mol.GetAtoms())
        v_total += sum(a.GetTotalNumHs() for a in mol.GetAtoms()) * 8.71
        v_total -= (6.56 * mol.GetNumBonds())

        return {
            "pred_Tm": pred_Tm, "abraham_A": (rd_hbd * 0.1 + acid_ref * 0.4), 
            "abraham_B": (rd_hba * 0.1 + base_ref * 0.3), "abraham_S": proxy_S, 
            "abraham_E": proxy_E, "abraham_V": v_total / 100.0 #,
            # "pred_S": self._joback_entropy(mol)  # Estimated molar entropy
        }
    
    def _joback_entropy(self, mol):
        """
        Joback-Reid group contribution method for standard molar entropy (S°).
        S° = 41.7 + Σ(group contributions)
        Units: J/(mol·K)
        """
        # Joback-Reid entropy group contributions (J/mol·K)
        joback_S = {
            # Non-ring groups
            "ch3": ("[CH3;X4;!R]", 23.06),      # -CH3
            "ch2_c": ("[CH2;X4;!R]", 22.88),    # -CH2- (chain)
            "ch_c": ("[CH1;X4;!R]", 21.74),     # >CH- (chain)
            "c_c": ("[CH0;X4;!R]", 21.32),      # >C< (chain)
            # Ring groups
            "ch2_r": ("[CH2;X4;R]", 25.14),     # -CH2- (ring)
            "ch_r": ("[CH1;X4;R]", 25.04),      # >CH- (ring)
            "c_r": ("[CH0;X4;R]", 25.00),       # >C< (ring)
            # Double bonds
            "c=c_c": ("[CX3;!R]=[CX3;!R]", 24.28),   # =CH2, =CH-, =C<
            "c=c_r": ("[c,C;R]=[c,C;R]", 24.96),     # aromatic/ring double bond
            # Halogens
            "F": ("[F]", 13.33),
            "Cl": ("[Cl]", 33.36),
            "Br": ("[Br]", 42.00),
            "I": ("[I]", 51.28),
            # Oxygen groups
            "oh_a": ("[OH;!#6a]", 28.12),       # -OH (aliphatic)
            "oh_p": ("[OH;a]", 32.00),          # -OH (phenolic)
            "ether_c": ("[OD2;!R]([#6])[#6]", 25.05),   # -O- (chain)
            "ether_r": ("[OD2;R]([#6])[#6]", 28.00),    # -O- (ring)
            "co": ("[CX3]=[OX1]", 31.93),       # >C=O
            "ester": ("[CX3](=[OX1])[OX2H0]", 38.00),   # -COO-
            # Nitrogen groups
            "nh2": ("[NH2]", 27.78),            # -NH2
            "nh_c": ("[NH1;!R]", 26.20),        # >NH (chain)
            "nh_r": ("[NH1;R]", 29.20),         # >NH (ring)
            "nitro": ("[NX3](=[OX1])=[OX1]", 45.00),   # -NO2
            "nitrile": ("[NX1]#[CX2]", 28.00),  # -C≡N
            # Sulfur
            "sh": ("[SH]", 35.00),              # -SH
            "s_c": ("[SX2;!R]([#6])[#6]", 32.00),  # -S- (chain)
        }
        
        # Base value
        S_sum = 41.7
        
        # Sum contributions
        for name, (smarts, contrib) in joback_S.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = len(mol.GetSubstructMatches(pattern))
                S_sum += matches * contrib
        
        return S_sum

    def _calc_feats(self, smiles):
        mol = self._get_mol(smiles)
        if not mol: return {}
        feats = {}
        # feats.update(self._atom_counts(mol))  # Commented out - redundant with other features
        # feats.update({f"Morgan_{i}": int(b) for i, b in enumerate(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))})
        # feats.update({f"MACCS_{i}": int(b) for i, b in enumerate(MACCSkeys.GenMACCSKeys(mol))})

        # try:
        #     autos = rdMolDescriptors.CalcAUTOCORR2D(mol)
        #     feats.update({f"AUTOCORR2D_{i}": val for i, val in enumerate(autos)})
        # except: pass

        feats.update(self._mose_features(mol))
        feats.update(self._thermo_proxies(mol))
        for name, func in self.desc_map.items():
            try: feats[name] = func(mol)
            except: feats[name] = 0.0
        return feats
