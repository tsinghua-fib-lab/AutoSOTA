# council.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

GLOBAL_DESCRIPTORS = [
    'MolLogP', 'MolWt', 'TPSA', 
    'NumHDonors', 'NumHAcceptors', 
    'NumRotatableBonds', 'HallKierAlpha', 'LabuteASA',
    'HeavyAtomCount', 'NumAromaticRings', 
    'MaxPartialCharge', 'MinPartialCharge',
    'pred_Tm', 'abraham_A', 'abraham_B', 
    'abraham_S', 'abraham_E', 'abraham_V'
]

# functional super-groups
SUPER_GROUP_MAP = {
    "Acidic": ["fr_COO", "fr_COO2", "fr_phos_acid", "fr_phos_ester"],
    "Basic":  ["fr_NH0", "fr_NH1", "fr_NH2", "fr_aniline", "fr_pyridine", "fr_quatN", "fr_amidine", "fr_guanido", "fr_amide", "fr_lactam", "fr_imide"],
    "Protic": ["fr_Al_OH", "fr_Ar_OH", "fr_phenol", "fr_SH"],
    "Polar":  ["fr_ketone", "fr_aldehyde", "fr_ester", "fr_nitro", "fr_nitroso", "fr_sulfone", "fr_sulfonamd", "fr_ether", "fr_nitrile", "fr_oxime", "fr_methoxy"],
    "Halogen":["fr_halogen", "fr_alkyl_halide"],
    "Aromatic":["fr_benzene", "fr_imidazole", "fr_furan", "fr_thiophene", "fr_thiazole", "fr_oxazole"]
}

class CouncilExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = GLOBAL_DESCRIPTORS + list(SUPER_GROUP_MAP.keys())
        self.is_fitted = False

    def fit(self, df):
        # learns the scaling parameters from the training data
        X = self._extract_raw(df)
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df):
        # extracts and scales data ready for the transformer
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transform")
        
        X = self._extract_raw(df)
        # scale to mean 0, std 1
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names)

    def _extract_raw(self, df):
        # internal helper
        # grab global physics
        available_global = [c for c in GLOBAL_DESCRIPTORS if c in df.columns]
        data = df[available_global].copy()
        
        # fill missing global cols with 0
        for col in GLOBAL_DESCRIPTORS:
            if col not in data.columns:
                data[col] = 0.0
        
        # aggregate super groups
        for group_name, cols in SUPER_GROUP_MAP.items():
            # Sum the columns. If a column is missing, ignore it.
            available_cols = [c for c in cols if c in df.columns]
            if not available_cols:
                data[group_name] = 0.0
            else:
                data[group_name] = df[available_cols].sum(axis=1)
        
        # Ensure column order matches feature_names
        return data[self.feature_names]
