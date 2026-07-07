import numpy as np
import random

class SeedSelector:
    def __init__(self, initial_molecules):
        self.mols = {m.smiles: m for m in initial_molecules}
        self.stats = {m.smiles: {"alpha": 1.0, "beta": 1.0} for m in initial_molecules}

    def add_new_molecule(self, new_mol, parent_smiles=None):

        if new_mol.smiles in self.mols:
            return

        self.mols[new_mol.smiles] = new_mol

        if parent_smiles and parent_smiles in self.stats:
            p_stat = self.stats[parent_smiles]
            self.stats[new_mol.smiles] = {
                "alpha": 1.0 + (p_stat["alpha"] - 1.0) * 0.5,
                "beta": 1.0 + (p_stat["beta"] - 1.0) * 0.5
            }
        else:
            self.stats[new_mol.smiles] = {"alpha": 1.0, "beta": 1.0}

    def update_node(self, smiles, child_score, max_incr=3.0, scale=3):

        if smiles not in self.stats: return

        delta = self.mols[smiles].score - child_score
        increment = min(abs(delta) * scale, max_incr)

        if delta > 0:
            self.stats[smiles]["alpha"] += increment
        elif delta < 0:
            self.stats[smiles]["beta"] += increment

    def select_seed(self, tau=2.0, use_thompson=True, use_bolzmann=True):
        # upper-level
        best_val = -float('inf')
        selected_mol = None
        if not use_thompson:
            smiles_list = list(self.mols.keys())
            smiles = random.choice(smiles_list)
            return self.mols[smiles]

        for smiles, stat in self.stats.items():

            theta = np.random.beta(stat["alpha"], stat["beta"])

            if use_bolzmann:
                weight = np.exp(-self.mols[smiles].score / tau)
                score_ts = theta * weight
            else:
                score_ts = theta

            if score_ts > best_val:
                best_val = score_ts
                selected_mol = self.mols[smiles]

        return selected_mol