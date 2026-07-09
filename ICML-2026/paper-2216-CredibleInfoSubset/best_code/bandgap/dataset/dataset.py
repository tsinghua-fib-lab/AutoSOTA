import torch
from torch.utils.data import Dataset
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
class CompositionDataset(Dataset):
    def __init__(self, df, formula_col, bandgap_col, state_col):
        self.df = df.reset_index(drop=True)
        self.formula_col = formula_col
        self.bandgap_col = bandgap_col
        self.state_col = state_col
        self.featurizer = ElementProperty.from_preset("magpie", impute_nan=True)

        element_list = ['Kr', 'B', 'P', 'La', 'Ti', 'Tc', 'Ce', 'Sr', 'Ni', 'N', 'Al', 
                        'Ru', 'Hf', 'Ne', 'Mn', 'H', 'Cu', 'Mg', 'Lu', 'Au', 'Ir', 'F', 
                        'Sn', 'Pt', 'He', 'O', 'Ar', 'Nb', 'Li', 'Rh', 'Zn', 'Ca', 'Be', 
                        'I', 'C', 'Os', 'Co', 'Na', 'Ge', 'Se', 'Y', 'Tl', 'Cr', 'Ta', 'Zr', 
                        'S', 'Ag', 'Mo', 'Ba', 'Cd', 'Dy', 'Ga', 'Xe', 'As', 'Si', 'Pb', 
                        'Rb', 'In', 'Fe', 'Bi', 'Pd', 'Th', 'Cs', 'Sc', 'K', 'Sb', 'W',
                        'Re', 'Cl', 'Hg', 'V', 'Te', 'Br']

        self.compositions = []
        self.bandgaps = []
        self.total_features = []
        self.states = []

        element_to_idx = {el: i for i, el in enumerate(element_list)}
        num_elements = len(element_list)

        for _, row in self.df.iterrows():
            formula = row[self.formula_col]
            bandgap = float(row[self.bandgap_col])
            state = int(row[self.state_col])

            comp = Composition(formula)
            comp_vec = torch.zeros(num_elements, dtype=torch.float32)

            x_total_feat = self.featurizer.featurize(comp)
            self.total_features.append(x_total_feat)

            for el, amt in comp.get_el_amt_dict().items():
                if el in element_to_idx:
                    idx = element_to_idx[el]
                    comp_vec[idx] = amt

            total = comp_vec.sum()
            comp_frac = comp_vec / total if total > 0 else comp_vec

            self.compositions.append(comp_frac)
            self.bandgaps.append([bandgap])
            self.states.append(state)

        self.total_features = torch.tensor(self.total_features, dtype=torch.float32)
        self.compositions = torch.stack(self.compositions)
        self.bandgaps = torch.tensor(self.bandgaps, dtype=torch.float32)
        self.states = torch.tensor(self.states, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x_comp": self.compositions[idx],
            "x_total_feats": self.total_features[idx],
            "y_bandgap": self.bandgaps[idx],
            "y_comp": self.compositions[idx],
            "state": self.states[idx],
        }


if __name__ == "__main__":
    csv_path = "./data/sim.csv"
    from torch.utils.data import DataLoader
    dataset = CompositionDataset(csv_path, "material formula", "Band_gap_GGA")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for batch in dataloader:
        print("x_comp:", batch["x_comp"].shape)
        print("x_total_feats:", batch["x_total_feats"].shape)
        print("y_bandgap:", batch["y_bandgap"].shape)
        print("y_comp:", batch["y_comp"].shape)

        print("x_comp first sample:", batch["x_comp"][0])
        print("x_total_feats first sample:", batch["x_total_feats"][0])
        print("y_bandgap first sample:", batch["y_bandgap"][0])
        print("y_comp first sample:", batch["y_comp"][0])
        break
