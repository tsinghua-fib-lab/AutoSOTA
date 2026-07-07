import os
import pandas as pd
import pickle
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
from utils.docking import calc_affinity_crossdocked


class Evaluator:
    def __init__(self, output_file="1_result.csv", pose_pkl="poses.pkl", clear_old=True):
        self.output_file = output_file
        self.pose_pkl = pose_pkl

        self.columns = [
            "SMILES", "docking_score", "qed", "sa", "logp", "mw",
            "parent_smiles", "user_prompt", "rationale"
        ]


        if not clear_old and os.path.exists(self.pose_pkl):
            with open(self.pose_pkl, 'rb') as f:
                self.pose_dict = pickle.load(f)
            print(f">> Loaded {len(self.pose_dict)} poses from {self.pose_pkl}")
        else:
            self.pose_dict = {}


        if clear_old or not os.path.exists(self.output_file):
            print(f">> Initializing new CSV at {self.output_file}")
            df_header = pd.DataFrame(columns=self.columns)
            df_header.to_csv(self.output_file, index=False, mode='w')
        else:
            print(f">> Continuing with existing CSV at {self.output_file}")

    def validate_and_standardize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        std_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if "." in std_smiles:
            std_smiles = max(std_smiles.split('.'), key=len)
            mol = Chem.MolFromSmiles(std_smiles)
            if mol is None: return None, None
            std_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return std_smiles, mol

    def save_pose_checkpoint(self):

        with open(self.pose_pkl, 'wb') as f:
            pickle.dump(self.pose_dict, f)

    def run(self, new_smiles, m_seed, rationale, user_prompt, protein_name, history_set):

        std_smiles, mol = self.validate_and_standardize(new_smiles)
        if not std_smiles:
            return None

        if std_smiles in history_set:
            return None

        qed_val = QED.qed(mol)
        sa_norm = max(0.0, min(1.0, (10.0 - sascorer.calculateScore(mol)) / 9.0))
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        print(f">> [Docking] Running Smina for {std_smiles}...")
        score, pose_str = calc_affinity_crossdocked(std_smiles, protein_name, dir_out="./Smina")
        if score >= 500:
            return None

        if pose_str:
            self.pose_dict[std_smiles] = pose_str
            self.save_pose_checkpoint()

        new_entry = {
            "SMILES": std_smiles,
            "docking_score": score,
            "qed": round(qed_val, 4),
            "sa": round(sa_norm, 4),
            "logp": round(logp, 2),
            "mw": round(mw, 2),
            "parent_smiles": m_seed.smiles,
            "user_prompt": user_prompt.replace("\n", " ").strip(),
            "rationale": rationale.replace("\n", " ").strip()
        }

        df_row = pd.DataFrame([new_entry])
        df_row.to_csv(
            self.output_file,
            mode='a',
            index=False,
            header=False,
            encoding='utf-8',
            quoting=1
        )

        history_set.add(std_smiles)
        return score, std_smiles, qed_val, sa_norm