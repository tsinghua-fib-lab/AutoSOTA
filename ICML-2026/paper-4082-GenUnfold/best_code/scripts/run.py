import os
import numpy as np
from tqdm import tqdm

from Gen_Unfold.src.data_processing.prtein_feature_extractor import load_structure, residue_contact_map
from Gen_Unfold.src import (train_pipeline, curve_prediction)

def build_features(pdb_dir, save_dir):
    filenames = os.listdir(pdb_dir)
    for filename in tqdm(filenames):
            try:
                path = os.path.join(pdb_dir, filename)
                structure = load_structure(path)
                map = residue_contact_map(structure)
                output_path = os.path.join(save_dir, filename[:4], 'res_map')
                np.save(output_path, map)
            except Exception as exc:
                print(filename, exc)

if __name__ == '__main__':
    # Example usage of feature building
    pdb_directory = r'../data/pdb_files'
    features_save_directory = r'../data/features'

    if os.path.exists(r'../data/pdb_files/pdb.cif'):
        raise "Please put your pdb files in the folder '../data/pdb_files' before running the script."

    build_features(pdb_directory, features_save_directory) # After building features, you can comment this line to avoid rebuilding.
    train_pipeline("../config/config.yaml", seed=42)







