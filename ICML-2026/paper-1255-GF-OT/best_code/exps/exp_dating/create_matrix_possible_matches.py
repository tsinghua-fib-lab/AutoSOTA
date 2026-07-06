import os
import numpy as np
import pandas as pd
from match_matrix import get_match_matrix

df = pd.read_csv("data/processed_dating_data.csv", index_col=0)

binary_matrix = np.zeros((len(df), len(df)), dtype=int)
match_matrix = get_match_matrix()


indices = df.index.to_list()

for idx_i, i in enumerate(indices):
    print(f"Processing row {idx_i + 1} of {len(indices)}")
    row_i = df.loc[i]
    key_i = f"({row_i['gender']}, {row_i['sexual_orientation']})"
    for j in indices[idx_i + 1 :]:
        row_j = df.loc[j]
        key_j = f"({row_j['gender']}, {row_j['sexual_orientation']})"
        value = match_matrix.loc[key_i, key_j]
        binary_matrix[i, j] = value
        binary_matrix[j, i] = value

# save it as a numpy binary file for faster loading
os.makedirs("exps/exp_dating/data/", exist_ok=True)
np.save("exps/exp_dating/data/possible_individual_matches.npy", binary_matrix)
