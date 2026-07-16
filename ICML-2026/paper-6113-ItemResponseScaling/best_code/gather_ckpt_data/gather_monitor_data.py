import pickle
import numpy as np

if __name__ == "__main__":
    repo_id = "EleutherAI/pythia-6.9b"
    model_name = repo_id.split("/")[1]
    # n_cols = 50
    
    with open(f"../data/gather_ckpt_data/results_{model_name}.pkl", "rb") as f:
        results = pickle.load(f)

    # Keep only columns with non-NaN z-values
    keep_cols = ~results.columns.get_level_values("z").isna()
    results = results.loc[:, keep_cols]
    
    print(results.columns.get_level_values("scenario").nunique())

    # Select legalbench scenario
    legalbench_cols = results.columns[results.columns.get_level_values("scenario") == "legalbench"]
    legalbench_data = results.loc[:, legalbench_cols]

    # Drop columns with any NaNs
    legalbench_data = legalbench_data.dropna(axis=1)

    # # Drop columns that contain only 0s or only 1s
    # mask = ~((legalbench_data.nunique(dropna=False) == 1) & 
    #          ((legalbench_data == 0).all() | (legalbench_data == 1).all()))
    # legalbench_data = legalbench_data.loc[:, mask]

    # # Randomly select 50 columns
    # np.random.seed(42)
    # selected_cols = np.random.permutation(legalbench_data.columns)[:50]
    # results_subset = legalbench_data.loc[:, selected_cols]

    # # Flatten multi-index to just 'z'
    # results_subset.columns = results_subset.columns.get_level_values("z")
    # results_subset.index.name = None

    # # Save without row index and column headers
    # results_subset.to_csv(f"legalbench_subset_{model_name}_{n_cols}.csv", index=False, header=False)

    with open(f"../data/gather_ckpt_data/results_{model_name}_legalbench.pkl", "wb") as f:
        pickle.dump(legalbench_data, f)