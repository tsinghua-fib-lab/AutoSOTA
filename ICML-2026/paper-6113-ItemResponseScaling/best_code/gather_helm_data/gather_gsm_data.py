import pickle

benchmark = "classic"
scenario = "math"

with open(f"../data/gather_helm_data/results_with_z.pkl", "rb") as f:
    results = pickle.load(f)
results = results.loc[:, results.columns.get_level_values("benchmark") == benchmark]
results = results.loc[:, results.columns.get_level_values("scenario") == scenario]
results = results[~results.isna().all(axis=1)]

print(results.shape)
print(f"missing percentage: {results.isna().values.sum() / (results.shape[0] * results.shape[1])}")

with open("gsm_results.pkl", "wb") as f:
    pickle.dump(results, f)