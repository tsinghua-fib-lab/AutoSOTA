# Read this file to pandas

# /dfs/scratch1/sttruong/scaling_data/models_math_evals_2024-10-10.parquet
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

compute_sum_score_per_group = False


df = pd.read_parquet("/dfs/scratch1/sttruong/scaling_data/models_math_evals_2024-10-10.parquet")

if compute_sum_score_per_group:
    # Compute the sum of "Score" for each (Model Nickname, prompt_idx) pair
    grouped_sums = df.groupby(["Model Nickname", "prompt_idx"], sort=False)["Score"].sum()

    # Count how many groups have a non-zero sum of Score
    nonzero_count = (grouped_sums != 0).sum()

    print(f"Number of (Model Nickname, prompt_idx) pairs with non-zero Score sum: {nonzero_count}")
    print(f"Tota number of (Model Nickname, prompt_idx) pairs: {grouped_sums.shape[0]}")

    matrix = grouped_sums.unstack()

    plt.imshow(matrix, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Grouped Sums (0=black, 1=white)')
    plt.xlabel('Column Label')
    plt.ylabel('Row Label')
    plt.savefig('grouped_sums.png')

# plot the histogram of max response_id for each (Model Nickname, prompt_idx) pair
max_response_ids = df.groupby(["Model Nickname", "prompt_idx"], sort=False)["response_idx"].max()
max_response_ids.hist(bins=100)
plt.title('Histogram of Max Response ID')
plt.xlabel('Max Response ID')
plt.ylabel('Frequency')
plt.savefig('histogram_max_response_id.png')
print(max_response_ids)