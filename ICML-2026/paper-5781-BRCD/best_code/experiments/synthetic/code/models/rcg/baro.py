import time
import argparse

from sklearn.preprocessing import RobustScaler

from utils import base_utils as bu

current_time = lambda: time.perf_counter() * 1e3


def rank_variables(n_df, a_df):
    start = current_time()
    ranks = []
    for col in n_df.columns:
        a = n_df[col].to_numpy()
        b = a_df[col].to_numpy()
        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    sorted_nodes = [x[0] for x in ranks]
    end = current_time() - start
    return {'time': end, 'root_cause': sorted_nodes, 'tests': n_df.shape[1] - 1}

def run(n_df, a_df):
    return rank_variables(n_df, a_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run mutual information on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    n_df, a_df = bu.load_datasets(path)
    result = rank_variables(n_df, a_df)
    print(result)
