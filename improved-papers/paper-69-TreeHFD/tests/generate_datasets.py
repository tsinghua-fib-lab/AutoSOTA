"""Generate and store datasets for tests."""


import numpy as np

from treehfd.validation import sample_data

if __name__ == "__main__":

    seed_list = [11, 21, 41, 51, 11, 61]
    nsample_list = [10, 10, 100, 100, 100, 1000]

    for k, seed in enumerate(seed_list):
        nsample = nsample_list[k]
        np.random.default_rng(seed)
        X, y = sample_data(nsample=nsample)
        np.savetxt(f"datasets/dataset_X_n{nsample}_seed{seed}.csv", X,
                   delimiter=",")
        np.savetxt(f"datasets/dataset_y_n{nsample}_seed{seed}.csv", y,
                   delimiter=",")
        if seed in (51, 11) and nsample == 100:
            X_new, _ = sample_data(nsample=3)
            np.savetxt(f"datasets/dataset_Xnew_n3_seed{seed}.csv",
                       X_new,  delimiter=",")
        if seed == 61 and nsample == 1000:
            X_new, _ = sample_data(nsample=1000)
            np.savetxt(f"datasets/dataset_Xnew_n1000_seed{seed}.csv",
                       X_new,  delimiter=",")
