"""
Execution example :
Compares Standard K-means, C-LO_K_means, D-LO_K_means, and Min_D_LO_K_means by loss and steps.
"""

import numpy as np
from LO_K_means import LO_K_means


def make_unique(X: np.ndarray):
    unique_X, weight = np.unique(X, axis=0, return_counts=True)
    return unique_X, weight


def main():
    # Read dataset
    with open("../data/Iris.txt") as f:
        N, D = map(int, f.readline().split())
        data = np.loadtxt(f, delimiter=None)
        data = data.reshape(N, D)

    X_unique, weight = make_unique(data)

    K = 50

    # K-means setting
    kmeans = LO_K_means(
        X=X_unique,
        weight=weight,
        K=K,
        init_="kmeans++",
        breg="squared",
        random_state=None,
        eps=1e-10,
    )

    # Standard K-means
    assignment, centers, loss = kmeans.K_means()
    print(f"Standard K_means loss: {loss:.4f} (steps: {kmeans.step_num})")

    # C_LO_K_means (Function 1)
    assignment_c, centers_c, loss_c = kmeans.C_LO_K_means()
    print(f"C_LO_K_means loss: {loss_c:.4f}, (steps: {kmeans.step_num})")

    # D_LO_K_means (Function 2)
    assignment_d, centers_d, loss_d = kmeans.D_LO_K_means()
    print(f"D_LO_K_means loss: {loss_d:.4f}, (steps: {kmeans.step_num})")

    # Min_D_LO_K_means (Function 3)
    assignment_min_d, centers_min_d, loss_min_d = kmeans.Min_D_LO_K_means()
    print(f"Min_D_LO_K_means loss: {loss_min_d:.4f}, (steps: {kmeans.step_num})")


if __name__ == "__main__":
    main()
