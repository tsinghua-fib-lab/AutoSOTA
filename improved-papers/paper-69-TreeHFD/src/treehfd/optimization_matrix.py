"""Function to build the least square optimization problem to fit TreeHFD."""


import numpy as np
from scipy import sparse


def build_constr_mat(y_tree: np.ndarray, interaction_list: list,
                     X_bin: np.ndarray, main_variables: np.ndarray,
                     partition_index: np.ndarray) -> tuple:
    """Build the matrix and target vector for the least square optimization.

    Parameters
    ----------
    y_tree : np.ndarray
        1d array with the tree output for the input training data.
    interaction_list : list
        List of variable index pairs for interactions.
    X_bin : np.ndarray
        Array with the cell indices of all Cartesian partitions
        for each training point.
    main_variables : np.ndarray
        list of variable indices for main effects.
    partition_index : np.ndarray
        cell indices of all partitions.

    Returns
    -------
    tuple
        constr_mat : scipy.sparse.csr_matrix
            scipy sparse matrix for the least square problem.
        target : np.ndarray
            array (1d) for the target vector of the least square problem.
    """
    # Handle empty partitions.
    if main_variables.size == 0:
        return(sparse.csr_matrix(np.empty((0, 0))), np.empty(0))

    # Retrieve parameters.
    num_func_main = len(main_variables)
    partition_size = partition_index[-1]
    nsample = y_tree.shape[0]

    # Build constraint matrix for hierarchical orthogonality.
    constr_ortho_list: list[np.ndarray] = []
    for i, variable_interaction in enumerate(interaction_list):
        for j in variable_interaction:
            idx_j = list(main_variables).index(j)
            cells = set(X_bin[:, idx_j])
            for k in cells:
                x = np.zeros(partition_size)
                values, counts = np.unique(
                    X_bin[X_bin[:, idx_j] == k, num_func_main + i],
                    return_counts=True)
                prob = np.sum(counts) / nsample
                x[values] = counts / np.sqrt(prob)
                constr_ortho_list.append(x)
    constr_ortho = np.array(constr_ortho_list).reshape(len(constr_ortho_list),
                                                       partition_size)

    # Build constraint matrix for zero mean of functional components.
    constr_mean = np.zeros((X_bin.shape[1], partition_size))
    for k in range(X_bin.shape[1]):
        values, counts = np.unique(X_bin[:, k], return_counts=True)
        constr_mean[k, values] = counts

    # Build constraint matrix for residual variance minimization.
    values, index_unique, counts = np.unique(X_bin, return_index=True,
                                             return_counts=True, axis=0)
    num_cells = counts.shape[0]
    constr_resid = np.zeros((num_cells, partition_size))
    for k in range(num_cells):
        constr_resid[k, values[k, :]] = np.sqrt(nsample * counts[k])

    # Concatenate all constraints in sparse matrix.
    constr_mat = np.concatenate((constr_ortho, constr_mean, constr_resid),
                                axis=0)
    constr_mat = sparse.csr_matrix(constr_mat)

    # Compute target.
    num_zero = constr_ortho.shape[0] + constr_mean.shape[0]
    target = np.zeros(num_zero + num_cells)
    target[num_zero:] = y_tree[index_unique] * np.sqrt(nsample*counts)

    return(constr_mat, target)
