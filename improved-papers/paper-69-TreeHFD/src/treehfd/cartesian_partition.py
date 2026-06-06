"""Class to define the Cartesian tree partitions of a tree ensemble."""


import numpy as np


class CartesianTreePartition:
    """Cartesian tree partition.

    This class defines the Cartesian tree partitions of a given tree from
    the splitting variables and values.

    Attributes
    ----------
    main_variables : np.ndarray
        List of variable indices for main effects.
    partition_index : np.ndarray
        Indices of all main effect partitions.
    split_list : list
        List of splitting values for all main effects.
    cell_list : list
        List of array of cell indices for each component.
    counts_list : list
        List of sizes of each cell of interaction partitions.
    """

    def __init__(self, main_variables: np.ndarray) -> None:
        """Initialize CartesianTreePartition."""
        self.main_variables: np.ndarray = main_variables
        self.partition_index: np.ndarray = np.empty(0)
        self.split_list: list[np.ndarray] = []
        self.cell_list: list[np.ndarray] = []
        self.counts_list: list[np.ndarray] = []

    def compute_cartesian_partition(self, X: np.ndarray, tree_structure:
                                    tuple[np.ndarray, np.ndarray, np.ndarray],
                                    interaction_list: list[list[int]],
                                    ) -> np.ndarray:
        """Compute Cartesian tree partitions from tree structure and data.

        Parameters
        ----------
        X : np.ndarray
            The input data used to train the xgboost model.
        tree_structure : tuple
            Tuple containing the splitting variables, children node indices,
            and splitting values of the tree.
        interaction_list : list
            List of variable index pairs, with a pair for each interaction.

        Returns
        -------
        X_bin : np.ndarray
            Array with the cell indices of all Cartesian partitions for each
            training point.
        """
        # Handle empty partitions.
        if self.main_variables.size == 0:
            return np.empty((X.shape[0], 0), dtype=int)

        # Compute Cartesian partition for main effects.
        X_bin_main = self.compute_partition_main(X, tree_structure[0],
                                                 tree_structure[2])

        # Compute Cartesian partition for second-order interactions.
        X_bin_order2 = self.compute_partition_order2(X_bin_main,
                                                     interaction_list)

        # Merge and return cell indices of main effects and interactions.
        return np.concatenate((X_bin_main, X_bin_order2), axis=1)

    def compute_partition_main(self, X: np.ndarray, variables: np.ndarray,
                               split_values: np.ndarray) -> np.ndarray:
        """Compute Cartesian tree partitions of main effects.

        Parameters
        ----------
        X : np.ndarray
            The input data used to train the xgboost model.
        variables: np.ndarray
            Array with the splitting variables of the tree.
        split_values: np.ndarray
            Array with the splitting values of the tree.

        Returns
        -------
        X_bin_main : np.ndarray
            Array with the cell indices of main-effect partitions for each
            training point.
        """
        X_bin_main = np.zeros((X.shape[0], len(self.main_variables)),
                              dtype=int)
        num_splits: list[int] = []
        for idx, j in enumerate(self.main_variables):
            split_j = np.unique(split_values[variables == j])
            Xj_bin = np.digitize(X[:, j], bins=split_j, right=False)
            # remove empty bins
            bins_empty = [k for k in range(len(split_j), -1, -1)
                          if k not in set(Xj_bin)]
            for k in bins_empty:
                if k == len(split_j):
                    split_j = np.delete(split_j, k - 1)
                if k == 0:
                    split_j = np.delete(split_j, k)
                if 0 < k < len(split_j):
                    split_j[k - 1] = np.mean(split_j[(k - 1):(k + 1)])
                    split_j = np.delete(split_j, k)
            Xj_bin = np.digitize(X[:, j], bins=split_j, right=False)
            self.split_list.append(split_j)
            X_bin_main[:, idx] = Xj_bin
            num_splits.append(len(split_j) + 1)
        self.partition_index = np.array(np.insert(np.cumsum(num_splits), 0, 0),
                                   dtype=int)

        return X_bin_main + self.partition_index[:-1]

    def compute_partition_order2(self, X_bin_main: np.ndarray, interaction_list:
                                 list[list[int]]) -> np.ndarray:
        """Compute Cartesian tree partitions of second-order interactions.

        Parameters
        ----------
        X_bin_main : np.ndarray
            Array with the cell indices of main-effect partitions for each
            training point.
        interaction_list : list
            List of variable index pairs, with a pair for each interaction.

        Returns
        -------
        X_bin_order2 : np.ndarray
            Array with the cell indices of second-order partitions for each
            training point.
        """
        X_bin_order2 = np.zeros((X_bin_main.shape[0], len(interaction_list)),
                                dtype=int)
        index_order2_list: list[int] = []
        cell_index = self.partition_index[-1]
        for k, variable_interaction in enumerate(interaction_list):
            variables_index = [list(self.main_variables).index(j) for j in
                               variable_interaction]
            cell_array, index_unique, counts = np.unique(
                X_bin_main[:, variables_index], return_inverse=True,
                return_counts=True, axis=0)
            self.cell_list.append(cell_array)
            self.counts_list.append(counts)
            for i in range(cell_array.shape[0]):
                X_bin_order2[index_unique == i, k] = cell_index
                cell_index += 1
            index_order2_list.append(cell_index)
        partition_index_order2 = np.array(index_order2_list, dtype=int)

        # Merge main and interaction partitions.
        self.partition_index = np.concatenate((self.partition_index,
                                               partition_index_order2))

        return X_bin_order2

    def predict_partition(self, X_new: np.ndarray, interaction_list:
                          list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Predict cells of the Cartesian tree partitions for new input data.

        Parameters
        ----------
        X_new : np.ndarray
            New input data where TreeHFD predictions are computed.
        interaction_list : list
            List of variable index pairs, with a pair for each interaction.

        Returns
        -------
        tuple
            X_bin_main : np.ndarray
                Array with the cell indices of main-effect partitions for
                the new input data.
            X_bin_order2 : np.ndarray
                Array with the cell indices of interaction partitions for
                the new input data.
        """
        # Handle empty partitions.
        if self.main_variables.size == 0:
            return(np.empty((X_new.shape[0], 0), dtype=int),
                   np.empty((X_new.shape[0], 0), dtype=int))

        # Main effects.
        num_func_main = len(self.main_variables)
        X_bin_main = np.zeros((X_new.shape[0], num_func_main), dtype=int)
        for idx, j in enumerate(self.main_variables):
            X_bin_main[:, idx] = np.digitize(X_new[:, j],
                                             bins=self.split_list[idx],
                                             right=False)
        X_bin_main = X_bin_main + self.partition_index[:num_func_main]

        # Second-order interactions.
        X_bin_order2 = np.zeros((X_new.shape[0], len(interaction_list)),
                                dtype=int)
        for k, variable_interaction in enumerate(interaction_list):
            variables_index = [list(self.main_variables).index(j) for j in
                               variable_interaction]
            cell_array, index_unique = np.unique(
                X_bin_main[:, variables_index], return_inverse=True, axis=0)
            cell_array_train = self.cell_list[k]
            idx_init = int(self.partition_index[num_func_main + k])
            for i in range(cell_array.shape[0]):
                cell = cell_array[i, :]
                cell_index_train = np.where(np.all(cell_array_train == cell,
                                                   axis=1))[0]
                # Merge empty cells.
                if len(cell_index_train) == 0:
                    cell_distance = np.sum(np.abs(cell_array_train - cell),
                                           axis=1)
                    index_min = np.where(cell_distance
                                         == np.min(cell_distance))[0]
                    counts = self.counts_list[k][index_min]
                    index_rand = np.random.default_rng().choice(
                        np.where(counts == np.max(counts))[0], size=1)[0]
                    cell_index_train = index_min[index_rand]
                else:
                    cell_index_train = cell_index_train[0]
                cell_index = idx_init + cell_index_train
                X_bin_order2[index_unique == i, k] = cell_index

        return(X_bin_main, X_bin_order2)
