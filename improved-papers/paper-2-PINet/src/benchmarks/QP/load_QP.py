"""Loading functionality for simple QP benchmark."""

import os
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split


# Load Instance Dataset
class SimpleQPDataset(Dataset):
    """Dataset for simple QP benchmark."""

    def __init__(self, filepath):
        """Initialize dataset.

        Args:
            filepath (str): Path to the dataset file.
        """
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Optimal objectives and solutions for all problem instances
        self.objectives = data["objectives"]
        self.Ystar = data["Ystar"]

    def __len__(self) -> int:
        """Length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get item from dataset.

        Args:
            idx (int): Index of the item to retrieve.
        """
        return self.X[idx], self.objectives[idx]


def create_dataloaders(
    filepath: str,
    batch_size: int = 512,
    val_split: float = 0.0,
    test_split: float = 0.1,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Dataset loaders for training, validation and test.

    Args:
        filepath (str): Path to the dataset file.
        batch_size (int): Size of each batch.
        val_split (float): Proportion of data to use for validation.
        test_split (float): Proportion of data to use for testing.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tuple: DataLoaders for training, validation, and test datasets.
    """
    dataset = SimpleQPDataset(filepath)
    size = len(dataset)

    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def collate_fn(batch):
        X, obj = zip(*batch)
        return jnp.array(X), jnp.array(obj)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=val_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=test_size, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


class DC3Dataset(Dataset):
    """Dataset for importing DC3 problems."""

    def __init__(self, filepath: str, use_convex: bool):
        """Initialize dataset.

        Args:
            filepath (str): Path to the dataset file.
            use_convex (bool): Whether to use convex problems.
        """
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Problem solutions
        self.Ystar = data["Ystar"]

        # Compute objectives
        if use_convex:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ y

        else:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ jnp.sin(y)

        self.obj_fun = jax.vmap(obj_fun, in_axes=[0])
        self.objectives = self.obj_fun(self.Ystar[:, :, 0])

    def __len__(self) -> int:
        """Length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get item from dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing
                - the input features; and
                - the corresponding objective value.
        """
        return self.X[idx], self.objectives[idx]


class JaxDataLoader:
    """Dataloader for DC3 dataset implemented in JAX."""

    def __init__(
        self,
        filepath: str,
        use_convex: bool,
        batch_size: int,
        shuffle: bool = True,
        rng_key: Optional[jax.Array] = None,
    ):
        """Initialize JaxDataLoader.

        Args:
            filepath (str): Path to the dataset file.
            use_convex (bool): Whether to use convex problems.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the dataset.
            rng_key (Optional[jax.Array]): Random key for shuffling.
        """
        self.dataset = DC3Dataset(filepath, use_convex)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Batch indices for the current epoch
        self._perm = self._get_perm() if self.shuffle else jnp.arange(len(self.dataset))

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            int: Number of batches in the dataset.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over the dataset."""
        for start in range(0, len(self.dataset), self.batch_size):
            batch_idx = self._perm[start : start + self.batch_size]
            yield self.dataset[batch_idx]

        if self.shuffle:
            self._perm = self._get_perm()

    def _get_perm(self) -> jax.Array:
        self.rng_key, last_key = jax.random.split(self._rng_key)
        perm = jax.random.permutation(last_key, len(self.dataset))
        return perm


def dc3_dataloader(
    filepath: str,
    use_convex: bool,
    batch_size: int = 512,
    shuffle: bool = True,
):
    """Dataset loader for training, or validation, or test."""
    dataset = DC3Dataset(filepath, use_convex)

    def collate_fn(batch):
        X, obj = zip(*batch)
        return jnp.array(X), jnp.array(obj)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return loader


def non_DC3_dataset_setup(
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int,
    use_jax_loader: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """Setup function for datasets generated with our script.

    Args:
        use_convex (bool): Whether to use convex problems.
        problem_seed (int): Seed for random number generation.
        problem_var (int): Variance of the problem.
        problem_nineq (int): Number of inequality constraints.
        problem_neq (int): Number of equality constraints.
        problem_examples (int): Number of examples in the dataset.
        rng_key (jax.Array): Random key for JAX operations.
            Unused in this function but kept for consistency with other loaders.
        batch_size (int): Size of each batch.
        use_jax_loader (bool): Whether to use JAX DataLoader.
            Unused in this function but kept for consistency with other loaders.

    Returns:
        tuple: A tuple containing:
            - Q (jnp.ndarray): Quadratic coefficient matrix.
            - p (jnp.ndarray): Linear coefficient vector.
            - A (jnp.ndarray): Equality constraint matrix.
            - G (jnp.ndarray): Inequality constraint matrix.
            - h (jnp.ndarray): Inequality constraint vector.
            - X (jnp.ndarray): Input features.
            - train_loader (DataLoader): DataLoader for training data.
            - valid_loader (DataLoader): DataLoader for validation data.
            - test_loader (DataLoader): DataLoader for test data.
    """
    # Choose problem parameters
    if use_convex:
        filename = (
            f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_examples{problem_examples}.npz"
        )
    else:
        raise NotImplementedError()
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)

    QPDataset = SimpleQPDataset(dataset_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=batch_size, val_split=0.1, test_split=0.1
    )
    Q, p, A, G, h = QPDataset.const
    p = p[0, :, :]
    X = QPDataset.X

    return Q, p, A, G, h, X, train_loader, valid_loader, test_loader


def DC3_dataset_setup(
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int,
    use_jax_loader: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    JaxDataLoader,
    JaxDataLoader,
    JaxDataLoader,
]:
    """Setup function for datasets generated with the DC3 script.

    Args:
        use_convex (bool): Whether to use convex problems.
        problem_seed (int): Seed for random number generation.
        problem_var (int): Variance of the problem.
        problem_nineq (int): Number of inequality constraints.
        problem_neq (int): Number of equality constraints.
        problem_examples (int): Number of examples in the dataset.
        rng_key (jax.Array): Random key for JAX operations.
        batch_size (int): Size of each batch.
        use_jax_loader (bool): Whether to use JAX DataLoader.

    Returns:
        tuple: A tuple containing:
            - Q (jnp.ndarray): Quadratic coefficient matrix.
            - p (jnp.ndarray): Linear coefficient vector.
            - A (jnp.ndarray): Equality constraint matrix.
            - G (jnp.ndarray): Inequality constraint matrix.
            - h (jnp.ndarray): Inequality constraint vector.
            - X (jnp.ndarray): Input features.
            - train_loader (JaxDataLoader): DataLoader for training data.
            - valid_loader (JaxDataLoader): DataLoader for validation data.
            - test_loader (JaxDataLoader): DataLoader for test data.
    """
    # Choose the filename here
    if use_convex:
        filename = (
            f"dc3_random_simple_dataset_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_ex{problem_examples}"
        )
    else:
        filename = (
            f"dc3_random_nonconvex_dataset_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_ex{problem_examples}"
        )
    filename_train = filename + "train.npz"
    dataset_path_train = os.path.join(
        os.path.dirname(__file__), "datasets", filename_train
    )
    filename_valid = filename + "valid.npz"
    dataset_path_valid = os.path.join(
        os.path.dirname(__file__), "datasets", filename_valid
    )
    filename_test = filename + "test.npz"
    dataset_path_test = os.path.join(
        os.path.dirname(__file__), "datasets", filename_test
    )
    if not use_jax_loader:
        train_loader = dc3_dataloader(
            dataset_path_train, use_convex, batch_size=batch_size
        )
        valid_loader = dc3_dataloader(
            dataset_path_valid, use_convex, batch_size=1024, shuffle=False
        )
        test_loader = dc3_dataloader(
            dataset_path_test, use_convex, batch_size=1024, shuffle=False
        )
    else:
        loader_keys = jax.random.split(rng_key, 3)
        train_loader = JaxDataLoader(
            dataset_path_train,
            use_convex,
            batch_size=batch_size,
            rng_key=loader_keys[0],
        )
        valid_loader = JaxDataLoader(
            dataset_path_valid,
            use_convex,
            batch_size=1024,
            shuffle=False,
            rng_key=loader_keys[1],
        )
        test_loader = JaxDataLoader(
            dataset_path_test,
            use_convex,
            batch_size=1024,
            shuffle=False,
            rng_key=loader_keys[2],
        )
    Q, p, A, G, h = train_loader.dataset.const
    p = p[0, :, :]
    X = train_loader.dataset.X

    return Q, p, A, G, h, X, train_loader, valid_loader, test_loader


def load_data(
    use_DC3_dataset: bool,
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int = 2048,
    use_jax_loader: bool = True,
    penalty: float = 0.0,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
]:
    """Load problem data.

    Args:
        use_DC3_dataset (bool): Whether to use the DC3 dataset.
        use_convex (bool): Whether to use convex problems.
        problem_seed (int): Seed for random number generation.
        problem_var (int): Variance of the problem.
        problem_nineq (int): Number of inequality constraints.
        problem_neq (int): Number of equality constraints.
        problem_examples (int): Number of examples in the dataset.
        rng_key (jax.Array): Random key for JAX operations.
        batch_size (int): Size of each batch.
        use_jax_loader (bool): Whether to use JAX DataLoader.
        penalty (float): Penalty term for the loss function.

    Returns:
        tuple: A tuple containing:
            - A (jnp.ndarray): Equality constraint matrix.
            - G (jnp.ndarray): Inequality constraint matrix.
            - h (jnp.ndarray): Inequality constraint vector.
            - X (jnp.ndarray): Input features.
            - batched_objective (Callable): Batched objective function.
            - train_loader (DataLoader | JaxDataLoader): DataLoader for training data.
            - valid_loader (DataLoader | JaxDataLoader): DataLoader for validation data.
            - test_loader (DataLoader | JaxDataLoader): DataLoader for test data.
            - batched_loss (Callable): Batched loss function.

    """
    if not use_DC3_dataset:
        setup = non_DC3_dataset_setup
    else:
        setup = DC3_dataset_setup

    Q, p, A, G, h, X, train_loader, valid_loader, test_loader = setup(
        use_convex=use_convex,
        problem_seed=problem_seed,
        problem_var=problem_var,
        problem_nineq=problem_nineq,
        problem_neq=problem_neq,
        problem_examples=problem_examples,
        rng_key=rng_key,
        batch_size=batch_size,
        use_jax_loader=use_jax_loader,
    )

    # Define loss/objective function
    # Predictions is of shape (batch_size, Y_DIM) and Q is of shape (Y_DIM, Y_DIM)
    def quadratic_form(prediction):
        """Evaluate the quadratic objective."""
        return 0.5 * prediction.T @ Q @ prediction + p.T @ prediction

    def quadratic_form_sine(prediction):
        """Evaluate the quadratic objective plus sine."""
        return 0.5 * prediction.T @ Q @ prediction + p.T @ jnp.sin(prediction)

    if use_convex:
        objective_function = quadratic_form
    else:
        objective_function = quadratic_form_sine

    # Vectorize the quadratic form computation over the batch dimension
    batched_objective = jax.vmap(objective_function, in_axes=[0])

    def penalty_form(predictions, X):
        eq_cv = jnp.max(
            jnp.abs(
                A[0].reshape(1, A.shape[1], A.shape[2])
                @ predictions.reshape(X.shape[0], A.shape[2], 1)
                - X
            ),
            axis=1,
        )
        ineq_cv = jnp.max(
            jnp.maximum(
                G[0].reshape(1, G.shape[1], G.shape[2])
                @ predictions.reshape(X.shape[0], G.shape[2], 1)
                - h,
                0,
            ),
            axis=1,
        )

        return eq_cv + ineq_cv

    def batched_loss(predictions, X):
        if penalty > 0:
            return batched_objective(predictions) + penalty * penalty_form(
                predictions, X
            )
        else:
            return batched_objective(predictions)

    return (
        A,
        G,
        h,
        X,
        batched_objective,
        train_loader,
        valid_loader,
        test_loader,
        batched_loss,
    )
