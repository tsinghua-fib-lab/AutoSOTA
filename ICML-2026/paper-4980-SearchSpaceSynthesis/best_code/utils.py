from pathlib import Path
import re
import warnings
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import accumulate

from pathlib import Path

from grakel.utils import graph_from_networkx
import networkx as nx

import matplotlib.pyplot as plt


from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import RandomLimitedDepthFirstInitialization

from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (edgelist_algebra, hierarchy_algebra,
                                                                        pretty_term_algebra, pytorch_function_algebra)

from bayesian_optimization.utils import to_grakel_graph, to_indexed_nx_digraph
from bayesian_optimization.initial_sampling import lazy_dpp_sample_optimized
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.examples.damg_nas.damg_targets import target_to_name


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "trapezoid_dataset.pt"

DEFAULT_PRESAMPLE_SIZE = 50

PRESAMPLE_DIR = Path(__file__).resolve().parent / "presamples"
DEFAULT_PRESAMPLE_PATH = PRESAMPLE_DIR / f"presample_{DEFAULT_PRESAMPLE_SIZE}.pt"
MAX_TREE_DEPTH = 1000

def get_num_parameters(model):
    """
    Calculate the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze
        
    Returns:
        int: Total number of parameters in the model
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> num_params = get_num_parameters(model)
        >>> print(f"Model has {num_params} parameters")
    """
    return sum(p.numel() for p in model.parameters())

def generate_data(true_model, n_samples, xmin=-10, xmax=10, eps=0.0):
    """
    Generate synthetic data by evaluating a true model on a grid of points and adding Gaussian noise.
    
    Note: 
        model evaluation is not batched!

    Args:
        true_model (torch.nn.Module): The true model to evaluate
        n_samples (int): Number of samples to generate
        xmin (float, optional): Minimum x value. Defaults to -10.
        xmax (float, optional): Maximum x value. Defaults to 10.
        eps (float, optional): Variance of Gaussian noise to add. Defaults to 0.0.
        
    Returns:
        tuple: A tuple containing (x, y) where x is the input tensor and y is the output tensor with optional noise added
        
    Example:
        >>> x, y = generate_data(trapezoid_model, 100, eps=0.1)
        >>> print(f"Generated {len(x)} samples")
    """
    x = torch.linspace(xmin, xmax, n_samples).view(-1,1)
    y = true_model(x).detach().view(-1)
    
    if eps > 0:
        noise = torch.normal(mean=0, std=torch.sqrt(torch.tensor(eps)), size=y.shape)
        y = y + noise
    
    return x,y

def fit_model(model, x, y, n_epochs=2_000, verbose=True, name="model"):
    """
    Train a PyTorch model on given data using Adam optimizer and MSE loss. 
     
    Note:
        Technically, this function implements Gradient Descent and not Stochastic Gradient Descent. There is no proper batching performed. 
        
    Args:
        model (torch.nn.Module): The model to train
        x (torch.Tensor): Input tensor
        y (torch.Tensor): Target tensor
        n_epochs (int, optional): Number of training epochs. Defaults to 2000.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        name (str, optional): Name of the model for progress bar display. Defaults to "model".
        
    Returns:
        torch.nn.Module: The trained model
        
    Example:
        >>> trained_model = fit_model(model, x_train, y_train, n_epochs=1000)
        >>> print("Model training completed")
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    
    pbar = tqdm.tqdm(range(n_epochs), total=n_epochs, desc=f"Training {name}", disable=not verbose)
    
    for _ in pbar:
        optimizer.zero_grad()
        pred = model(x).ravel()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return model


def create_and_save_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    true_model=None,
    n_samples=1_000,
    train_xmin=-10,
    train_xmax=10,
    test_xmin=-15,
    test_xmax=15,
    eps=1e-4,
):
    """
    Generate train/test data and persist it to disk.

    Returns:
        tuple: (x, y, x_test, y_test, dataset_path)
    """
    if true_model is None:
        true_model = TrapezoidNetPure()

    x, y = generate_data(true_model, xmin=train_xmin, xmax=train_xmax, n_samples=n_samples, eps=eps)
    x_test, y_test = generate_data(true_model, xmin=test_xmin, xmax=test_xmax, n_samples=n_samples, eps=eps)

    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "x": x,
            "y": y,
            "x_test": x_test,
            "y_test": y_test,
        },
        dataset_path,
    )

    return x, y, x_test, y_test, dataset_path


def load_dataset(dataset_path=DEFAULT_DATASET_PATH, map_location="cpu"):
    """
    Load a previously saved train/test dataset.

    Returns:
        tuple: (x, y, x_test, y_test)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    payload = torch.load(dataset_path, map_location=map_location)
    required_keys = ("x", "y", "x_test", "y_test")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Dataset file is missing keys: {missing_keys}")

    return payload["x"], payload["y"], payload["x_test"], payload["y_test"]

class TrapezoidNetPure(nn.Module):
        def __init__(self, random_weights=False, sharpness=None):
            super().__init__()

            self.split = nn.Linear(1, 1, bias=True)
            self.left = nn.Linear(1, 1, bias=True)
            self.right = nn.Linear(1, 1, bias=True)
            self.sharpness = sharpness

            if not random_weights:
                with torch.no_grad():
                    # For left branch (x <= 0): we want output = x + 10
                    # So left(x) = x + 10 => weight = 1, bias = 10
                    self.left.weight.data.fill_(1.0)
                    self.left.bias.data.fill_(10.0)

                    # For right branch (x > 0): we want output = 10 - x
                    # So right(x) = 10 - x => weight = -1, bias = 10
                    self.right.weight.data.fill_(-1.0)
                    self.right.bias.data.fill_(10.0)

                self.split.weight.data.fill_(1.0)
                self.split.bias.data.fill_(0.0)

        def forward(self, x):
            if not self.sharpness:
                gate = (self.split(x) <= 0).float()
            else:
                gate = torch.sigmoid(-self.sharpness * self.split(x))

            left_out = self.left(x) * gate
            right_out = self.right(x) * (1 - gate)

            return left_out + right_out

def as_DAMG(t: Tree, verbose=False):
    if verbose:
        edgelist, posA = t.interpret(edgelist_algebra(verbose))
    else:
        edgelist = t.interpret(edgelist_algebra(verbose))

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: re.sub(r"[)][(][-]*[0-9]*[.][0-9]*[,]\s[-]*[0-9]*[.][0-9]*[)]", ")", n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    if verbose:
        return gk_graph, G, posA, relabel

    return gk_graph

# To visualize the effect of the Hierarchy
def plot_tree(tree: Tree):
    gk_graph, G, pos_A, relabel = as_DAMG(tree, verbose=True)

    connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.3] * 4)]

    fig = plt.figure(figsize=(25, 25))

    #pos_G = nx.bfs_layout(G, "input")
    node_size = 3000
    nx.draw_networkx_nodes(G, pos_A, node_size=node_size, node_color='lightblue', alpha=0.5, margins=0.05)
    nx.draw_networkx_labels(G, pos_A, labels=relabel, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(G, pos_A, edge_color="black", connectionstyle=connectionstyle, node_size=node_size,
                           width=2)
    plt.figtext(0.01, 0.02, tree.interpret(pretty_term_algebra()), fontsize=14)

    id = tree.__hash__()

    path = Path(__file__).resolve().parent / "results"
    out_path = path / f"damg_{id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    T, label = to_indexed_nx_digraph(tree, 0)

    pos_T = nx.bfs_layout(T, 0)

    fig = plt.figure(figsize=(25, 25))

    node_size = 3000
    nx.draw_networkx_nodes(T, pos_T, node_size=node_size, node_color='lightblue', alpha=0.5, margins=0.05)
    nx.draw_networkx_labels(T, pos_T, labels=label, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(T, pos_T, edge_color="black", connectionstyle=connectionstyle, node_size=node_size,
                           width=2)
    plt.figtext(0.01, 0.02, tree.interpret(pretty_term_algebra()), fontsize=14)

    id = tree.__hash__()

    out_path = path / f"tree_{id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_pre_samples(search_space, target, size: int, path=None):
    """Generate pre-samples (X values only, no objective function evaluations).

    Pre-samples are a fixed, reproducible set of Tree structures used across
    different kernel and BO experiments for fair comparison.

    Args:
        search_space: The COSY SolutionSpace to sample from
        target: The target type for the search space
        size: Number of samples to generate

    Returns:
        tuple: (pre_sample_x, metadata) where metadata contains target_name, size, file_name, path
    """
    init = RandomLimitedDepthFirstInitialization(search_space, target, MAX_TREE_DEPTH)
    sample_space = init.initialize_population(size * 100)
    pre_sample_x = lazy_dpp_sample_optimized(sample_space, WeisfeilerLehmanKernel(), size)
    #pre_sample_x = list(init.initialize_population(size))
    target_name = target_to_name(target)
    file_name = f"presample_{target_name}_{size}.pt"
    if path is not None:
        metadata = {"target_name": target_name, "sample_size": size,
                    "file_name": file_name,
                    "path": path / file_name}
    else:
        metadata = {"target_name": target_name, "sample_size": size,
                    "file_name": file_name,
                    "path": PRESAMPLE_DIR / file_name}
    return pre_sample_x, metadata

def save_pre_samples(pre_samples_x, metadata=None, path=None):
    """Save pre-samples (X values only) to disk.

    Args:
        pre_samples_x: List of Tree objects (the samples)
        metadata: Dictionary with metadata (optional)
        path: Path to save to (optional, uses metadata["path"] if not provided)
    """
    if path is None:
        if metadata is not None and "path" in metadata:
            path = metadata["path"]
        else:
            path = DEFAULT_PRESAMPLE_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pre_samples_x": pre_samples_x,
        "metadata": metadata if metadata is not None else {},
    }
    torch.save(payload, path)
    return path

def load_pre_samples(path, map_location="cpu"):
    """Load pre-samples (X values only) from disk.

    Args:
        path: Path to pre-samples file
        map_location: torch device mapping

    Returns:
        tuple: (pre_samples_x, metadata)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-sample file not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "pre_samples_x" not in payload:
        raise ValueError("Pre-sample file is missing 'pre_samples_x' key.")

    return payload["pre_samples_x"], payload.get("metadata", {})


def _load_or_generate_initial_pre_samples(search_space, target, pre_sample_path=DEFAULT_PRESAMPLE_PATH,
                                          size: int = DEFAULT_PRESAMPLE_SIZE):
    pre_sample_path = Path(pre_sample_path)

    if pre_sample_path.is_dir():
        base_dir = pre_sample_path
    else:
        base_dir = pre_sample_path.parent

    target_name = target_to_name(target)
    expected_filename = f"presample_{target_name}_{size}.pt"
    candidate = base_dir / expected_filename

    try:
        return load_pre_samples(candidate), candidate
    except (FileNotFoundError, ValueError):
        print(f"Pre-sample not found at {candidate}. Generating new pre-samples.")
        return generate_pre_samples(search_space, target, size), candidate


def _dedupe_pre_samples(pre_samples_x):
    unique_pre_samples_x = []
    seen = set()
    duplicates_removed = 0

    for tree in pre_samples_x:
        if tree in seen:
            duplicates_removed += 1
            continue
        seen.add(tree)
        unique_pre_samples_x.append(tree)

    return unique_pre_samples_x, seen, duplicates_removed


def get_or_create_unique_pre_samples(search_space, target, pre_sample_path=DEFAULT_PRESAMPLE_PATH,
                                     size: int = DEFAULT_PRESAMPLE_SIZE,
                                     max_attempts_factor: int = 5,
                                     cache_tag: str | None = None):
    """Get or create deduplicated pre-samples for a given target and size.

    The returned sample set is guaranteed to be unique up to the available search
    space diversity. A dedicated cache file named
    ``presample_{target_name}_{size}_unique.pt`` is used for the deduplicated set.
    """
    pre_sample_path = Path(pre_sample_path)

    if pre_sample_path.is_dir():
        base_dir = pre_sample_path
    else:
        base_dir = pre_sample_path.parent

    target_name = target_to_name(target)
    if cache_tag is None:
        unique_filename = f"presample_{target_name}_{size}_unique.pt"
    else:
        safe_cache_tag = str(cache_tag).strip().replace(" ", "_")
        unique_filename = f"presample_{target_name}_{size}_{safe_cache_tag}_unique.pt"
    unique_candidate = base_dir / unique_filename

    try:
        return load_pre_samples(unique_candidate)
    except (FileNotFoundError, ValueError):
        pass

    (initial_pre_samples_x, metadata), _ = _load_or_generate_initial_pre_samples(
        search_space=search_space,
        target=target,
        pre_sample_path=pre_sample_path,
        size=size,
    )
    unique_pre_samples_x, seen, duplicates_removed = _dedupe_pre_samples(initial_pre_samples_x)

    attempts = len(initial_pre_samples_x)
    additional_generated = 0
    batch_metadata = metadata

    while len(unique_pre_samples_x) < size and attempts < size * max_attempts_factor:
        batch_size = max(size - len(unique_pre_samples_x), 5)
        batch_pre_samples_x, batch_metadata = generate_pre_samples(search_space, target, batch_size)
        additional_generated += len(batch_pre_samples_x)
        attempts += len(batch_pre_samples_x)

        for tree in batch_pre_samples_x:
            if tree in seen:
                duplicates_removed += 1
                continue
            seen.add(tree)
            unique_pre_samples_x.append(tree)
            if len(unique_pre_samples_x) >= size:
                break

    if len(unique_pre_samples_x) < size:
        warnings.warn(
            f"Only {len(unique_pre_samples_x)} unique pre-samples could be generated for target '{target_name}' "
            f"after {attempts} attempts (requested {size}).",
            RuntimeWarning,
            stacklevel=2,
        )

    unique_pre_samples_x = unique_pre_samples_x[:size]
    unique_metadata = dict(batch_metadata or metadata or {})
    unique_metadata.update(
        {
            "target_name": target_name,
            "sample_size": len(unique_pre_samples_x),
            "requested_sample_size": size,
            "unique_sample_size": len(unique_pre_samples_x),
            "duplicate_samples_removed": duplicates_removed,
            "additional_samples_generated": additional_generated,
            "max_attempts_factor": max_attempts_factor,
            "file_name": unique_filename,
            "path": unique_candidate,
            "cache_kind": "unique",
            "cache_tag": cache_tag,
        }
    )
    save_pre_samples(unique_pre_samples_x, unique_metadata, path=unique_candidate)
    return unique_pre_samples_x, unique_metadata

def get_or_create_pre_samples(search_space, target, pre_sample_path=DEFAULT_PRESAMPLE_PATH,
                              size: int = DEFAULT_PRESAMPLE_SIZE,
                              cache_tag: str | None = None):
    """Backward-compatible alias for the deduplicated pre-sample cache.

    All callers now receive unique samples via
    :func:`get_or_create_unique_pre_samples`.
    """
    if cache_tag is None:
        return get_or_create_unique_pre_samples(search_space, target, pre_sample_path=pre_sample_path, size=size)
    return get_or_create_unique_pre_samples(
        search_space,
        target,
        pre_sample_path=pre_sample_path,
        size=size,
        cache_tag=cache_tag,
    )
