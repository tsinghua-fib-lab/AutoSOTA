#!/usr/bin/env python3
"""
Parallel processing script for computing and saving VDW 
Data objects, or just their P and Q sparse tensor data 
(if `proc_kwargs['return_data_object'] == False`).

To add a dataset, do:
(1) import its module in the imports section
(2) add logic for its `dataset_key` in `init_dataset` 
    and the "__main__" call

NOTE: if mean recentering is desired instead of reference point recentering,
use the --use_mean_recentering flag.

Example call (using HDF5):
python3 "./scripts/parallel_process_dataset.py" \
--dataset_key QM9 \
--dataset_dir "/path/to/QM9" \
--h5_path "/path/to/QM9/{hdf5_filename}.h5" \
--num_workers 26 \
--local_pca_kernel cosine_cutoff \
--subsample_n 100

python3 "/path/to/codecode/scripts/python/parallel_process_dataset.py" \
--dataset_key QM9 \
--dataset_dir "/path/to/codedata/QM9" \
--h5_path "/path/to/codedata/QM9/pq_tensor_data_radial.h5" \
--hdf5_tensor_dtype float16 \
--graph_construction radius \
--num_workers 16 \
--local_pca_kernel cosine_cutoff \
--adjacency_rbf_eps 4.0 \
--adjacency_max_num_neighbors 16 \
--num_edge_radial_fns 16 \
--r_cut 5.0 \
--include_diracs

python3 "/path/to/codecode/scripts/python/parallel_process_dataset.py" \
--dataset_key QM9 \
--dataset_dir "/path/to/codedata/QM9" \
--h5_path "/path/to/codedata/QM9/pq_tensor_data_chem.h5" \
--hdf5_tensor_dtype float16 \
--num_workers 48 \
--local_pca_kernel epanechnikov \
--adjacency_rbf_eps 4.0 \
--adjacency_max_num_neighbors 4 \
--num_edge_radial_fns 0 \
--r_cut 5.0 \
--include_diracs
"""

# basic modules
import os
import sys
from pathlib import Path

# Determine the absolute path to the project root (two levels up from this file)
_SCRIPT_DIR = Path(__file__).resolve().parent  # .../scripts/python
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent      # .../ (project root)

# Prepend the project root to sys.path so that intra-repo imports (e.g., data_processing)
# work regardless of the working directory from which this script is executed.
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

import time
import argparse
import multiprocessing as mp
from typing import Tuple, List, Dict, Any, Optional
from itertools import islice
import logging
import h5py
import numpy as np
from tqdm import tqdm

# numpy, pytorch modules
from numpy.random import RandomState
from torch_geometric.data import Data
from torch_geometric.datasets import QM9, MD17
import torch

# our modules
import data_processing.process_pyg_data as process


# define functions used in parallel processing
def init_dataset(
    dataset_dir: str, 
    dataset_key: str
) -> None:
    global dataset
    global proc_kwargs

    if 'qm9' in dataset_key.lower():
        dataset = QM9(root=dataset_dir)
        proc_kwargs['vector_feat_key'] = 'pos'
    elif 'md17' in dataset_key.lower():
        # Expect format md17-<name>-<sample_size>, or at least <name> after first hyphen
        dataset = MD17(root=dataset_dir, name=dataset_key.split('-')[1])
        proc_kwargs['vector_feat_key'] = 'pos'
    else:
        raise NotImplementedError(f"Dataset {dataset_key} not yet supported.")


def process_one(i) -> Data:
    import data_processing.process_pyg_data as process
    return process.process_pyg_data(
        dataset[i], 
        data_i=i,
        **proc_kwargs
    )


def process_batch(
    batch_indices: List[int], 
    batch_idx: int,
    tensor_data_save_dir: str,
) -> Dict[str, int]:
    """
    Process a batch of graphs and save them together in a single file.
    
    Args:
        batch_indices: List of indices to process
        batch_idx: Index of this batch
        tensor_data_save_dir: Directory to save the processed data
    
    Returns:
        Dictionary containing statistics about the batch processing
    """
    # Process all graphs in the batch and collect their dictionaries
    batch_data = []
    failed_indices = []
    
    for i in batch_indices:
        try:
            tensor_dict = process.process_pyg_data(
                dataset[i],
                data_i=i,
                **proc_kwargs
            )
            if tensor_dict is not None:  # Only append if processing was successful
                batch_data.append(tensor_dict)
            else:
                failed_indices.append(i)
                logging.warning(f"Processing returned None for graph {i} in batch {batch_idx}")
        except Exception as e:
            failed_indices.append(i)
            logging.error(f"Failed to process graph {i} in batch {batch_idx}: {str(e)}")
    
    if batch_data:  # Only save if we have data to save
        # Save batched dictionaries in a single file
        try:
            torch.save(
                batch_data,
                os.path.join(tensor_data_save_dir, f"batch_{batch_idx}.pt")
            )
            logging.info(f"Successfully saved batch {batch_idx} with {len(batch_data)} graphs")
        except Exception as e:
            logging.error(f"Failed to save batch {batch_idx}: {str(e)}")
    
    if failed_indices:
        logging.warning(f"Batch {batch_idx} had {len(failed_indices)} failed graphs: {failed_indices}")
    
    return {
        'successful': len(batch_data),
        'failed': len(failed_indices),
        'total': len(batch_indices)
    }


def process_and_save_to_hdf5(
    indices, 
    temp_h5_path, 
    original_idx_key: str = 'original_idx',
    scalar_key: str = 'P',
    vector_key: str = 'Q',
    line_key: str = 'P_line',
) -> str:
    # print(f"[DEBUG] Worker writing temp file: {temp_h5_path} for indices: {indices}")
    # assert len(set(indices)) == len(indices), f"Duplicate indices in chunk: {indices}"
    # Each worker writes to its own temp HDF5 file
    with h5py.File(temp_h5_path, 'w') as h5f:
        for i in indices:
            tensor_dict = process.process_pyg_data(
                dataset[i],
                data_i=i,
                **proc_kwargs
            )

            # Add P_line if present in tensor_dict
            keys_to_save = [
                original_idx_key,
                scalar_key, 
                vector_key, 
                line_key, 
                'edge_index', 
                'edge_weight', 
                'edge_features', 
                'dirac_nodes'
            ]
            for key in keys_to_save:
                if key not in tensor_dict:
                    continue
                tensor_data = tensor_dict[key]
                grp = h5f.require_group(f"{key}/{i}")

                # ----------------------------------------------------------
                # Original index
                # ----------------------------------------------------------
                if key == original_idx_key:
                    # Ensure non-scalar dataset for compression: wrap scalar into 1D array
                    _vals = np.asarray(tensor_data, dtype='int64')
                    if _vals.shape == ():
                        _vals = _vals.reshape(1)
                    grp.create_dataset('values', data=_vals, compression="gzip")

                # ----------------------------------------------------------
                # Sparse COO tensors (P, Q)
                # ----------------------------------------------------------
                if key in [scalar_key, vector_key, line_key]:
                    # Sparse COO tensors share same structure
                    for dset_name, dset_data in [
                        ('indices', tensor_data['indices']),
                        ('values', tensor_data['values']),
                        ('size', np.array(tensor_data['size'], dtype='int64'))
                    ]:
                        grp.create_dataset(dset_name, data=dset_data, compression="gzip")

                # ----------------------------------------------------------
                # Dense tensors
                # ----------------------------------------------------------
                elif key == 'edge_index':
                    # Save as a single dataset 'values'
                    grp.create_dataset('values', data=tensor_data, compression="gzip")

                elif key == 'edge_weight':
                    grp.create_dataset('values', data=tensor_data, compression="gzip")

                elif key == 'edge_features':
                    # Dict with 'values' and 'shape'
                    grp.create_dataset('values', data=tensor_data['values'], compression="gzip")
                    grp.create_dataset('shape', data=tensor_data['shape'], compression="gzip")

                elif key == 'dirac_nodes':
                    grp.create_dataset('values', data=tensor_data, compression="gzip")
                else:
                    # Unhandled key – silently skip (future-proof)
                    pass
    return temp_h5_path


def _process_and_save_to_hdf5_star(args):
    """A thin wrapper around `process_and_save_to_hdf5` for `Pool.imap_*` calls."""
    return process_and_save_to_hdf5(*args)


def merge_hdf5_files(h5_path, temp_files):
    with h5py.File(h5_path, 'w') as h5f:
        for temp in temp_files:
            with h5py.File(temp, 'r') as tf:
                def copy_group(name, obj):
                    if isinstance(obj, h5py.Group):
                        h5f.require_group(name)
                    elif isinstance(obj, h5py.Dataset):
                        src_parent = tf[name.rsplit('/', 1)[0]] if '/' in name else tf
                        dst_parent = h5f[name.rsplit('/', 1)[0]] if '/' in name else h5f
                        if name.split('/')[-1] not in dst_parent:
                            src_parent.copy(obj, dst_parent)
                tf.visititems(copy_group)
            os.remove(temp)
    print(f"Final HDF5 written to {h5_path}.")


def parallel_process_to_hdf5(
    dataset_dir, 
    dataset_key, 
    idx, 
    num_workers, 
    h5_path, 
    original_idx_key,
    scalar_key, 
    vector_key,
    line_key,
) -> None:
    idx_chunks = np.array_split(idx, num_workers)
    temp_files = [f"{h5_path}.tmp_worker_{i}" for i in range(num_workers)]

    # Prepare tasks as (indices_chunk, temp_path, scalar_key, vector_key)
    chunk_args = [
        (idx_chunk, temp_files[i], original_idx_key, scalar_key, vector_key, line_key) \
        for i, idx_chunk in enumerate(idx_chunks)
    ]

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=num_workers,
        initializer=init_dataset,
        initargs=(dataset_dir, dataset_key)
    ) as pool:
        # Iterate over tasks as they complete and display progress
        for _ in tqdm(
            pool.imap_unordered(_process_and_save_to_hdf5_star, chunk_args),
            total=len(chunk_args),
            desc="Processing chunks",
        ):
            pass  # tqdm handles progress display

    # Merge temp files into final HDF5
    print(f"Merging {len(temp_files)} temp files into final HDF5 file...")
    merge_hdf5_files(h5_path, temp_files)


# run processing in __main__
if __name__ == "__main__":
    
    # command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', default=None, type=str, 
        help="dataset_dir directory where dataset is saved"
    )
    parser.add_argument(
        '--tensor_data_save_dir', default='pq_tensor_data', type=str, 
        help="subdirectory of 'dataset_dir' for saving tensor data files, when not using HDF5"
    )
    parser.add_argument(
        '-d', '--dataset_key', default=None, type=str, 
        help="dataset key, e.g. 'QM9'"
    )
    parser.add_argument(
        '-w', '--num_workers', default=None, type=int, 
        help="number of workers for parallel processing (defaults to CPU count - 1)"
    )
    parser.add_argument(
        '-s', '--subsample_n', default=None, type=int, 
        help="number of samples to process; if None, entire dataset will be processed"
    )
    parser.add_argument(
        '--seed', default=542920, type=int, 
        help="random seed for dataset subsampling"
    )
    parser.add_argument(
        '--n_graphs_per_file', default=1, type=int,
        help="number of graphs to save in each file (when not using HDF5)"
    )
    parser.add_argument(
        '--h5_path', type=str, required=True, 
        help='Output HDF5 file path (e.g., for P and Q tensors)'
    )
    parser.add_argument(
        '--graph_construction', type=str, default=None,
        choices=['k-nn', 'radius', 'unweighted_knn'],
        help='Graph construction method (default: None, to use existing edge_index)'
    )
    parser.add_argument(
        '--use_mean_recentering', action='store_true', default=False,
        help='Enable mean recentering in processing (default: False)'
    )
    parser.add_argument(
        '--local_pca_kernel', type=str, default='cosine_cutoff',
        choices=['chemical', 'gaussian', 'cosine_cutoff', 'epanechnikov'],
        help='Distance kernel function to use in node local PCA/SVD (default: cosine_cutoff)'
    )
    parser.add_argument(
        '--gaussian_eps', type=float, default=4.0,
        help='Local PCA/SVD Gaussian kernel epsilon (=sigma^2) (default: 4.0)'
    )
    parser.add_argument(
        '--hdf5_tensor_dtype', type=str, default='float16',
        help='Tensor data type for saving to HDF5 (default: float16)'
    )
    parser.add_argument(
        '--num_edge_radial_fns', type=int, default=0,
        help='Number of radial functions per edge (default: 0)'
    )
    parser.add_argument(
        '--adjacency_rbf_eps', type=float, default=4.0,
        help='Scale eps (=sigma^2) for Gaussian RBF used in adjacency matrix construction (default: 4.0)'
    )
    parser.add_argument(
        '--adjacency_max_num_neighbors', type=int, default=16,
        help='Maximum number of neighbors per node in adjacency matrix construction (default: 16)'
    )
    parser.add_argument(
        '--r_cut', type=float, default=5.0,
        help='Cutoff radius for adjacency matrix construction AND local (node) PCA/SVD kernel (default: 5.0)'
    )   
    parser.add_argument(
        '--include_diracs', action='store_true', default=False,
        help='Include Dirac nodes in processing (default: False)'
    )
    parser.add_argument(
        '--edge_feature_type', type=str, default='bessel',
        help="Type of edge features to compute (default: 'bessel'). Options: 'bessel' (sinusoidal radial basis)."
    )
    parser.add_argument(
        '--compute_line_operator', action='store_true', default=False,
        help='Compute line-graph diffusion operator P_line and include it in outputs (default: False)'
    )
    parser.add_argument(
        '--no_save_p', action='store_true', default=False,
        help='Do not save scalar operator P to HDF5 (default: False)'
    )
    clargs = parser.parse_args()

    # Set default num_workers to CPU count - 1 if not specified
    if clargs.num_workers is None:
        clargs.num_workers = max(1, mp.cpu_count() - 1)
        logging.info(f"Using {clargs.num_workers} workers (CPU count - 1)")

    # output save directory (if not using HDF5)
    if clargs.n_graphs_per_file > 1: 
        save_dir = os.path.join(clargs.dataset_dir, clargs.tensor_data_save_dir)
        os.makedirs(save_dir, exist_ok=True)
    
    # init vars made global in parallel processing (in `init_dataset`)
    dataset = None
    # Map the requested kernel to the actual callable function rather than a string
    # if clargs.local_pca_kernel == 'chemical':
    #     kernel_fn = process.K_chemical_weights
    if 'gauss' in clargs.local_pca_kernel.lower():
        local_pca_kernel_fn_kwargs = {
            'kernel': 'gaussian',
            'gaussian_eps': clargs.gaussian_eps,
            'r_cut': clargs.r_cut,
        }
    elif 'cosine' in clargs.local_pca_kernel:
        local_pca_kernel_fn_kwargs = {
            'kernel': 'cosine_cutoff',
            'r_cut': clargs.r_cut,
        }
    elif 'epane' in clargs.local_pca_kernel.lower():
        local_pca_kernel_fn_kwargs = {
            'kernel': 'epanechnikov',
            'r_cut': clargs.r_cut,
        }
    else:
        raise ValueError(f"Invalid distance kernel: {clargs.local_pca_kernel}")
    
    # generate index list of graphs to process in parallel
    # first, need to lazy-load dataset metadata to get its length
    if 'qm9' in clargs.dataset_key.lower():
        dataset_meta = QM9(root=clargs.dataset_dir)
        dataset_len = len(dataset_meta)
        del dataset_meta

        if clargs.include_diracs:
            from data_processing.process_pyg_data import select_molecule_dirac_nodes
            dirac_fn = select_molecule_dirac_nodes
            dirac_kwargs: Dict[str, Any] = {
                'coords_key': 'pos',
                'atom_types_key': 'z',
                'k': 3
            }
    elif 'md17' in clargs.dataset_key.lower():
        # Expect format md17-<name>-<sample_size>, or at least <name> after first hyphen
        dataset_meta = MD17(root=clargs.dataset_dir, name=clargs.dataset_key.split('-')[1])
        dataset_len = len(dataset_meta)
        del dataset_meta
    else:
        raise NotImplementedError(f"Dataset {clargs.dataset_key} not yet supported.")
    
    # generate index
    if clargs.subsample_n is None:
        idx = list(range(dataset_len))
    else:
        rand_state = np.random.RandomState(seed=clargs.seed)
        idx = rand_state.choice(dataset_len, size=clargs.subsample_n, replace=False).tolist()

    # Initialize process kwargs
    # NOTE: made global in `init_dataset`
    proc_kwargs = {
        'use_mean_recentering': clargs.use_mean_recentering,
        'enforce_sign': True,
        'device': 'cpu',
        'return_data_object': False,
        'graph_construction': clargs.graph_construction,
        'graph_construction_kwargs': {
            'r_cutoff': clargs.r_cut,
            'gaussian_eps': clargs.adjacency_rbf_eps,
            'max_num_neighbors': clargs.adjacency_max_num_neighbors,
        },
        'local_pca_kernel_fn_kwargs': local_pca_kernel_fn_kwargs,
        'hdf5_tensor_dtype': clargs.hdf5_tensor_dtype,
        'num_edge_features': clargs.num_edge_radial_fns,
        'include_diracs': clargs.include_diracs,
        'dirac_fn': locals().get('dirac_fn'),
        'dirac_kwargs': locals().get('dirac_kwargs'),
        'edge_feature_type': clargs.edge_feature_type,
        'compute_line_operator': clargs.compute_line_operator,
        'save_scalar_operator': (not clargs.no_save_p),
    }

    # Execute parallel processing
    t0 = time.time()
    parallel_process_to_hdf5(
        dataset_dir=clargs.dataset_dir,
        dataset_key=clargs.dataset_key,
        idx=idx,
        num_workers=clargs.num_workers,
        h5_path=clargs.h5_path,
        scalar_key=proc_kwargs.get('scalar_operator_key', 'P'),
        vector_key=proc_kwargs.get('vector_operator_key', 'Q'),
        line_key=proc_kwargs.get('line_operator_key', 'P_line'),
        original_idx_key=proc_kwargs.get('original_idx_key', 'original_idx'),
    )
    time_elapsed = time.time() - t0
    time_min, time_sec = time_elapsed // 60, time_elapsed % 60
    print(
        f"\nProcessing complete:"
        f"\n- Total graphs processed: {len(idx)}"
        f"\n- Final HDF5 file: {clargs.h5_path}"
        f"\n- Processing time ({clargs.num_workers} workers): {time_min} min, {time_sec:.2f} sec"
    )

