#!/usr/bin/env python3
"""
Description: Script to calculate statistics for multi-modal datasets.
Supports different normalization strategies (1/sqrt(d) and 1/d)
per modality. Used for Latent modality reconstruction in SMF.
"""

import os

import h5py
import numpy as np
import pandas as pd

# HDF5 file paths for input (extracted features from backbone)
HDF5_PATH = "/h5_files/svi_data/place-pulse-2.0/legendre_polys_10_25_05_2025.h5"

# Output directory for saved statistics
OUTPUT_DIR = "/svi_data/place-pulse-2.0/"
TRAINING_GDF_PATH = "/svi_data/place-pulse-2.0/gdf_training.csv"

# List of modalities to load and concatenate
MODALITIES = ["coords_original", "SVI", "sentinel2", "OSM", "POI"]

# Normalization method: "sqrt" for 1/sqrt(d), "linear" for 1/d
NORMALIZATION_METHOD = "sqrt"  # or "linear"

postfix = "bge-m3_OSM_30_buffer_200"


def get_scaling_factors(dimensions: list, method: str = "sqrt") -> list:
    """
    Compute scaling factors for each modality dimension.

    Parameters
    ----------
    dimensions : list of int
        List containing the dimensionality of each modality.
    method : str, optional
        Normalization method: "sqrt" for 1/sqrt(d), "linear" for 1/d.
        Default is "sqrt".

    Returns
    -------
    scaling_factors : list of float
        List of scaling factors for each modality.
    """
    if method == "sqrt":
        return [1 / np.sqrt(dim) for dim in dimensions]
    elif method == "linear":
        return [1 / dim for dim in dimensions]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_and_concatenate_modalities(
    hdf5_path: str, modalities: str, df_path: str
) -> tuple:
    """
    Load and concatenate specified modalities for selected indices from HDF5
    file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    modalities : list of str
        List of modality dataset names to load and concatenate.
    df_path : str
        Path to the CSV file containing 'h5_index' and 'split' columns.

    Returns
    -------
    concatenated : np.ndarray
        Concatenated array of all modalities for selected indices.
    modality_dims : list of int
        Dimensionality of each modality.
    h5_indices : np.ndarray
        Array of selected indices.
    """
    df = pd.read_csv(df_path)
    df = df[df["split"] == "training"]
    h5_indices = df["h5_index"].values
    print(len(h5_indices), "indices selected for training.")
    with h5py.File(hdf5_path, "r") as f:
        modality_dims = []
        datasets = []
        print("Keys in the HDF5 file:")
        for key in modalities:
            if key in f:
                data = f[key][h5_indices, :]
                print(data.shape, "data shape for", key)
                datasets.append(data)
                modality_dims.append(data.shape[1])
            else:
                print(f"Warning: {key} not found in the file!")
        concatenated = np.concatenate(datasets, axis=1)
    return concatenated, modality_dims


def compute_normalization_params(array: np.ndarray) -> tuple:
    """
    Compute mean and standard deviation vectors for normalization.

    Parameters
    ----------
    array : np.ndarray
        Array to normalize.

    Returns
    -------
    mean_vector : np.ndarray
        Mean vector of the array.
    std_vector : np.ndarray
        Standard deviation vector of the array.
    """
    mean_vector = np.mean(array, axis=0)
    std_vector = np.std(array, axis=0)
    return mean_vector, std_vector


def build_scaling_vector(
    modality_dims: list, scaling_factors: list
) -> np.ndarray:
    """
    Build scaling vector to match concatenated array dimensions.

    Parameters
    ----------
    modality_dims : list of int
        Dimensionality of each modality.
    scaling_factors : list of float
        Scaling factor for each modality.

    Returns
    -------
    scaling_vector : np.ndarray
        Scaling vector matching the full feature axis.
    """
    scaling_vector = np.concatenate(
        [
            np.full(dim, scale)
            for dim, scale in zip(modality_dims, scaling_factors)
        ]
    )
    return scaling_vector


def normalize_array(
    array: np.array,
    mean_vector: np.array,
    std_vector: np.array,
    scaling_vector: np.array,
) -> np.ndarray:
    """
    Normalize array using mean, std, and scaling vector.

    Parameters
    ----------
    array : np.ndarray
        Input data array.
    mean_vector : np.ndarray
        Mean vector for normalization.
    std_vector : np.ndarray
        Standard deviation vector for normalization.
    scaling_vector : np.ndarray
        Scaling vector per feature.

    Returns
    -------
    normalized_array : np.ndarray
        The normalized array.
    """
    return ((array - mean_vector) / std_vector) * scaling_vector


def save_vectors(
    mean_vector: np.array,
    std_vector: np.array,
    scaling_vector: np.array,
    output_dir: str,
) -> None:
    """
    Save normalization vectors to disk.

    Parameters
    ----------
    mean_vector : np.ndarray
        Mean vector to save.
    std_vector : np.ndarray
        Std vector to save.
    scaling_vector : np.ndarray
        Scaling vector to save.
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(
        os.path.join(output_dir, f"mean_vector_{postfix}.npy"),
        mean_vector,
    )
    np.save(
        os.path.join(output_dir, f"std_vector_{postfix}.npy"),
        std_vector,
    )
    np.save(
        os.path.join(output_dir, f"scaling_vector_{postfix}.npy"),
        scaling_vector,
    )
    print(f"Saved normalization vectors to: {output_dir}")


if __name__ == "__main__":
    # Load and concatenate data
    concatenated, modality_dims = load_and_concatenate_modalities(
        HDF5_PATH, MODALITIES, TRAINING_GDF_PATH
    )
    print("Shape of concatenated array:", concatenated.shape)

    # Compute normalization parameters
    mean_vector, std_vector = compute_normalization_params(concatenated)
    print("Mean vector shape:", mean_vector.shape)
    print("Standard deviation vector shape:", std_vector.shape)

    # Get scaling factors and build scaling vector
    scaling_factors = get_scaling_factors(
        modality_dims, method=NORMALIZATION_METHOD
    )
    scaling_vector = build_scaling_vector(modality_dims, scaling_factors)
    print("Scaling vector shape:", scaling_vector.shape)

    # Normalize the concatenated array
    normalized_concatenated = normalize_array(
        concatenated, mean_vector, std_vector, scaling_vector
    )
    print(
        "Shape of normalized concatenated array:",
        normalized_concatenated.shape,
    )
    print(normalized_concatenated[0])  # Print first row for verification

    # Save normalization vectors for later use
    save_vectors(mean_vector, std_vector, scaling_vector, OUTPUT_DIR)
