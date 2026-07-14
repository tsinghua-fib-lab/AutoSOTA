import argparse
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import os
import pandas as pd
from jax import random
from jax.scipy.spatial.transform import Rotation
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_cuda_visible_devices", "")
jax.config.update("jax_default_matmul_precision", "highest")


def randomly_choose_indices(key, n, m):
    """Choose m indices from jnp.arange(n), keeping duplicates to a minimum.

    This is done by concatenating one or more random permutations of
    jnp.arange(n), shuffling the result, and taking the first m entries.

    Args:
        key: JAX PRNGKey used for random permutations.
        n: Number of indices to choose from.
        m: Number of indices to randomly choose.

    Returns:
        A JAX array of shape (m,) containing integer indices in [0, n).
    """
    ceil = int(np.ceil(m / n))
    keys = random.split(key, ceil + 1)
    indices = jnp.arange(n)
    permutations = [random.permutation(k, indices) for k in keys[:-1]]
    permutations = jnp.concatenate(permutations)
    permutations = random.permutation(keys[-1], permutations)
    choice = permutations[:m]
    return choice


def process_and_save_wild_places(
    base_path, save_path, sequence, step, voxel_size, number_of_points
):
    """Process a Wild Places dataset sequence and save samples to a HDF5 file.

    Samples are formed by reading pose entries from submap_poses.csv, pairing
    poses step frames apart, loading corresponding point clouds from .bin files,
    voxel downsampling with Open3D, and selecting number_of_points points from
    each cloud using randomly chosen indices with low duplicate rate. Ground
    truth rotation matrix R and translation vector t from X to Y are computed
    from pose translations and quaternions.

    Output is written incrementally to save_path under groups named
    "sample_000000", "sample_000001", etc., each with datasets "X", "Y", "R",
    and "t".

    Args:
        base_path: Base directory of Wild Places dataset.
        save_path: Output path for the HDF5 file written by this function.
        sequence: Sequence name under base_path.
        step: Stride between pose entries used to form point cloud pairs.
        voxel_size: Voxel size in meters used for Open3D voxel downsampling.
        number_of_points: Number of points to choose from each point cloud.
    """
    key = random.key(0)
    pose_path = os.path.join(base_path, sequence, "submap_poses.csv")
    pose_data = pd.read_csv(pose_path, dtype=str)
    pose_data.rename(columns={"%time": "time"}, inplace=True)
    pose_list = list(pose_data.itertuples(index=False, name=None))
    sample_index = 0
    for idx in trange(0, len(pose_list) - step, step, mininterval=10.0):
        key, key_x, key_y = random.split(key, num=3)
        X_time, X_x, X_y, X_z, X_qx, X_qy, X_qz, X_qw = pose_list[idx]
        Y_time, Y_x, Y_y, Y_z, Y_qx, Y_qy, Y_qz, Y_qw = pose_list[idx + step]
        # Load source (X) and target (Y) point clouds from .bin files.
        X_path = os.path.join(base_path, sequence, "Clouds", f"{X_time}.bin")
        Y_path = os.path.join(base_path, sequence, "Clouds", f"{Y_time}.bin")
        X = np.fromfile(X_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        Y = np.fromfile(Y_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        # Voxel down sample both point clouds.
        X_pcd = o3d.geometry.PointCloud()
        X_pcd.points = o3d.utility.Vector3dVector(X)
        Y_pcd = o3d.geometry.PointCloud()
        Y_pcd.points = o3d.utility.Vector3dVector(Y)
        X_pcd = X_pcd.voxel_down_sample(voxel_size=voxel_size)
        Y_pcd = Y_pcd.voxel_down_sample(voxel_size=voxel_size)
        # Choose number_of_points points from X and Y.
        X = jnp.asarray(X_pcd.points)
        Y = jnp.asarray(Y_pcd.points)
        X_choice = randomly_choose_indices(key_x, len(X), number_of_points)
        Y_choice = randomly_choose_indices(key_y, len(Y), number_of_points)
        X = X[X_choice]
        Y = Y[Y_choice]
        # Calculate rotation matrix R and translation vector t from X to Y.
        X_t = jnp.array([float(X_x), float(X_y), float(X_z)])
        Y_t = jnp.array([float(Y_x), float(Y_y), float(Y_z)])
        X_q = jnp.array([float(X_qx), float(X_qy), float(X_qz), float(X_qw)])
        Y_q = jnp.array([float(Y_qx), float(Y_qy), float(Y_qz), float(Y_qw)])
        X_R = Rotation.from_quat(X_q).as_matrix()
        Y_R = Rotation.from_quat(Y_q).as_matrix()
        R = Y_R.T @ X_R
        t = Y_R.T @ (X_t - Y_t)
        # Save sample.
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        R = np.array(R, dtype=np.float32)
        t = np.array(t, dtype=np.float32)
        s = f"sample_{sample_index:06d}"
        with h5py.File(save_path, "x" if sample_index == 0 else "r+") as f:
            f.create_dataset(f"{s}/X", dtype=np.float32, data=X)
            f.create_dataset(f"{s}/Y", dtype=np.float32, data=Y)
            f.create_dataset(f"{s}/R", dtype=np.float32, data=R)
            f.create_dataset(f"{s}/t", dtype=np.float32, data=t)
        sample_index += 1


def get_args():
    """Parse command-line arguments for processing the Wild Places dataset.

    This function defines input and output paths, the selected dataset sequence,
    and key processing parameters. It also validates that the output file does
    not already exist.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    # Sequence V-01 has missing data.
    sequens = ["K-01", "K-02", "K-03", "K-04", "V-02", "V-03", "V-04"]
    default_base_path = os.path.join("datasets", "wild_places")
    default_save_path = os.path.join("datasets", "processed", "wild.hdf5")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=default_base_path)
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--sequence", type=str, default="K-04", choices=sequens)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--voxel_size", type=float, default=0.15)
    parser.add_argument("--number_of_points", type=int, default=50000)
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        raise FileExistsError(f"File '{args.save_path}' already exists.")
    return args


def main():
    """Run the processing pipeline for a Wild Places dataset sequence.

    Parse command-line arguments, process the selected Wild Places dataset
    sequence, and save samples to a HDF5 file.
    """
    args = get_args()
    print(f"Processing Wild Places {args.sequence}, saving to {args.save_path}")
    process_and_save_wild_places(
        args.base_path,
        args.save_path,
        args.sequence,
        args.step,
        args.voxel_size,
        args.number_of_points,
    )


if __name__ == "__main__":
    main()
