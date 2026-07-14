import argparse
import glob
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import os
from jax import random
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


def process_and_save_kitti(
    base_path, save_path, sequence, step, voxel_size, number_of_points
):
    """Process a KITTI dataset sequence and save samples to a HDF5 file.

    Samples are formed by reading KITTI odometry poses from
    poses/{sequence}.txt, pairing frames step frames apart, loading
    corresponding Velodyne point clouds from .bin files, voxel downsampling
    with Open3D, and selecting number_of_points points from each cloud using
    randomly chosen indices with low duplicate rate. Ground truth rotation
    matrix R and translation vector t from X to Y are computed from the KITTI
    pose matrices and the Velodyne-to-camera calibration matrix.

    Output is written incrementally to save_path under groups named
    "sample_000000", "sample_000001", etc., each with datasets "X", "Y", "R",
    and "t".

    Args:
        base_path: Base directory of KITTI dataset.
        save_path: Output path for the HDF5 file written by this function.
        sequence: Sequence name under base_path.
        step: Stride between pose entries used to form point cloud pairs.
        voxel_size: Voxel size in meters used for Open3D voxel downsampling.
        number_of_points: Number of points to choose from each point cloud.
    """
    key = random.key(0)
    data_root = os.path.join(base_path, "odometry", "dataset")
    pose_path = os.path.join(data_root, "poses", f"{sequence}.txt")
    pose_data = np.loadtxt(pose_path, dtype=np.float32)
    sequ_root = os.path.join(data_root, "sequences", sequence)
    bin_files = glob.glob(os.path.join(sequ_root, "velodyne", "*.bin"))
    cali_path = os.path.join(sequ_root, "calib.txt")
    with open(cali_path, "r") as file:
        cali_lines = file.readlines()
    tr_line = cali_lines[4].strip()
    assert tr_line.startswith("Tr: ")
    tr = np.fromstring(tr_line[4:], sep=" ", dtype=np.float32)
    cali_T = np.eye(4, dtype=np.float32)
    cali_T[:3, :4] = tr.reshape((3, 4))
    sample_index = 0
    for idx in trange(0, len(bin_files) - step, step, mininterval=10.0):
        key, key_x, key_y = random.split(key, num=3)
        X_idx = idx
        Y_idx = idx + step
        # Load source (X) and target (Y) point clouds from .bin files.
        X_path = os.path.join(sequ_root, "velodyne", f"{X_idx:06d}.bin")
        Y_path = os.path.join(sequ_root, "velodyne", f"{Y_idx:06d}.bin")
        X = np.fromfile(X_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        Y = np.fromfile(Y_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        # Voxel down sample both point clouds.
        X_pcd = o3d.geometry.PointCloud()
        X_pcd.points = o3d.utility.Vector3dVector(X)
        Y_pcd = o3d.geometry.PointCloud()
        Y_pcd.points = o3d.utility.Vector3dVector(Y)
        X_pcd = X_pcd.voxel_down_sample(voxel_size=voxel_size)
        Y_pcd = Y_pcd.voxel_down_sample(voxel_size=voxel_size)
        X = jnp.asarray(X_pcd.points)
        Y = jnp.asarray(Y_pcd.points)
        # Choose number_of_points points from X and Y.
        X_choice = randomly_choose_indices(key_x, len(X), number_of_points)
        Y_choice = randomly_choose_indices(key_y, len(Y), number_of_points)
        X = X[X_choice]
        Y = Y[Y_choice]
        # Calculate rotation matrix R and translation vector t from X to Y.
        X_T = np.eye(4, dtype=np.float32)
        X_T[:3, :4] = pose_data[X_idx].reshape((3, 4))
        Y_T = np.eye(4, dtype=np.float32)
        Y_T[:3, :4] = pose_data[Y_idx].reshape((3, 4))
        T = np.linalg.inv(Y_T @ cali_T) @ (X_T @ cali_T)
        R = T[:3, :3]
        t = T[:3, 3]
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
    """Parse command-line arguments for processing the KITTI Odometry dataset.

    This function defines input and output paths, the selected dataset sequence,
    and key processing parameters. It also validates that the output file does
    not already exist.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    sequens = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    default_base_path = os.path.join("datasets", "kitti")
    default_save_path = os.path.join("datasets", "processed", "kitti.hdf5")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=default_base_path)
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--sequence", type=str, default="10", choices=sequens)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.15)
    parser.add_argument("--number_of_points", type=int, default=30000)
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        raise FileExistsError(f"File '{args.save_path}' already exists.")
    return args


def main():
    """Run the processing pipeline for a KITTI Odometry sequence.

    Parse command-line arguments, process the selected KITTI Odometry sequence,
    and save samples to a HDF5 file.
    """
    args = get_args()
    print(f"Processing KITTI {args.sequence}, saving to {args.save_path}")
    process_and_save_kitti(
        args.base_path,
        args.save_path,
        args.sequence,
        args.step,
        args.voxel_size,
        args.number_of_points,
    )


if __name__ == "__main__":
    main()
