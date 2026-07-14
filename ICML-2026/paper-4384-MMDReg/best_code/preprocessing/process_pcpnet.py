import argparse
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import os
from jax import random
from jax.scipy.spatial.transform import Rotation
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_cuda_visible_devices", "")
jax.config.update("jax_default_matmul_precision", "highest")


def get_random_rotation_matrix(key, min_degrees, max_degrees):
    """Generate a rotation matrix from randomly chosen Euler angles.

    Euler angles are sampled independently and uniformly in degrees from
    [min_degrees, max_degrees) and interpreted using the "zyx" convention.

    Args:
        key: JAX PRNGKey used to sample Euler angles.
        min_degrees: Minimum value for each Euler angle in degrees.
        max_degrees: Maximum value for each Euler angle in degrees.

    Returns:
        A 3x3 rotation matrix.
    """
    euler_angles = random.uniform(
        key, shape=(3,), minval=min_degrees, maxval=max_degrees
    )
    R = Rotation.from_euler("zyx", euler_angles, degrees=True).as_matrix()
    return R


def random_rotate_source(key, X, R, min_degrees=0.0, max_degrees=45.0):
    """Randomly rotate the source point cloud and update the rotation matrix.

    Args:
        key: JAX PRNGKey used to sample a random rotation.
        X: Matrix of points for the source point cloud.
        R: True rotation matrix from X to Y.
        min_degrees: Minimum value for each Euler angle in degrees.
        max_degrees: Maximum value for each Euler angle in degrees.

    Returns:
        A tuple (X, R) where X is the rotated source point cloud and R is the
        updated true rotation matrix.
    """
    # If ``x, y`` are three-dimensional vectors,
    # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
    # then ``y = (R @ Q.T) @ (Q @ x) + t``.
    Q = get_random_rotation_matrix(key, min_degrees, max_degrees)
    X = jnp.matmul(X, Q.T)
    R = jnp.matmul(R, Q.T)
    return X, R


def get_random_translation_vector(key, lower_bound, upper_bound):
    """Generate a random translation vector with uniformly sampled components.

    Each axis is sampled independently and uniformly from
    [lower_bound, upper_bound).

    Args:
        key: JAX PRNGKey used to sample a random translation vector.
        lower_bound: Minimum value for each axis of the translation vector.
        upper_bound: Maximum value for each axis of the translation vector.

    Returns:
        A translation vector.
    """
    t = random.uniform(key, shape=(3,), minval=lower_bound, maxval=upper_bound)
    return t


def random_translate_source(key, X, R, t, lower_bound=-0.5, upper_bound=0.5):
    """Randomly translate the source point cloud and update translation vector.

    Args:
        key: JAX PRNGKey used to sample a random translation vector.
        X: Matrix of points for the source point cloud.
        R: True rotation matrix from X to Y.
        t: True translation vector from X to Y.
        lower_bound: Minimum value for each axis of the translation vector.
        upper_bound: Maximum value for each axis of the translation vector.

    Returns:
        A tuple (X, t) where X is the translated source point cloud and t is the
        updated true translation vector.
    """
    # If ``x, y, c`` are three-dimensional vectors,
    # ``R`` is a rotation matrix and ``y = R @ x + t``,
    # then ``y = R @ (x + c) + (t - R @ c)``.
    c = get_random_translation_vector(key, lower_bound, upper_bound)
    X = X + c
    t = t - jnp.matmul(R, c)
    return X, t


def insert_outliers(key, X, number_of_outliers, radius):
    """Replace randomly selected points with outliers sampled from a ball.

    Outliers are sampled from a 3D unit ball and scaled by radius, then written
    into randomly chosen rows of X without replacement.

    Args:
        key: JAX PRNGKey used to sample indices and outlier points.
        X: Matrix of points for the source point cloud.
        number_of_outliers: Number of points in X to replace with outliers.
        radius: Radius of the ball from which outliers are sampled.

    Returns:
        The matrix of points with outliers inserted.
    """
    key_choice, key_ball = random.split(key, num=2)
    choice = random.choice(key_choice, len(X), (number_of_outliers,), False)
    points = random.ball(key_ball, 3, shape=(number_of_outliers,)) * radius
    X = X.at[choice].set(points)
    return X


def get_splits_dict():
    """Return mapping from split name to corresponding text file name.

    Returns:
        A dictionary mapping a data split identifier to a text file that
        contains a list of point cloud names from that data split.
    """
    splits_dict = {
        "test_all": "testset_all.txt",
        "test_no_noise": "testset_no_noise.txt",
        "test_low_noise": "testset_low_noise.txt",
        "test_med_noise": "testset_med_noise.txt",
        "test_high_noise": "testset_high_noise.txt",
        "test_var_density_striped": "testset_vardensity_striped.txt",
        "test_var_density_gradient": "testset_vardensity_gradient.txt",
        "train_no_noise": "trainingset_no_noise.txt",
        "train_noise": "trainingset_whitenoise.txt",
        "train_var_density": "trainingset_vardensity.txt",
        "train_var_density_noise": "trainingset_vardensity_whitenoise.txt",
        "val_no_noise": "validationset_no_noise.txt",
        "val_noise": "validationset_whitenoise.txt",
        "val_var_density": "validationset_vardensity.txt",
        "val_var_density_noise": "validationset_vardensity_whitenoise.txt",
    }
    return splits_dict


def get_symmetric_categories():
    """Return list of symmetric shape categories.

    Returns:
        The list of symmetric shape categories.
    """
    symmetric_categories = [
        "column100k",
        "column_head100k",
        "cylinder100k",
        "cylinder_analytic100k",
        "flower100k",
        "icosahedron100k",
        "pipe100k",
        "sheet_analytic100k",
        "sphere100k",
        "sphere_analytic100k",
        "star_halfsmooth100k",
        "star_sharp100k",
        "star_smooth100k",
    ]
    return symmetric_categories


def process_and_save_pcpnet(
    base_path,
    save_path,
    split,
    number_of_passes,
    number_of_points,
    number_of_outliers,
):
    """Process a PCPNet dataset split and save samples to a HDF5 file.

    Each sample is built by loading a 100k-point shape, centering and scaling it
    to the unit sphere, randomly subsampling 2 * number_of_points points and
    splitting them into source and target point clouds, optionally inserting
    outliers, and applying a random rigid transformation to the source. Samples
    from symmetric categories are skipped.

    Output is written incrementally to save_path under groups named
    "sample_000000", "sample_000001", etc., each with datasets "X", "Y", "R",
    and "t".

    Args:
        base_path: Base directory of PCPNet dataset.
        save_path: Output path for the HDF5 file written by this function.
        split: Split identifier used to select the split file name.
        number_of_passes: Number of passes over the dataset split list.
        number_of_points: Number of points to choose from each point cloud.
        number_of_outliers: Number of outliers inserted into each point cloud.
    """
    key = random.key(0)
    split_file_path = os.path.join(base_path, get_splits_dict()[split])
    with open(split_file_path) as split_file:
        split_file_names = split_file.read().split("\n")[:-1]
        split_file_names.sort()
    symmetric_categories = get_symmetric_categories()
    sample_index = 0
    for _ in trange(number_of_passes, mininterval=10.0):
        for file_name in split_file_names:
            assert "100k" in file_name
            file_category = file_name[: file_name.find("100k") + 4]
            if file_category in symmetric_categories:
                continue
            key, key_p, key_r, key_t, key_x, key_y = random.split(key, num=6)
            points_path = os.path.join(base_path, file_name + ".xyz")
            points = np.loadtxt(points_path, dtype=np.float32)
            points = points - np.mean(points, axis=0)
            points = points / np.max(np.linalg.norm(points, axis=1))
            assert len(points) == 100000
            points = jnp.array(points)
            points = random.choice(
                key_p, points, (2 * number_of_points,), False
            )
            X = points[:number_of_points]
            Y = points[number_of_points:]
            R = jnp.eye(3)
            t = jnp.zeros(3)
            if number_of_outliers > 0:
                X = insert_outliers(key_x, X, number_of_outliers, 1.0)
                Y = insert_outliers(key_y, Y, number_of_outliers, 1.0)
            X, R = random_rotate_source(key_r, X, R)
            X, t = random_translate_source(key_t, X, R, t)
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
    """Parse command-line arguments for processing the PCPNet dataset.

    This function defines input and output paths, available dataset splits, and
    key processing parameters. It also validates that the output file does not
    already exist, number_of_points does not exceed 50000, and
    number_of_outliers does not exceed number_of_points.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    default_base_path = os.path.join("datasets", "pcpnet")
    default_save_path = os.path.join("datasets", "processed", "pcpnet.hdf5")
    splits = get_splits_dict().keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=default_base_path)
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--split", type=str, choices=splits, default="test_all")
    parser.add_argument("--number_of_passes", type=int, default=100)
    parser.add_argument("--number_of_points", type=int, default=50000)
    parser.add_argument("--number_of_outliers", type=int, default=0)
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        raise FileExistsError(f"File '{args.save_path}' already exists.")
    if args.number_of_points > 50000:
        raise ValueError("--number_of_points > 50000")
    if args.number_of_outliers > args.number_of_points:
        raise ValueError("--number_of_outliers > --number_of_points")
    return args


def main():
    """Run the processing pipeline for a PCPNet dataset split.

    Parse command-line arguments, process the selected PCPNet dataset split,
    and save samples to a HDF5 file.
    """
    args = get_args()
    print(f"Processing PCPNet split {args.split}, saving to {args.save_path},")
    print(f"Points {args.number_of_points}, Outliers {args.number_of_outliers}")
    process_and_save_pcpnet(
        args.base_path,
        args.save_path,
        args.split,
        args.number_of_passes,
        args.number_of_points,
        args.number_of_outliers,
    )


if __name__ == "__main__":
    main()
