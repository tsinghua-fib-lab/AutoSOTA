import argparse
import glob
import h5py
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as SciPyRotation
from tqdm import tqdm


def mend_modelnet40_off_file(path):
    """When missing, add newline character after 'OFF' in first line of file.

    Args:
        path: Path to a ModelNet40 OFF file.
    """
    with open(path, "r") as file:
        lines = file.readlines()
    if lines[0] != "OFF\n":
        assert lines[0][:3] == "OFF"
        lines[0] = lines[0][3:]
        lines = ["OFF\n"] + lines
        with open(path, "w") as file:
            file.writelines(lines)


def random_sample_and_crop(
    mesh, number_of_points, same_sample, crop, crop_number_of_points_left
):
    """Randomly sample two point clouds from a mesh and optionally crop them.

    Args:
        mesh: An Open3D triangle mesh.
        number_of_points: The number of points to sample for both point clouds.
        same_sample: If True, the same sample is used for both point clouds.
        crop: If True, both point clouds are independently cropped after
            sampling.
        crop_number_of_points_left: The number of points to retain in each
            point cloud after cropping.

    Returns:
        A tuple (X, Y, R, t, Xo, Yo) where X is the matrix of points for the
        source point cloud, Y is the matrix of points for the target point
        cloud, R is the true rotation matrix, t is the true translation vector,
        Xo is the overlap mask for X and Yo is the overlap mask for Y.
    """
    if same_sample:
        pcd = mesh.sample_points_uniformly(number_of_points)
        pcd_points = np.array(pcd.points).astype(np.float32)
        pcd_points = pcd_points - np.mean(pcd_points, axis=0)
        pcd_points = pcd_points / np.max(np.linalg.norm(pcd_points, axis=1))
        X_permutation = np.random.permutation(len(pcd_points))
        Y_permutation = np.random.permutation(len(pcd_points))
        X = pcd_points[X_permutation]
        Y = pcd_points[Y_permutation]
    else:
        pcd = mesh.sample_points_uniformly(2 * number_of_points)
        pcd_points = np.array(pcd.points).astype(np.float32)
        pcd_points = pcd_points - np.mean(pcd_points, axis=0)
        pcd_points = pcd_points / np.max(np.linalg.norm(pcd_points, axis=1))
        permutation = np.random.permutation(len(pcd_points))
        pcd_points = pcd_points[permutation]
        X = pcd_points[:number_of_points]
        Y = pcd_points[number_of_points:]
    if crop:
        # Generate normal vectors for two random hyperplanes.
        normal_vector_1 = np.random.randn(3)
        normal_vector_1 = normal_vector_1 / np.linalg.norm(normal_vector_1)
        normal_vector_2 = np.random.randn(3)
        normal_vector_2 = normal_vector_2 / np.linalg.norm(normal_vector_2)
        # Compute the points in X that lie below both hyperplanes, these points
        # are in the overlapping region. Then, based on the first hyperplane,
        # crop X and its overlap mask.
        X_dot_products_1 = np.linalg.matmul(X, normal_vector_1)
        X_dot_products_2 = np.linalg.matmul(X, normal_vector_2)
        X_left_1 = np.argsort(X_dot_products_1)[:crop_number_of_points_left]
        X_left_2 = np.argsort(X_dot_products_2)[:crop_number_of_points_left]
        Xo = np.zeros(len(X))
        Xo[np.intersect1d(X_left_1, X_left_2)] = 1.0
        X = X[X_left_1]
        Xo = Xo[X_left_1].astype(np.bool)
        # Compute the points in Y that lie below both hyperplanes, these points
        # are in the overlapping region. Then, based on the second hyperplane,
        # crop Y and its overlap mask.
        Y_dot_products_1 = np.linalg.matmul(Y, normal_vector_1)
        Y_dot_products_2 = np.linalg.matmul(Y, normal_vector_2)
        Y_left_1 = np.argsort(Y_dot_products_1)[:crop_number_of_points_left]
        Y_left_2 = np.argsort(Y_dot_products_2)[:crop_number_of_points_left]
        Yo = np.zeros(len(Y))
        Yo[np.intersect1d(Y_left_1, Y_left_2)] = 1.0
        Y = Y[Y_left_2]
        Yo = Yo[Y_left_2].astype(np.bool)
    else:
        Xo = np.ones(len(X)).astype(np.bool)
        Yo = np.ones(len(Y)).astype(np.bool)
    R = np.eye(3)
    t = np.zeros(3)
    return X, Y, R, t, Xo, Yo


def random_jitter(X, scale=0.01, min_clip=-0.05, max_clip=0.05):
    """Add random noise to a point cloud.

    Args:
        X: The matrix of points for a point cloud.
        scale: The standard deviation for the random noise.
        min_clip: The minimum value to clip the noise.
        max_clip: The maximum value to clip the noise.

    Returns:
        The matrix of points for the point cloud with added random noise.
    """
    # Same as in GeoTransformer and Deep Closest Point
    noise = np.random.normal(scale=scale, size=X.shape)
    noise = np.clip(noise, min_clip, max_clip)
    return X + noise


def get_random_rotation_matrix(min_degrees, max_degrees):
    """Return a rotation matrix using randomly chosen Euler angles.

    Args:
        min_degrees: The minimum value for each Euler angle in degrees.
        max_degrees: The maximum value for each Euler angle in degrees.

    Returns:
        A rotation matrix.
    """
    euler_angles = np.random.uniform(min_degrees, max_degrees, size=3)
    # Same as using anglez, angley, anglex = np.deg2rad(euler_angles) in DCP.
    R = SciPyRotation.from_euler("zyx", euler_angles, degrees=True).as_matrix()
    return R


def random_rotate_source(X, R, min_degrees=0.0, max_degrees=45.0):
    """Randomly rotate the source point cloud and update the rotation matrix.

    Args:
        X: The matrix of points for the source point cloud.
        R: The true rotation matrix.
        min_degrees: The minimum value for each Euler angle in degrees.
        max_degrees: The maximum value for each Euler angle in degrees.

    Returns:
        A tuple (X, R) where X is the new matrix of points for the source point
        cloud and R is the new true rotation matrix.
    """
    # If ``x, y`` are three-dimensional vectors,
    # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
    # then ``y = (R @ Q.T) @ (Q @ x) + t``.
    Q = get_random_rotation_matrix(min_degrees, max_degrees)
    X = np.matmul(X, Q.T)
    R = np.matmul(R, Q.T)
    return X, R


def get_random_translation_vector(lower_bound, upper_bound):
    """Return a random translation vector, where each axis is uniformly sampled.

    Args:
        lower_bound: The minimum value for each axis of the translation vector.
        upper_bound: The maximum value for each axis of the translation vector.

    Returns:
        A translation vector.
    """
    t = np.random.uniform(lower_bound, upper_bound, 3)
    return t


def random_translate_source(X, R, t, lower_bound=-0.5, upper_bound=0.5):
    """Randomly translate the source point cloud and update translation vector.

    Args:
        X: The matrix of points for the source point cloud.
        R: The true rotation matrix.
        t: The true translation vector.
        lower_bound: The minimum value for each axis of the translation vector.
        upper_bound: The maximum value for each axis of the translation vector.

    Returns:
        A tuple (X, R, t) where X is the new matrix of points for the source
        point cloud, R is the new true rotation matrix and t is the new true
        translation vector.
    """
    # If ``x, y, c`` are three-dimensional vectors,
    # ``R`` is a rotation matrix and ``y = R @ x + t``,
    # then ``y = R @ (x + c) + (t - R @ c)``.
    c = get_random_translation_vector(lower_bound, upper_bound)
    X = X + c
    t = t - np.matmul(R, c)
    return X, t


def process_and_save_modelnet40_sampled(
    base_path,
    save_path,
    split,
    categories,
    number_of_points,
    same_sample,
    crop,
    crop_number_of_points_left,
    jitter,
    rotate,
    translate,
):
    """Process ModelNet40 dataset categories and save samples to a HDF5 file.

    Output is written incrementally to save_path under groups named
    "sample_000000", "sample_000001", etc., each with datasets "X", "Y", "R",
    "t", "Xo", and "Yo".

    Args:
        base_path: The path to the ModelNet40 data.
        save_path: The path where the HDF5 file will be saved.
        split: The dataset split to process.
        categories: The list of categories to process.
        number_of_points: The number of points to sample for both point clouds.
        same_sample: If True, the same sample is used for both point clouds.
        crop: If True, both point clouds are independently cropped after
            sampling.
        crop_number_of_points_left: The number of points to retain in each
            point cloud after cropping.
        jitter: If True, random noise is independently added to both point
            clouds.
        rotate: If True, the source point cloud is randomly rotated.
        translate: If True, the source point cloud is randomly translated.
    """
    sample_index = 0
    for category in tqdm(categories):
        category_path = os.path.join(base_path, category)
        test_path = os.path.join(category_path, "test")
        test_mesh_paths = glob.glob(os.path.join(test_path, "*.off"))
        test_mesh_paths.sort(key=lambda x: int(x[-8:-4]))
        train_path = os.path.join(category_path, "train")
        train_mesh_paths = glob.glob(os.path.join(train_path, "*.off"))
        train_mesh_paths.sort(key=lambda x: int(x[-8:-4]))
        if split == "all":
            mesh_paths = test_mesh_paths + train_mesh_paths
        elif split == "test":
            mesh_paths = test_mesh_paths
        else:
            mesh_paths = train_mesh_paths
        for mesh_path in mesh_paths:
            mend_modelnet40_off_file(mesh_path)
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            np.random.seed(sample_index)
            X, Y, R, t, Xo, Yo = random_sample_and_crop(
                mesh,
                number_of_points,
                same_sample,
                crop,
                crop_number_of_points_left,
            )
            if jitter:
                X = random_jitter(X)
                Y = random_jitter(Y)
            if rotate:
                X, R = random_rotate_source(X, R)
            if translate:
                X, t = random_translate_source(X, R, t)
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            R = R.astype(np.float32)
            t = t.astype(np.float32)
            Xo = Xo.astype(np.bool)
            Yo = Yo.astype(np.bool)
            s = f"sample_{sample_index:06d}"
            with h5py.File(save_path, "x" if sample_index == 0 else "r+") as f:
                f.create_dataset(f"{s}/X", dtype=np.float32, data=X)
                f.create_dataset(f"{s}/Y", dtype=np.float32, data=Y)
                f.create_dataset(f"{s}/R", dtype=np.float32, data=R)
                f.create_dataset(f"{s}/t", dtype=np.float32, data=t)
                f.create_dataset(f"{s}/Xo", dtype=np.float32, data=Xo)
                f.create_dataset(f"{s}/Yo", dtype=np.float32, data=Yo)
            sample_index += 1


def get_all_categories():
    """Return a list of all categories of ModelNet40.

    Returns:
        A list of all categories of ModelNet40.
    """
    all_categories = [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    ]
    return all_categories


def get_args():
    """Parse command-line arguments for processing the ModelNet40 dataset.

    This function defines input and output paths, the selected dataset
    categories, and key processing parameters. It also validates that the
    output file does not already exist.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    default_base_path = os.path.join("datasets", "modelnet40")
    default_save_path = os.path.join("datasets", "processed", "modelnet40.hdf5")
    all_categories = get_all_categories()
    splits = ["all", "test", "train"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=default_base_path)
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--split", type=str, choices=splits, default="all")
    parser.add_argument(
        "--categories",
        choices=all_categories,
        nargs="*",
        default=all_categories,
    )
    parser.add_argument("--number_of_points", type=int, default=1024)
    parser.add_argument("--same_sample", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--crop_number_of_points_left", type=int, default=717)
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        raise FileExistsError(f"File '{args.save_path}' already exists.")
    if args.crop and args.crop_number_of_points_left > args.number_of_points:
        raise ValueError("--crop_number_of_points_left > --number_of_points")
    return args


def main():
    """Run the processing pipeline for the ModelNet40 dataset categories.

    Parse command-line arguments, process the selected ModelNet40 dataset
    categories, and save samples to a HDF5 file.
    """
    args = get_args()
    print(f"Processing ModelNet40 categories, saving to {args.save_path}")
    process_and_save_modelnet40_sampled(
        args.base_path,
        args.save_path,
        args.split,
        args.categories,
        args.number_of_points,
        args.same_sample,
        args.crop,
        args.crop_number_of_points_left,
        args.jitter,
        args.rotate,
        args.translate,
    )


if __name__ == "__main__":
    main()
