import h5py
import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import probreg
from mmd_reg.mmd_unweighted import expm_skew, inner_objective_solution


def generate_orthogonal_frequencies(key, D, d, length_scale):
    """Generate orthogonal random Fourier feature frequencies.

    Uses structured orthogonal matrices (QR of random Gaussian) to reduce
    kernel approximation variance compared to i.i.d. Gaussian sampling.
    Reference: Yu et al., "Orthogonal Random Features", NeurIPS 2016.

    Args:
        key: JAX PRNG key.
        D: Number of random Fourier features.
        d: Input dimensionality (3 for 3D point clouds).
        length_scale: Kernel length scale (bandwidth).

    Returns:
        Frequency matrix W of shape (D, d), with approximately orthonormal
        rows scaled by 1/length_scale.
    """
    # Generate a random orthogonal matrix of size max(D, d) x max(D, d)
    M = max(D, d)
    key, gauss_key = jax.random.split(key)
    G = jax.random.normal(gauss_key, (M, M))
    Q, _ = jnp.linalg.qr(G)
    # Take the first D rows and first d columns
    # Each row of Q[:D, :d] has expected squared norm = d/D.
    # For d-dim standard Gaussian, expected squared norm = d.
    # Multiply by sqrt(D) to correct: E[||row||^2] = (d/D) * D = d.
    W = Q[:D, :d] * jnp.sqrt(float(D))
    # Apply random signs for radial symmetry (preserves Gaussian kernel property)
    key, sign_key = jax.random.split(key)
    signs = jax.random.rademacher(sign_key, (D, 1))
    W = W * signs
    # Scale by inverse length scale
    W = W / length_scale
    return W


def load_data_to_list(data_path):
    """Load samples from a HDF5 file into a Python list of tuples.

    Each item in the list is a tuple (X, Y, R, t), where X is the matrix of
    points for the source point cloud, Y is the matrix of points for the target
    point cloud, R is the true rotation matrix from X to Y, and t is the true
    translation vector from X to Y.

    The file is expected to contain groups named "sample_000000",
    "sample_000001", etc., each with datasets "X", "Y", "R", and "t".

    Args:
        data_path: Path to a HDF5 file containing the processed data.

    Returns:
        A list of tuples (X, Y, R, t), one per sample.
    """
    with h5py.File(data_path, "r") as data_file:
        num_samples = len(data_file)
    data_list = []
    for sample_index in range(num_samples):
        sample_string = f"sample_{sample_index:06d}"
        with h5py.File(data_path, "r") as data_file:
            X = np.array(data_file[f"{sample_string}/X"], dtype=np.float32)
            Y = np.array(data_file[f"{sample_string}/Y"], dtype=np.float32)
            R = np.array(data_file[f"{sample_string}/R"], dtype=np.float32)
            t = np.array(data_file[f"{sample_string}/t"], dtype=np.float32)
        data_list.append((X, Y, R, t))
    return data_list


def get_rotation_error(pred_rotation_matrix, true_rotation_matrix):
    """Compute the angular error in degrees between two rotation matrices.

    Args:
        pred_rotation_matrix: Predicted rotation matrix.
        true_rotation_matrix: Ground truth rotation matrix.

    Returns:
        The angular error in degrees.
    """
    K = np.sum(pred_rotation_matrix * true_rotation_matrix)
    error = np.clip(0.5 * (K - 1.0), -1.0, 1.0)
    error = np.arccos(error)
    error = np.rad2deg(error)
    return error


def get_translation_error(pred_translation_vector, true_translation_vector):
    """Compute the L2 (Euclidean) distance between two translation vectors.

    Args:
        pred_translation_vector: Predicted translation vector.
        true_translation_vector: Ground truth translation vector.

    Returns:
        The L2 (Euclidean) distance.
    """
    K = pred_translation_vector - true_translation_vector
    error = np.linalg.norm(K)
    return error


@jax.jit
def mmd_reg(Ws, X, Y):
    """Run MMD-Reg to predict the rigid transform from X to Y.

    Args:
        Ws: List of matrices of random weights for the random feature maps.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector.
    """
    inner_params = jnp.zeros((6,))
    for W in Ws:
        inner_params = inner_objective_solution(
            inner_params,
            W,
            X,
            Y,
            tol=2e-5,
            solver="cholesky",
            materialize_jac=True,
            maxiter=100,
            damping_parameter=1.0,
        )
    pred_r = inner_params[:3]
    pred_t = inner_params[3:]
    pred_R = expm_skew(pred_r)
    return pred_R, pred_t


@jax.jit
def mmd_reg_adaptive(Ws, X, Y):
    """Run MMD-Reg with adaptive two-pass damping per frequency scale.

    For each W, first runs LM with high damping (lambda=5.0) for robust
    convergence from identity initialization, then refines with low damping
    (lambda=0.1) for precise Gauss-Newton convergence near the optimum.

    This Fletcher-style gain-ratio schedule improves convergence by using
    gradient-descent-like behavior early and Newton-like behavior late.

    Args:
        Ws: List of matrices of random weights for the random feature maps.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector.
    """
    inner_params = jnp.zeros((6,))
    for W in Ws:
        # Coarse pass: high damping for robust wide-basin convergence
        inner_params = inner_objective_solution(
            inner_params,
            W,
            X,
            Y,
            tol=2e-5,
            solver="cholesky",
            materialize_jac=True,
            maxiter=100,
            damping_parameter=5.0,
        )
        # Fine pass: low damping for precise Gauss-Newton refinement
        inner_params = inner_objective_solution(
            inner_params,
            W,
            X,
            Y,
            tol=2e-5,
            solver="cholesky",
            materialize_jac=True,
            maxiter=100,
            damping_parameter=0.1,
        )
    pred_r = inner_params[:3]
    pred_t = inner_params[3:]
    pred_R = expm_skew(pred_r)
    return pred_R, pred_t


def eval_mmd_reg(Ws, X, Y, adaptive_damping=False):
    """Run MMD-Reg to predict the rigid transform from X to Y.

    This is a wrapper that converts inputs to JAX arrays, calls the jitted
    MMD-Reg implementation, and converts the predicted rotation matrix and
    translation vector back to NumPy arrays.

    Args:
        Ws: List of matrices of random weights for the random feature maps.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        adaptive_damping: If True, use two-pass coarse-fine damping schedule.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    Ws = [jnp.asarray(W) for W in Ws]
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    if adaptive_damping:
        pred_R, pred_t = mmd_reg_adaptive(Ws, X, Y)
    else:
        pred_R, pred_t = mmd_reg(Ws, X, Y)
    pred_R = np.asarray(pred_R)
    pred_t = np.asarray(pred_t)
    return pred_R, pred_t


def eval_icp_cpu(X, Y, max_corr_dist=0.75, use_point_to_plane=False):
    """Run ICP on CPU to predict the rigid transform from X to Y.

    This function uses Open3D's legacy (non-tensor) ICP implementation, which we
    found to be faster on CPU than the newer tensor-based implementation.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        max_corr_dist: Maximum correspondence distance.
        use_point_to_plane: Use point-to-plane instead of point-to-point.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    X_pcd = o3d.geometry.PointCloud()
    X_pcd.points = o3d.utility.Vector3dVector(X)
    Y_pcd = o3d.geometry.PointCloud()
    Y_pcd.points = o3d.utility.Vector3dVector(Y)
    if use_point_to_plane:
        Y_pcd.estimate_normals()
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ini = np.eye(4, dtype=np.float32)
    cri = o3d.pipelines.registration.ICPConvergenceCriteria()
    result = o3d.pipelines.registration.registration_icp(
        X_pcd, Y_pcd, max_corr_dist, ini, est, cri
    )
    pred_T = np.asarray(result.transformation, dtype=np.float32)
    pred_R = pred_T[:3, :3]
    pred_t = pred_T[:3, 3]
    return pred_R, pred_t


def eval_icp_gpu(X, Y, max_corr_dist=0.75, use_point_to_plane=False):
    """Run ICP on GPU to predict the rigid transform from X to Y.

    This function uses Open3D's tensor-based ICP implementation on "CUDA:0".

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        max_corr_dist: Maximum correspondence distance.
        use_point_to_plane: Use point-to-plane instead of point-to-point.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    dtype = o3d.core.float32
    device = o3d.core.Device("CUDA:0")
    X_pcd = o3d.t.geometry.PointCloud(device)
    X_pcd.point.positions = o3d.core.Tensor(X, dtype=dtype, device=device)
    Y_pcd = o3d.t.geometry.PointCloud(device)
    Y_pcd.point.positions = o3d.core.Tensor(Y, dtype=dtype, device=device)
    if use_point_to_plane:
        Y_pcd.estimate_normals()
        est = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    ini = o3d.core.Tensor(
        np.eye(4, dtype=np.float32), dtype=dtype, device=device
    )
    cri = o3d.t.pipelines.registration.ICPConvergenceCriteria()
    result = o3d.t.pipelines.registration.icp(
        X_pcd, Y_pcd, max_corr_dist, ini, est, cri
    )
    pred_T = np.asarray(result.transformation.cpu().numpy(), dtype=np.float32)
    pred_R = pred_T[:3, :3]
    pred_t = pred_T[:3, 3]
    return pred_R, pred_t


def eval_multi_scale_icp_gpu(
    X,
    Y,
    voxel_sizes=[0.35, 0.3, 0.25, 0.2, 0.15],
    max_corr_dists=[4.0, 2.0, 1.0, 0.5, 0.25],
    use_point_to_plane=False,
):
    """Run multi scale ICP on GPU to predict the rigid transform from X to Y.

    This function uses Open3D's tensor-based ICP implementation on "CUDA:0".

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        voxel_sizes: List of voxel sizes.
        max_corr_dists: List of maximum correspondence distances.
        use_point_to_plane: Use point-to-plane instead of point-to-point.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria()
        for _ in voxel_sizes
    ]
    voxel_sizes = o3d.utility.DoubleVector(voxel_sizes)
    max_corr_dists = o3d.utility.DoubleVector(max_corr_dists)
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    dtype = o3d.core.float32
    device = o3d.core.Device("CUDA:0")
    X_pcd = o3d.t.geometry.PointCloud(device)
    X_pcd.point.positions = o3d.core.Tensor(X, dtype=dtype, device=device)
    Y_pcd = o3d.t.geometry.PointCloud(device)
    Y_pcd.point.positions = o3d.core.Tensor(Y, dtype=dtype, device=device)
    if use_point_to_plane:
        Y_pcd.estimate_normals()
        est = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    ini = o3d.core.Tensor(
        np.eye(4, dtype=np.float32), dtype=dtype, device=device
    )
    result = o3d.t.pipelines.registration.multi_scale_icp(
        X_pcd, Y_pcd, voxel_sizes, criteria_list, max_corr_dists, ini, est
    )
    pred_T = np.asarray(result.transformation.cpu().numpy(), dtype=np.float32)
    pred_R = pred_T[:3, :3]
    pred_t = pred_T[:3, 3]
    return pred_R, pred_t


def eval_gicp(X, Y, max_corr_dist=0.75):
    """Run Generalized ICP on CPU to predict the rigid transform from X to Y.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        max_corr_dist: Maximum correspondence distance.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    X_pcd = o3d.geometry.PointCloud()
    X_pcd.points = o3d.utility.Vector3dVector(X)
    Y_pcd = o3d.geometry.PointCloud()
    Y_pcd.points = o3d.utility.Vector3dVector(Y)
    ini = np.eye(4, dtype=np.float32)
    est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    cri = o3d.pipelines.registration.ICPConvergenceCriteria()
    result = o3d.pipelines.registration.registration_generalized_icp(
        X_pcd, Y_pcd, max_corr_dist, ini, est, cri
    )
    pred_T = np.asarray(result.transformation, dtype=np.float32)
    pred_R = pred_T[:3, :3]
    pred_t = pred_T[:3, 3]
    return pred_R, pred_t


def eval_cpd(X, Y):
    """Run CPD on CPU to predict the rigid transform from X to Y.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    result = probreg.cpd.registration_cpd(X, Y, maxiter=10)
    pred_R = np.asarray(result.transformation.rot, dtype=np.float32)
    pred_t = np.asarray(result.transformation.t, dtype=np.float32)
    return pred_R, pred_t


def eval_filterreg(X, Y):
    """Run FilterReg on CPU to predict the rigid transform from X to Y.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    result = probreg.filterreg.registration_filterreg(X, Y)
    pred_R = np.asarray(result.transformation.rot, dtype=np.float32)
    pred_t = np.asarray(result.transformation.t, dtype=np.float32)
    return pred_R, pred_t


def eval_gmmreg(X, Y):
    """Run GMMReg on CPU to predict the rigid transform from X to Y.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    result = probreg.l2dist_regs.registration_gmmreg(X, Y, n_gmm_components=100)
    pred_R = np.asarray(result.rot, dtype=np.float32)
    pred_t = np.asarray(result.t, dtype=np.float32)
    return pred_R, pred_t


def eval_svr(X, Y):
    """Run SVR on CPU to predict the rigid transform from X to Y.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector as NumPy arrays.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    result = probreg.l2dist_regs.registration_svr(X, Y, nu=0.01)
    pred_R = np.asarray(result.rot, dtype=np.float32)
    pred_t = np.asarray(result.t, dtype=np.float32)
    return pred_R, pred_t
