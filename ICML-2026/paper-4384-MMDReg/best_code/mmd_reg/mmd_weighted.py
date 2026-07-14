import functools
import jax
import jax.numpy as jnp
import open3d as o3d
from jaxopt import LevenbergMarquardt


def skew(x):
    """Compute the skew-symmetric cross-product matrix of vector `x`.

    Args:
        x: Three-dimensional vector.

    Returns:
        The skew-symmetric cross-product matrix of vector `x`.
    """
    K = jnp.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])
    return K


def expm_skew(x):
    """Compute the rotation matrix from the matrix exponential of `skew(x)`.

    Args:
        x: Three-dimensional vector.

    Returns:
        The rotation matrix from the matrix exponential of `skew(x)`.
    """
    return jax.scipy.linalg.expm(skew(x))


def transform(params, X):
    """Transform the points in matrix `X` using `params`.

    Args:
        params: Six-dimensional vector that is the concatenation of a rotation
            vector and a translation vector.
        X: Matrix of points for a point cloud.

    Returns:
        The matrix of transformed points.
    """
    rotation_vector = params[:3]
    translation_vector = params[3:]
    return jnp.matmul(X, expm_skew(rotation_vector).T) + translation_vector


def random_feature_map(x, W):
    """Compute the random feature map of vector `x`.

    Args:
        x: Three-dimensional vector.
        W: Matrix of random weights.

    Returns:
        The random feature map of vector `x`.
    """
    Wx = jnp.matmul(W, x).reshape((-1, 1))
    v = jnp.concatenate([jnp.sin(Wx), jnp.cos(Wx)], axis=1).reshape((-1,))
    D = jnp.size(W, axis=0)
    return jnp.sqrt(1.0 / D) * v


def approx_maximum_mean_discrepancy_residual(W, X, Y, X_weights, Y_weights):
    """Compute the residual values of an approximated maximum mean discrepancy.

    Args:
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        X_weights: Column vector of weights for the weighted sum involving `X`.
        Y_weights: Column vector of weights for the weighted sum involving `Y`.

    Returns:
        The residual values of an approximated maximum mean discrepancy.
    """
    vmap_rfm = jax.vmap(random_feature_map, (0, None), 0)
    X_weights = X_weights.reshape((-1, 1))
    Y_weights = Y_weights.reshape((-1, 1))
    X_weighted_rfm = X_weights * vmap_rfm(X, W)  # (N, 1) x (N, 2 * D)
    Y_weighted_rfm = Y_weights * vmap_rfm(Y, W)  # (M, 1) x (M, 2 * D)
    return jnp.mean(X_weighted_rfm, axis=0) - jnp.mean(Y_weighted_rfm, axis=0)


def inner_objective_residual(inner_params, W, X, Y, X_weights, Y_weights):
    """Compute the residual values of the inner objective function.

    Args:
        inner_params: Six-dimensional vector that is the concatenation of a
            rotation vector and a translation vector.
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        X_weights: Column vector of weights for the weighted sum involving `X`.
        Y_weights: Column vector of weights for the weighted sum involving `Y`.

    Returns:
        The residual values of the inner objective function.
    """
    X = transform(inner_params, X)
    return approx_maximum_mean_discrepancy_residual(
        W, X, Y, X_weights, Y_weights
    )


def inner_objective_solution(
    init_inner_params, W, X, Y, X_weights, Y_weights, **kwargs
):
    """Run the Levenberg Marquardt solver on the inner objective function.

    Args:
        init_inner_params: For the LM solver, the initial six-dimensional
            vector that is the concatenation of a rotation vector and a
            translation vector.
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        X_weights: Column vector of weights for the weighted sum involving `X`.
        Y_weights: Column vector of weights for the weighted sum involving `Y`.
        **kwargs: Additional keyword arguments passed to the LM solver.

    Returns:
        Six-dimensional vector returned by the LM solver.
    """
    inner_solver = LevenbergMarquardt(inner_objective_residual, **kwargs)
    inner_params = inner_solver.run(
        init_inner_params, W, X, Y, X_weights, Y_weights
    ).params
    return inner_params


def batched_inner_objective_solutions(
    init_inner_params, Ws, Xs, Ys, Xs_weights, Ys_weights, **kwargs
):
    """In parallel run Levenberg Marquardt solvers on inner objective functions.

    Args:
        init_inner_params: For the LM solvers, batch of initial
            six-dimensional vectors that are each the concatenation of a
            rotation vector and a translation vector.
        Ws: Batch of matrices of random weights for the random feature maps.
        Xs: Batch of matrices of points for the source point clouds.
        Ys: Batch of matrices of points for the target point clouds.
        Xs_weights: Batch of column vectors of weights for the weighted sums
            involving `Xs`.
        Ys_weights: Batch of column vectors of weights for the weighted sums
            involving `Ys`.
        **kwargs: Additional keyword arguments passed to the LM solvers.

    Returns:
        Batch of six-dimensional vectors returned by the LM solvers.
    """
    inner_solver = functools.partial(inner_objective_solution, **kwargs)
    inner_solver = jax.vmap(inner_solver)
    inner_params = inner_solver(
        init_inner_params, Ws, Xs, Ys, Xs_weights, Ys_weights
    )
    return inner_params
    # mmd_rotation_vectors = inner_params[:, :3]
    # mmd_translation_vectors = inner_params[:, 3:]
    # return mmd_rotation_vectors, mmd_translation_vectors


if __name__ == "__main__":  # Example point cloud registration.
    # Generate ground truth rotation vector and translation vector.
    key = jax.random.key(0)
    key, r_key, t_key = jax.random.split(key, num=3)
    true_r = jax.random.normal(r_key, (3,))
    true_r = true_r / jnp.linalg.norm(true_r)
    true_r = true_r * jnp.deg2rad(25.0)
    true_t = jax.random.uniform(t_key, (3,), minval=-0.5, maxval=0.5)

    # Generate source point cloud and target point cloud.
    num_points = 10000
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    XY = mesh.sample_points_poisson_disk(2 * num_points, init_factor=5).points
    XY = jnp.array(XY)
    XY = XY / jnp.max(jnp.linalg.norm(XY, axis=1))
    X = XY[:num_points, :]
    Y = XY[num_points:, :]
    Y = jnp.matmul(Y, expm_skew(true_r).T) + true_t

    # Visualize source point cloud and target point cloud.
    X_pcd = o3d.geometry.PointCloud()
    X_pcd.points = o3d.utility.Vector3dVector(X)
    X_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    Y_pcd = o3d.geometry.PointCloud()
    Y_pcd.points = o3d.utility.Vector3dVector(Y)
    o3d.visualization.draw([X_pcd, Y_pcd])

    # Point cloud registration solution using MMD.
    key, W_key = jax.random.split(key, num=2)
    W = jax.random.laplace(W_key, (128, 3)) / 0.75
    init_inner_params = jnp.zeros((6,))
    X_weights = jnp.ones((num_points, 1))
    Y_weights = jnp.ones((num_points, 1))
    inner_params = inner_objective_solution(
        init_inner_params,
        W,
        X,
        Y,
        X_weights,
        Y_weights,
        tol=2e-5,
        maxiter=100,
        solver="cholesky",
        materialize_jac=True,
        damping_parameter=1.0,
    )
    pred_r = inner_params[:3]  # Rotation vector.
    pred_t = inner_params[3:]  # Translation vector.

    # Visualize solution.
    X = jnp.matmul(X, expm_skew(pred_r).T) + pred_t
    X_pcd = o3d.geometry.PointCloud()
    X_pcd.points = o3d.utility.Vector3dVector(X)
    X_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    Y_pcd = o3d.geometry.PointCloud()
    Y_pcd.points = o3d.utility.Vector3dVector(Y)
    o3d.visualization.draw([X_pcd, Y_pcd])
