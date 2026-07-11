import jax
import jax.numpy as jnp

ONE_OVER_2SQRT2 = 1.0 / (2 * jnp.sqrt(2))


@jax.jit
def rotmat_geodesic_distance(R1, R2, clamping=1.0):
    r"""
    Returns the angular distance alpha between a pair of rotation matrices.
    Based on the equality :math:`|R_2 - R_1|_F = 2 \sqrt{2} sin(alpha/2)`.
    Args:
        R1, R2 (...x3x3 array): batch of 3x3 rotation matrices.
        clamping: clamping value applied to the input of :func:`jax.numpy.arcsin()`.
                Use 1.0 to ensure valid angular distances.
                Use a value strictly smaller than 1.0 to ensure finite gradients.
    Returns:
        batch of angles in radians (... array).
    """
    return 2.0 * jnp.arcsin(
        jnp.minimum(jnp.linalg.norm(R2 - R1, axis=(-2, -1)) * ONE_OVER_2SQRT2, clamping)
    )


@jax.jit
def symmetric_orthogonalization(x):
    """Maps 9D input vector onto SO(3) via symmetric orthogonalization.

    Args:
        x: Input tensor of shape (9,) - single 9D vector

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Reshape to 3x3
    m = x.reshape(3, 3)

    # SVD
    u, _, vh = jnp.linalg.svd(m, full_matrices=False)

    # Calculate determinant
    det = jnp.linalg.det(u @ vh)

    # Ensure proper rotation (det=1)
    vh_adjusted = vh.at[2].multiply(det)

    # Compute rotation matrix
    r = u @ vh_adjusted

    return r


@jax.jit
def gramschmidt_to_rotmat(inp):
    """Convert input to rotation matrix via Gram-Schmidt process.

    Args:
        inp: Input tensor - either (..., 6) for 6D representation or (..., 3, 3) for matrices

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Handle 6D representation: reshape to (..., 3, 2)
    if inp.shape[-1] == 6:
        # Reshape 6D vector to 3x2 matrix
        m = inp.reshape(inp.shape[:-1] + (3, 2))
    else:
        # Extract first two columns from matrix
        m = inp[..., :2]

    # Determine if we have batch dimensions
    if len(m.shape) == 2:
        # Single sample: (3, 2) -> call single function
        return special_gramschmidt_single(m)
    else:
        # Batch: (..., 3, 2) -> call vmapped function
        # Need to vmap over all but the last two dimensions
        num_batch_dims = len(m.shape) - 2
        if num_batch_dims == 1:
            return special_gramschmidt(m)
        else:
            # Multiple batch dimensions - apply vmap recursively
            vmap_fn = special_gramschmidt_single
            for _ in range(num_batch_dims):
                vmap_fn = jax.vmap(vmap_fn)
            return vmap_fn(m)


@jax.jit
def special_gramschmidt_single(M, epsilon=1e-10):
    """Returns a single 3x3 rotation matrix via Gram-Schmidt orthonormalization.

    Args:
        M: Input matrix of shape (3, 2)

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Extract first two columns
    x = M[:, 0]  # Shape: (3,)
    y = M[:, 1]  # Shape: (3,)

    # Normalize x
    x_norm = jnp.maximum(jnp.linalg.norm(x), epsilon)
    x = x / x_norm

    # Make y orthogonal to x and normalize
    y = y - jnp.dot(x, y) * x
    y_norm = jnp.maximum(jnp.linalg.norm(y), epsilon)
    y = y / y_norm

    # Compute third basis vector using cross product
    z = jnp.cross(x, y)  # Works on 3D vectors

    # Stack to form rotation matrix
    R = jnp.stack([x, y, z], axis=1)  # Shape: (3, 3)

    return R


# Vectorized version for batched inputs
special_gramschmidt = jax.vmap(special_gramschmidt_single)


@jax.jit
def map_to_lie_algebra(v: jnp.ndarray) -> jnp.ndarray:
    r"""Hat operator (\(\mathbb R^3 \to \mathfrak{so}(3)\)).

    Args:
        v: Array of shape (..., 3) representing rotation vectors.

    Returns:
        Skew-symmetric matrices of shape (..., 3, 3).
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    zeros = jnp.zeros_like(vx)

    row0 = jnp.stack([zeros, -vz, vy], axis=-1)
    row1 = jnp.stack([vz, zeros, -vx], axis=-1)
    row2 = jnp.stack([-vy, vx, zeros], axis=-1)

    return jnp.stack([row0, row1, row2], axis=-2)


@jax.jit
def map_to_lie_vector(X: jnp.ndarray) -> jnp.ndarray:
    r"""Vee operator (\(\mathfrak{so}(3) \to \mathbb R^3\)).

    Args:
        X: Skew-symmetric matrices of shape (..., 3, 3).

    Returns:
        Corresponding rotation vectors of shape (..., 3).
    """
    return jnp.stack([-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]], axis=-1)


@jax.jit
def rodrigues(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    r"""Exponential map \(\exp: \mathbb R^3 \to SO(3)\) via Rodrigues formula.

    Args:
        v: Rotation vectors of shape (..., 3).
        eps: Numerical threshold for small angles.

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    theta = jnp.linalg.norm(v, axis=-1, keepdims=True)  # (..., 1)

    # Broadcast helpers
    I = jnp.eye(3, dtype=v.dtype)
    I = jnp.broadcast_to(I, v.shape[:-1] + (3, 3))

    K = map_to_lie_algebra(v)

    # Series expansions for small angles
    theta2 = theta**2
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    A = jnp.where(
        theta < eps, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, sin_theta / theta
    )
    B = jnp.where(
        theta < eps,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - cos_theta) / theta2,
    )

    A = A[..., None]  # (..., 1, 1)
    B = B[..., None]

    R = I + A * K + B * (K @ K)
    return R


@jax.jit
def log_map(R: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    r"""Logarithm map \(\log: SO(3) \to \mathbb R^3\).

    Args:
        R: Rotation matrices of shape (..., 3, 3).
        eps: Numerical threshold.

    Returns:
        Rotation vectors of shape (..., 3).
    """
    trace = jnp.trace(R, axis1=-2, axis2=-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)[..., None]  # (..., 1)

    # For theta ~ 0 use series expansion (theta/(2*sin(theta)) -> 0.5 + ...)
    sin_theta = jnp.sin(theta)
    factor = jnp.where(theta < eps, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta))
    factor = factor[..., None]  # (..., 1, 1)

    skew = (R - jnp.swapaxes(R, -1, -2)) * factor
    return map_to_lie_vector(skew)


@jax.jit
def ddexp_so3(x: jnp.ndarray, z: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Derivative of the exponential map on SO(3).

    This computes the derivative of the exponential map dexp_x(z), which is needed
    for second-order dynamics using the derivative of the exponential map.

    Args:
        x: Rotation vector of shape (..., 3)
        z: Direction vector of shape (..., 3)
        eps: Numerical threshold for small angles

    Returns:
        Derivative matrix of shape (..., 3, 3)
    """
    hatx = map_to_lie_algebra(x)  # (..., 3, 3)
    hatz = map_to_lie_algebra(z)  # (..., 3, 3)

    phi = jnp.linalg.norm(x, axis=-1, keepdims=True)[..., None]  # (..., 1, 1)

    # Handle small angles with series expansions
    half_phi = phi / 2.0
    sin_half_phi = jnp.sin(half_phi)
    sin_phi = jnp.sin(phi)

    # beta = sin(phi/2)^2 / (phi/2)^2
    beta = jnp.where(
        phi < eps,
        1.0 - (phi**2) / 12.0 + (phi**4) / 240.0,  # Series expansion
        (sin_half_phi / half_phi) ** 2,
    )

    # alpha = sin(phi) / phi
    alpha = jnp.where(
        phi < eps,
        1.0 - (phi**2) / 6.0 + (phi**4) / 120.0,  # Series expansion
        sin_phi / phi,
    )

    # Compute dot product x·z
    x_dot_z = jnp.sum(x * z, axis=-1, keepdims=True)[..., None]  # (..., 1, 1)

    # Terms in the derivative formula
    term1 = 0.5 * beta * hatz

    # Handle division by phi^2 carefully
    phi_sq = phi**2
    coeff2 = jnp.where(
        phi < eps,
        1.0 / 6.0 - (phi**2) / 120.0,  # Series expansion of (1-alpha)/phi^2
        (1.0 - alpha) / phi_sq,
    )
    term2 = coeff2 * (hatx @ hatz + hatz @ hatx)

    coeff3 = jnp.where(
        phi < eps,
        -1.0 / 12.0 + (phi**2) / 240.0,  # Series expansion of (alpha-beta)/phi^2
        (alpha - beta) / phi_sq,
    )
    term3 = coeff3 * x_dot_z * hatx

    coeff4 = jnp.where(
        phi < eps,
        -1.0 / 24.0 + (phi**2) / 720.0,  # Series expansion
        (beta / 2.0 - 3.0 / phi_sq * (1.0 - alpha)) / phi_sq,
    )
    term4 = coeff4 * x_dot_z * (hatx @ hatx)

    return term1 + term2 + term3 + term4
