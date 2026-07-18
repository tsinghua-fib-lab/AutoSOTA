import torch
from torch.nn.functional import normalize


def _skew_from_vector(vector):
    """Build skew-symmetric matrices from vectors with final dimension 3."""
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _vee_from_rotmat(rotmat):
    """Return twice the axial vector of the skew part of a rotation matrix."""
    return torch.stack(
        [
            rotmat[..., 2, 1] - rotmat[..., 1, 2],
            rotmat[..., 0, 2] - rotmat[..., 2, 0],
            rotmat[..., 1, 0] - rotmat[..., 0, 1],
        ],
        dim=-1,
    )


def _trace_cos_theta(rotmat):
    """Return cos(theta) inferred from the trace of a rotation matrix."""
    trace = torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(dim=-1)
    return torch.clamp((trace - 1) / 2, -1, 1)


def rotvec_to_rotmat(rotvec):
    """Convert rotation vectors in axis-angle form to rotation matrices."""
    theta = torch.linalg.norm(rotvec, dim=-1)
    safe_theta = torch.where(theta == 0, torch.ones_like(theta), theta)
    generator = _skew_from_vector(rotvec) / safe_theta[..., None, None]
    return generator_to_rotmat(theta, generator)


def rotmat_to_theta(rotmat):
    """Return the rotation angle of each rotation matrix."""
    vee = _vee_from_rotmat(rotmat)
    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)
    return torch.atan2(sin_theta, _trace_cos_theta(rotmat))


def rotmat_to_rotvec(rotmat):
    """Convert rotation matrices to axis-angle rotation vectors."""
    generator, theta = rotmat_to_generator(rotmat)
    axis = torch.stack(
        [
            generator[..., 2, 1],
            generator[..., 0, 2],
            generator[..., 1, 0],
        ],
        dim=-1,
    )
    return theta[..., None] * axis


def _near_pi_axis(rotmat, vee):
    """Estimate a stable rotation axis for rotations close to pi."""
    diag = torch.diagonal(rotmat, dim1=-2, dim2=-1)
    eps = torch.finfo(rotmat.dtype).eps

    x = torch.sqrt(torch.clamp((rotmat[..., 0, 0] + 1) / 2, min=eps))
    y_from_x = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / (4 * x)
    z_from_x = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / (4 * x)
    axis_x = torch.stack([x, y_from_x, z_from_x], dim=-1)

    y = torch.sqrt(torch.clamp((rotmat[..., 1, 1] + 1) / 2, min=eps))
    x_from_y = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / (4 * y)
    z_from_y = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / (4 * y)
    axis_y = torch.stack([x_from_y, y, z_from_y], dim=-1)

    z = torch.sqrt(torch.clamp((rotmat[..., 2, 2] + 1) / 2, min=eps))
    x_from_z = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / (4 * z)
    y_from_z = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / (4 * z)
    axis_z = torch.stack([x_from_z, y_from_z, z], dim=-1)

    max_diag = torch.argmax(diag, dim=-1)
    axis = torch.zeros_like(vee)
    axis = torch.where((max_diag == 0).unsqueeze(-1), axis_x, axis)
    axis = torch.where((max_diag == 1).unsqueeze(-1), axis_y, axis)
    axis = torch.where((max_diag == 2).unsqueeze(-1), axis_z, axis)
    axis = normalize(axis, dim=-1)

    sign = torch.where(torch.sum(axis * vee, dim=-1, keepdim=True) < 0, -1, 1)
    return axis * sign


def rotmat_to_generator(rotmat):
    """Return the unit skew generator and angle for each rotation matrix."""
    skew_part = 0.5 * (rotmat - rotmat.transpose(-1, -2))
    vee = _vee_from_rotmat(rotmat)
    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)
    theta = torch.atan2(sin_theta, _trace_cos_theta(rotmat))

    small = theta < 1e-6
    near_pi = torch.abs(torch.pi - theta) < 1e-4
    safe_sin_theta = torch.where(small | near_pi, torch.ones_like(sin_theta), sin_theta)
    generator = skew_part / safe_sin_theta[..., None, None]
    generator = torch.where(small[..., None, None], torch.zeros_like(generator), generator)

    pi_generator = _skew_from_vector(_near_pi_axis(rotmat, vee))
    generator = torch.where(near_pi[..., None, None], pi_generator, generator)
    return generator, theta


def generator_to_rotmat(theta, generator):
    """Exponentiate skew generators scaled by angles into rotation matrices."""
    if torch.is_tensor(theta):
        theta = theta.to(dtype=generator.dtype, device=generator.device)
    else:
        theta = torch.as_tensor(theta, dtype=generator.dtype, device=generator.device)

    theta2 = theta * theta
    small = torch.abs(theta) < 1e-6
    sin_theta = torch.where(
        small,
        theta - theta * theta2 / 6 + theta * theta2 * theta2 / 120,
        torch.sin(theta),
    )
    one_minus_cos = torch.where(
        small,
        theta2 / 2 - theta2 * theta2 / 24 + theta2 * theta2 * theta2 / 720,
        1 - torch.cos(theta),
    )

    eye = torch.eye(3, dtype=generator.dtype, device=generator.device)
    return (
        eye
        + sin_theta[..., None, None] * generator
        + one_minus_cos[..., None, None] * torch.matmul(generator, generator)
    )
