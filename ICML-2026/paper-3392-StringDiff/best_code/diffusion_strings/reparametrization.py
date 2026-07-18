import torch

from .rotate_align import align_translations_and_rotations_to_index
from .so3 import (
    generator_to_rotmat,
    rotmat_to_generator,
    rotmat_to_rotvec,
    rotmat_to_theta,
    rotvec_to_rotmat,
)


def _segment_lengths(segments):
    """Return Euclidean lengths after flattening all non-string dimensions."""
    if segments.shape[0] == 0:
        return torch.empty(0, dtype=segments.dtype, device=segments.device)
    return torch.linalg.vector_norm(
        segments.reshape(segments.shape[0], -1), ord=2, dim=1
    )


def _merge_close_points(string, segment_lengths, merge_tol):
    """Merge contiguous groups whose cumulative neighboring-segment length is below threshold."""
    total_length = torch.sum(segment_lengths)
    if merge_tol is None or string.shape[0] <= 1:
        return string

    threshold = merge_tol * total_length
    if threshold <= 0:
        return string

    assert string.shape[0] == segment_lengths.shape[0] + 1

    n_points = string.shape[0]
    merged_points = []
    i = 0
    while i < n_points:
        if i == n_points - 1:
            merged_points.append(string[i])
            break

        cum_distance = torch.zeros(
            (), dtype=segment_lengths.dtype, device=segment_lengths.device
        )
        j = i
        while j < n_points - 1 and (cum_distance + segment_lengths[j]) <= threshold:
            cum_distance = cum_distance + segment_lengths[j]
            j += 1

        if j > i:
            # Collapse points i..j (inclusive) into their average.
            merged_points.append(torch.mean(string[i : j + 1], dim=0))
            i = j + 1
        else:
            merged_points.append(string[i])
            i += 1

    return torch.stack(merged_points, dim=0)


def _constant_string(string, n_new):
    """Expand a degenerate string to the requested number of points."""
    return string[0].unsqueeze(0).expand(n_new, *string.shape[1:]).clone()


def _arc_length_positions(lengths, n_new):
    """Return old and new cumulative arc-length coordinates."""
    cumulative = torch.cat(
        [
            torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
            torch.cumsum(lengths, dim=0),
        ],
        dim=0,
    )
    new_cumulative = (
        torch.linspace(0, 1, n_new, dtype=lengths.dtype, device=lengths.device)
        * cumulative[-1]
    )
    return cumulative, new_cumulative


def _segment_indices(cumulative, new_cumulative):
    """Find interpolation segment indices for new arc-length coordinates."""
    indices = torch.empty_like(new_cumulative, dtype=torch.long)
    torch.searchsorted(cumulative, new_cumulative, out=indices)
    indices = indices - 1
    return torch.clamp(indices, 0, cumulative.shape[0] - 2)


def uniform_string_repametrize_rn_linear(string, n_new, merge_tol=1e-6):
    """Resample a Euclidean string to uniformly spaced arc-length points."""
    segments = torch.diff(string, dim=0)
    segment_lengths = _segment_lengths(segments)
    total_length = torch.sum(segment_lengths)

    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    string = _merge_close_points(string, segment_lengths, merge_tol)
    segments = torch.diff(string, dim=0)
    segment_lengths = _segment_lengths(segments)
    total_length = torch.sum(segment_lengths)
    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    n_extradims = segments.ndim - 1
    cumulative, new_cumulative = _arc_length_positions(segment_lengths, n_new)
    indices = _segment_indices(cumulative, new_cumulative)
    relative_progression = (new_cumulative - cumulative[indices]) / segment_lengths[
        indices
    ]
    return string[indices] + segments[indices] * relative_progression.view(
        -1, *[1] * n_extradims
    )


def uniform_string_repametrize_so3_linear(string, n_new, merge_tol=1e-6):
    """Resample an SO(3) string to approximately uniform rotational spacing."""
    assert string.shape[-2:] == (3, 3)
    segments = torch.matmul(string[:-1].transpose(-1, -2), string[1:])
    segment_lengths = rotmat_to_theta(segments)
    total_length = torch.sum(segment_lengths)

    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    string = _merge_close_points(string, segment_lengths, merge_tol)

    relative_segments = torch.matmul(string[:-1].transpose(-1, -2), string[1:])
    generators, segment_lengths = rotmat_to_generator(relative_segments)
    total_length = torch.sum(segment_lengths)
    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    cumulative, new_cumulative = _arc_length_positions(segment_lengths, n_new)
    indices = _segment_indices(cumulative, new_cumulative)
    relative_progression = (new_cumulative - cumulative[indices]) / segment_lengths[
        indices
    ]
    delta = generator_to_rotmat(
        relative_progression * segment_lengths[indices], generators[indices]
    )
    return torch.matmul(string[indices], delta)


def uniform_string_repametrize_se3_linear(
    translations, rotations, n_new, corr_coeff=1.0, merge_tol=1e-6
):
    """Placeholder for uniform reparametrization of coupled translations and rotations."""
    assert translations.shape[-1] == 3
    assert rotations.shape[-2:] == (3, 3)
    assert translations.shape[:-1] == rotations.shape[:-2]
    first_translation = translations[0].clone()
    last_translation = translations[-1].clone()
    first_rotation = rotations[0].clone()
    last_rotation = rotations[-1].clone()
    trans_segments = torch.diff(translations, dim=0)
    trans_lengths = _segment_lengths(trans_segments)
    rot_segments = torch.matmul(rotations[:-1].transpose(-1, -2), rotations[1:])
    rot_lengths = _segment_lengths(rotmat_to_theta(rot_segments).unsqueeze(-1))
    segment_lengths = trans_lengths + corr_coeff * rot_lengths
    total_length = torch.sum(segment_lengths)

    if (
        translations.shape[0] == 1
        or total_length <= torch.finfo(translations.dtype).eps
    ):
        return (
            _constant_string(translations, n_new),
            _constant_string(rotations, n_new),
        )

    translations = _merge_close_points(translations, segment_lengths, merge_tol)
    rotation_vectors = rotmat_to_rotvec(rotations)
    rotation_vectors = _merge_close_points(rotation_vectors, segment_lengths, merge_tol)
    rotations = rotvec_to_rotmat(rotation_vectors)

    trans_segments = torch.diff(translations, dim=0)
    trans_lengths = _segment_lengths(trans_segments)
    rot_segments = torch.matmul(rotations[:-1].transpose(-1, -2), rotations[1:])
    rot_generators, rot_thetas = rotmat_to_generator(rot_segments)
    rot_lengths = _segment_lengths(rot_thetas.unsqueeze(-1))
    segment_lengths = trans_lengths + corr_coeff * rot_lengths
    total_length = torch.sum(segment_lengths)

    if (
        translations.shape[0] == 1
        or total_length <= torch.finfo(translations.dtype).eps
    ):
        return _constant_string(translations, n_new), _constant_string(rotations, n_new)

    cumulative, new_cumulative = _arc_length_positions(segment_lengths, n_new)
    indices = _segment_indices(cumulative, new_cumulative)
    relative_progression = (new_cumulative - cumulative[indices]) / segment_lengths[
        indices
    ]
    theta_delta = relative_progression * rot_lengths[indices]
    theta_delta = theta_delta.view(-1, *[1] * (rot_generators[indices].ndim - 3))
    delta = generator_to_rotmat(theta_delta, rot_generators[indices])

    new_translations = (
        translations[indices]
        + relative_progression.view(-1, *[1] * (translations.ndim - 1))
        * trans_segments[indices]
    )
    new_rotations = torch.matmul(rotations[indices], delta)
    new_translations[0] = first_translation
    new_translations[-1] = last_translation
    new_rotations[0] = first_rotation
    new_rotations[-1] = last_rotation
    return new_translations, new_rotations


def reparaametrize_kabsh_se3(
    translations,
    rotations,
    n_new,
    index=0,
    corr_coeff=1.0,
    merge_tol=1e-6,
):
    """Reparametrize protein shape and global Kabsch pose, then restore global frames.

    The Kabsch transform uses the row-vector convention
    ``aligned = translations @ global_rotation + global_translation``.
    Protein configurations are first reparametrized in the aligned frame. The
    saved global rotations and translations are reparametrized as a second SE(3)
    string, and their inverses restore each new protein image to its global pose.
    """
    aligned_t, aligned_r, global_r, global_t = (
        align_translations_and_rotations_to_index(
            translations,
            rotations,
            index=index,
            return_transform=True,
        )
    )

    reparam_aligned_t, reparam_aligned_r = uniform_string_repametrize_se3_linear(
        aligned_t,
        aligned_r,
        n_new=n_new,
        corr_coeff=corr_coeff,
        merge_tol=merge_tol,
    )
    reparam_global_t, reparam_global_r = uniform_string_repametrize_se3_linear(
        global_t.squeeze(-2),
        global_r,
        n_new=n_new,
        corr_coeff=corr_coeff,
        merge_tol=merge_tol,
    )

    global_r_inv = reparam_global_r.transpose(-1, -2)
    restored_t = torch.matmul(
        reparam_aligned_t - reparam_global_t.unsqueeze(-2),
        global_r_inv,
    )
    restored_r = torch.matmul(
        reparam_global_r.unsqueeze(-3),
        torch.matmul(reparam_aligned_r, global_r_inv.unsqueeze(-3)),
    )
    return restored_t, restored_r


def uniform_string_repametrize_rn_cubic(string, n_new, merge_tol=1e-6):
    """Resample a Euclidean string with cubic Hermite interpolation by arc length."""
    segments = torch.diff(string, dim=0)
    segment_lengths = _segment_lengths(segments)
    total_length = torch.sum(segment_lengths)

    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    string = _merge_close_points(string, segment_lengths, merge_tol)
    segments = torch.diff(string, dim=0)
    segment_lengths = _segment_lengths(segments)
    total_length = torch.sum(segment_lengths)
    if string.shape[0] == 1 or total_length <= torch.finfo(string.dtype).eps:
        return _constant_string(string, n_new)

    n_extradims = segments.ndim - 1

    if string.shape[0] == 2:
        return uniform_string_repametrize_rn_linear(string, n_new, merge_tol=None)

    tangents = torch.empty_like(string)
    tangent_shape = (-1, *[1] * n_extradims)
    cumulative, new_cumulative = _arc_length_positions(segment_lengths, n_new)
    tangents[0] = (string[1] - string[0]) / (cumulative[1] - cumulative[0])
    tangents[-1] = (string[-1] - string[-2]) / (cumulative[-1] - cumulative[-2])
    tangents[1:-1] = (string[2:] - string[:-2]) / (
        cumulative[2:] - cumulative[:-2]
    ).view(*tangent_shape)

    indices = _segment_indices(cumulative, new_cumulative)

    interval_lengths = cumulative[indices + 1] - cumulative[indices]
    relative_progression = (new_cumulative - cumulative[indices]) / interval_lengths
    t = relative_progression.view(-1, *[1] * n_extradims)
    h = interval_lengths.view(-1, *[1] * n_extradims)
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return (
        h00 * string[indices]
        + h10 * h * tangents[indices]
        + h01 * string[indices + 1]
        + h11 * h * tangents[indices + 1]
    )


uniform_string_repametrize_rn = uniform_string_repametrize_rn_linear
uniform_string_repametrize_so3 = uniform_string_repametrize_so3_linear
uniform_string_repametrize_se3 = uniform_string_repametrize_se3_linear
