import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.reparametrization import (  # noqa: E402
    _merge_close_points,
    _segment_lengths,
    reparaametrize_kabsh_se3,
    rotvec_to_rotmat,
    uniform_string_repametrize_rn_linear,
    uniform_string_repametrize_rn_cubic,
    uniform_string_repametrize_se3_linear,
    uniform_string_repametrize_so3_linear,
)
from diffusion_strings.so3 import rotmat_to_theta  # noqa: E402


def test_uniform_string_reparametrize_rn_line_segment():
    string = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)

    result = uniform_string_repametrize_rn_linear(string, 3)

    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(result, expected)


def test_uniform_string_reparametrize_rn_constant_string():
    string = torch.tensor([[1.0, -2.0], [1.0, -2.0]], dtype=torch.float64)

    result = uniform_string_repametrize_rn_linear(string, 4)

    assert result.shape == (4, 2)
    assert torch.allclose(result, string[:1].expand(4, 2))


def test_uniform_string_reparametrize_rn_cubic_preserves_line_segment():
    string = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)

    result = uniform_string_repametrize_rn_cubic(string, 5)

    expected = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(result, expected)


def test_uniform_string_reparametrize_rn_cubic_preserves_endpoints():
    string = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [3.0, -1.0], [4.0, 0.0]],
        dtype=torch.float64,
    )

    result = uniform_string_repametrize_rn_cubic(string, 7)

    assert result.shape == (7, 2)
    assert torch.allclose(result[0], string[0])
    assert torch.allclose(result[-1], string[-1])


def test_uniform_string_reparametrize_so3_preserves_endpoints_and_midpoint():
    identity = torch.eye(3, dtype=torch.float64)
    end = rotvec_to_rotmat(torch.tensor([0.0, 0.0, torch.pi], dtype=torch.float64))
    string = torch.stack([identity, end])

    result = uniform_string_repametrize_so3_linear(string, 3)
    midpoint = rotvec_to_rotmat(
        torch.tensor([0.0, 0.0, torch.pi / 2], dtype=torch.float64)
    )

    assert torch.allclose(result[0], identity, atol=1e-10)
    assert torch.allclose(result[1], midpoint, atol=1e-10)
    assert torch.allclose(result[2], end, atol=1e-10)


def _so3_relative_angles(rotations):
    rel = torch.matmul(rotations[:-1].transpose(-1, -2), rotations[1:])
    return rotmat_to_theta(rel)


def test_uniform_string_reparametrize_se3_preserves_endpoints():
    translations = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2], [1.0, 0.4, 0.6]],
        dtype=torch.float64,
    )
    rotvecs = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 1.3]],
        dtype=torch.float64,
    )
    rotations = rotvec_to_rotmat(rotvecs)

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=11, corr_coeff=1.0
    )

    assert new_t.shape == (11, 3)
    assert new_r.shape == (11, 3, 3)
    assert torch.allclose(new_t[0], translations[0], atol=1e-12)
    assert torch.allclose(new_t[-1], translations[-1], atol=1e-12)
    assert torch.allclose(new_r[0], rotations[0], atol=1e-12)
    assert torch.allclose(new_r[-1], rotations[-1], atol=1e-12)


def test_uniform_string_reparametrize_se3_preserves_random_rotation_endpoints():
    torch.manual_seed(0)
    translations = torch.randn((30, 10, 3), dtype=torch.float64)
    rotations = rotvec_to_rotmat(torch.randn((30, 10, 3), dtype=torch.float64))

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=30
    )

    assert torch.equal(new_t[0], translations[0])
    assert torch.equal(new_t[-1], translations[-1])
    assert torch.equal(new_r[0], rotations[0])
    assert torch.equal(new_r[-1], rotations[-1])


def test_uniform_string_reparametrize_se3_preserves_merged_boundary_endpoints():
    translations = torch.zeros((4, 2, 3), dtype=torch.float64)
    translations[0] = 1.0
    translations[1] = 1.0 + 1.0e-8
    translations[2] = 3.0
    translations[3] = 3.0 + 1.0e-8
    rotations = rotvec_to_rotmat(torch.randn((4, 2, 3), dtype=torch.float64))

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=4
    )

    assert torch.equal(new_t[0], translations[0])
    assert torch.equal(new_t[-1], translations[-1])
    assert torch.equal(new_r[0], rotations[0])
    assert torch.equal(new_r[-1], rotations[-1])


def test_reparaametrize_kabsh_se3_restores_original_endpoints():
    reference_t = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.1, 0.3],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float64,
    )
    reference_r = rotvec_to_rotmat(
        torch.tensor(
            [
                [0.1, 0.2, -0.1],
                [0.3, -0.2, 0.1],
                [-0.2, 0.1, 0.4],
                [0.2, 0.3, -0.3],
            ],
            dtype=torch.float64,
        )
    )
    global_r = rotvec_to_rotmat(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.1, 0.3],
                [-0.3, 0.2, 0.1],
            ],
            dtype=torch.float64,
        )
    )
    global_t = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 1.5, 2.0]],
        dtype=torch.float64,
    )
    translations = torch.matmul(
        reference_t.unsqueeze(0), global_r.transpose(-1, -2)
    ) + global_t.unsqueeze(1)
    rotations = torch.matmul(
        global_r.unsqueeze(1),
        torch.matmul(
            reference_r.unsqueeze(0),
            global_r.transpose(-1, -2).unsqueeze(1),
        ),
    )

    new_t, new_r = reparaametrize_kabsh_se3(translations, rotations, n_new=9, index=0)

    assert torch.allclose(new_t[0], translations[0], atol=1e-10)
    assert torch.allclose(new_t[-1], translations[-1], atol=1e-10)
    assert torch.allclose(new_r[0], rotations[0], atol=1e-10)
    assert torch.allclose(new_r[-1], rotations[-1], atol=1e-10)


def test_reparaametrize_kabsh_se3_preserves_rigid_protein_geometry():
    reference_t = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.1, 0.3],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float64,
    )
    reference_r = torch.eye(3, dtype=torch.float64).expand(4, 3, 3).clone()
    global_r = rotvec_to_rotmat(
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.3, -0.1], [0.4, 0.6, -0.2]],
            dtype=torch.float64,
        )
    )
    global_t = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -0.5, 0.2], [2.0, -1.0, 0.4]],
        dtype=torch.float64,
    )
    translations = torch.matmul(
        reference_t.unsqueeze(0), global_r.transpose(-1, -2)
    ) + global_t.unsqueeze(1)
    rotations = torch.matmul(
        global_r.unsqueeze(1),
        torch.matmul(
            reference_r.unsqueeze(0),
            global_r.transpose(-1, -2).unsqueeze(1),
        ),
    )

    new_t, _ = reparaametrize_kabsh_se3(translations, rotations, n_new=11, index=0)
    reference_distances = torch.cdist(reference_t, reference_t)

    assert torch.allclose(
        torch.cdist(new_t, new_t),
        reference_distances.unsqueeze(0).expand(11, -1, -1),
        atol=1e-10,
    )


def test_uniform_string_reparametrize_se3_constant_path_expands_single_state():
    translations = torch.tensor([[1.5, -2.0, 0.3]], dtype=torch.float64)
    rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=7
    )

    assert new_t.shape == (7, 3)
    assert new_r.shape == (7, 3, 3)
    assert torch.allclose(new_t, translations.expand_as(new_t))
    assert torch.allclose(new_r, rotations.expand_as(new_r))


def test_uniform_string_reparametrize_se3_translation_only_matches_rn_linear_when_corr_zero():
    translations = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.4, 0.0], [2.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    rotations = (
        torch.eye(3, dtype=torch.float64)
        .unsqueeze(0)
        .expand(translations.shape[0], 3, 3)
        .clone()
    )

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=25, corr_coeff=0.0
    )
    rn_ref = uniform_string_repametrize_rn_linear(translations, n_new=25)

    assert torch.allclose(new_t, rn_ref, atol=1e-10)
    assert torch.allclose(new_r, rotations[:1].expand_as(new_r), atol=1e-12)


def test_uniform_string_reparametrize_se3_rotation_only_matches_so3_linear():
    n_pts = 5
    translations = torch.zeros((n_pts, 3), dtype=torch.float64)
    axis = torch.tensor([1.0, -0.3, 0.4], dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis)
    angles = torch.tensor([0.0, 0.2, 1.1, 1.7, 2.3], dtype=torch.float64)
    rotations = rotvec_to_rotmat(angles.unsqueeze(-1) * axis)

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=31, corr_coeff=1.0
    )
    so3_ref = uniform_string_repametrize_so3_linear(rotations, n_new=31)

    assert torch.allclose(new_t, torch.zeros_like(new_t), atol=1e-12)
    assert torch.allclose(new_r, so3_ref, atol=1e-10)


def test_uniform_string_reparametrize_se3_segment_costs_are_near_uniform():
    translations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.7, 0.2, 0.0],
            [1.2, 0.6, 0.2],
            [2.0, 1.0, 0.4],
        ],
        dtype=torch.float64,
    )
    yaw = torch.tensor([0.0, 0.1, 1.2, 1.4, 2.4], dtype=torch.float64)
    rotvec = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=-1)
    rotations = rotvec_to_rotmat(rotvec)
    coeff = 1.5

    new_t, new_r = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=80, corr_coeff=coeff
    )

    trans_lengths = _segment_lengths(torch.diff(new_t, dim=0))
    rot_lengths = _so3_relative_angles(new_r)
    combined = trans_lengths + coeff * rot_lengths

    assert combined.numel() > 1
    # Ignore boundaries to reduce endpoint clamping effects in searchsorted bins.
    interior = combined[1:-1]
    rel_std = interior.std(unbiased=False) / interior.mean()
    assert rel_std < 0.15


def test_uniform_string_reparametrize_se3_corr_coeff_increases_rotation_uniformity_weight():
    t = torch.tensor([0.0, 0.2, 0.55, 0.7, 1.0], dtype=torch.float64)
    translations = torch.stack(
        [torch.cos(2 * torch.pi * t), torch.sin(2 * torch.pi * t), t], dim=1
    )
    yaw = torch.tensor([0.0, 0.2, 1.8, 2.0, 3.0], dtype=torch.float64)
    rotvec = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], dim=1)
    rotations = rotvec_to_rotmat(rotvec)

    _, rot_low = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=80, corr_coeff=0.1
    )
    _, rot_high = uniform_string_repametrize_se3_linear(
        translations, rotations, n_new=80, corr_coeff=5.0
    )

    low_std = _so3_relative_angles(rot_low).std(unbiased=False)
    high_std = _so3_relative_angles(rot_high).std(unbiased=False)
    assert high_std < low_std


def test_merge_close_points_returns_input_when_merge_tol_is_none():
    string = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]], dtype=torch.float64)
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=None)

    assert torch.allclose(merged, string)


def test_merge_close_points_returns_input_when_threshold_non_positive():
    string = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]], dtype=torch.float64)
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=0.0)

    assert torch.allclose(merged, string)


def test_merge_close_points_collapses_contiguous_group_by_average():
    # Segments: 0.1, 0.1, 0.8 (total=1.0), threshold=0.25
    # The first two segments form one contiguous group (points 0..2).
    string = torch.tensor([[0.0], [0.1], [0.2], [1.0]], dtype=torch.float64)
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=0.25)

    expected = torch.tensor([[0.1], [1.0]], dtype=torch.float64)
    assert merged.shape == expected.shape
    assert torch.allclose(merged, expected, atol=1e-12)


def test_merge_close_points_collapses_all_points_when_threshold_large():
    string = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=torch.float64)
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=2.0)

    expected = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    assert merged.shape == expected.shape
    assert torch.allclose(merged, expected, atol=1e-12)


def test_merge_close_points_does_not_merge_when_segment_is_above_threshold():
    # Segments: 0.3, 0.7 (total=1.0), threshold=0.25 -> no merge.
    string = torch.tensor([[0.0], [0.3], [1.0]], dtype=torch.float64)
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=0.25)

    assert merged.shape == string.shape
    assert torch.allclose(merged, string, atol=1e-12)


def test_merge_close_points_merges_multiple_disjoint_groups():
    # Segments: 0.05, 0.05, 0.8, 0.05, 0.05 (total=1.0), threshold=0.1
    # Groups merged: points 0..2 and points 3..5.
    string = torch.tensor(
        [[0.0], [0.05], [0.10], [0.90], [0.95], [1.0]], dtype=torch.float64
    )
    seglenghts = _segment_lengths(torch.diff(string, dim=0))

    merged = _merge_close_points(string, seglenghts, merge_tol=0.1)

    expected = torch.tensor([[0.05], [0.95]], dtype=torch.float64)
    assert merged.shape == expected.shape
    assert torch.allclose(merged, expected, atol=1e-12)


def test_merge_close_points_validates_segment_length_count():
    string = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    wrong_seglenghts = torch.tensor([1.0], dtype=torch.float64)  # should have length 2

    with pytest.raises(AssertionError):
        _merge_close_points(string, wrong_seglenghts, merge_tol=1.0)
