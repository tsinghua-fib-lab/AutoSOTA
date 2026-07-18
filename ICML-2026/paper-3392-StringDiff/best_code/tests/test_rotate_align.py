import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.rotate_align import kabsch_align
from diffusion_strings.so3 import rotvec_to_rotmat


def test_kabsch_align_identity_returns_same_points():
    points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [0.5, -0.2, 3.0]],
            [[-1.0, 0.5, 2.0], [2.0, -1.0, 0.0], [0.1, 0.2, 0.3]],
        ],
        dtype=torch.float64,
    )

    aligned = kabsch_align(points, points)

    assert aligned.shape == points.shape
    assert torch.allclose(aligned, points, atol=1e-12)


def test_kabsch_align_recovers_known_rigid_transform():
    torch.manual_seed(0)
    t_batch, n_points = 4, 16
    reference = torch.randn(t_batch, n_points, 3, dtype=torch.float64)
    rotvec = torch.randn(t_batch, 3, dtype=torch.float64)
    rotation = rotvec_to_rotmat(rotvec)
    translation = torch.randn(t_batch, 1, 3, dtype=torch.float64)

    # Build moving so that the true mapping back to reference is:
    # reference = moving @ rotation + translation
    moving = reference @ rotation.transpose(-1, -2) - translation @ rotation.transpose(-1, -2)

    aligned, estimated_rotation, estimated_translation = kabsch_align(
        moving, reference, return_transform=True
    )

    assert aligned.shape == reference.shape
    assert estimated_rotation.shape == (t_batch, 3, 3)
    assert estimated_translation.shape == (t_batch, 1, 3)
    assert torch.allclose(aligned, reference, atol=1e-10, rtol=1e-10)
    assert torch.allclose(estimated_rotation, rotation, atol=1e-10, rtol=1e-10)
    assert torch.allclose(estimated_translation, translation, atol=1e-10, rtol=1e-10)


def test_kabsch_align_returns_proper_rotations_for_reflection_case():
    reference = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.2, -0.1], [0.3, 1.1, 0.4], [0.5, -0.3, 1.2]]],
        dtype=torch.float64,
    )
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
    moving = reference @ reflection

    _, estimated_rotation, _ = kabsch_align(moving, reference, return_transform=True)
    dets = torch.det(estimated_rotation)

    assert torch.all(dets > 0.0)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-12)


def test_kabsch_align_validates_input_shapes():
    reference = torch.zeros(2, 3, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="shape"):
        kabsch_align(torch.zeros(3, 3, dtype=torch.float64), reference)

    with pytest.raises(ValueError, match="same shape"):
        kabsch_align(torch.zeros(2, 4, 3, dtype=torch.float64), reference)

    with pytest.raises(ValueError, match="Last dimension"):
        kabsch_align(torch.zeros(2, 3, 4, dtype=torch.float64), torch.zeros(2, 3, 4, dtype=torch.float64))
