import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.so3 import (
    generator_to_rotmat,
    rotmat_to_generator,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)


def test_rotvec_rotmat_roundtrip():
    rotvec = torch.tensor(
        [[0.1, -0.2, 0.3], [1.0, 0.2, -0.4]],
        dtype=torch.float64,
    )

    rotmat = rotvec_to_rotmat(rotvec)
    identity = torch.eye(3, dtype=rotmat.dtype)

    assert torch.allclose(rotmat @ rotmat.transpose(-1, -2), identity, atol=1e-10)
    assert torch.allclose(rotmat_to_rotvec(rotmat), rotvec, atol=1e-10)


def test_generator_roundtrip_matches_rotation_matrix():
    rotvec = torch.tensor([[0.2, 0.3, -0.4]], dtype=torch.float64)
    rotmat = rotvec_to_rotmat(rotvec)

    generator, theta = rotmat_to_generator(rotmat)
    recovered = generator_to_rotmat(theta, generator)

    assert torch.allclose(recovered, rotmat, atol=1e-10)
