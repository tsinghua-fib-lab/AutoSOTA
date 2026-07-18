import inspect

import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.dynamics_se3 import (  # noqa: E402
    b_transport,
    dynamic_string_method_T0_continuous,
    dynamic_string_method_T0_singular,
    dynamic_string_method_Tfinite_continuous,
    dynamic_string_method_Tfinite_singular,
    pure_string_sampling,
    static_string_method_Tfinite,
)
from diffusion_strings.so3 import rotmat_to_rotvec, rotvec_to_rotmat  # noqa: E402


def _identity_rotations(shape, dtype=torch.float64):
    return torch.eye(3, dtype=dtype).expand(*shape, 3, 3).clone()


def test_b_transport_integrates_constant_se3_field():
    translations = torch.zeros((2, 3), dtype=torch.float64)
    rotations = _identity_rotations((2,))
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    def field(translations, rotations, t):
        trans_delta = torch.ones_like(translations)
        rot_delta = torch.zeros_like(translations)
        rot_delta[..., 2] = 0.5
        return trans_delta, rot_delta

    new_t, new_r = b_transport(translations, rotations, field, times)

    assert torch.allclose(new_t, torch.ones_like(translations))
    assert torch.allclose(
        rotmat_to_rotvec(new_r),
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=torch.float64),
        atol=1e-12,
    )


def test_pure_string_sampling_reparametrizes_se3_string():
    translations = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    rotations = _identity_rotations((3,))

    def field(translations, rotations, t):
        return torch.zeros_like(translations), torch.zeros_like(translations)

    new_t, new_r = pure_string_sampling(
        translations, rotations, field, 1, reparam_every=1
    )

    assert new_t.shape == translations.shape
    assert new_r.shape == rotations.shape
    assert torch.allclose(new_t[0], translations[0])
    assert torch.allclose(new_t[-1], translations[-1])
    assert torch.allclose(torch.matmul(new_r, new_r.transpose(-1, -2)), rotations)


def test_se3_string_method_optional_defaults():
    methods = (
        pure_string_sampling,
        dynamic_string_method_T0_singular,
        dynamic_string_method_T0_continuous,
        static_string_method_Tfinite,
        dynamic_string_method_Tfinite_singular,
        dynamic_string_method_Tfinite_continuous,
    )

    for method in methods:
        assert inspect.signature(method).parameters["align_index"].default is None
        assert inspect.signature(method).parameters["keep_centered"].default is False


def test_pure_string_sampling_centers_each_chain_before_reparametrization():
    translations = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 1.0, 0.0], [14.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    rotations = _identity_rotations((2, 2))

    def zero_field(translations, rotations, t):
        return torch.zeros_like(translations), torch.zeros_like(translations)

    new_t, _ = pure_string_sampling(
        translations,
        rotations,
        zero_field,
        n_timesteps=1,
        reparam_every=1,
        keep_centered=True,
    )

    assert torch.allclose(
        new_t.mean(dim=-2), torch.zeros((2, 3), dtype=torch.float64), atol=1e-12
    )


def test_pure_string_sampling_centers_and_restores_kabsch_global_rotations():
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.1, 0.3],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float64,
    )
    frame = rotvec_to_rotmat(torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64))
    translations = torch.stack([reference + 3.0, reference @ frame - 4.0])
    rotations = _identity_rotations((2, reference.shape[0]))

    def zero_field(translations, rotations, t):
        return torch.zeros_like(translations), torch.zeros_like(translations)

    new_t, _ = pure_string_sampling(
        translations,
        rotations,
        zero_field,
        n_timesteps=1,
        reparam_every=1,
        align_index=0,
        keep_centered=True,
    )

    expected = translations - translations.mean(dim=-2, keepdim=True)
    assert torch.allclose(new_t, expected, atol=1e-10)


def test_pure_string_sampling_kabsch_reparametrization_restores_global_pose():
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.1, 0.3],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float64,
    )
    frame = rotvec_to_rotmat(torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64))
    shift = torch.tensor([2.0, -1.0, 0.4], dtype=torch.float64)
    translations = torch.stack([reference, reference @ frame + shift])
    rotations = _identity_rotations((2, reference.shape[0]))

    def zero_field(translations, rotations, t):
        return torch.zeros_like(translations), torch.zeros_like(translations)

    new_t, _ = pure_string_sampling(
        translations,
        rotations,
        zero_field,
        n_timesteps=1,
        reparam_every=1,
        align_index=0,
    )

    assert torch.allclose(new_t, translations, atol=1e-10)


def test_dynamic_string_method_t0_continuous_runs_on_se3():
    translations = torch.zeros((3, 3), dtype=torch.float64)
    rotations = _identity_rotations((3,))
    schedule = torch.tensor([0.0, 1.0], dtype=torch.float64)

    def zero_field(translations, rotations, t):
        return torch.zeros_like(translations), torch.zeros_like(translations)

    def score(translations, rotations, t):
        trans_delta = torch.zeros_like(translations)
        trans_delta[..., 0] = 1.0
        return trans_delta, torch.zeros_like(translations)

    new_t, new_r = dynamic_string_method_T0_continuous(
        translations,
        rotations,
        zero_field,
        score,
        n_timesteps=2,
        schedule=schedule,
        gamma_norm=1.0,
        max_algostep=None,
    )

    assert new_t.shape == translations.shape
    assert new_r.shape == rotations.shape
    assert torch.allclose(new_t[..., 0], torch.ones(3, dtype=torch.float64) * 0.5)
