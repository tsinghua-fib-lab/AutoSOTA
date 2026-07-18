import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.dynamics import (
    b_transport,
    dynamic_string_method_T0_singular,
    pure_string_sampling,
)


def test_b_transport_integrates_constant_field():
    x = torch.tensor([0.0, 1.0])
    times = torch.linspace(0.0, 1.0, 5)

    result = b_transport(x, lambda y, t: torch.ones_like(y), times)

    assert torch.allclose(result, x + 1.0)


def test_b_transport_forwards_extra_field_arguments():
    x = torch.tensor([0.0, 1.0])
    times = torch.linspace(0.0, 1.0, 3)

    result = b_transport(
        x,
        lambda y, t, label, scale=1.0: torch.full_like(y, label * scale),
        times,
        2.0,
        scale=0.5,
    )

    assert torch.allclose(result, x + 1.0)


def test_pure_string_sampling_constant_field():
    string = torch.tensor([[0.0], [1.0], [2.0]])

    result = pure_string_sampling(
        string,
        lambda y, t: torch.ones_like(y),
        n_timesteps=2,
        reparam_every=None,
    )

    assert torch.allclose(result, string + 1.0)


def test_pure_string_sampling_forwards_extra_field_arguments():
    string = torch.tensor([[0.0], [1.0], [2.0]])

    result = pure_string_sampling(
        string,
        lambda y, t, label, scale=1.0: torch.full_like(y, label * scale),
        2,
        None,
        2.0,
        scale=0.5,
    )

    assert torch.allclose(result, string + 1.0)


def test_dynamic_string_method_t0_singular_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError):
        dynamic_string_method_T0_singular(
            torch.zeros(2, 1),
            lambda x, t: x,
            lambda x, t: x,
            10,
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            1.0,
            None,
            None,
        )
