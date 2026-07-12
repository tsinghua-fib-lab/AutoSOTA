# test_so2_tensor_product.py
import math
import torch
import pytest
from so2_tensor_product import so2_tensor_product


def rand_complex(shape, *, device="cpu"):
    return torch.randn(*shape, dtype=torch.cfloat, device=device)


@pytest.mark.parametrize("batch,Ca,Cb", [(6, 2, 3), (1, 4, 1)])
def test_keys_and_channel_count(batch, Ca, Cb):
    a = {-1: rand_complex((batch, Ca)), 2: rand_complex((batch, Ca))}
    b = {0: rand_complex((batch, Cb)), 3: rand_complex((batch, Cb))}
    out = so2_tensor_product(a, b)

    # expected keys: sums of every pair
    expected_keys = {-1 + 0, -1 + 3, 2 + 0, 2 + 3}
    assert set(out) == expected_keys

    # channel counts = Ca*Cb per contributing pair
    assert out[-1].shape[-1] == Ca * Cb  # (-1) = (-1)+0
    assert out[5].shape[-1] == Ca * Cb  # ( 5) =  2 + 3
    # key 2 appears twice: (-1)+3 and 2+0  → concat → 2*Ca*Cb
    assert out[2].shape[-1] == 2 * Ca * Cb


@pytest.mark.parametrize("batch,Ca,Cb", [(5, 3, 4)])
def test_equivariance(batch, Ca, Cb, tol=1e-5):
    a = {-2: rand_complex((batch, Ca)), 1: rand_complex((batch, Ca))}
    b = {0: rand_complex((batch, Cb)), 2: rand_complex((batch, Cb))}
    c = so2_tensor_product(a, b)  # reference

    # rotate by random α
    alpha = torch.rand(1).item() * 2 * math.pi
    eia = torch.cos(torch.tensor(alpha)) + 1j * torch.sin(torch.tensor(alpha))
    phase = lambda m: eia**m  # <- tensor, not Python complex

    a_rot = {m: phase(m) * t for m, t in a.items()}
    b_rot = {m: phase(m) * t for m, t in b.items()}
    c_rot = so2_tensor_product(a_rot, b_rot)

    # expected ρ(α) acting on each key
    for m in c:
        assert torch.allclose(c_rot[m], phase(m) * c[m], atol=tol, rtol=tol)
