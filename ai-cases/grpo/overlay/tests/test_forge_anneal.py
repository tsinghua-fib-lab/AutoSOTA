import pytest

from verl.utils.dataset.rl_dataset import forge_anneal_multiplier


def test_forge_anneal_piecewise_linear_schedule():
    assert forge_anneal_multiplier(1, 75, 100) == 1.0
    assert forge_anneal_multiplier(75, 75, 100) == 1.0
    assert forge_anneal_multiplier(80, 75, 100) == 0.8
    assert forge_anneal_multiplier(90, 75, 100) == 0.4
    assert forge_anneal_multiplier(100, 75, 100) == 0.0
    assert forge_anneal_multiplier(125, 75, 100) == 0.0


@pytest.mark.parametrize('start,end', [(-1, 100), (75, 75), (100, 75)])
def test_forge_anneal_rejects_invalid_intervals(start, end):
    with pytest.raises(ValueError):
        forge_anneal_multiplier(1, start, end)
