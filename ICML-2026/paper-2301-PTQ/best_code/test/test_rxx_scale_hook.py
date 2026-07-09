import sys
from pathlib import Path

src_path = Path(__file__).parents[1].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"

sys.path.append(src_path.as_posix())
import torch

from qera.statistic_profiler.scale import ScaleHookFactoryRxx, sqrtm_newton_schulz

if __name__ == "__main__":
    hook_factory = ScaleHookFactoryRxx()

    hook = hook_factory.get_scale_hook("test")

    x = torch.randn(100, 5).to("cuda")

    hook(None, (x,), None)

    scales = hook_factory.get_scale_dict()

    print(scales["test"].shape)

    # do it manually

    Exx_manual = torch.zeros(5, 5, device=x.device)

    for x_i in x:
        Exx_manual += torch.ger(x_i, x_i)

    Exx_manual /= x.shape[0]

    print(torch.allclose(Exx_manual, scales["test"].cuda(), atol=1e-6))

    print(Exx_manual.shape)

    Exx_manual_unsqueezed = Exx_manual.unsqueeze(0)

    print(sqrtm_newton_schulz(Exx_manual_unsqueezed, numIters=200))
