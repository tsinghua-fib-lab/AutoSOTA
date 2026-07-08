import os
import sys

import torch


_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PMF_REPO_DIR = os.path.join(_THIS_FILE_DIR, "external", "pMF")
if _PMF_REPO_DIR not in sys.path:
    sys.path.insert(0, _PMF_REPO_DIR)

from pmf import pixelMeanFlow  # noqa: E402


DEFAULT_PMF_B16_CHECKPOINT = os.path.join(
    _THIS_FILE_DIR, "checkpoints", "pMF", "pMF-B-16.pt"
)


def build_pmf_module(model_name: str = "pmfDiT_B_16") -> pixelMeanFlow:
    img_size = 16 * int(model_name.split("_")[-1])
    return pixelMeanFlow(model_name, img_size=img_size)


def load_pmf_checkpoint(
    model: pixelMeanFlow, checkpoint_path: str
) -> pixelMeanFlow:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


class PixelMeanFlowOneStepModel:
    # x_t = (1 - t) * x_0 + t * noise_scale * eps. For pMF-B/16 the noise_scale is 1.0,
    # which matches calibrated_guidance.inference.flow_matching. The L/32 and H/32
    # checkpoints use noise_scale > 1, in which case the outer flow_matching loop
    # would need to draw scaled initial noise to stay in distribution.
    def __init__(
        self,
        pmf_module: pixelMeanFlow,
        cfg_omega: float = 7.5,
        interval_min: float = 0.1,
        interval_max: float = 0.8,
    ):
        self._pmf = pmf_module
        self._cfg_omega = float(cfg_omega)
        self._interval_min = float(interval_min)
        self._interval_max = float(interval_max)

    @property
    def img_size(self) -> int:
        return self._pmf.img_size

    @property
    def img_channels(self) -> int:
        return self._pmf.img_channels

    @property
    def num_classes(self) -> int:
        return self._pmf.num_classes

    @property
    def noise_scale(self) -> float:
        return float(self._pmf.noise_scale)

    def predict_x0_one_step(self, x_t, t_scalar, labels):
        batch_size = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        t = torch.full((batch_size,), float(t_scalar), device=device, dtype=dtype)
        h = t
        omega = torch.full((batch_size,), self._cfg_omega, device=device, dtype=dtype)
        t_min = torch.full(
            (batch_size,), self._interval_min, device=device, dtype=dtype
        )
        t_max = torch.full(
            (batch_size,), self._interval_max, device=device, dtype=dtype
        )

        u, _v = self._pmf.u_fn(
            x_t, t, h, omega, t_min, t_max, y=labels.to(device)
        )
        return x_t - t.view(-1, 1, 1, 1) * u
