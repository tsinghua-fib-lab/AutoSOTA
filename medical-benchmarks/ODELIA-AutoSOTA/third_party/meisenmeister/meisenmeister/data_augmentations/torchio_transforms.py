from __future__ import annotations

import numpy as np
import torch


def _require_torchio():
    try:
        import torchio as tio
    except ImportError as error:
        raise ImportError(
            "TorchIO augmentations require the optional 'torchio' dependency. "
            "Install it with `pip install torchio`."
        ) from error
    return tio


class TorchIOTransform3D:
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        image_tensor = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
        transformed = self.transform(image_tensor)
        transformed_image = (
            transformed.detach().cpu().numpy().astype(image.dtype, copy=False)
        )
        return {**sample, "image": transformed_image}


def build_default_mri_torchio_pipeline():
    tio = _require_torchio()
    return TorchIOTransform3D(
        tio.Compose(
            [
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5, p=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=(10.0, 10.0, 10.0),
                    translation=(8.0, 8.0, 8.0),
                    image_interpolation="linear",
                    p=0.5,
                ),
                tio.OneOf(
                    {
                        tio.RandomMotion(
                            degrees=10.0,
                            translation=8.0,
                            num_transforms=2,
                            p=1.0,
                        ): 0.30,
                        tio.RandomGhosting(
                            num_ghosts=(4, 8),
                            intensity=(0.3, 0.8),
                            p=1.0,
                        ): 0.20,
                        tio.RandomSpike(p=1.0): 0.15,
                        tio.RandomBiasField(p=1.0): 0.35,
                    },
                    p=0.45,
                ),
                tio.OneOf(
                    {
                        tio.RandomAnisotropy(
                            axes=(0, 1, 2), downsampling=(1.5, 2.5), p=1.0
                        ): 0.30,
                        tio.RandomBlur(std=(0.25, 1.0), p=1.0): 0.20,
                        tio.RandomNoise(mean=0.0, std=(0.0, 0.08), p=1.0): 0.30,
                        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=1.0): 0.20,
                    },
                    p=0.35,
                ),
            ]
        )
    )
