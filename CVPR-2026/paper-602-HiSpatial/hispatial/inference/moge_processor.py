"""MoGe depth estimation processor for HiSpatial inference."""

import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 448


class MoGeProcessor:
    """Wraps the MoGe v2 depth model to produce XYZ point maps from RGB images.

    The output ``xyz_values`` tensor has shape [4, H, W] where the first 3
    channels are (x, y, z) coordinates and the 4th channel is a validity mask.
    """

    def __init__(self, device_name="cuda", **kwargs):
        from moge.model.v2 import MoGeModel

        self.device = torch.device(device_name)
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        self.img_size = IMAGE_SIZE

    def infer_depth(self, image):
        """Run raw MoGe inference, returning the full output dict."""
        image = self._to_numpy_rgb(image)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        output = self.model.infer(image_tensor, fov_x=None)
        return output

    def apply_transform(self, image):
        """Produce a [4, H, W] xyz_values tensor ready for HiSpatialVLM.

        Returns:
            xyz_values: Tensor of shape [4, img_size, img_size].
        """
        image = self._to_numpy_rgb(image)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        output = self.model.infer(image_tensor, fov_x=None)
        xyz = output["points"].cpu().numpy()
        mask = output["mask"].cpu().numpy().astype(np.int8)

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask)
        xyz = torch.from_numpy(xyz)

        xyz[xyz > 250] = 250
        xyz[torch.isinf(xyz)] = 250
        xyz[torch.isnan(xyz)] = 250

        xyz_values = xyz.permute(2, 0, 1)  # [3, H, W]
        xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)  # [4, H, W]

        return xyz_values

    @staticmethod
    def _to_numpy_rgb(image):
        """Convert various image inputs to a numpy RGB array."""
        if isinstance(image, str):
            return cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError("Unsupported image type. Please provide a file path, PIL Image, or numpy array.")
