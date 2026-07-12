from pathlib import Path
from typing import List

import numpy as np
import torch
from aim.sdk.objects.image import Image, convert_to_aim_image_list


class MediaLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def save_image(self, filename: str, image: Image):
        filename = Path(self.logdir) / f"{filename}.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        image.to_pil_image().save(filename)

    def save_images(self, filename: str, images: List[Image]):
        for idx, image in enumerate(images):
            self.save_image(f"{filename}_{idx}", image)

    def save_tensor_as_images(self, filename: str, data: torch.Tensor):
        aim_images = convert_to_aim_image_list(data)
        self.save_images(filename, aim_images)

    def save_tensor_as_npy(self, filename: str, data: torch.Tensor):
        filename = Path(self.logdir) / f"{filename}.npy"
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, data.cpu().detach().numpy())

    def save_tensors_as_npz(self, filename: str, data_dict: dict):
        filename = Path(self.logdir) / f"{filename}.npz"
        filename.parent.mkdir(parents=True, exist_ok=True)
        array_dict = {k: v.cpu().detach().numpy() for k, v in data_dict.items()}
        np.savez(filename, **array_dict)
