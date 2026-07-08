import os
import re

import torch
from PIL import Image
from torchvision import transforms as T

_CLASS_ID_FROM_FILENAME_RE = re.compile(r"^(\d+)")


def _build_center_crop_to_256_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
        ]
    )


def _parse_class_id_from_filename(filename):
    match = _CLASS_ID_FROM_FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"Filename must start with integer ImageNet class id: got {filename!r}"
        )
    return int(match.group(1))


def list_supported_image_files(directory):
    supported_extensions = (".jpg", ".jpeg", ".png", ".webp")
    return sorted(
        filename
        for filename in os.listdir(directory)
        if filename.lower().endswith(supported_extensions)
    )


def load_imagenet_test_images_and_class_labels(directory, device="cuda"):
    to_tensor = _build_center_crop_to_256_transform()
    filenames = list_supported_image_files(directory)
    if len(filenames) == 0:
        raise FileNotFoundError(f"No test images found in {directory}")
    image_tensors = []
    class_ids = []
    for filename in filenames:
        class_ids.append(_parse_class_id_from_filename(filename))
        image = Image.open(os.path.join(directory, filename)).convert("RGB")
        tensor_in_0_1 = to_tensor(image)
        tensor_in_m1_1 = tensor_in_0_1 * 2.0 - 1.0
        image_tensors.append(tensor_in_m1_1)
    return (
        torch.stack(image_tensors, dim=0).to(device),
        torch.tensor(class_ids, dtype=torch.long, device=device),
    )
