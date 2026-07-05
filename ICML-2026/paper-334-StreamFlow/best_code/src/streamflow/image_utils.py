from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision


def denormalize_perflow(images: Union[torch.Tensor, np.ndarray], method: str = "normalize") -> torch.Tensor:
    """
    为PeRFlow优化的去归一化函数
    
    Args:
        images: 图像张量，通常在[-1.5, 1.5]范围内（VAE解码输出）
        method: 处理方法
            - "normalize": 标准归一化 (image / 2 + 0.5)
            - "dynamic": 动态范围归一化
            - "clamp": 直接截断（不推荐，会偏暗）
    """
    if method == "normalize":
        # 标准方法：假设输入在[-1,1]附近
        return (images / 2 + 0.5).clamp(0, 1)
    elif method == "dynamic":
        # 动态方法：保持完整动态范围
        min_val = images.min()
        max_val = images.max()
        if max_val > min_val:
            return (images - min_val) / (max_val - min_val)
        else:
            return images.clamp(0, 1)
    elif method == "clamp":
        # 直接截断（保留用于对比）
        return images.clamp(0, 1)
    else:
        raise ValueError(f"Unknown method: {method}")


def denormalize(images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    保持原始的denormalize函数以兼容性
    """
    return denormalize_perflow(images, method="normalize")


def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            PIL.Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images


def postprocess_image(
    image: torch.Tensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
    denormalize_method: str = "normalize",
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image]:
    """
    为PeRFlow优化的后处理函数
    
    Args:
        denormalize_method: PeRFlow专用的去归一化方法
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )

    if output_type == "latent":
        return image

    do_normalize_flg = True
    if do_denormalize is None:
        do_denormalize = [do_normalize_flg] * image.shape[0]

    # 使用PeRFlow优化的去归一化
    image = torch.stack(
        [
            denormalize_perflow(image[i], method=denormalize_method) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ]
    )

    if output_type == "pt":
        return image

    image = pt_to_numpy(image)

    if output_type == "np":
        return image

    if output_type == "pil":
        return numpy_to_pil(image)


def process_image(
    image_pil: PIL.Image.Image, range: Tuple[int, int] = (-1, 1)
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    image = torchvision.transforms.ToTensor()(image_pil)
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image[None, ...], image_pil


def pil2tensor(image_pil: PIL.Image.Image) -> torch.Tensor:
    height = image_pil.height
    width = image_pil.width
    imgs = []
    img, _ = process_image(image_pil)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear"
    )
    image_tensors = images.to(torch.float16)
    return image_tensors