import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, GaussianBlur


class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor``.

    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

    """

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.

        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``

        Returns:
        A ``torch.LongTensor``.

        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,
                                             2).contiguous().long().squeeze_()


class ToOnehotGaussianBlur(object):
    """
    Convert the label map to onehots and then use GaussianBlur to smooth the onehots.
    """
    epsilon = 1e-6

    def __init__(self, kernel_size, num_classes):
        assert kernel_size // 2 != 0, "the kernel size should be odd!"
        self.num_classes = num_classes
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # adopt from cv2.getGaussianKernel
        self.gaussian_blur = GaussianBlur(kernel_size, sigma)

    def convert_onehot(self, label):  # label (H, W)
        size = list(label.shape)
        label = label.view(-1)  # (H*W,)
        ones = torch.eye(self.num_classes).float()
        ones = ones * (1 - self.epsilon) + (1 - ones) * self.epsilon
        onehots = ones.index_select(0, label).view(*size, self.num_classes).permute(2, 0, 1)
        return onehots

    def __call__(self, label: Tensor) -> Tensor:
        """
        Args:
            label (Tensor): image to be blurred. (H, W)
        Returns:
            Tensor: Gaussian blurred Label. (H, W)
        """
        onehots = self.convert_onehot(label)  # (self.num_classes, H, W)
        blurred = self.gaussian_blur(onehots)  # (self.num_classes, H, W)
        # the first log makes the value doman [-inf, 0], the second log makes it [-inf, inf]
        double_log_blurred = torch.log(-torch.log(blurred))
        return label, double_log_blurred