import numpy as np
import torch
from math import pi
from torch import nn


class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super(Resizer, self).__init__()

        # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

        # Choose interpolation method, each method has the matching kernel size
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)  # set default interpolation method as cubic
        }.get(kernel)

        # Antialiasing is only used when downscaling
        antialiasing *= (np.any(np.array(scale_factor) < 1))

        # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
            # weights that multiply the values there to get its result.
            weights, field_of_view = self.contributions(in_shape[dim], output_shape[dim], scale_factor[dim], method,
                                                        kernel_width, antialiasing)

            # convert to torch tensor
            weights = torch.tensor(weights.T, dtype=torch.float32)

            # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
            # tmp_im[field_of_view.T], (bsxfun style)
            weights_list.append(
                nn.Parameter(torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                             requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
            x = torch.transpose(x, dim, 0)

            # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
            # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
            # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
            # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
            # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
            # same number
            x = torch.sum(x[fov] * w, dim=0)

            # Finally we swap back the axes to the original order
            x = torch.transpose(x, dim, 0)

        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
        # same size as the number of input dimensions)
        if scale_factor is not None:
            # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]

            # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

        # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
        # to all the unspecified dimensions
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))

        # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
        # sub-optimal, because there can be different scales to the same output-shape.
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

        # Dealing with missing output-shape. calculating according to scale-factor
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
        # such that each position from the field_of_view will be multiplied with a matching filter from the
        # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
        # around it. This is only done for one dimension of the image.

        # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
        # 1/sf. this means filtering is more 'low-pass filter'.
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0

        # These are the coordinates of the output image
        out_coordinates = np.arange(1, out_length + 1)

        # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
        # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
        # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

        # These are the matching positions of the output-coordinates on the input image coordinates.
        # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
        # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
        # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
        # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
        # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
        # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
        # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
        # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

        # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
        left_boundary = np.floor(match_coordinates - kernel_width / 2)

        # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
        # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
        expanded_kernel_width = np.ceil(kernel_width) + 2

        # Determine a set of field_of_view for each each output position, these are the pixels in the input image
        # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
        # vertical dim is the pixels it 'sees' (kernel_size + 2)
        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

        # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
        # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
        # 'field_of_view')
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

        # Normalize weights to sum up to 1. be careful from dividing by 0
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        # We use this mirror structure as a trick for reflection padding at the boundaries
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        # Get rid of  weights and pixel positions that are of zero weight
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
        return weights, field_of_view


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))

# === End: forward_operator/resizer.py ===


# === Begin: forward_operator/fastmri_utils.py ===

"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore


def fft2c_old(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Whether to include normalization. Must be one of ``"backward"``
            or ``"ortho"``. See ``torch.fft.fft`` on PyTorch 1.9.0 for details.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if norm not in ("ortho", "backward"):
        raise ValueError("norm must be 'ortho' or 'backward'.")
    normalized = True if norm == "ortho" else False

    data = ifftshift(data, dim=[-3, -2])
    data = torch.fft(data, 2, normalized=normalized)
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_old(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Whether to include normalization. Must be one of ``"backward"``
            or ``"ortho"``. See ``torch.fft.ifft`` on PyTorch 1.9.0 for
            details.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if norm not in ("ortho", "backward"):
        raise ValueError("norm must be 'ortho' or 'backward'.")
    normalized = True if norm == "ortho" else False

    data = ifftshift(data, dim=[-3, -2])
    data = torch.ifft(data, 2, normalized=normalized)
    data = fftshift(data, dim=[-3, -2])

    return data


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

# === End: forward_operator/fastmri_utils.py ===


# === Begin: forward_operator/motionblur/motionblur.py ===

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve

# tiny error used for nummerical stability
eps = 0.1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


class Kernel(object):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: tuple = (100, 100), intensity: float=0):

        # checking if size is correctly given
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        # check if intensity is float (int) between 0 and 1
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        # saving args
        self.SIZE = size
        self.INTENSITY = intensity

        # deriving quantities

        # we super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2

        # getting length of kernel diagonal
        self.DIAGONAL = (self.x**2 + self.y**2)**0.5

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False

    def _createPath(self):
        """[summary]
        creates a motion blur path with the given intensity.
        [description]
        Proceede in 5 steps
        1. Get a random number of random step sizes
        2. For each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions

        NOTE: "random" means random but might depend on the given intensity
        """

        # first we find the lengths of the motion blur steps
        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * \
                (uniform() + uniform(0, self.INTENSITY**2))

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """

            # same as with the steps

            # first we get the max angle in radians
            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)

            # now we sample "jitter" which is the probability that the
            # next angle has a different sign than the previous one
            self.JITTER = beta(2, 20)

            # initialising angles (and sign of angle)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                # sample next angle (absolute value)
                angle = triangular(0, self.INTENSITY *
                                   self.MAX_ANGLE, self.MAX_ANGLE + eps)

                # with jitter probability change sign wrt previous angle
                if uniform() < self.JITTER:
                    angle *= - np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            # save angles
            self.ANGLES = np.asarray(angles)

        # Get steps and angles
        getSteps()
        getAngles()

        # Turn them into a path
        ####

        # we turn angles and steps into complex numbers
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        # generate path as the cumsum of these increments
        self.path_complex = np.cumsum(complex_increments)

        # find center of mass of path
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        # Shift path s.t. center of mass lies in the middle of
        # the kernel and a apply a random rotation
        ###

        # center it on COM
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        # randomly rotate path by an angle a in (0, pi)
        self.path_complex *= np.exp(1j * uniform(0, pi))

        # center COM on center of kernel
        self.path_complex += center_of_kernel

        # convert complex path to final list of coordinate tuples
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self, save_to: Path=None, show: bool=False):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        use displayKernel to actually see the kernel.

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # get the path
        self._createPath()

        # Initialise an image with super-sized dimensions
        # (pillow Image object)
        self.kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        # we can use to draw on our kernel_image
        self.painter = ImageDraw.Draw(self.kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))

        # Resize to actual size
        self.kernel_image = self.kernel_image.resize(
            self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True

    def displayKernel(self, save_to: Path=None, show: bool=True):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        Saves the kernel to save_to if needed or shows it
        is show true

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # generate kernel if needed
        self._createKernel()

        # save if needed
        if save_to is not None:

            save_to_file = Path(save_to)

            # save Kernel image
            self.kernel_image.save(save_to_file)
        else:
            # Show kernel
            self.kernel_image.show()

    @property
    def kernelMatrix(self) -> np.ndarray:
        """[summary]
        Kernel matrix of motion blur of given intensity.
        [description]
        Once generated, it stays the same.
        Returns:
            numpy ndarray
        """

        # generate kernel if needed
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)

        return kernel

    @kernelMatrix.setter
    def kernelMatrix(self, *kargs):
        raise NotImplementedError("Can't manually set kernel matrix yet")

    def applyTo(self, image, keep_image_dim: bool = False) -> Image:
        """[summary]
        Applies kernel to one of the following:

        1. Path to image file
        2. Pillow image object
        3. (H,W,3)-shaped numpy array
        [description]

        Arguments:
            image {[str, Path, Image, np.ndarray]}
            keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

        Returns:
            Image -- [description]
        """
        # calculate kernel if haven't already
        self._createKernel()

        def applyToPIL(image: Image, keep_image_dim: bool = False) -> Image:
            """[summary]
            Applies the kernel to an PIL.Image instance
            [description]
            converts to RGB and applies the kernel to each
            band before recombining them.
            Arguments:
                image {Image} -- Image to convolve
                keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

            Returns:
                Image -- blurred image
            """
            # convert to RGB
            image = image.convert(mode="RGB")

            conv_mode = "valid"
            if keep_image_dim:
                conv_mode = "same"

            result_bands = ()

            for band in image.split():

                # convolve each band individually with kernel
                result_band = convolve(
                    band, self.kernelMatrix, mode=conv_mode).astype("uint8")

                # collect bands
                result_bands += result_band,

            # stack bands back together
            result = np.dstack(result_bands)

            # Get image
            return Image.fromarray(result)

        # If image is Path
        if isinstance(image, str) or isinstance(image, Path):

            # open image as Image class
            image_path = Path(image)
            image = Image.open(image_path)

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, Image.Image):

            # apply kernel
            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, np.ndarray):

            # ASSUMES we have an array of the form (H, W, 3)
            ###

            # initiate Image object from array
            image = Image.fromarray(image)

            return applyToPIL(image, keep_image_dim)

        else:

            raise ValueError("Cannot apply kernel to this type.")


if __name__ == '__main__':
    image = Image.open("./images/moon.png")
    image.show()
    k = Kernel()

    k.applyTo(image, keep_image_dim=True).show()


# === End: forward_operator/motionblur/motionblur.py ===


# === Begin: forward_operator/__init__.py (edited for single-file merge) ===

from abc import ABC, abstractmethod
import torch.nn.functional as F

import torch
import torch.nn as nn
import scipy
import numpy as np
import yaml
import warnings
from torch.autograd import grad

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            if __OPERATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __OPERATOR__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    """
    Abstract base class for operators in diffusion processes.

    Attributes:
        sigma (float): Standard deviation of measurement noise.
    """
    def __init__(self, sigma=0.05):
        """
        Initializes the operator with a noise standard deviation.

        Args:
            sigma (float, optional): Measurement noise level. Defaults to 0.05.
        """
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        """
        Abstract method: apply operator to input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output after applying the operator.
        """
        pass

    def measure(self, x):
        """
        Measures input data by applying the operator and adding Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy measurement result.
        """
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def loss(self, x, y):
        """
        Computes squared-error loss between operator output and observed data.

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Observed measurement tensor.

        Returns:
            torch.Tensor: Loss values (one per sample).
        """
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def gradient(self, x, y, return_loss=False):
        """
        Computes gradient of the loss with respect to input x.

        Args:
            x (torch.Tensor): Input tensor requiring gradient.
            y (torch.Tensor): Observed measurements.
            return_loss (bool, optional): If True, returns both gradient and loss. Defaults to False.

        Returns:
            torch.Tensor or tuple: Gradient tensor (and optionally the loss value).
        """
        x_tmp = x.clone().detach().requires_grad_(True)
        loss = self.loss(x_tmp, y).sum()
        x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if return_loss:
            return x_grad, loss
        return x_grad

    def log_likelihood(self, x, y):
        return -self.loss(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))


# Linear Operator
@register_operator(name='down_sampling')
class DownSampling(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        in_shape = [1, 3, resolution, resolution]
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def __call__(self, x):
        return self.down_sample(x)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


@register_operator(name='inpainting')
class Inpainting(Operator):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
                 sigma=0.05):
        super().__init__(sigma)
        self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution)
        self.mask = None  # [B, 1, H, W]

    def __call__(self, x):
        if self.mask is None:
            self.mask = self.mask_gen(x)
            self.mask = self.mask[0:1, 0:1, :, :]
        return x * self.mask


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


@register_operator(name='gaussian_blur')
class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


@register_operator(name='motion_blur')
class MotionBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        # A^T * A
        return self.conv(data)


# Non-linear Operator
@register_operator(name='phase_retrieval')
class PhaseRetrieval(Operator):
    def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.pad = int((oversample / 8.0) * resolution)

    def __call__(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        # amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
        return amplitude


@register_operator(name='nonlinear_blur')
class NonlinearBlur(Operator):
    def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        """Load BKSE KernelWizard for nonlinear blur.

        `opt_yml_path` may be a relative path from upstream presets. We resolve it
        relative to this codebase so the operator works regardless of the current
        working directory.
        """
        from pathlib import Path
        from forward_operator.bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        base_dir = Path(__file__).resolve().parent
        opt_path = Path(opt_yml_path)
        if not opt_path.is_absolute():
            cand = base_dir / opt_path
            if cand.exists():
                opt_path = cand
            else:
                cand2 = base_dir.parent / opt_path
                if cand2.exists():
                    opt_path = cand2
        opt_path = opt_path.resolve()

        with open(opt_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = Path(opt["pretrained"])

        if not model_path.is_absolute():
            cand = opt_path.parent / model_path
            if cand.exists():
                model_path = cand
            else:
                cand2 = base_dir / model_path
                if cand2.exists():
                    model_path = cand2
                else:
                    cand3 = base_dir.parent / model_path
                    if cand3.exists():
                        model_path = cand3
        model_path = model_path.resolve()

        # If the BKSE checkpoint was moved into the shared assets folder
        # (<repo_root>/inverse_2d_assets/checkpoints), fall back to that location.
        if not model_path.exists():
            try:
                from misc import resolve_checkpoint_path
                alt = Path(resolve_checkpoint_path(model_path.name))
                if alt.exists():
                    model_path = alt
            except Exception:
                pass

        if not model_path.exists():
            raise FileNotFoundError(
                f"BKSE pretrained checkpoint not found: {model_path}. "
                f"Update the 'KernelWizard.pretrained' field in {opt_path} "
                f"or place '{model_path.name}' under inverse_2d_assets/checkpoints."
            )


        blur_model = KernelWizard(opt)
        blur_model.eval()
        state = torch.load(str(model_path), map_location="cpu")
        blur_model.load_state_dict(state)
        blur_model = blur_model.to(self.device)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        return blur_model

    def call_old(self, data):
        # random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = []
        for i in range(data.shape[0]):
            single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
            blurred.append(single_blurred)
        blurred = torch.cat(blurred, dim=0)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def __call__(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]

        # blurred = []
        # for i in range(data.shape[0]):
        #     single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
        #     blurred.append(single_blurred)
        # blurred = torch.cat(blurred, dim=0)
        # blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


@register_operator(name='high_dynamic_range')
class HighDynamicRange(Operator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def __call__(self, data):
        return torch.clip((data * self.scale), -1, 1)


class LatentWrapper(Operator):
    def __init__(self, op, model):
        super().__init__(sigma=op.sigma)
        self.op = op
        self.model = model

    def __call__(self, x):
        decoded = self.model.decode(x)
        return self.op(decoded)


    def loss(self, pred, observation):
        decoded = self.model.decode(pred)
        return self.op.loss(decoded.float(), observation)

    def gradient(self, pred, observation, return_loss=False):
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = grad(loss, pred_tmp)[0]
        pred_grad = pred_grad.to(pred.dtype)
        # clip the gradient
        pred_grad = torch.clamp(pred_grad, -1, 1)
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad
        

# === End: forward_operator/__init__.py ===
