# pylint: disable=missing-module-docstring,missing-function-docstring,invalid-name
# TRANSFORMATION ENGINE / BACKEND
from typing import Union, Optional, Tuple, List
from enum import Enum
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional


def identity(
    batch_size: Union[int, tuple] = 1,
    device: Union[str, torch.device] = 'cuda',
    truncate: bool = False
) -> Tensor:
    if isinstance(batch_size, tuple):
        i = torch.eye(3).to(device)
        for size in batch_size:
            i = i.unsqueeze(0).expand(size, *i.shape)
    else:
        assert isinstance(batch_size, int)
        i = torch.eye(3)[None].expand(batch_size, 3, 3).to(device)
    if truncate:
        return i[:, :2, :2]
    return i


def translate(T: Optional[Tensor], translation: Tensor) -> Tensor:
    """ translation by 1 is shift by half as this is the relative coordinate system
    """
    # clone required as otherwise raises: `unsupported operation:
    # more than one element of the written-to tensor refers to a single memory location.`
    translation_matrix = identity(batch_size=translation.shape[0], device=translation.device).clone()
    # clone is essential as otherwise copy by reference and then translation_matrix = T at return
    T = T if T is not None else translation_matrix.clone()
    translation_matrix[:, 0, -1] = translation[:, 0]
    translation_matrix[:, 1, -1] = translation[:, 1]
    # this is the correct order as starting with translation, the translation bias
    # is with this version not altered after multiplication
    return T @ translation_matrix


def scale(T: Optional[Tensor], scaling: Tensor) -> Tensor:
    scaling_matrix = identity(batch_size=scaling.shape[0], device=scaling.device).clone()
    T = T if T is not None else scaling_matrix.clone()
    scaling_matrix[:, 0, 0] += scaling[:, 0]
    scaling_matrix[:, 1, 1] += scaling[:, 1]
    return T @ scaling_matrix


def shear(T: Optional[Tensor], shearing: Tensor) -> Tensor:
    shearing_matrix = identity(batch_size=shearing.shape[0], device=shearing.device).clone()
    T = T if T is not None else shearing_matrix.clone()
    shearing_matrix[:, 0, 1] += shearing[:, 0]
    shearing_matrix[:, 1, 0] += shearing[:, 1]
    return T @ shearing_matrix


def rotate(T: Optional[Tensor], angle: Tensor) -> Tensor:
    rotation_matrix = identity(batch_size=angle.shape[0], device=angle.device).clone()
    T = T if T is not None else rotation_matrix.clone()
    rotation_matrix[:, 0, 0] = torch.cos(angle)
    rotation_matrix[:, 0, 1] = -torch.sin(angle)
    rotation_matrix[:, 1, 0] = torch.sin(angle)
    rotation_matrix[:, 1, 1] = torch.cos(angle)
    return T @ rotation_matrix


def reflect(T: Optional[Tensor], reflection: Tensor) -> Tensor:
    reflection_matrix = identity(batch_size=reflection.shape[0], device=reflection.device).clone()
    T = T if T is not None else reflection_matrix.clone()
    reflection_matrix[:, 0, 0] = -1 if reflection else 1
    return T @ reflection_matrix


def compose(
    translation: Optional[Tensor] = None,
    scaling: Optional[Tensor] = None,
    rotation: Optional[Tensor] = None,
    shearing: Optional[Tensor] = None,
    reflection: bool = False
):
    T = None
    # first translate then scale for instance -> should be more robust
    if translation is not None:
        T = translate(T, translation)
    if scaling is not None:
        T = scale(T, scaling)
    if rotation is not None:
        T = rotate(T, rotation)
    if shearing is not None:
        T = shear(T, shearing)
    if reflection:
        T = reflect(T, reflection)
    return T


def resampling(x: Tensor, transformation_mat: Tensor, cropping: bool = False) -> Tensor:
    """
    :param x: input image
    :param cropping: todo does not work for MNIST!
    :return: transformed output image of same shape
    """
    # remove the last row of the matrix due to torch conventions (it will be always (0, 0, 1)
    affine_transform_mat = transformation_mat[:, :2, :]
    # transform the affine grid according to the transformation matrix and resample
    x = x if x.dim() == 4 else x[None, ...]
    # padding_const = x[:, :, -1, -1]
    grid = F.affine_grid(affine_transform_mat, x.size(), align_corners=False).type(x.dtype).to(x.device)
    # reflection, zeros, border
    x = F.grid_sample(x, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
    # x = torchvision.transforms.functional.normalize(x, 0.1307, 0.3081)
    # x = torch.where(x == 0, x.min(), x)
    # requires zero padding
    if cropping:
        # detect zero pixels (padding)
        padding_mask = x.to(x.device) == torch.zeros((1, x.shape[1], 1, 1)).to(x.device)
        # count zero pixel along a center line through the image (mean over channels)
        crop_size = padding_mask[:, :, int(x.shape[-2] / 2), :].sum(axis=(1, -1)) / x.shape[1]
        crop_size = x.shape[2] - crop_size
        resize = torchvision.transforms.Resize(x.shape[-2])
        for b in range(x.shape[0]):
            if int(crop_size[b]) > 0:
                x[b] = resize(torchvision.transforms.functional.center_crop(
                    x[b], (int(crop_size[b]), int(crop_size[b]))))
        # x = torch.where(padding_mask, padding_const[..., None, None], x)
    return x


def get_rotation_param(
    n: Union[int, Tensor],
    domain: Optional[Union[Tensor, float]] = None,
    n_samples: int = 16
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param domain: (float) parameters are generated within [-domain, +domain]
    :param n_samples: int
    :return:
    """
    domain = torch.tensor(torch.pi) if domain is None else domain
    array = torch.linspace(-domain, domain, n_samples)
    if isinstance(n, int) or n.shape[0] == 1:
        return array
    array = array[None].tile(n.shape[0], 1).to(n.device)
    return array[torch.arange(n.shape[0], device=n.device), n]


def get_rotation_param_orbit(
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None,
    extend: int = 0,
    shift: int = 1
) -> Tensor:
    """ Returns a list of parameters to construct an orbit.
    :param n_samples: int
    :param extend: (int) extends over the orbit by `extend`-many increment steps to the left and the right
    :param shift: (int) shifts the parameters by n increments
                        might be used to center the canonical form in the middle of the list
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return: list of radii - for consistency reasons [B, 1]
    """
    domain = torch.tensor(torch.pi) if domain is None else domain
    orbit = torch.linspace(-domain, domain, n_samples)
    if extend > 0:
        left_extend = orbit[0] + torch.tensor([(-2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        right_extend = orbit[-1] + torch.tensor([(2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        orbit = torch.concat([left_extend, orbit, right_extend], dim=0)
    return orbit[:, None] + shift * (2*domain / n_samples)


def get_translation_hor_param(
    n: Tensor,
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param n_samples: int
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return:
    """
    n = n[:, None]
    domain = 0.5 if domain is None else domain
    param = torch.linspace(-domain, domain, n_samples)
    param = param.to(n.device)[n]
    return torch.concat([param, torch.zeros_like(param)], dim=-1)


def get_translation_hor_param_orbit(
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None,
    extend: int = 0,
    shift: int = 0
 ) -> Tensor:
    """ Returns a list of parameters to construct an orbit.
    :param n_samples: int
    :param extend: (int) extends over the orbit
    :param shift: (int) shifts the parameters by n increments
                        might be used to center the canonical form in the middle of the list
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return: parameter list
    """
    domain = 0.5 if domain is None else domain
    orbit = torch.linspace(-domain, domain, n_samples)
    if extend > 0:
        left_extend = orbit[0] + torch.tensor([(-2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        right_extend = orbit[-1] + torch.tensor([(2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        orbit = torch.concat([left_extend, orbit, right_extend], dim=0)
    orbit = orbit[:, None] + shift * (2*domain / n_samples)
    return torch.concat([orbit, torch.zeros_like(orbit)], dim=-1)


def get_translation_ver_param(
    n: Tensor,
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param domain: (float) parameters are generated within [-domain, +domain]
    :param n_samples: int
    :return:
    """
    n = n[:, None]
    domain = 0.5 if domain is None else domain
    param = torch.linspace(-domain, domain, n_samples)
    param = param.to(n.device)[n]
    return torch.concat([torch.zeros_like(param), param], dim=-1)


def get_translation_ver_param_orbit(
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None,
    extend: int = 0,
    shift: int = 0
) -> Tensor:
    """ Returns a list of parameters to construct an orbit.
    :param n_samples: int
    :param extend: (int) extends over the orbit
    :param shift: (int) shifts the parameters by n increments
                        might be used to center the canonical form in the middle of the list
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return: parameter list
    """
    domain = 0.5 if domain is None else domain
    orbit = torch.linspace(-domain, domain, n_samples)
    if extend > 0:
        left_extend = orbit[0] + torch.tensor([(-2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        right_extend = orbit[-1] + torch.tensor([(2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        orbit = torch.concat([left_extend, orbit, right_extend], dim=0)
    orbit = orbit[:, None] + shift * (2*domain / n_samples)
    return torch.concat([torch.zeros_like(orbit), orbit], dim=-1)


def get_scaling_param(
    n: Tensor,
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param domain: (float) parameters are generated within [-domain, +domain]
    :param n_samples: int
    :return:
    """
    n = n[:, None]
    domain = 0.5 if domain is None else domain
    param = torch.linspace(-domain, domain, n_samples)
    param = param.to(n.device)[n]
    return torch.concat([param, param], dim=-1)


def get_scaling_param_orbit(
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None,
    extend: int = 0,
    shift: int = 0
) -> Tensor:
    """ Returns a list of parameters to construct an orbit.
    :param n_samples: int
    :param extend: (int) extends over the orbit
    :param shift: (int) shifts the parameters by n increments
                        might be used to center the canonical form in the middle of the list
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return: parameter list
    """
    domain = 0.25 if domain is None else domain
    orbit = torch.linspace(-domain, domain, n_samples)
    if extend > 0:
        left_extend = orbit[0] + torch.tensor([(-2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        right_extend = orbit[-1] + torch.tensor([(2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        orbit = torch.concat([left_extend, orbit, right_extend], dim=0)
    orbit = orbit[:, None] + shift * (2*domain / n_samples)
    return torch.concat([orbit, orbit], dim=-1)


def get_shearing_param(
    n: Tensor,
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param domain: (float) parameters are generated within [-domain, +domain]
    :param n_samples: int
    :return:
    """
    n = n[:, None]
    domain = 0.2 if domain is None else domain
    param = torch.linspace(-domain, domain, n_samples)
    param = param.to(n.device)[n]
    return torch.concat([param, param], dim=-1)


def get_shearing_param_orbit(
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None,
    extend: int = 0,
    shift: int = 0
) -> Tensor:
    """ Returns a list of parameters to construct an orbit.
    :param n_samples: int
    :param extend: (int) extends over the orbit
    :param shift: (int) shifts the parameters by n increments
                        might be used to center the canonical form in the middle of the list
    :param domain: (float) parameters are generated within [-domain, +domain]
    :return: parameter list
    """
    domain = 0.2 if domain is None else domain
    orbit = torch.linspace(-domain, domain, n_samples)
    if extend > 0:
        left_extend = orbit[0] + torch.tensor([(-2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        right_extend = orbit[-1] + torch.tensor([(2*domain / n_samples) * i for i in torch.arange(1, extend + 1)])
        orbit = torch.concat([left_extend, orbit, right_extend], dim=0)
    orbit = orbit[:, None] + shift * (2*domain / n_samples)
    return torch.concat([orbit, torch.zeros_like(orbit)], dim=-1)


def get_reflection_param(
    n: int,
    n_samples: int = 16,
    domain: Optional[Union[Tensor, float]] = None
) -> Tensor:
    """ Creates the discrete sample space on the orbit according to `n_samples`
    and selects the sample at the `n` position.
    :param n: if int: a single sample is returned
              if (batch, ): multiple samples are returned
    :param todo domain: (float) parameters are generated within [-domain, +domain]
    :param n_samples: int
    :return:
    """
    # todo not used yet
    # todo continuous samples are not possible only three options
    discrete_space = torch.randint(-1, 2, (n_samples,))  # -1, 0 or 1
    reflection = torch.concat([discrete_space[n, None], discrete_space[n, None]], axis=-1)
    return reflection


class AffineTransformation(Enum):
    """
    Example: The following calls the `get_translation_hor_param` method with `n=7`:
             AffineTransformation.TRANSLATION.value['param'][0](7)
    """
    TRANSLATION = {
       "matrix": translate,
       "param": (get_translation_hor_param, get_translation_ver_param),
       "orbit": (get_translation_hor_param_orbit, get_translation_ver_param_orbit)
    }
    ROTATION = {
       "matrix": rotate,
       "param": (get_rotation_param, ),
       "orbit": (get_rotation_param_orbit, )
    }
    SCALING = {
       "matrix": scale,
       "param": (get_scaling_param, ),
       "orbit": (get_scaling_param_orbit, )
    }
    SHEARING = {
       "matrix": shear,
       "param": (get_shearing_param, ),
       "orbit": (get_shearing_param_orbit, )
    }
    REFLECTION = { # TODO implement
       "matrix": reflect,
       "param": (get_reflection_param, ),
       # "orbit": (get_reflection_param_orbit, )
    }


def multi_transform(
    x: Tensor,
    trans_func: list,
    n: Tensor,
    domain: Optional[Union[Tensor, list]] = None,
    n_samples: int = 16
) -> Tuple[Tensor, List[Tensor], Tensor]:
    """ This method transforms the given input signal (assumed image)
    according to the specified list of transformations and the number of samples on each orbit.
    The resulting transformation matrix is a composition of these transformations
    with samples along their orbits. This can be used during training and/or testing.
    The order the transformations are applied are always constant following `transformations`
    with a strict following of `n`.
    :param: x: vanilla signal (batch x channel x height x width)
    :param trans_func: list of `AffineTransformation` values
    :param n: sample index along orbits (batch x len(transformations))
    :param domain: (len(trans_func), ) parameters are generated within [-domain, +domain]
    :return: transformed signal (batch x channel x height x width) and transformation (batch x 3 x 3)
    """
    domain = [None for _ in range(len(trans_func))] if domain is None else domain
    T = [identity(x.shape[0], x.device, truncate=False) for _ in range(len(trans_func))]
    T_composed = identity(x.shape[0], x.device, truncate=False)
    # loop over transformation functions
    for t, func in enumerate(trans_func):
        # this is a real loop for ver and hor translation for instance
        for o, param_func in enumerate(func['param']):
            if domain[t][o] != 0.:
                param = param_func(n[:, t], n_samples=n_samples, domain=domain[t][o])
                T[t] = func['matrix'](T[t], param)
                T_composed = func['matrix'](T_composed, param)
    return resampling(x, T_composed), T, T_composed


def orbit_sampling(
    x: Tensor,
    trans_func: Union[List, Tensor],
    n_samples: int = 16,
    T: Optional[Union[List, Tensor]] = None,
    domain: Optional[Union[List, Tensor]] = None,
    extend: int = 1
) -> Tuple[Tensor, List[Tensor]]:
    """ Returns the orbit samples given a signal x along its orbits.
    :param x: (batch x channel x width x height)
    :param trans_func: (batch x orbits) with the latter being `AffineTransformation` values
    :param n_samples: (int) for all batches
    :param T: if None: the identity is used as a base matrix len(trans_func) x (batch x samples x 3 x 3)
              else T is used instead
    :param domain: (len(trans_func), ) parameters are generated within [-domain, +domain]
    :return: orbit samples (batch x orbits x samples x channel x width x height)
             transformation matrices (batch x samples x 3 x 3) for len(trans_func)
    """
    if not isinstance(trans_func, list):
        trans_func = [trans_func, ]
    B, O = x.shape[0], len(trans_func)
    domain = [None for _ in range(len(trans_func))] if domain is None else domain
    T = [identity((B, n_samples + 2 * extend), x.device, truncate=False)
         if T is None else T for _ in range(len(trans_func))]
    for t, func in enumerate(trans_func):
        for o, param_func in enumerate(func['orbit']):
            if domain[t][o] != 0.:
                T[t] = T[t].flatten(end_dim=1)
                param = param_func(n_samples, domain=domain[t][o], extend=extend, shift=0)#, shift=int(n_samples/2))
                param = param.tile((int(T[t].shape[0] / param.shape[0]), 1))
                T[t] = func['matrix'](T[t], param.squeeze(-1).to(T[t].device))
                T[t] = T[t].unflatten(0, (x.shape[0], n_samples + 2 * extend))
    # (orbits x batch x samples x channel x width x height) -> (orbits * batch * samples x channel x width x height)
    x = x[None, :, None].expand((len(trans_func), -1, n_samples + 2 * extend, -1, -1, -1)).flatten(end_dim=2)
    Tx = resampling(x, torch.stack(T).flatten(end_dim=2))
    # (orbits * batch * samples x channel x width x height) -> (batch x orbits x samples x channel x width x height)
    return Tx.unflatten(0, (O, B, n_samples + 2 * extend)).swapaxes(0, 1), T


def get_n(
    n_transformations: int,
    _max: int = 16,
    batch_size: int = 128,
    device: str = 'cuda'
) -> Tensor:
    if _max > 0:
        n = torch.randint(0, _max, size=(batch_size, n_transformations))#.cuda()
    else:
        n = torch.zeros((batch_size, n_transformations), dtype=torch.long)#.cuda()
    return n.to(device)
