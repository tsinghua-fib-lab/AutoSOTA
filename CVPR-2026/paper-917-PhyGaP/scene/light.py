import torch
import imageio
import numpy as np
import torch.nn.functional as F
from . import renderutils as ru
from .light_utils import *
import nvdiffrast.torch as dr
import imageio
import numpy as np


def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.clip(linear,1e-4,255) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)

def srgb_to_linear_torch(srgb, eps=None):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = ((200 * srgb + 11) / (211)).clamp_min(eps) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def convert_to_real_num_space(x):
    positive_mask = torch.where(x > 0.5)
    negative_mask = torch.where(x <= 0.5)
    x[positive_mask] = x[positive_mask] - 0.5
    x[negative_mask] = inverse_sigmoid(x[negative_mask].clamp(0.001, 0.999)) 
    return x

def convert_to_linear_color(x):
    return torch.where(x > 0, torch.relu(x+0.5), torch.sigmoid(x))

class EnvLight(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )
        
        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()


    def load(self, path, yaw_deg: float = 0.0, pitch_deg: float = 0.0):
        """
        Load an .hdr or .exr environment light map file (equirectangular/latlong),
        optionally rotate it by yaw/pitch (in degrees), then convert it to a cubemap.
        """
        # # load latlong env map from file
        # image = imageio.imread(path)  # Load .hdr file
        # if image.dtype != np.float32:
        #     image = image.astype(np.float32) / 255.0  # Scale to [0,1] if not already in float
        # 从文件中加载图像
        hdr_image = imageio.imread(path)
        # import pdb;pdb.set_trace()
        if hdr_image.dtype != np.float32:
            raise ValueError("HDR image should be in float32 format.")
        image = torch.from_numpy(hdr_image).to(self.device) *  self.scale
        positive_mask = torch.where(image > 0.5)
        negative_mask = torch.where(image <= 0.5)
        image[positive_mask] = image[positive_mask] - 0.5
        image[negative_mask] = inverse_sigmoid(image[negative_mask].clamp(0.001, 0.999)) 
        # 确保图像为浮点类型
        # import pdb;pdb.set_trace()

        # Optional rotation in latlong space (default 0,0 keeps original)
        if (yaw_deg != 0.0) or (pitch_deg != 0.0):
            image = rotate_latlong(image, yaw_deg=yaw_deg, pitch_deg=pitch_deg)

        # Convert from latlong to cubemap format
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)
        # import pdb;pdb.set_trace()
        # Assign the cubemap to the base parameter
        self.base.data = cubemap 

    def build_mips(self, cutoff=0.9999):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        spec_color = convert_to_linear_color(self.specular[-1])#convert from real num parameter space to linear color space
        self.diffuse = convert_to_real_num_space(ru.diffuse_cubemap(spec_color))
        #NOTE: Here remains a problem, the specular cubemap is still in real num space, should be converted to linear color space before use
        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
    
    def get_cube_map(self):
        return torch.where(self.base > 0, torch.relu(self.base+0.5), torch.sigmoid(self.base))

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )
            if torch.isnan(light).any():
                print("NaN detected in direct_light computation!")
                import pdb; pdb.set_trace()

        light = light.view(*prefix, -1)
        
        return torch.where(light > 0, torch.relu(light+0.5), torch.sigmoid(light))
#
# Convenience subclass: GridMapEnvLight
# Keeps all behavior of EnvLight but stores an additional `xyz` attribute
# and provides `load_envmap` to set a cubemap tensor directly into `self.base`
# and rebuild mipmaps.
class GridMapEnvLight(EnvLight):
    """
    GridMapEnvLight extends EnvLight by adding:
      - self.xyz: optional user data (e.g. point cloud xyz) stored on the object
      - load_envmap(cubemap): load a cubemap (numpy or torch tensor) into self.base
        and call build_mips() to regenerate specular/diffuse mips.

    Expected cubemap shapes accepted:
      - (6, H, W, 3)  (faces, height, width, channels)
      - (6, 3, H, W)  (faces, channels, height, width)
    The method will convert/resize to (6, max_res, max_res, 3) on this object's device.
    """

    def __init__(self, *args, xyz=None, **kwargs):
        # forward all arguments to EnvLight
        super().__init__(*args, **kwargs)
        # extra attribute reserved for user (default None)
        self.xyz = torch.from_numpy(xyz)

    def load_cubemap(self, cubemap):
        """
        Load a cubemap (numpy array or torch tensor) into self.base and rebuild mips.

        cubemap: numpy.ndarray or torch.Tensor with shape (6,H,W,3) or (6,3,H,W).
        """
        device = self.device
        # import pdb;pdb.set_trace()
        # convert numpy -> torch if needed
        if isinstance(cubemap, np.ndarray):
            cubemap = torch.from_numpy(cubemap)

        if not torch.is_tensor(cubemap):
            raise TypeError("cubemap must be a numpy array or torch tensor")

        cubemap = cubemap.to(device=device)
        cubemap = cubemap.float()

        # normalize shape conventions to (6, C, H, W)
        if cubemap.ndim == 4 and cubemap.shape[-1] == 3:
            # (6, H, W, 3) -> (6, 3, H, W)
            cubemap_ch = cubemap.permute(0, 3, 1, 2).contiguous()
        elif cubemap.ndim == 4 and cubemap.shape[1] == 3:
            cubemap_ch = cubemap.contiguous()
        else:
            raise ValueError(f"Unsupported cubemap shape: {tuple(cubemap.shape)}")
        
        # resize to this instance's max_res if necessary
        _, _, H, W = cubemap_ch.shape
        if H != self.max_res or W != self.max_res:
            cubemap_ch = F.interpolate(cubemap_ch, size=(self.max_res, self.max_res), mode='bilinear', align_corners=False)
        positive_mask = torch.where(cubemap_ch > 0.5)
        negative_mask = torch.where(cubemap_ch <= 0.5)
        cubemap_ch[positive_mask] = cubemap_ch[positive_mask] - 0.5
        cubemap_ch[negative_mask] = inverse_sigmoid(cubemap_ch[negative_mask].clamp(0.001, 0.999))
        # convert back to stored format (6, H, W, 3)
        cubemap_final = cubemap_ch.permute(0, 2, 3, 1).contiguous()

        with torch.no_grad():
            # assign into parameter buffer so gradients to base are preserved if trainable
            self.base.data = cubemap_final.to(self.device)

        # rebuild mipmaps / specular/diffuse representations
        self.build_mips()
        
#torch.where(self.specular[0]>0, torch.relu(self.specular[0]+0.5), torch.sigmoid(self.specular[0]))
#torch.where(self.diffuse>0, torch.relu(self.diffuse+0.5), torch.sigmoid(self.diffuse))

