#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from abc import ABC, abstractmethod
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import utils.general_utils as utils

lr_scale_fns = {
    "linear": lambda x: x,
    "sqrt": lambda x: np.sqrt(x),
}


class BaseGaussianModel(ABC):
    """Base class for Gaussian models with common functionality"""

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, only_for_rendering: bool = False):
        args = utils.get_args()
        self.args = args

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._parameters = torch.empty(0)
        self.param_dims = torch.empty(0)
        self.param_dims_presum_rshift = torch.empty(0)
        self.col2attr = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.parameters_buffer = torch.empty(0)
        self.parameters_grad_buffer = torch.empty(0)
        self.only_for_rendering = only_for_rendering
        self.setup_functions()
        self.device = self._get_device()

    @abstractmethod
    def _get_device(self):
        """Return the device type for this model"""
        pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @abstractmethod
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """Initialize parameters from point cloud data"""
        pass

    @abstractmethod
    def all_parameters(self):
        """Return list of all parameters for this model"""
        pass

    @abstractmethod
    def training_setup(self, training_args):
        """Setup optimizer and training parameters"""
        pass

    def log_gaussian_stats(self):
        # log the statistics of the gaussian model
        # number of total 3dgs on this rank
        num_3dgs = self._xyz.shape[0]
        # average size of 3dgs
        avg_size = torch.mean(torch.max(self.get_scaling, dim=1).values).item()
        # average opacity
        avg_opacity = torch.mean(self.get_opacity).item()
        stats = {
            "num_3dgs": num_3dgs,
            "avg_size": avg_size,
            "avg_opacity": avg_opacity,
        }

        # get the exp_avg, exp_avg_sq state for all parameters
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    exp_avg_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg"], dim=-1)
                    ).item()
                    exp_avg_sq_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg_sq"], dim=-1)
                    ).item()
        return stats, exp_avg_dict, exp_avg_sq_dict

    @abstractmethod
    def update_learning_rate(self, iteration):
        """Update learning rate for the current iteration"""
        pass

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        features_dc_elems = (
            self._features_dc.shape[1] * self._features_dc.shape[2]
            if len(self._features_dc.shape) == 3
            else self._features_dc.shape[1]
        )
        features_rest_elems = (
            self._features_rest.shape[1] * self._features_rest.shape[2]
            if len(self._features_rest.shape) == 3
            else self._features_rest.shape[1]
        )
        for i in range(features_dc_elems):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest_elems):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        """Save model to PLY file"""
        args = utils.get_args()
        _xyz = _features_dc = _features_rest = _opacity = _scaling = _rotation = None
        utils.log_cpu_memory_usage("start save_ply")

        # Directly use local tensors
        _xyz = self._xyz
        _features_dc = self._features_dc
        _features_rest = self._features_rest
        _opacity = self._opacity
        _scaling = self._scaling
        _rotation = self._rotation

        mkdir_p(os.path.dirname(path))

        xyz = _xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            _features_dc.detach()
            .contiguous()
            .view(-1, 1, 3)
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            _features_rest.detach()
            .contiguous()
            .view(-1, 15, 3)
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = _opacity.detach().cpu().numpy()
        scale = _scaling.detach().cpu().numpy()
        rotation = _rotation.detach().cpu().numpy()

        utils.log_cpu_memory_usage("after change gpu tensor to cpu numpy")

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        utils.log_cpu_memory_usage(
            "after change numpy to plyelement before writing ply file"
        )
        PlyData([el]).write(path)
        utils.log_cpu_memory_usage("finish write ply file")

    def load_raw_ply(self, path):
        print("Loading ", path)
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        args = utils.get_args()

        if args.drop_initial_3dgs_p > 0.0:
            # drop each point with probability args.drop_initial_3dgs_p
            drop_mask = np.random.rand(xyz.shape[0]) > args.drop_initial_3dgs_p
            xyz = xyz[drop_mask]
            features_dc = features_dc[drop_mask]
            features_extra = features_extra[drop_mask]
            scales = scales[drop_mask]
            rots = rots[drop_mask]
            opacities = opacities[drop_mask]

        return xyz, features_dc, features_extra, opacities, scales, rots

    @abstractmethod
    def load_ply(self, path):
        """Load model from PLY file"""
        pass

    @abstractmethod
    def reset_opacity(self):
        """Reset opacity values"""
        pass

    def prune_based_on_opacity(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        utils.LOG_FILE.write(
            "Pruning based on opacity. Percent: {:.2f}\n".format(
                100 * prune_mask.sum().item() / prune_mask.shape[0]
            )
        )
        self.prune_points(prune_mask)

    @abstractmethod
    def prune_points(self, mask):
        """Prune points based on mask"""
        pass

    @abstractmethod
    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_parameters=None,
    ):
        """Post-processing after densification"""
        pass

    @abstractmethod
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split Gaussians based on gradients"""
        pass

    @abstractmethod
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians based on gradients"""
        pass

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        args = utils.get_args()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # NOTE: this is bug in its implementation.
            assert torch.all(
                self.max_radii2D == 0
            ), "In its implementation, max_radii2D is all 0. This is a bug."
            assert torch.all(
                big_points_vs == False
            ), "In its implementation, big_points_vs is all False. This is a bug."
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    @abstractmethod
    def gsplat_add_densification_stats_exact_filter(
        self,
        viewspace_point_tensor_grad,
        radii,
        send2gpu_final_filter_indices,
        width,
        height,
    ):
        pass
