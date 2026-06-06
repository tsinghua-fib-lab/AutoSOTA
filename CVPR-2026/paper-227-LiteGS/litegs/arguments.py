from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    
    @classmethod
    def add_cmdline_arg(cls, DefaultObj:GroupParams, parser: ArgumentParser, fill_none = False):
        group = parser.add_argument_group(cls.__name__)
        for key, value in vars(cls).items():
            if hasattr(value,"__call__") or value.__class__==classmethod:
                continue
            if key.startswith("__"):
                continue

            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = getattr(DefaultObj,key,None) if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)
        return

    @classmethod
    def extract(cls, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(cls) or ("_" + arg[0]) in vars(cls):
                setattr(group, arg[0], arg[1])
        return group
    
    @classmethod
    def get_class_default_obj(cls):
        group = GroupParams()
        for key, value in vars(cls).items():
            if hasattr(value,"__call__") or value.__class__==classmethod:
                continue
            if key.startswith("__"):
                continue
            if key.startswith("_"):
                key = key[1:]
            setattr(group, key, value)
        return group

class ModelParams(ParamGroup): 

    sh_degree = 3
    _source_path = ""
    _model_path = ""
    _images = "images"
    _resolution = -1
    _white_background = False
    data_device = "cuda"
    eval = False

class PipelineParams(ParamGroup):
    cluster_size = 128
    tile_size = 8
    sparse_grad = True
    spatial_refine_interval = 5
    device_preload = True
    enable_transmitance=False
    enable_depth=False
    # DashGaussian parameters
    max_n_gaussian = -1
    max_gaussians_per_tile_at_beginning = 150
    def __init__(self, parser):
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    iterations = 30000
    iterations_per_epoch = 200
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    lambda_dssim = 0.2
    lambda_entropy = 0.0  # Entropy regularization weight
    scale_reset_factor = 0.0
    def __init__(self, parser):
        super().__init__(parser, "Optimization Parameters")

class DensifyParams(ParamGroup):
    densification_interval = 1
    densify_from = 3
    densify_until = -1
    prune_interval = 5
    opacity_reset_interval = 10
    densify_grad_threshold = 0.0001 # small enough so that we can reach to final count with our new strategies
    final_gaussian_count = -1
    opacity_threshold=0.005
    prune_large_point_from=40  # not used
    screen_size_threshold=128#tile
    percent_dense = 0.01
    def __init__(self, parser):
        super().__init__(parser, "Densify Parameters")
