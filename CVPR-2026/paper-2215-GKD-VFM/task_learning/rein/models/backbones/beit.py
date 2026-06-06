# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmseg.models.builder import BACKBONES

from mmengine.logging import MMLogger
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# Copyright (c) Open-MMLab. All rights reserved.
import io
import math
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import mmcv
import numpy as np
import torch
import torchvision
from mmengine.fileio import FileClient
from mmengine.fileio import load as load_file
from mmengine.dist import get_dist_info
from mmengine.model import is_model_wrapper
from mmengine import mkdir_or_exist
from scipy import interpolate
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils import model_zoo

ENV_MMCV_HOME = "MMCV_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "mmcv"),
        )
    )

    mkdir_or_exist(mmcv_home)
    return mmcv_home


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_url_dist(url, model_dir=None, map_location="cpu"):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(
            url, model_dir=model_dir, map_location=map_location
        )
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(
                url, model_dir=model_dir, map_location=map_location
            )
    return checkpoint


def load_pavimodel_dist(model_path, map_location=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError("Please install pavi to load checkpoint from modelcloud.")
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            downloaded_file = osp.join(tmp_dir, model.name)
            model.download(downloaded_file)
            checkpoint = torch.load(downloaded_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                downloaded_file = osp.join(tmp_dir, model.name)
                model.download(downloaded_file)
                checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    allowed_backends = ["ceph"]
    if backend not in allowed_backends:
        raise ValueError(f"Load from Backend {backend} is not supported.")
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f"torchvision.models.{name}")
        if hasattr(_zoo, "model_urls"):
            _urls = getattr(_zoo, "model_urls")
            model_urls.update(_urls)
    return model_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], "model_zoo/open_mmlab.json")
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, "open_mmlab.json")
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], "model_zoo/mmcls.json")
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0], "model_zoo/deprecated.json")
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.
    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith("modelzoo://"):
        warnings.warn(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead'
        )
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith("torchvision://"):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith("open-mmlab://"):
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(
                f"open-mmlab://{model_name} is deprecated in favor "
                f"of open-mmlab://{deprecated_urls[model_name]}"
            )
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        # check if is url
        if model_url.startswith(("http://", "https://")):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f"{filename} is not a checkpoint file")
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith("mmcls://"):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(("http://", "https://")):
        checkpoint = load_url_dist(filename)
    elif filename.startswith("pavi://"):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith("s3://"):
        checkpoint = load_fileclient_dist(
            filename, backend="ceph", map_location=map_location
        )
    else:
        if not osp.isfile(filename):
            raise IOError(f"{filename} is not a checkpoint file")
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    # reshape absolute position embedding for Swin
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2
            ).permute(0, 3, 1, 2)

    rank, _ = get_dist_info()
    if "rel_pos_bias.relative_position_bias_table" in state_dict:
        if rank == 0:
            print("Expand the shared relative position embedding to each layers. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = state_dict["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                state_dict[
                    "blocks.%d.attn.relative_position_bias_table" % i
                ] = rel_pos_bias.clone()

        state_dict.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if rank == 0:
                    print(
                        "Position interpolate for %s from %dx%d to %dx%d"
                        % (key, src_size, src_size, dst_size, dst_size)
                    )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.13492:
                #     q = 1.13492

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                if rank == 0:
                    print("x = {}".format(x))
                    print("dx = {}".format(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias

    if "pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if rank == 0:
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict["pos_embed"] = new_pos_embed

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode="bicubic",
                )
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(
                    1, 0
                )

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix="", keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.
    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_model_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict or None, but got {type(meta)}")
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_model_wrapper(model):
        model = model.module

    if hasattr(model, "CLASSES") and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(get_state_dict(model))}
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint["optimizer"] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint["optimizer"] = {}
        for name, optim in optimizer.items():
            checkpoint["optimizer"][name] = optim.state_dict()

    if filename.startswith("pavi://"):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError("Please install pavi to load checkpoint from modelcloud.")
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, "wb") as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, "wb") as f:
            torch.save(checkpoint, f)
            f.flush()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the original BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        #qkv: B,N,3,K,C->3,B,K,N,C
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # attn : B,K,N,C@B,K,N,C->B,K,N,N

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            # relative_position_bias = relative_position_bias[:, 1:, 1:]
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        with_cp=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W, rel_pos_bias=None):
        def _inner_forward(x):
            if self.gamma_1 is None:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
                )
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
                )
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, Hp, Wp


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


# @BACKBONES.register_module()
class BEiT(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        init_values=None,
        use_checkpoint=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        pretrained=None,
        with_cp=False,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm_layer = norm_layer
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.drop_path_rate = drop_path_rate
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_cp=with_cp,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape
                    if use_rel_pos_bias
                    else None,
                )
                for i in range(depth)
            ]
        )

        # if self.pos_embed is not None:
        #     trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.init_weights(pretrained)

        # self.fix_init_weight()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # pretrained = 'pretrained/beit_large_patch16_512_pt22k_ft22kto1k.pth'
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
