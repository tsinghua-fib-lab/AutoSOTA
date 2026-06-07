import logging
import sys
import os
import os.path as osp

from torch import nn
from diffusers.models.embeddings import Timesteps
from diffusers.models.activations import FP32SiLU
from typing import List, Optional, Tuple, Union

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.activations import get_activation


class TimestepEmbedding(nn.Module):
    r"""Modified from TimestepEmbedding in diffusers.models.embeddings
            from diffusers.models.embeddings import TimestepEmbedding
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "hardswish",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class PixArtAlphaTextProjection(nn.Module):
    r"""Modified from PixArtAlphaTextProjection in diffusers.models.embeddings
            from diffusers.models.embeddings import PixArtAlphaTextProjection
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        elif act_fn == "hardswish":
            self.act_1 = nn.Hardswish()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(nn.Module):
    r"""Modified from CombinedTimestepTextProjEmbeddings in diffusers.models.embeddings
            from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
    """

    def __init__(
        self,
        embedding_dim,
        pooled_projection_dim,
        act_fn: str = "hardswish",
        use_pooled_projection: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, act_fn=act_fn)
        if use_pooled_projection:
            self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn=act_fn)
        else:
            self.text_embedder = None

    def forward(
        self,
        timestep,
        pooled_projection=None,
        weight_type=None
    ):
        timesteps_proj = self.time_proj(timestep)
        if pooled_projection is not None:
            timesteps_proj = timesteps_proj.to(dtype=pooled_projection.dtype)
        else:
            timesteps_proj = timesteps_proj.to(dtype=weight_type)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        if pooled_projection is not None:
            pooled_projections = self.text_embedder(pooled_projection)
            conditioning = timesteps_emb + pooled_projections
        else:
            conditioning = timesteps_emb
        return conditioning


TIME_TEXT_EMBEDDING_MODULES = {
    "combinedtimesteptextprojembeddings": CombinedTimestepTextProjEmbeddings,
}

def get_time_text_embedding_module(time_text_embedding_module: str, *args, **kwargs):
    """
    Get the time text embedding model based on the type.

    Args:
        time_text_embedding_module (str): The type of the time text embedding model.
        *args: Positional arguments for the model.
        **kwargs: Keyword arguments for the model.
    Returns:
        nn.Module: The time text embedding model.
    """
    time_text_embedding_module = time_text_embedding_module.lower()
    if time_text_embedding_module in TIME_TEXT_EMBEDDING_MODULES:
        return TIME_TEXT_EMBEDDING_MODULES[time_text_embedding_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported time text embedding type: {time_text_embedding_module}. Supported types are {list(TIME_TEXT_EMBEDDING_MODULES.keys())}")

TIME_EMBEDDING_MODULES = {
    "timestepembedding": TimestepEmbedding,
}


def get_time_embedding_module(time_embedding_module: str, *args, **kwargs):
    """
    Get the time embedding model based on the type.

    Args:
        time_embedding_module (str): The type of the time embedding model.
        *args: Positional arguments for the model.
        **kwargs: Keyword arguments for the model.
    Returns:
        nn.Module: The time embedding model.
    """
    time_embedding_module = time_embedding_module.lower()
    if time_embedding_module in TIME_EMBEDDING_MODULES:
        return TIME_EMBEDDING_MODULES[time_embedding_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported time embedding type: {time_embedding_module}. Supported types are {list(TIME_EMBEDDING_MODULES.keys())}")


TEXT_PROJECTION_MODULES = {
    "pixartalphatextprojection": PixArtAlphaTextProjection,
}


def get_text_projection_module(text_projection_module: str, *args, **kwargs):
    """
    Get the text projection model based on the type.

    Args:
        text_projection_module (str): The type of the text projection model.
        *args: Positional arguments for the model.
        **kwargs: Keyword arguments for the model.
    Returns:
        nn.Module: The text projection model.
    """
    text_projection_module = text_projection_module.lower()
    if text_projection_module in TEXT_PROJECTION_MODULES:
        return TEXT_PROJECTION_MODULES[text_projection_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported text projection type: {text_projection_module}. Supported types are {list(TEXT_PROJECTION_MODULES.keys())}")
