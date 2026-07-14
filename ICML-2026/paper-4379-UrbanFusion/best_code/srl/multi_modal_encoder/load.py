import torch
import typing
from omegaconf import ListConfig, DictConfig
from omegaconf.base import ContainerMetadata

from srl.multi_modal_encoder.multi_modal_transformer import (
    MultiModalTransformer,
)
from srl.encoders.coordinate_encoder.coordinate_encoder import (
    CoordinateEncoder
)
from srl.encoders.SVI_encoder.svi_encoder import SVIEncoder
from srl.encoders.RS_encoder.RS_encoder import RSEncoder
from srl.encoders.POI_encoder.text_encoders import TextTransformer
from srl.encoders.OSM_encoder.OSM_encoder import OSMEncoder


def get_urbanfusion(
        checkpoint_path: str,
        precomputed_features: bool = True,
        device: str = "cpu"
) -> torch.nn.Module:
    """
    Initialize the UrbanFusion model and load weights from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .ckpt file (can come from hf_hub_download).
    precomputed_features : bool
        Whether to use precomputed features.
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    torch.nn.Module
        Loaded UrbanFusion model in eval mode.
    """
    # Allow OmegaConf and typing globals for safe unpickling
    torch.serialization.add_safe_globals([
        ListConfig, DictConfig, ContainerMetadata, typing.Any
    ])

    # Define encoders
    coords = CoordinateEncoder(
        positional_encoding_name="direct_no_rad",
        neural_network_name="rff",
        embed_dim=768,
        seq_len=1,
        dim_hidden=512,
        num_layers=2,
        dropout=0.0,
        legendre_polys=10,
        harmonics_calculation="analytic",
        min_radius=None,
        max_radius=None,
        frequency_num=None,
        return_encoding=False,
        precomputed_features=precomputed_features,
        cartesian_3d_branch=False,
        cartesian_3d_input_dropout=1.0
    )

    svi = SVIEncoder(
        embed_dim=768,
        seq_len=1,
        return_encoding=False,
        precomputed_features=precomputed_features
    )

    rs = RSEncoder(
        embed_dim=768,
        seq_len=1,
        return_encoding=False,
        precomputed_features=precomputed_features
    )

    osm = OSMEncoder(
        pretrained_model_name="facebook/vit-mae-base",
        checkpoint_path=None,
        embed_dim=768,
        seq_len=1,
        return_encoding=False,
        precomputed_features=precomputed_features
    )

    poi = TextTransformer(
        model_name="BAAI/bge-small-en-v1.5",
        embed_dim=768,
        head="linear",
        head_hidden_dim=512,
        seq_len=1,
        trainable_layers=0,
        return_encoding=False,
        precomputed_features=precomputed_features
    )

    # Load full checkpoint (requires trust)
    try:
        raw = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        print("✅ Checkpoint loaded:", checkpoint_path)
    except Exception as e:
        print("❌ Failed to load checkpoint:", checkpoint_path)
        raise e

    # Extract and clean state dict
    state_dict = raw.get("state_dict", raw)

    def strip_prefix(k: str) -> str:
        for p in ("model.", "urbanfusion.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    state_dict = {
        strip_prefix(k): v for k, v in state_dict.items()
        if not k.startswith("loss.")
    }

    # Build the model
    model = MultiModalTransformer(
        encoders=[coords, svi, rs, osm, poi],
        transformer_type="single",
        embed_dim=768,
        only_cls=True,
        avg_pool=True,
        depth=1,
        num_heads=8,
        head="hourglass_tiny",
        head_contrastive_dim=512,
        head_hidden_dim=330,
        hourglass_dim=330,
        name_vit_architecture=None,
        pretrained=False,
        first_n_layers=3,
        reg_tokens=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        proj_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        add_positional_encodings=True,
        lora_init="xavier_uniform",
        lora_init_settings=None,
        lora_rank=8,
        lora_chunk_size=None,
        reconstruction_head_dim=3842,
        normalize_embedding=True
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print("❗ Missing keys in state_dict:", missing)
    if unexpected:
        print("❗ Unexpected keys in state_dict:", unexpected)
    else:
        print("✅ Weights loaded successfully.")

    return model.to(device).eval()
