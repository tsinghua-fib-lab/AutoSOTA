"""Helper functions for the T2T-ViT attention experiments."""

import os
import argparse
import torch

CHECKPOINTPATH = "checkpoints"
DATASETPATH = "data/imagenet"


def get_base_parser() -> argparse.ArgumentParser:
    """Get the base parser for the T2T-ViT attention experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        "-s",
        default=1,
        type=int,
        help="random seed for both pytorch and thinformer",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        help="PyTorch device: e.g., cuda or cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--ckpt_path",
        "-cp",
        default=CHECKPOINTPATH,
        help="directory containing 82.6_T2T_ViTt_24.pth.tar",
    )
    parser.add_argument(
        "--dataset_path",
        "-dp",
        default=DATASETPATH,
        help="directory containing ImageNet val folder",
    )
    parser.add_argument(
        "--output_path", "-op", default="out", help="directory for storing output"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="if set, overwrite existing output file even when it exists",
    )
    parser.add_argument(
        "--rank1",
        "-r1",
        default=224,
        type=int,
        help="wildcat rank parameter for attention layer 1",
    )
    parser.add_argument(
        "--rank2",
        "-r2",
        default=196,
        type=int,
        help="wildcat rank parameter for attention layer 2",
    )
    parser.add_argument(
        "--bins1",
        "-b1",
        default=224,
        type=int,
        help="wildcat bins parameter for attention layer 1",
    )
    parser.add_argument(
        "--bins2",
        "-b2",
        default=196,
        type=int,
        help="wildcat bins parameter for attention layer 2",
    )
    return parser


#
# Model utils
#
def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    """Helper for loading the T2T-ViT checkpoint"""
    # fix from Insu >>>>
    state_dict = torch.load(checkpoint_path, map_location=device)

    # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
    import re

    if state_dict.keys() == {"state_dict_ema"}:
        state_dict = state_dict["state_dict_ema"]

    # Replace the names of some of the submodules
    def key_mapping(key: str) -> str:
        if key == "pos_embed":
            return "pos_embed.pe"
        elif key.startswith("tokens_to_token."):
            return re.sub("^tokens_to_token.", "patch_embed.", key)
        else:
            return key

    state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    # <<<< END

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # NOTE: Performer & Scatterbrain also have a projection_matrix term
    # that is not in the original T2T checkpoint
    # these projection matrices are initialized by the module
    # so we don't have to worry about them
    print(f"missing keys: {missing_keys}")
    assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"


# Load model using specified attention method
def get_model(
    method1: str,
    method2: str,
    args: argparse.Namespace,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Load the T2T-ViT model with the specified attention methods."""
    from model.t2t_vit import t2t_vit_t_24
    from model.attn_cfgs import get_attn_cfg1, get_attn_cfg2

    attn_cfg_1 = get_attn_cfg1(method1, args)
    attn_cfg_2 = get_attn_cfg2(method2, args)
    print(f"attn_cfg_1: {attn_cfg_1}")
    print(f"attn_cfg_2: {attn_cfg_2}")

    checkpoint_path = os.path.join(ckpt_path, "82.6_T2T_ViTt_24.pth.tar")
    model = t2t_vit_t_24(
        **{
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "img_size": 224,
            "t2tattn1_cfg": attn_cfg_1,
            "t2tattn2_cfg": attn_cfg_2,
        }
    )
    load_checkpoint(model, checkpoint_path, device)
    # put model (e.g., dropout, batch norm layers) in evaluation mode
    model.eval()
    model = model.to(device=device, dtype=dtype)

    return model


#
# Timing utils
#
def get_modules(model: torch.nn.Module) -> dict:
    """Get the modules of the T2T-ViT model."""
    modules = {
        "attention1.attn.attention_layer": model.patch_embed.attention1.attn.attention_layer,
        "attention2.attn.attention_layer": model.patch_embed.attention2.attn.attention_layer,
    }
    return modules
