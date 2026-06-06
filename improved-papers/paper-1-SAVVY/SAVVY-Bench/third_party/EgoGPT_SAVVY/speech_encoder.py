# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py
"""
We modified the path of speech_encoder in speech_encoder.py to adapt SAVVY-Bench evaluation.
"""

import types

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from egogpt.utils import rank0_print

from .model import ModelDimensions, Whisper


def load_zero_partitions(
    model,
    state_dict,
    is_deepspeed_zero3_enabled,
    pretrained_model_path,
    ignore_mismatched_sizes=False,
):
    """
    adept from pytorch lightning and transformers
    with deepspeed.zero.Init():
        model = MyModel()
    state_dict = torch.load(model_path, map_location="cpu")
    load_zero_partitions(model, prefix="")
    """

    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    loaded_keys = list(state_dict.keys())
    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
    # matching the weights in the model.
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            model_key = checkpoint_key

            if (
                model_key in model_state_dict
                and state_dict[checkpoint_key].shape
                != model_state_dict[model_key].shape
            ):
                mismatched_keys.append(
                    (
                        checkpoint_key,
                        state_dict[checkpoint_key].shape,
                        model_state_dict[model_key].shape,
                    )
                )
                del state_dict[checkpoint_key]
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if is_deepspeed_zero3_enabled:
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(
                list(module.parameters(recurse=False)), modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    load(model_to_load, prefix=start_prefix)
    del state_dict
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
        )
    if len(unexpected_keys) > 0:
        rank0_print(
            f"Some weights of the model checkpoint at {pretrained_model_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        rank0_print(
            f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
        )
    if len(missing_keys) > 0:
        rank0_print(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        rank0_print(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        rank0_print(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )


class WhisperWrappedEncoder(nn.Module):
    def __init__(self, config, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.speech_encoder_name = config.speech_encoder

        if not delay_load:
            rank0_print(f"Loading speech encoder: {self.speech_encoder_name}")
            self.load_model(config)

    def load_model(self, model_config):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.speech_encoder_name
                )
            )
            return

        def replace_layer_norm(module):
            from whisper.model import LayerNorm

            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(
                        child.normalized_shape,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                    )
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        # import whisper
        self.speech_encoder_name = "data/ckpt/large-v3.pt"
        # self.encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        checkpoint = torch.load(self.speech_encoder_name, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        deepspeed3_enabled = True
        load_zero_partitions(
            model,
            checkpoint["model_state_dict"],
            deepspeed3_enabled,
            self.speech_encoder_name,
        )
        self.encoder = model.encoder
        replace_layer_norm(self.encoder)
        self.encoder.requires_grad_(False)

        self.is_loaded = True

    def forward(self, audio):
        return self.encoder(audio)
