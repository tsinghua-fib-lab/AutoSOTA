import logging
from copy import deepcopy

from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Layer,
    DebertaV2Encoder,
    DebertaV2ForSequenceClassification,
    DebertaV2ForMaskedLM,
)

from ..quantize import get_quantized_layer_cls
from ..utils import find_matched_pattern, get_layer_name, set_layer_by_name

logger = logging.getLogger(__name__)


def build_qera_config_deberta_v2(
    model: DebertaV2ForSequenceClassification, qera_config: dict
):
    assert isinstance(model, DebertaV2ForSequenceClassification)
    parsed_config = {}

    encoder: DebertaV2Encoder = model.deberta.encoder

    for module in encoder.modules():
        if not isinstance(module, nn.Linear):
            continue

        fc_name = get_layer_name(model, module)
        matched_entry = find_matched_pattern(fc_name, qera_config.keys())
        assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
        if isinstance(qera_config[matched_entry], str):
            matched_entry = qera_config[matched_entry]

        parsed_config[fc_name] = deepcopy(qera_config[matched_entry])
    return parsed_config


def quantize_deberta_v2(model: DebertaV2ForSequenceClassification, qera_config: dict):
    assert isinstance(model, (DebertaV2ForSequenceClassification, DebertaV2ForMaskedLM))
    qera_config = build_qera_config_deberta_v2(model, qera_config)

    for module in model.deberta.encoder.modules():
        if not isinstance(module, nn.Linear):
            continue

        fc_name = get_layer_name(model, module)
        if fc_name not in qera_config:
            raise ValueError(f"Cannot find {fc_name} in qera_config")

        quantized_fc_cls = get_quantized_layer_cls("linear", qera_config[fc_name])
        new_fc = quantized_fc_cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            q_config=qera_config[fc_name],
        )
        new_fc.load_state_dict(module.state_dict())
        set_layer_by_name(model, fc_name, new_fc)

    model._no_split_modules = ["DebertaV2Layer"]

    return model


def find_layers_to_register_scale_hook_deberta_v2(
    model: DebertaV2ForSequenceClassification,
):
    assert isinstance(model, (DebertaV2ForSequenceClassification, DebertaV2ForMaskedLM))
    assert model.config._attn_implementation == "eager"
    layers_to_register = []

    decoder_layer: DebertaV2Layer
    for decoder_layer in model.deberta.encoder.layer:
        k_name = get_layer_name(model, decoder_layer.attention.self.key_proj)
        q_name = get_layer_name(model, decoder_layer.attention.self.query_proj)
        v_name = get_layer_name(model, decoder_layer.attention.self.value_proj)
        layers_to_register.append(
            dict(target_layer=k_name, layers_sharing_scale=[q_name, v_name])
        )

        o_name = get_layer_name(model, decoder_layer.attention.output.dense)
        layers_to_register.append(dict(target_layer=o_name, layers_sharing_scale=[]))

        fc1_name = get_layer_name(model, decoder_layer.intermediate.dense)
        layers_to_register.append(dict(target_layer=fc1_name, layers_sharing_scale=[]))

        fc2_name = get_layer_name(model, decoder_layer.output.dense)
        layers_to_register.append(dict(target_layer=fc2_name, layers_sharing_scale=[]))

    return layers_to_register


def find_layers_to_approximate_deberta_v2(model: DebertaV2ForSequenceClassification):
    assert isinstance(model, (DebertaV2ForSequenceClassification, DebertaV2ForMaskedLM))
    layers_to_approximate = []
    for layer_name, layer in model.deberta.encoder.layer.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        layer_name = get_layer_name(model, layer)
        layers_to_approximate.append(layer_name)
    return layers_to_approximate
