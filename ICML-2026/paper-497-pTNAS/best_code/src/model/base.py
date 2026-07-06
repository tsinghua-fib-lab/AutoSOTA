import torch
import torch_frame

from typing import Any, Dict

default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
    torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
    torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
    torch_frame.multicategorical: (
        torch_frame.nn.MultiCategoricalEmbeddingEncoder,
        {},
    ),
    torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
    torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
}


def construct_stype_encoder_dict(
        stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any],
) -> Dict[torch_frame.stype, torch.nn.Module]:
    stype_encoder_dict = {
        stype: stype_encoder_cls_kwargs[stype][0](
            **stype_encoder_cls_kwargs[stype][1]
        )
        for stype in stype_encoder_cls_kwargs.keys()
    }
    return stype_encoder_dict
