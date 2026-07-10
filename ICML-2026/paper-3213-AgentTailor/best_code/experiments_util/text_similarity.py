

from typing import Any

import torch

from AgentTailor.ATNetwork.Critics import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _encode_text(encoder: Encoder, text: Any) -> torch.Tensor:
    vec = encoder.model.encode(
        str(text),
        convert_to_numpy=True,
        device=DEVICE,
        normalize_embeddings=True,
    )
    return torch.tensor(vec, device=DEVICE, dtype=torch.float32, requires_grad=False)


def compute_text_similarity(encoder: Encoder, text_a: str, text_b: str) -> float:

    if not text_a or not text_b:
        return 0.0
    vec_a = _encode_text(encoder, text_a)
    vec_b = _encode_text(encoder, text_b)
    return torch.dot(vec_a, vec_b).item()


__all__ = ["compute_text_similarity"]


