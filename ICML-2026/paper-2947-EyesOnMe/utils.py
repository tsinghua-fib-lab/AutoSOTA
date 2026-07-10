import json
import random
from typing import List, Tuple, Dict, Optional
import config
import re

import torch
import torch.nn.functional as F


def decode_ids(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def embed_text(text: str, tokenizer, model, device=config.DEVICE) -> torch.Tensor:
    """Return L2-normalized embedding [1, d] via mean pooling."""
    toks = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**toks, output_attentions=False)
        hid = out.last_hidden_state
        attn_mask = toks["attention_mask"].unsqueeze(-1).float()
        pooled = (hid * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-6)
        normed = F.normalize(pooled, p=2, dim=-1)
    return normed


@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).item())


def attn_score_one_head(outputs, layer_idx: int, head_idx: int, core_range: range) -> torch.Tensor:
    """
    outputs.attentions: tuple(len=num_layers) of [B, H, S, S]
    Return attention mass directed to core_range (sum over all query positions).
    """
    att = outputs.attentions[layer_idx]
    att_one = att[0, head_idx]
    return att_one[:, list(core_range)].sum()


def generate_random_ids(length: int, emb_layer, device=config.DEVICE) -> torch.Tensor:
    vocab_size = emb_layer.weight.size(0)
    if length <= vocab_size:
        ids = torch.tensor(random.sample(range(vocab_size), length), device=device, dtype=torch.long)
    else:
        ids = torch.randint(0, vocab_size, (length,), device=device)
    return ids


def build_sequence(
    prefix_len: Optional[int],
    suffix_len: Optional[int],
    core_text: str,
    tokenizer,
    model,
    device=config.DEVICE,
    init_mode: str = config.PREFIX_INIT_MODE,
    init_phrase: str = config.GEN_PREFIX_CONTENT,
) -> Tuple[torch.Tensor, range, range, range]:
    """
    Build a token sequence: [prefix | core | suffix].

    init_mode:
        "random" -> random token IDs for prefix/suffix
        "phrase" -> tokenized init_phrase, repeated/truncated to length
    """
    emb_layer = model.get_input_embeddings()

    def make_segment(length: int) -> torch.Tensor:
        if length == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if init_mode == "random":
            vocab_size = emb_layer.weight.size(0)
            return torch.randint(0, vocab_size, (length,), device=device, dtype=torch.long)
        elif init_mode == "phrase":
            init_ids = tokenizer(
                init_phrase, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)[0]
            return init_ids
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    prefix_ids = make_segment(prefix_len)
    core_ids = tokenizer(core_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
    suffix_ids = make_segment(suffix_len)

    seq = torch.cat([prefix_ids, core_ids, suffix_ids], dim=0).to(torch.int64)
    p_range = range(0, prefix_ids.numel())
    c_range = range(prefix_ids.numel(), prefix_ids.numel() + core_ids.numel())
    s_range = range(prefix_ids.numel() + core_ids.numel(), seq.numel())
    return seq, p_range, c_range, s_range


def find_subtensor_index(main_tensor, sub_tensor):
    for i in range(main_tensor.size(0) - sub_tensor.size(0) + 1):
        if torch.equal(main_tensor[i:i + sub_tensor.size(0)], sub_tensor):
            return i
    return None


def is_jailbreak(response_text: str) -> bool:
    return response_text.strip().lower().startswith("hackers")
