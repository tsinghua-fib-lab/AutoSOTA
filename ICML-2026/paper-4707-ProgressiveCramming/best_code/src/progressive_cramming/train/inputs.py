import torch


def build_compression_attention_mask(
    batch_size: int,
    num_compression_tokens: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compression tokens attention mask."""
    compression_attention_mask = torch.ones(
        (batch_size, num_compression_tokens),
        dtype=dtype,
        device=device,
    )
    return compression_attention_mask


def build_united_input(
    compression_token_embeddings: torch.Tensor,  # [batch, compression, hidden]
    compression_attention_mask: torch.Tensor,  # [batch, compression]
    token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
    attention_mask: torch.Tensor,  # [batch, sequence]
    prefix_token_embeddings: torch.Tensor | None = None,  # [batch, prefix, hidden]
    prefix_attention_mask: torch.Tensor | None = None,  # [batch, prefix]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather up compression, optional uncompressed prefix, and sequence token embeddings + masks.

    Layout: ``[compression] [prefix?] [sequence]``. The prefix block (real, uncompressed tokens the
    model attends to but never compresses) is inserted between the compression tokens and the
    sequence when ``prefix_token_embeddings`` is provided. With no prefix this is identical to the
    original ``[compression] [sequence]`` concatenation.
    """
    compression_token_embeddings = compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype)
    embedding_blocks = [compression_token_embeddings]
    mask_blocks = [compression_attention_mask]
    if prefix_token_embeddings is not None:
        embedding_blocks.append(prefix_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype))
        if prefix_attention_mask is None:
            raise ValueError("prefix_attention_mask is required when prefix_token_embeddings is provided!")
        mask_blocks.append(prefix_attention_mask)
    embedding_blocks.append(token_embeddings)
    mask_blocks.append(attention_mask)
    united_token_embeddings = torch.cat(embedding_blocks, dim=1)
    united_attention_mask = torch.cat(mask_blocks, dim=1)
    return united_token_embeddings, united_attention_mask
