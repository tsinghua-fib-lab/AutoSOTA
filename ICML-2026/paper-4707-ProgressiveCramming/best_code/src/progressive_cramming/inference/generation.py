from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast


@torch.no_grad()
def generate_from_compression(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    compression_token_embeddings: torch.Tensor,  # [1, compression, hidden]
    max_new_tokens: int,
    num_return_sequences: int = 1,
    add_noise=False,
    random_position_ids: bool = False,
    return_generated_ids: bool = False,
) -> list[str] | tuple[list[str], torch.Tensor]:
    """Generates a sequence starting from compressed embeddings."""
    # Cast to the same device
    device = compression_token_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    # Add pad_token to a tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eos_token_id = tokenizer.eos_token_id

    # Prepare batch of prefixes
    if num_return_sequences > 1:
        compression_token_embeddings = compression_token_embeddings.expand(
            num_return_sequences, -1, -1
        )  # [batch, sequence, hidden]

    if add_noise:
        noise = torch.randn_like(compression_token_embeddings) * 0.01
        compression_token_embeddings += noise

    batch_size, num_compression_tokens, hidden_size = compression_token_embeddings.shape

    # Container for generated token ids
    generated_token_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)  # [batch, 0]
    # Model's input embeddings layer
    input_embeddings = model.get_input_embeddings()
    torch_dtype = input_embeddings.weight.dtype

    for _ in range(max_new_tokens):
        # Embeddings
        if generated_token_ids.size(1) == 0:
            generated_embeddings = torch.empty((batch_size, 0, hidden_size), device=device)  # [batch, 0, hidden]
        else:
            generated_embeddings = input_embeddings(generated_token_ids)  # [batch, sequence, hidden]
        united_token_embeddings = torch.cat(
            (compression_token_embeddings, generated_embeddings), dim=1
        )  # [batch, compression + sequence, hidden]
        united_token_embeddings = united_token_embeddings.to(torch_dtype)

        # Attention mask
        compression_attention_mask = torch.ones(
            (batch_size, num_compression_tokens), dtype=torch.long, device=device
        )  # [batch, compression]
        attention_mask = torch.ones(
            (batch_size, generated_embeddings.size(1)), dtype=torch.long, device=device
        )  # [batch, sequence]
        united_attention_mask = torch.cat(
            (compression_attention_mask, attention_mask), dim=1
        )  # [batch, compression + sequence]

        if random_position_ids:
            position_ids = (
                torch.randperm(united_token_embeddings.size(1), device=device).unsqueeze(dim=0).repeat(batch_size, 1)
            )  # [batch, compression + sequence]
            outputs = model(
                inputs_embeds=united_token_embeddings,
                attention_mask=united_attention_mask,
                position_ids=position_ids,
            )
        else:
            outputs = model(
                inputs_embeds=united_token_embeddings,
                attention_mask=united_attention_mask,
            )
        logits = outputs.logits[:, -1, :]  # [batch, vocabulary]
        next_token_ids = torch.argmax(logits, dim=-1)  # [batch]

        # If a sequence already reached EOS token leave EOS to the end
        if eos_token_id is not None:
            if generated_token_ids.size(1) > 0:
                reached_eos = generated_token_ids[:, -1].eq(eos_token_id)
                next_token_ids = torch.where(
                    reached_eos,
                    torch.full_like(next_token_ids, eos_token_id),
                    next_token_ids,
                )

        generated_token_ids = torch.cat((generated_token_ids, next_token_ids.unsqueeze(-1)), dim=-1)  # [batch, sequence]

        # Stop early if all sequences just produced eos and had eos previously
        if eos_token_id is not None and torch.all(next_token_ids.eq(eos_token_id)):
            break

    texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    return texts, generated_token_ids if return_generated_ids else texts


@torch.no_grad()
def calculate_logits(
    model: PreTrainedModel,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    sequence_embeddings: torch.Tensor,  # [1, sequence, hidden]
    attention_mask: torch.Tensor,  # [1, sequence]
) -> torch.Tensor:
    """Calculate logits for a sequence."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    united_embeddings = torch.cat(
        (compressed_embeddings, sequence_embeddings),
        dim=1,
    )  # [1, mem + sequence, hidden]
    united_attention_mask = torch.cat(
        (
            torch.ones(
                compressed_embeddings.size(0),
                compressed_embeddings.size(1),
                dtype=torch.long,
                device=device,
            ),
            attention_mask,
        ),
        dim=1,
    )  # [1, mem + sequence]
    outputs = model(
        inputs_embeds=united_embeddings,
        attention_mask=united_attention_mask,
    )
    logits = outputs.logits  # [batch, mem + sequence, vocabulary]
    return logits


@torch.no_grad()
def calculate_outputs(
    model: PreTrainedModel,
    compressed_embeddings: torch.Tensor,
    sequence_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> CausalLMOutputWithPast:
    """Calculate outputs for a sequence."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)

    # Required to enable output_attentions
    model.set_attn_implementation("eager")
    model.eval()

    united_embeddings = torch.cat(
        (compressed_embeddings, sequence_embeddings),
        dim=1,
    )
    united_attention_mask = torch.cat(
        (
            torch.ones(
                compressed_embeddings.size(0),
                compressed_embeddings.size(1),
                dtype=torch.long,
                device=device,
            ),
            attention_mask,
        ),
        dim=1,
    )
    outputs = model(
        inputs_embeds=united_embeddings,
        attention_mask=united_attention_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    return outputs
