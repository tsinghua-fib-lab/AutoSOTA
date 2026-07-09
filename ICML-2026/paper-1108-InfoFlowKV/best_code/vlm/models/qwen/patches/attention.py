"""Patch for extracting attention from Qwen3-VL during generation."""

import torch
import json
from typing import List, Tuple, Optional

from ...base import BasePatch


class AttentionPatch(BasePatch):
    """
    Extract attention scores during autoregressive generation.
    Only extracts attention from generated tokens to image tokens.

    Usage:
        with AttentionPatch(model, processor, layer_indices=[0, 10, 20]) as patch:
            outputs = model.generate(..., output_attentions=True, return_dict_in_generate=True)
            attention = patch.extract(outputs, input_ids, input_len)
    """

    def __init__(self, model, processor, layer_indices: Optional[List[int]] = None):
        self.model = model
        self.processor = processor
        self.layer_indices = layer_indices
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Captured data
        self.attention = None
        self.attention_info = None

    def apply(self):
        # No forward patching needed, just prepare for extraction
        pass

    def remove(self):
        pass

    def clear(self):
        self.attention = None
        self.attention_info = None

    def get_image_token_ranges(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """Find contiguous ranges of image tokens in input_ids."""
        if input_ids.dim() == 2:
            input_ids = input_ids[0]

        input_ids = input_ids.cpu()
        is_image_token = (input_ids == self.image_token_id).numpy()

        ranges = []
        start = None
        for i, is_img in enumerate(is_image_token):
            if is_img and start is None:
                start = i
            elif not is_img and start is not None:
                ranges.append((start, i))
                start = None
        if start is not None:
            ranges.append((start, len(is_image_token)))

        return ranges

    def extract(self, outputs, input_ids: torch.Tensor, input_len: int) -> torch.Tensor:
        """
        Extract attention from generation outputs.

        Args:
            outputs: GenerateOutput with attentions attribute
            input_ids: Input token ids (to find image token positions)
            input_len: Length of input sequence before generation

        Returns:
            Tensor of shape (num_generated_tokens, num_layers, num_heads, num_image_tokens)
        """
        image_ranges = self.get_image_token_ranges(input_ids)
        attentions = outputs.attentions

        num_generated_tokens = len(attentions)
        total_layers = len(attentions[0])
        selected_layers = self.layer_indices if self.layer_indices else list(range(total_layers))
        num_selected_layers = len(selected_layers)
        total_image_tokens = sum(end - start for start, end in image_ranges)
        num_heads = attentions[0][0].shape[1]

        result = torch.zeros(
            num_generated_tokens, num_selected_layers, num_heads, total_image_tokens,
            dtype=attentions[0][0].dtype,
            device=attentions[0][0].device
        )

        for gen_idx, gen_step_attentions in enumerate(attentions):
            for out_idx, layer_idx in enumerate(selected_layers):
                attn = gen_step_attentions[layer_idx][0, :, 0, :]  # (heads, seq_len)
                img_attention = torch.cat([attn[:, s:e] for s, e in image_ranges], dim=1)
                result[gen_idx, out_idx] = img_attention

        self.attention = result
        self.attention_info = {
            "num_image_tokens": total_image_tokens,
            "num_generated_tokens": num_generated_tokens,
            "image_ranges": image_ranges,
            "attention_shape": list(result.shape),
            "layer_indices": selected_layers,
        }

        return result

    def save(self, output_dir: str, sample_idx: int):
        """Save attention and info to files."""
        import os
        if self.attention is not None:
            torch.save(self.attention.cpu(), os.path.join(output_dir, f"sample_{sample_idx}.pt"))
        if self.attention_info is not None:
            with open(os.path.join(output_dir, f"sample_{sample_idx}_info.json"), "w") as f:
                json.dump(self.attention_info, f, indent=2)
