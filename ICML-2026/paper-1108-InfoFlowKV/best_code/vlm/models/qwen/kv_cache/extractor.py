"""VLM-aware KV cache extraction for Qwen3-VL."""

import torch
from typing import Dict, List, Optional, Tuple
from transformers.cache_utils import DynamicCache

from .base import KVCacheData
from .chunker import ImageChunker
from .chunk_prefiller import ChunkPrefiller, KVCacheConcatenator
from ..patches import TextPatch


class VLMKVCacheExtractor:
    """
    Extract KV cache from Qwen3-VL during prefill.

    Handles mixed image + text inputs and captures all information
    needed for later recomputation (position_ids, position_embeddings, etc.).

    Usage:
        extractor = VLMKVCacheExtractor(model)
        kv_data = extractor.extract(inputs)
    """

    # Qwen3-VL uses these special tokens for image placeholders
    IMAGE_TOKEN_ID = 151655  # <|image_pad|>
    VIDEO_TOKEN_ID = 151656  # <|video_pad|>

    def __init__(self, model):
        """
        Args:
            model: The Qwen3-VL model (AutoModelForImageTextToText)
        """
        self.model = model
        self.language_model = model.model.language_model

    @torch.no_grad()
    def extract(
        self,
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
        chunk_k: Optional[int] = None,
    ) -> KVCacheData:
        """
        Run prefill and extract KV cache with all necessary metadata.

        Args:
            inputs: Processor output dict (input_ids, pixel_values, etc.)
            return_logits: Whether to include logits in output (for debugging)

        Returns:
            KVCacheData containing:
                - past_key_values: DynamicCache with K, V tensors
                - position_ids: [3, batch, seq_len] for MRoPE
                - position_embeddings: (cos, sin) from rotary embedding
                - input_embeds: [batch, seq_len, hidden_size]
                - image_ranges: List of (start, end) for image tokens
        """
        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("inputs must contain 'input_ids'")

        # Find image token ranges before forward pass
        image_ranges = self.find_image_token_ranges(input_ids)

        # For chunk_k=0 or chunk_k=1 with images, use chunked extraction with SDPA
        # This ensures both use the same attention backend for fair comparison
        if len(image_ranges) >= 1 and self._can_chunk(inputs, 1):
            # chunk_k=0 internally uses chunk_k=1 (single chunk) for consistent SDPA attention
            effective_chunk_k = chunk_k if chunk_k else 1
            return self._extract_chunked(inputs, input_ids, image_ranges, effective_chunk_k)

        if chunk_k and len(image_ranges) >= 1:
            raise ValueError("chunk_k set but image_grid_thw is missing; cannot chunk with padding.")

        # Fallback: Use native forward for text-only inputs (no images)
        with TextPatch(self.language_model, capture_hidden_states=False) as patch:
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )

            # Get captured data
            position_ids = patch.outputs.position_ids
            position_embeddings = patch.outputs.position_embeddings
            input_embeds = patch.outputs.input_embeds

        # Get KV cache from outputs
        past_key_values = outputs.past_key_values
        seq_len = input_ids.shape[1]

        return KVCacheData(
            past_key_values=past_key_values,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            input_embeds=input_embeds,
            input_ids=input_ids.clone(),
            seq_len=seq_len,
            image_ranges=image_ranges,
        )

    def _extract_chunked(
        self,
        inputs: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        image_ranges: List[Tuple[int, int]],
        chunk_k: int,
    ) -> KVCacheData:
        """
        Run chunked prefill for image tokens and return chunk-ordered KVCacheData.
        Supports both single-image and multi-image inputs.
        """
        # Build chunk_info for each image
        image_grid_thw = inputs["image_grid_thw"]
        spatial_merge_size = 2  # Qwen3-VL uses 2x2 spatial merge
        chunker = ImageChunker(k=chunk_k)

        chunk_infos = []
        for i, (image_start_idx, image_end_idx) in enumerate(image_ranges):
            _, patch_h, patch_w = image_grid_thw[i].tolist()
            grid_h = patch_h // spatial_merge_size
            grid_w = patch_w // spatial_merge_size

            chunk_info = chunker.compute_chunk_indices(
                grid_h=grid_h,
                grid_w=grid_w,
                image_start_idx=image_start_idx,
                device=input_ids.device,
                allow_padding=True,
            )
            chunk_infos.append(chunk_info)

        with TextPatch(self.language_model, capture_hidden_states=False) as patch:
            self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

        input_embeds = patch.outputs.input_embeds
        position_ids = patch.outputs.position_ids
        position_embeddings = patch.outputs.position_embeddings
        visual_pos_masks = patch.outputs.visual_pos_masks
        deepstack_visual_embeds = patch.outputs.deepstack_visual_embeds

        prefiller = ChunkPrefiller(self.model)
        concatenator = KVCacheConcatenator()
        seq_len = input_ids.shape[1]

        if len(image_ranges) == 1:
            # Single-image: use original prefill_chunks path (verified equivalent to baseline)
            chunk_info = chunk_infos[0]
            image_start_idx, image_end_idx = image_ranges[0]

            chunked_cache = prefiller.prefill_chunks(
                input_embeds=input_embeds,
                position_embeddings=position_embeddings,
                chunk_info=chunk_info,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                include_text=True,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            past_key_values = concatenator.concatenate(chunked_cache)
            full_indices = self._build_chunked_indices(
                seq_len, image_start_idx, image_end_idx, chunk_info, input_ids.device
            )
        else:
            # Multi-image: use prefill_multi_image path
            multi_cache = prefiller.prefill_multi_image(
                input_embeds=input_embeds,
                position_embeddings=position_embeddings,
                chunk_infos=chunk_infos,
                image_ranges=image_ranges,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            past_key_values = concatenator.concatenate_multi_image(multi_cache)
            full_indices = self._build_chunked_indices_multi(
                seq_len, image_ranges, chunk_infos, input_ids.device
            )

        input_ids = input_ids.index_select(1, full_indices)
        input_embeds = input_embeds.index_select(1, full_indices)
        position_ids = position_ids.index_select(2, full_indices)
        position_embeddings = (
            self._reorder_position_embeddings(position_embeddings[0], full_indices),
            self._reorder_position_embeddings(position_embeddings[1], full_indices),
        )

        return KVCacheData(
            past_key_values=past_key_values,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            input_embeds=input_embeds,
            input_ids=input_ids,
            seq_len=seq_len,
            image_ranges=image_ranges,
        )

    def _build_position_ids(
        self,
        inputs: Dict[str, torch.Tensor],
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Build 3D position_ids compatible with Qwen3-VL MRoPE."""
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            cache_position = torch.arange(
                0, input_embeds.shape[1], device=input_embeds.device
            )
            position_ids = cache_position.view(1, 1, -1).expand(3, input_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        return position_ids

    def _can_chunk(self, inputs: Dict[str, torch.Tensor], chunk_k: int) -> bool:
        """Check if chunking metadata is available."""
        image_grid_thw = inputs.get("image_grid_thw")
        return image_grid_thw is not None

    def _build_chunked_indices(
        self,
        seq_len: int,
        image_start_idx: int,
        image_end_idx: int,
        chunk_info,
        device: torch.device,
    ) -> torch.Tensor:
        prefix = torch.arange(0, image_start_idx, device=device)
        suffix = torch.arange(image_end_idx, seq_len, device=device)
        return torch.cat([prefix, chunk_info.reorder_indices, suffix], dim=0)

    def _build_chunked_indices_multi(
        self,
        seq_len: int,
        image_ranges: List[Tuple[int, int]],
        chunk_infos: List,
        device: torch.device,
    ) -> torch.Tensor:
        """Build reordered indices for multi-image sequence.

        Output order: [prefix][img1_reordered][inter1][img2_reordered][inter2]...[imgN_reordered][suffix]
        """
        parts = []

        # Prefix (before first image)
        first_img_start = image_ranges[0][0]
        if first_img_start > 0:
            parts.append(torch.arange(0, first_img_start, device=device))

        # Process each image and inter-text
        for i, ((img_start, img_end), chunk_info) in enumerate(zip(image_ranges, chunk_infos)):
            # Add reordered image indices
            parts.append(chunk_info.reorder_indices)

            # Add inter-text indices (if not last image)
            if i < len(image_ranges) - 1:
                next_img_start = image_ranges[i + 1][0]
                if img_end < next_img_start:
                    parts.append(torch.arange(img_end, next_img_start, device=device))

        # Suffix (after last image)
        last_img_end = image_ranges[-1][1]
        if last_img_end < seq_len:
            parts.append(torch.arange(last_img_end, seq_len, device=device))

        return torch.cat(parts, dim=0)

    def _split_visual_masks(
        self,
        visual_pos_masks: Optional[torch.Tensor],
        image_ranges: List[Tuple[int, int]],
    ) -> Optional[List[torch.Tensor]]:
        """Split visual position masks by image.

        Assumes visual_pos_masks has shape [batch, total_visual_tokens] where
        visual tokens are concatenated in image order.
        """
        if visual_pos_masks is None:
            return None

        masks_list = []
        offset = 0
        for img_start, img_end in image_ranges:
            num_tokens = img_end - img_start
            # Extract mask for this image
            if visual_pos_masks.dim() == 2:
                mask = visual_pos_masks[:, offset:offset + num_tokens]
            else:
                mask = visual_pos_masks[offset:offset + num_tokens]
            masks_list.append(mask)
            offset += num_tokens

        return masks_list

    def _split_deepstack_embeds(
        self,
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        image_ranges: List[Tuple[int, int]],
    ) -> Optional[List[List[torch.Tensor]]]:
        """Split deepstack embeddings by image.

        Assumes each tensor in deepstack_visual_embeds has visual tokens concatenated
        in image order.
        """
        if deepstack_visual_embeds is None:
            return None

        # For each image, extract its portion from each layer's embeddings
        embeds_list = []
        for i, (img_start, img_end) in enumerate(image_ranges):
            num_tokens = img_end - img_start
            offset = sum(e - s for s, e in image_ranges[:i])

            image_embeds = []
            for layer_embeds in deepstack_visual_embeds:
                if layer_embeds.dim() == 3:
                    # [batch, seq, hidden]
                    image_embeds.append(layer_embeds[:, offset:offset + num_tokens, :])
                elif layer_embeds.dim() == 2:
                    # [seq, hidden]
                    image_embeds.append(layer_embeds[offset:offset + num_tokens, :])
                else:
                    image_embeds.append(layer_embeds)
            embeds_list.append(image_embeds)

        return embeds_list

    def _reorder_position_embeddings(
        self, pos_emb: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Reorder position embeddings along sequence dimension."""
        if pos_emb.dim() == 3:
            return pos_emb.index_select(1, indices)
        if pos_emb.dim() == 4:
            return pos_emb.index_select(2, indices)
        raise ValueError(f"Unexpected position embedding dim: {pos_emb.dim()}")

    def find_image_token_ranges(
        self, input_ids: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Find contiguous ranges of image/video tokens in input_ids.

        Args:
            input_ids: [batch, seq_len] input token IDs

        Returns:
            List of (start, end) tuples for each image/video region
        """
        # Flatten to 1D for simplicity (assuming batch_size=1)
        ids = input_ids[0].cpu().tolist()

        ranges = []
        start = None

        for i, token_id in enumerate(ids):
            is_visual = token_id in (self.IMAGE_TOKEN_ID, self.VIDEO_TOKEN_ID)

            if is_visual and start is None:
                start = i
            elif not is_visual and start is not None:
                ranges.append((start, i))
                start = None

        # Handle case where visual tokens extend to the end
        if start is not None:
            ranges.append((start, len(ids)))

        return ranges

    def get_num_image_tokens(self, input_ids: torch.Tensor) -> int:
        """Count total number of image/video tokens."""
        ids = input_ids[0]
        is_visual = (ids == self.IMAGE_TOKEN_ID) | (ids == self.VIDEO_TOKEN_ID)
        return is_visual.sum().item()
