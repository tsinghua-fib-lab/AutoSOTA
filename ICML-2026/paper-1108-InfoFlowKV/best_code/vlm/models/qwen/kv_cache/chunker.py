"""Image chunking for Qwen3-VL KV cache."""

import torch
from typing import List, Tuple
from dataclasses import dataclass
import math


@dataclass
class ChunkInfo:
    """Information about image chunks.
    
    Index mappings:
    - reorder_indices: Maps chunked positions to original positions
      reorder_indices[chunked_pos] = original_pos
      Usage: reordered = original[reorder_indices]
      
    - restore_indices: Maps original positions to chunked positions (inverse)
      restore_indices[original_pos] = chunked_pos  
      Usage: for applying correct position_ids to chunked KV cache
    """
    num_chunks: int  # k*k total chunks
    k: int  # k x k grid of chunks
    grid_h: int  # Original grid height
    grid_w: int  # Original grid width
    chunk_h: int  # Height of each chunk (grid_h // k)
    chunk_w: int  # Width of each chunk (grid_w // k)
    chunk_size: int  # chunk_h * chunk_w (may include padding)
    chunk_indices: List[torch.Tensor]  # [k*k] list of padded global index tensors (padding=-1)
    chunk_valid_masks: List[torch.Tensor]  # [k*k] list of bool masks for valid positions
    reorder_indices: torch.Tensor  # [total_image_tokens] chunked_pos -> original_pos
    restore_indices: torch.Tensor  # [total_image_tokens] original_pos -> chunked_pos
    padded_grid_h: int  # Padded grid height
    padded_grid_w: int  # Padded grid width


class ImageChunker:
    def __init__(self, k: int = 2):
        """
        Args:
            k: Number of chunks per dimension (total chunks = k*k)
        """
        self.k = k
    
    def compute_chunk_indices(
        self, 
        grid_h: int, 
        grid_w: int,
        image_start_idx: int = 0,
        device: torch.device = None,
        allow_padding: bool = False,
    ) -> ChunkInfo:
        """
        Compute indices for each chunk.
        
        Args:
            grid_h: Height of the image grid (number of patch rows)
            grid_w: Width of the image grid (number of patch columns)
            image_start_idx: Starting index of image tokens in the full sequence
            device: Device for tensors
            
        Returns:
            ChunkInfo with indices for each chunk and reordering
        """
        if device is None:
            device = torch.device('cpu')
        
        k = self.k
        
        if allow_padding:
            padded_grid_h = math.ceil(grid_h / k) * k
            padded_grid_w = math.ceil(grid_w / k) * k
        else:
            # Ensure grid is divisible by k
            assert grid_h % k == 0, f"grid_h ({grid_h}) must be divisible by k ({k})"
            assert grid_w % k == 0, f"grid_w ({grid_w}) must be divisible by k ({k})"
            padded_grid_h = grid_h
            padded_grid_w = grid_w

        chunk_h = padded_grid_h // k
        chunk_w = padded_grid_w // k
        chunk_size = chunk_h * chunk_w
        total_tokens = grid_h * grid_w

        # Build 2D grid of token indices (row-major order), padded with -1
        idx_grid = torch.full(
            (padded_grid_h, padded_grid_w),
            -1,
            device=device,
            dtype=torch.long,
        )
        idx_grid[:grid_h, :grid_w] = torch.arange(total_tokens, device=device).reshape(grid_h, grid_w)

        # Reshape to (k, chunk_h, k, chunk_w) then permute to (k, k, chunk_h, chunk_w)
        # Finally reshape to (k*k, chunk_size) - each row is one chunk
        chunks = idx_grid.reshape(k, chunk_h, k, chunk_w).permute(0, 2, 1, 3).reshape(k * k, chunk_size)

        # Build chunk_indices and chunk_valid_masks as lists
        chunk_indices = []
        chunk_valid_masks = []
        reorder_list = []

        for i in range(k * k):
            chunk_idx = chunks[i].clone()
            valid_mask = chunk_idx >= 0
            chunk_idx_shifted = chunk_idx + image_start_idx
            chunk_idx_shifted[~valid_mask] = -1
            chunk_indices.append(chunk_idx_shifted)
            chunk_valid_masks.append(valid_mask)
            reorder_list.append(chunk_idx_shifted[valid_mask])
        
        # import pdb; pdb.set_trace()

        # reorder_indices[chunked_pos] = original_pos
        if len(reorder_list) == 0 or all(r.numel() == 0 for r in reorder_list):
            raise ValueError("No valid image tokens found for chunking.")
        reorder_indices = torch.cat(reorder_list, dim=0)

        # restore_indices[original_pos] = chunked_pos (vectorized)
        restore_indices = torch.empty(total_tokens, device=device, dtype=torch.long)
        restore_indices[reorder_indices - image_start_idx] = torch.arange(total_tokens, device=device) + image_start_idx
        
        return ChunkInfo(
            num_chunks=k * k,
            k=k,
            grid_h=grid_h,
            grid_w=grid_w,
            chunk_h=chunk_h,
            chunk_w=chunk_w,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_valid_masks=chunk_valid_masks,
            reorder_indices=reorder_indices,
            restore_indices=restore_indices,
            padded_grid_h=padded_grid_h,
            padded_grid_w=padded_grid_w,
        )
    
    def reorder_sequence(
        self,
        sequence: torch.Tensor,
        chunk_info: ChunkInfo,
        image_start_idx: int,
        image_end_idx: int,
    ) -> torch.Tensor:
        """
        Reorder a sequence so image tokens are grouped by chunk.
        
        Args:
            sequence: [batch, seq_len, ...] or [seq_len, ...] tensor
            chunk_info: ChunkInfo from compute_chunk_indices
            image_start_idx: Start index of image tokens
            image_end_idx: End index of image tokens
            
        Returns:
            Reordered sequence with same shape
        """
        has_batch = sequence.dim() >= 2 and sequence.shape[0] == 1
        if has_batch:
            seq = sequence[0]
        else:
            seq = sequence
            
        # Extract parts
        prefix = seq[:image_start_idx]
        image_part = seq[image_start_idx:image_end_idx]
        suffix = seq[image_end_idx:]
        
        # Reorder image part
        local_reorder = chunk_info.reorder_indices - image_start_idx
        reordered_image = image_part[local_reorder]
        
        # Concatenate
        result = torch.cat([prefix, reordered_image, suffix], dim=0)
        
        if has_batch:
            result = result.unsqueeze(0)
        
        return result
    
    def get_chunk_ranges(
        self, 
        chunk_info: ChunkInfo,
        image_start_idx: int,
    ) -> List[Tuple[int, int]]:
        """
        Get (start, end) ranges for each chunk in the reordered sequence.
        
        After reordering, chunks are contiguous in the sequence.
        
        Args:
            chunk_info: ChunkInfo from compute_chunk_indices
            image_start_idx: Start index of image tokens
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        ranges = []
        if hasattr(chunk_info, "chunk_valid_masks") and chunk_info.chunk_valid_masks:
            start = image_start_idx
            for mask in chunk_info.chunk_valid_masks:
                length = int(mask.sum().item())
                end = start + length
                ranges.append((start, end))
                start = end
            return ranges

        chunk_size = chunk_info.chunk_h * chunk_info.chunk_w
        for i in range(chunk_info.num_chunks):
            start = image_start_idx + i * chunk_size
            end = start + chunk_size
            ranges.append((start, end))

        return ranges
