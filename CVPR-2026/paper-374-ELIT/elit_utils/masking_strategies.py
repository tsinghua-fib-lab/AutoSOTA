from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from .token_editors import (
    group_tokens_2d,
    group_tokens_2d_flatten,
    ungroup_tokens_2d_unflatten,
)

class BaseTokenMaskingStrategy:
    """
    Base class for token masking strategies.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, x, thw_shape, data_entries=None) -> Tuple[torch.Tensor, Dict]:
        """
        Mask the tokens according to the strategy.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class OrderedTokenMaskingStrategy(BaseTokenMaskingStrategy):
    """
    Randomly masks tokens based on a specified probability. in an ordered manner
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_mask_prob = config.get("max_mask_prob", None)
        self.min_mask_prob = config.get("min_mask_prob", self.max_mask_prob)
        self.window_size = config["window_size"]


    def sample_mask(self, x, thw_shape, window_size, mask_prob):
        B, N, _ = x.shape
        T, H, W = thw_shape
        
        # Handle both integer (backward compatibility) and tuple window_size
        if isinstance(window_size, (int, float)):
            zt, zh, zw = 1, int(window_size), int(window_size)
        else:
            assert len(window_size) == 3, "window_size must be a tuple of length 3 (zt, zh, zw), got {}".format(window_size)
            zt, zh, zw = window_size
            
        assert T % zt == 0, f"Temporal dimension {T} must be divisible by temporal window size {zt}"
        assert H % zh == 0, f"Height dimension {H} must be divisible by height window size {zh}"
        assert W % zw == 0, f"Width dimension {W} must be divisible by width window size {zw}"

        patch_keep = torch.zeros(1, T * H * W, 1, device=x.device, dtype=torch.bool)  # (1, Tg * Hg * Wg, 1)
        patch_keep = group_tokens_2d_flatten(patch_keep, thw_shape, window_size)  # (B, Tg, Hg, Wg, 1 * zt * zh * zw * 1)
        tokens_per_group = patch_keep.shape[-1]
        
        # TODO: @moayed make this implementation more efficient
        tokens_to_keep = int(tokens_per_group * (1 - mask_prob))
        patch_keep[:, :, :, :, :tokens_to_keep] = True   
                    
        keep_mask = ungroup_tokens_2d_unflatten(patch_keep, thw_shape, window_size)  # (B, N, 1)
        keep_mask = keep_mask.expand(B, -1, -1)
            
        num_masked = int((keep_mask).sum().item())
        
        # verify that the mask is correct
        keep_mask_reshaped = group_tokens_2d(keep_mask, thw_shape, window_size)  # (B, Tg, Hg, Wg, 1, zt, zh, zw, 1)
        assert torch.all(keep_mask_reshaped.sum(dim=(-4, -3, -2)) > 1e-6), "Masking strategy failed: some windows contain zero tokens."
        
        return keep_mask, {"num_masked": num_masked}
    
    def __call__(self, x, thw_shape, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Randomly mask tokens in the input tensor.
        thw_shape: Tuple[int, int, int] Frames, Height, Width for the patchificed input 
        """
        
        window_size = self.window_size
        return self.sample_mask(x, thw_shape, window_size, self.max_mask_prob)
    

        

class MultiOrderedTokenMaskingStrategy(OrderedTokenMaskingStrategy):
    def __init__(self, config):
        super().__init__(config)
        
        self.synchronized_budget_sampling = config.get("synchronized_budget_sampling", True)
        
        if isinstance(self.window_size, (int, float)):
            window_tokens = int(self.window_size) ** 2
        else:
            assert len(self.window_size) == 3, "window_size must be a tuple of length 3 (zt, zh, zw)"
            zt, zh, zw = self.window_size
            window_tokens = zt * zh * zw
        
        self.num_tokens = window_tokens
        
        # If max_mask_prob is not set, default to keeping at least 1 token per window
        if self.max_mask_prob is None:
            self.max_mask_prob = 1 - 1 / window_tokens
        # min_mask_prob defaults to max_mask_prob (single budget), already set in parent
        if self.min_mask_prob is None:
            self.min_mask_prob = self.max_mask_prob
        
        # Build the list of valid mask probabilities between min and max
        # Valid levels are multiples of 1/window_tokens
        all_levels = np.arange(0, window_tokens) / window_tokens  # [0, 1/wt, 2/wt, ..., (wt-1)/wt]
        self.mask_prob = all_levels[(all_levels >= self.min_mask_prob - 1e-9) & (all_levels <= self.max_mask_prob + 1e-9)]
        print(f"Sampling mask_prob from valid precision levels in [{self.min_mask_prob}, {self.max_mask_prob}]: {self.mask_prob}")
        
        self.sampling_prob = config.get('sampling_prob', None)
        if self.sampling_prob is None:
            self.sampling_prob = [1.0] * len(self.mask_prob)
        self.sampling_prob = np.array(self.sampling_prob) / sum(self.sampling_prob)
        
        
        budget_scheduler = config.get('budget_scheduler', None)
        if budget_scheduler is not None:
            budget_scheduler_target = budget_scheduler['target']
            self.budget_scheduler = budget_scheduler_target(budget_scheduler)
        else:
            self.budget_scheduler = None
            
        
        
    
    def __call__(self, x, thw_shape, train_step=None, inference_budget=None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        """
        if inference_budget is not None:
            # use inference budget from data entries
            budget = inference_budget
            assert budget is not None
            
            if budget > 1:
                max_budget = thw_shape[0] * thw_shape[1] * thw_shape[2]
                mask_prob = 1 - budget / max_budget
            else:
                mask_prob = 1 - budget
                
        elif self.budget_scheduler is not None:
                # use budget scheduler to get the budget
                budget = self.budget_scheduler(x, thw_shape, train_step=train_step)
                
                # convert budget to mask probability
                max_budget = thw_shape[0] * thw_shape[1] * thw_shape[2]
                mask_prob = 1 - budget / max_budget
        else:
            # sample masking probability
            if self.synchronized_budget_sampling:
                # get rank info 
                sampled_masking_prob = torch.zeros(1).to(x.device)
                if dist.get_rank() == 0:
                    masking_prob_idx = torch.multinomial(torch.tensor(self.sampling_prob, device=x.device), 1).item()
                    sampled_masking_prob[0] = self.mask_prob[masking_prob_idx]
                dist.broadcast(sampled_masking_prob, src=0)
                
                mask_prob = sampled_masking_prob[0].item()
            
            else:
                masking_prob_idx = torch.multinomial(torch.tensor(self.sampling_prob, device=x.device), 1).item()
                mask_prob = self.mask_prob[masking_prob_idx]


        return self.sample_mask(x, thw_shape, self.window_size, mask_prob)
