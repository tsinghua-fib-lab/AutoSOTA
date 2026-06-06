# Copyright (c) 2025 Youyu Chen
# SPDX-License-Identifier: MIT
import torch

from . import _C

def lanczos_resample(
	input: torch.Tensor, 
	scale_factor: float=None, 
	size: tuple=None, 
	kernel_size: int=2
) -> torch.Tensor:
	# Input shape default follows (H, W, C)
	input_h, input_w = input.shape[:2]
	assert scale_factor is not None or size is not None
	if scale_factor is not None:
		output_h, output_w = int(input_h / scale_factor), int(input_w / scale_factor)
	elif size is not None:
		output_h, output_w = size[0], size[1]
	return _C.fast_lanczos(input, output_h, output_w, kernel_size)