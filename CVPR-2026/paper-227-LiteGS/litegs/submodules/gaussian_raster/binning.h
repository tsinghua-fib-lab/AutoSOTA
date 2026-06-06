#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU,at::Tensor RD,at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
	at::Tensor large_index,int64_t allocate_size, int64_t TilesSizeX);
at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId);
std::vector<at::Tensor> create_ROI_AABB(at::Tensor ndc, at::Tensor eigen_val, at::Tensor eigen_vec, at::Tensor opacity,
	int64_t height, int64_t width, int64_t tilesize);
