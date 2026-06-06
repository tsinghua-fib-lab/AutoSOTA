#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> compact_visible_params_forward(at::Tensor visible_chunkid,at::Tensor position, at::Tensor scale, at::Tensor rotation, 
	at::Tensor sh_base, at::Tensor sh_rest, at::Tensor opacity);
std::vector<at::Tensor> compact_visible_params_backward(int64_t chunk_num, int64_t chunk_size, at::Tensor visible_chunkid,
    at::Tensor compacted_position_grad, at::Tensor compacted_scale_grad, at::Tensor compacted_rotation_grad,
    at::Tensor compacted_sh_base_grad, at::Tensor compacted_sh_rest_grad, at::Tensor compacted_opacity_grad);
void adamUpdate(torch::Tensor &param,torch::Tensor &param_grad,torch::Tensor &exp_avg,torch::Tensor &exp_avg_sq,torch::Tensor &visible,
    const double lr,
	const double b1,
	const double b2,
	const double eps
);