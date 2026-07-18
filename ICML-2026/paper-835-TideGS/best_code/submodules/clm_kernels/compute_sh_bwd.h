#pragma once

#include <torch/extension.h>

torch::Tensor compute_sh_bwd_inplace_tensor(const uint32_t K, const uint32_t degrees_to_use,
                                            torch::Tensor &dirs,               // [..., 3]
                                            torch::Tensor &coeffs,             // [..., K, 3]
                                            torch::Tensor &v_coeffs,
                                            at::optional<torch::Tensor> masks, // [...]
                                            torch::Tensor &v_colors,           // [..., 3]
                                            bool compute_v_dirs);

