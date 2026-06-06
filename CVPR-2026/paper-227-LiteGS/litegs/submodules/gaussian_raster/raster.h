#pragma once
#include <torch/extension.h>
std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,//
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    std::optional<at::Tensor> specific_tiles,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w,
    bool enable_trans,
    bool enable_depth,
    bool enable_entropy
);

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    std::optional<at::Tensor> specific_tiles,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    std::optional<at::Tensor> d_trans_img_arg,
    std::optional<at::Tensor> d_depth_img_arg,
    std::optional<at::Tensor> d_entropy_img_arg,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
);
