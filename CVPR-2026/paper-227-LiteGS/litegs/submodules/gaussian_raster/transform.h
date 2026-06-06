#pragma once
#include <torch/extension.h>
at::Tensor jacobianRayspace(at::Tensor translate_position,at::Tensor proj_matrix,int64_t output_h,int64_t output_w,bool bTranspose);

at::Tensor createTransformMatrix_forward(at::Tensor quaternion, at::Tensor scale);
std::vector<at::Tensor> createTransformMatrix_backward(at::Tensor transform_matrix_grad, at::Tensor quaternion, at::Tensor scale);

at::Tensor world2ndc_backword(at::Tensor view_project_matrix, at::Tensor position, at::Tensor repc_hom_w, at::Tensor grad_ndcpos);

at::Tensor createCov2dDirectly_forward(at::Tensor J, at::Tensor view_matrix,at::Tensor transform_matrix);
at::Tensor createCov2dDirectly_backward(at::Tensor cov2d_grad, at::Tensor J, at::Tensor view_matrix, at::Tensor transform_matrix);

at::Tensor sh2rgb_forward(int64_t degree, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor dir);
std::vector<at::Tensor> sh2rgb_backward(int64_t degree, at::Tensor rgb_grad, int64_t sh_rest_dim, at::Tensor dir, at::Tensor SH_base, at::Tensor SH_rest);

std::vector<at::Tensor> eigh_and_inv_2x2matrix_forward(at::Tensor input);
at::Tensor inv_2x2matrix_backward(at::Tensor inv_matrix, at::Tensor dL_dInvMatrix);