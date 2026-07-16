/*
 * C++ bindings for Hermite Hash Encoding CUDA kernels.
 *
 * This is a NEW file - does not modify original PyTorch implementation.
 */

#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor hermite_encoding_forward_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
);

std::vector<torch::Tensor> hermite_encoding_with_laplacian_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
);

std::vector<torch::Tensor> hermite_encoding_backward_cuda(
    torch::Tensor x,
    torch::Tensor grad_output,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
);

std::vector<torch::Tensor> hermite_encoding_backward_full_cuda(
    torch::Tensor x,
    torch::Tensor grad_enc,
    torch::Tensor grad_dx,
    torch::Tensor grad_dy,
    torch::Tensor grad_dz,
    torch::Tensor grad_dxx,
    torch::Tensor grad_dyy,
    torch::Tensor grad_dzz,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
);

// C++ interface with input validation

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor hermite_encoding_forward(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    CHECK_INPUT(x);
    CHECK_INPUT(hash_table_1);
    CHECK_INPUT(hash_table_2);
    CHECK_INPUT(hash_table_3);
    CHECK_INPUT(hash_table_4);
    CHECK_INPUT(resolutions);

    // Ensure float32
    x = x.to(torch::kFloat32);

    return hermite_encoding_forward_cuda(x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions);
}

std::vector<torch::Tensor> hermite_encoding_with_laplacian(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    CHECK_INPUT(x);
    CHECK_INPUT(hash_table_1);
    CHECK_INPUT(hash_table_2);
    CHECK_INPUT(hash_table_3);
    CHECK_INPUT(hash_table_4);
    CHECK_INPUT(resolutions);

    x = x.to(torch::kFloat32);

    return hermite_encoding_with_laplacian_cuda(x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions);
}

std::vector<torch::Tensor> hermite_encoding_backward(
    torch::Tensor x,
    torch::Tensor grad_output,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(hash_table_1);
    CHECK_INPUT(hash_table_2);
    CHECK_INPUT(hash_table_3);
    CHECK_INPUT(hash_table_4);
    CHECK_INPUT(resolutions);

    x = x.to(torch::kFloat32);

    return hermite_encoding_backward_cuda(x, grad_output, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions);
}

std::vector<torch::Tensor> hermite_encoding_backward_full(
    torch::Tensor x,
    torch::Tensor grad_enc,
    torch::Tensor grad_dx,
    torch::Tensor grad_dy,
    torch::Tensor grad_dz,
    torch::Tensor grad_dxx,
    torch::Tensor grad_dyy,
    torch::Tensor grad_dzz,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_enc);
    CHECK_INPUT(grad_dx);
    CHECK_INPUT(grad_dy);
    CHECK_INPUT(grad_dz);
    CHECK_INPUT(grad_dxx);
    CHECK_INPUT(grad_dyy);
    CHECK_INPUT(grad_dzz);
    CHECK_INPUT(hash_table_1);
    CHECK_INPUT(hash_table_2);
    CHECK_INPUT(hash_table_3);
    CHECK_INPUT(hash_table_4);
    CHECK_INPUT(resolutions);

    x = x.to(torch::kFloat32);

    return hermite_encoding_backward_full_cuda(
        x, grad_enc, grad_dx, grad_dy, grad_dz, grad_dxx, grad_dyy, grad_dzz, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hermite_encoding_forward, "Hermite encoding forward (CUDA)");
    m.def("forward_with_laplacian", &hermite_encoding_with_laplacian,
          "Hermite encoding forward with Laplacian (CUDA)");
    m.def("backward", &hermite_encoding_backward, "Hermite encoding backward (CUDA)");
    m.def("backward_full", &hermite_encoding_backward_full,
          "Hermite encoding backward with all derivatives (CUDA)");
}
