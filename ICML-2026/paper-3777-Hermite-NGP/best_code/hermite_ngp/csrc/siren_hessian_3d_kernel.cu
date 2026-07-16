/*
 * SIREN Hessian CUDA Kernel - Optimized for analytic Hessian
 *
 * Uses the trick: h'' = -ω² * h (reuses forward values!)
 *
 * For 2-layer SIREN: enc -> hidden -> output
 * Computes: u, du/d_enc, d²u/d_enc² in one efficient pass
 *
 * Much faster than propagating 7 derivative tensors!
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16

// =============================================================================
// Kernel 1: Forward pass + compute h, h', h''
// z = enc @ W1.T + b1
// h = sin(ω*z), h' = ω*cos(ω*z), h'' = -ω²*h
// =============================================================================

template <typename scalar_t>
__global__ void siren_forward_kernel(
    const scalar_t* __restrict__ enc,      // [N, D_in]
    const scalar_t* __restrict__ W1,       // [H, D_in]
    const scalar_t* __restrict__ b1,       // [H]
    scalar_t* __restrict__ h,              // [N, H] - sin(ω*z)
    scalar_t* __restrict__ h_p,            // [N, H] - ω*cos(ω*z)
    scalar_t* __restrict__ h_pp,           // [N, H] - -ω²*sin(ω*z)
    const int N,
    const int D_in,
    const int H,
    const scalar_t omega
) {
    __shared__ scalar_t tile_enc[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_w[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t z = 0;

    const int num_tiles = (D_in + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_col = t * TILE_SIZE + threadIdx.x;
        const int tile_row_w = t * TILE_SIZE + threadIdx.y;

        if (row < N && tile_col < D_in) {
            tile_enc[threadIdx.y][threadIdx.x] = enc[row * D_in + tile_col];
        } else {
            tile_enc[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < H && tile_row_w < D_in) {
            tile_w[threadIdx.y][threadIdx.x] = W1[col * D_in + tile_row_w];
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            z += tile_enc[threadIdx.y][k] * tile_w[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row >= N || col >= H) return;

    z += b1[col];

    const int idx = row * H + col;
    const scalar_t omega_z = omega * z;
    const scalar_t sin_val = sin(omega_z);
    const scalar_t cos_val = cos(omega_z);
    const scalar_t omega2 = omega * omega;

    h[idx] = sin_val;
    h_p[idx] = omega * cos_val;
    h_pp[idx] = -omega2 * sin_val;  // = -ω² * h (the trick!)
}

// =============================================================================
// Kernel 2: Output layer + Jacobian + Hessian diagonal
// u = h @ W2.T + b2
// du/d_enc = (h' * W2) @ W1
// d²u/d_enc² = (h'' * W2) @ W1²
// =============================================================================

template <typename scalar_t>
__global__ void siren_output_jacobian_hessian_kernel(
    const scalar_t* __restrict__ h,        // [N, H]
    const scalar_t* __restrict__ h_p,      // [N, H]
    const scalar_t* __restrict__ h_pp,     // [N, H]
    const scalar_t* __restrict__ W1,       // [H, D_in]
    const scalar_t* __restrict__ W2,       // [1, H]
    const scalar_t* __restrict__ b2,       // [1]
    scalar_t* __restrict__ u,              // [N, 1]
    scalar_t* __restrict__ du,             // [N, D_in]
    scalar_t* __restrict__ d2u,            // [N, D_in]
    const int N,
    const int D_in,
    const int H
) {
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // batch
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // D_in

    if (row >= N) return;

    // First compute u = h @ W2.T + b2 (only thread 0 per row)
    if (col == 0) {
        scalar_t sum_u = 0;
        for (int j = 0; j < H; j++) {
            sum_u += h[row * H + j] * W2[j];
        }
        u[row] = sum_u + b2[0];
    }

    if (col >= D_in) return;

    // Compute du[row, col] = sum_j (h'[row,j] * W2[j] * W1[j, col])
    // Compute d2u[row, col] = sum_j (h''[row,j] * W2[j] * W1[j, col]²)
    scalar_t sum_du = 0;
    scalar_t sum_d2u = 0;

    for (int j = 0; j < H; j++) {
        const scalar_t w2_j = W2[j];
        const scalar_t w1_jk = W1[j * D_in + col];
        const scalar_t w1_jk_sq = w1_jk * w1_jk;

        sum_du += h_p[row * H + j] * w2_j * w1_jk;
        sum_d2u += h_pp[row * H + j] * w2_j * w1_jk_sq;
    }

    du[row * D_in + col] = sum_du;
    d2u[row * D_in + col] = sum_d2u;
}

// =============================================================================
// Kernel 3: Compute Laplacian using chain rule (3D version)
// lap = sum_i [d2u_i * (dx_i² + dy_i² + dz_i²) + du_i * (dxx_i + dyy_i + dzz_i)]
// =============================================================================

template <typename scalar_t>
__global__ void compute_laplacian_3d_kernel(
    const scalar_t* __restrict__ du,       // [N, D]
    const scalar_t* __restrict__ d2u,      // [N, D]
    const scalar_t* __restrict__ dx,       // [N, D]
    const scalar_t* __restrict__ dy,       // [N, D]
    const scalar_t* __restrict__ dz,       // [N, D]
    const scalar_t* __restrict__ dxx,      // [N, D]
    const scalar_t* __restrict__ dyy,      // [N, D]
    const scalar_t* __restrict__ dzz,      // [N, D]
    scalar_t* __restrict__ laplacian,      // [N, 1]
    const int N,
    const int D
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    scalar_t lap = 0;
    for (int i = 0; i < D; i++) {
        const int idx = row * D + i;
        const scalar_t du_i = du[idx];
        const scalar_t d2u_i = d2u[idx];
        const scalar_t dx_i = dx[idx];
        const scalar_t dy_i = dy[idx];
        const scalar_t dz_i = dz[idx];

        // d²u/dx² = d²u/d_enc² * (d_enc/dx)² + du/d_enc * d²enc/dx²
        lap += d2u_i * (dx_i * dx_i + dy_i * dy_i + dz_i * dz_i);
        lap += du_i * (dxx[idx] + dyy[idx] + dzz[idx]);
    }

    laplacian[row] = lap;
}

// =============================================================================
// C++ Interface - Combined forward with Laplacian
// =============================================================================

std::vector<torch::Tensor> siren_forward_with_laplacian_3d_cuda(
    torch::Tensor enc,      // [N, D_in] encoding
    torch::Tensor dx,       // [N, D_in] d_enc/dx
    torch::Tensor dy,       // [N, D_in] d_enc/dy
    torch::Tensor dz,       // [N, D_in] d_enc/dz
    torch::Tensor dxx,      // [N, D_in] d²enc/dx²
    torch::Tensor dyy,      // [N, D_in] d²enc/dy²
    torch::Tensor dzz,      // [N, D_in] d²enc/dz²
    torch::Tensor W1,       // [H, D_in]
    torch::Tensor b1,       // [H]
    torch::Tensor W2,       // [1, H]
    torch::Tensor b2,       // [1]
    float omega
) {
    CHECK_INPUT(enc);
    CHECK_INPUT(dx);
    CHECK_INPUT(dy);
    CHECK_INPUT(dz);
    CHECK_INPUT(dxx);
    CHECK_INPUT(dyy);
    CHECK_INPUT(dzz);
    CHECK_INPUT(W1);
    CHECK_INPUT(b1);
    CHECK_INPUT(W2);
    CHECK_INPUT(b2);

    const int N = enc.size(0);
    const int D_in = enc.size(1);
    const int H = W1.size(0);

    auto options = enc.options();

    // Intermediate tensors
    auto h = torch::empty({N, H}, options);
    auto h_p = torch::empty({N, H}, options);
    auto h_pp = torch::empty({N, H}, options);

    // Output tensors
    auto u = torch::empty({N, 1}, options);
    auto du = torch::empty({N, D_in}, options);
    auto d2u = torch::empty({N, D_in}, options);
    auto laplacian = torch::empty({N, 1}, options);

    // Kernel 1: Forward through hidden layer
    dim3 threads1(TILE_SIZE, TILE_SIZE);
    dim3 blocks1((H + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(enc.scalar_type(), "siren_forward", ([&] {
        siren_forward_kernel<scalar_t><<<blocks1, threads1>>>(
            enc.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            b1.data_ptr<scalar_t>(),
            h.data_ptr<scalar_t>(),
            h_p.data_ptr<scalar_t>(),
            h_pp.data_ptr<scalar_t>(),
            N, D_in, H,
            static_cast<scalar_t>(omega)
        );
    }));

    // Kernel 2: Output + Jacobian + Hessian
    dim3 threads2(TILE_SIZE, TILE_SIZE);
    dim3 blocks2((D_in + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(enc.scalar_type(), "siren_jacobian_hessian", ([&] {
        siren_output_jacobian_hessian_kernel<scalar_t><<<blocks2, threads2>>>(
            h.data_ptr<scalar_t>(),
            h_p.data_ptr<scalar_t>(),
            h_pp.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            W2.data_ptr<scalar_t>(),
            b2.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            du.data_ptr<scalar_t>(),
            d2u.data_ptr<scalar_t>(),
            N, D_in, H
        );
    }));

    // Kernel 3: Laplacian via chain rule
    int threads3 = 256;
    int blocks3 = (N + threads3 - 1) / threads3;

    AT_DISPATCH_FLOATING_TYPES(enc.scalar_type(), "compute_laplacian_3d", ([&] {
        compute_laplacian_3d_kernel<scalar_t><<<blocks3, threads3>>>(
            du.data_ptr<scalar_t>(),
            d2u.data_ptr<scalar_t>(),
            dx.data_ptr<scalar_t>(),
            dy.data_ptr<scalar_t>(),
            dz.data_ptr<scalar_t>(),
            dxx.data_ptr<scalar_t>(),
            dyy.data_ptr<scalar_t>(),
            dzz.data_ptr<scalar_t>(),
            laplacian.data_ptr<scalar_t>(),
            N, D_in
        );
    }));

    return {u, laplacian, du, d2u, h, h_p, h_pp};
}

// =============================================================================
// Simple forward pass (for BC evaluation)
// =============================================================================

torch::Tensor siren_forward_cuda(
    torch::Tensor enc,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    float omega
) {
    CHECK_INPUT(enc);
    CHECK_INPUT(W1);
    CHECK_INPUT(b1);
    CHECK_INPUT(W2);
    CHECK_INPUT(b2);

    const int N = enc.size(0);
    const int D_in = enc.size(1);
    const int H = W1.size(0);

    auto options = enc.options();
    auto h = torch::empty({N, H}, options);
    auto h_p = torch::empty({N, H}, options);  // not used but needed for kernel
    auto h_pp = torch::empty({N, H}, options); // not used but needed for kernel

    // Forward through hidden layer
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((H + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(enc.scalar_type(), "siren_forward_simple", ([&] {
        siren_forward_kernel<scalar_t><<<blocks, threads>>>(
            enc.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            b1.data_ptr<scalar_t>(),
            h.data_ptr<scalar_t>(),
            h_p.data_ptr<scalar_t>(),
            h_pp.data_ptr<scalar_t>(),
            N, D_in, H,
            static_cast<scalar_t>(omega)
        );
    }));

    // Output: u = h @ W2.T + b2
    // W2 is passed as [H] (1D), reshape to [H, 1] for matmul
    auto W2_col = W2.unsqueeze(1);  // [H] -> [H, 1]
    auto u = torch::mm(h, W2_col) + b2;  // [N, H] @ [H, 1] = [N, 1]

    return u;
}

// =============================================================================
// Hidden layer forward only - returns h, h', h'' for use with PyTorch autograd
// =============================================================================

std::vector<torch::Tensor> siren_hidden_forward_cuda(
    torch::Tensor enc,      // [N, D_in]
    torch::Tensor W1,       // [H, D_in]
    torch::Tensor b1,       // [H]
    float omega
) {
    CHECK_INPUT(enc);
    CHECK_INPUT(W1);
    CHECK_INPUT(b1);

    const int N = enc.size(0);
    const int D_in = enc.size(1);
    const int H = W1.size(0);

    auto options = enc.options();
    auto h = torch::empty({N, H}, options);
    auto h_p = torch::empty({N, H}, options);
    auto h_pp = torch::empty({N, H}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((H + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(enc.scalar_type(), "siren_hidden_forward", ([&] {
        siren_forward_kernel<scalar_t><<<blocks, threads>>>(
            enc.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            b1.data_ptr<scalar_t>(),
            h.data_ptr<scalar_t>(),
            h_p.data_ptr<scalar_t>(),
            h_pp.data_ptr<scalar_t>(),
            N, D_in, H,
            static_cast<scalar_t>(omega)
        );
    }));

    return {h, h_p, h_pp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_with_laplacian", &siren_forward_with_laplacian_3d_cuda,
          "SIREN forward with Laplacian 3D (CUDA) - uses Hessian trick");
    m.def("forward", &siren_forward_cuda,
          "SIREN forward (CUDA)");
    m.def("hidden_forward", &siren_hidden_forward_cuda,
          "SIREN hidden layer forward (CUDA) - returns h, h', h''");
}
