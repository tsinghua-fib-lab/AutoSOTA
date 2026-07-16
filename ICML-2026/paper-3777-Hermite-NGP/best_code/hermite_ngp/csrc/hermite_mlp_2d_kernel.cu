/*
 * Hermite MLP CUDA Kernel V2 - With Custom Backward
 *
 * Saves intermediates in forward for efficient backward pass.
 * Both forward and backward are CUDA kernels.
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
// Forward Kernel - Saves intermediates for backward
// =============================================================================

template <typename scalar_t>
__global__ void hermite_layer_forward_v2_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ dh_dx,
    const scalar_t* __restrict__ dh_dy,
    const scalar_t* __restrict__ d2h_dxx,
    const scalar_t* __restrict__ d2h_dyy,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out_h,
    scalar_t* __restrict__ out_dh_dx,
    scalar_t* __restrict__ out_dh_dy,
    scalar_t* __restrict__ out_d2h_dxx,
    scalar_t* __restrict__ out_d2h_dyy,
    // Saved for backward
    scalar_t* __restrict__ save_z,
    scalar_t* __restrict__ save_dz_dx,
    scalar_t* __restrict__ save_dz_dy,
    scalar_t* __restrict__ save_d2z_dxx,
    scalar_t* __restrict__ save_d2z_dyy,
    const int N,
    const int D_in,
    const int D_out,
    const scalar_t omega,
    const bool apply_activation
) {
    __shared__ scalar_t tile_h[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_dx[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_dy[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_dxx[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_dyy[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_w[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t z = 0;
    scalar_t dz_dx = 0;
    scalar_t dz_dy = 0;
    scalar_t d2z_dxx = 0;
    scalar_t d2z_dyy = 0;

    const int num_tiles = (D_in + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_col = t * TILE_SIZE + threadIdx.x;
        const int tile_row_w = t * TILE_SIZE + threadIdx.y;

        if (row < N && tile_col < D_in) {
            const int idx = row * D_in + tile_col;
            tile_h[threadIdx.y][threadIdx.x] = h[idx];
            tile_dx[threadIdx.y][threadIdx.x] = dh_dx[idx];
            tile_dy[threadIdx.y][threadIdx.x] = dh_dy[idx];
            tile_dxx[threadIdx.y][threadIdx.x] = d2h_dxx[idx];
            tile_dyy[threadIdx.y][threadIdx.x] = d2h_dyy[idx];
        } else {
            tile_h[threadIdx.y][threadIdx.x] = 0;
            tile_dx[threadIdx.y][threadIdx.x] = 0;
            tile_dy[threadIdx.y][threadIdx.x] = 0;
            tile_dxx[threadIdx.y][threadIdx.x] = 0;
            tile_dyy[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < D_out && tile_row_w < D_in) {
            tile_w[threadIdx.y][threadIdx.x] = weight[col * D_in + tile_row_w];
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            scalar_t w = tile_w[k][threadIdx.x];
            z += tile_h[threadIdx.y][k] * w;
            dz_dx += tile_dx[threadIdx.y][k] * w;
            dz_dy += tile_dy[threadIdx.y][k] * w;
            d2z_dxx += tile_dxx[threadIdx.y][k] * w;
            d2z_dyy += tile_dyy[threadIdx.y][k] * w;
        }

        __syncthreads();
    }

    if (row >= N || col >= D_out) return;

    // Add bias to z only
    z += bias[col];

    const int out_idx = row * D_out + col;

    // Save intermediates for backward (before activation)
    save_z[out_idx] = z;
    save_dz_dx[out_idx] = dz_dx;
    save_dz_dy[out_idx] = dz_dy;
    save_d2z_dxx[out_idx] = d2z_dxx;
    save_d2z_dyy[out_idx] = d2z_dyy;

    if (apply_activation) {
        scalar_t omega_z = omega * z;
        scalar_t sin_val = sin(omega_z);
        scalar_t cos_val = cos(omega_z);
        scalar_t omega2 = omega * omega;

        scalar_t h_p = omega * cos_val;
        scalar_t h_pp = -omega2 * sin_val;

        out_h[out_idx] = sin_val;
        out_dh_dx[out_idx] = h_p * dz_dx;
        out_dh_dy[out_idx] = h_p * dz_dy;
        out_d2h_dxx[out_idx] = h_pp * dz_dx * dz_dx + h_p * d2z_dxx;
        out_d2h_dyy[out_idx] = h_pp * dz_dy * dz_dy + h_p * d2z_dyy;
    } else {
        out_h[out_idx] = z;
        out_dh_dx[out_idx] = dz_dx;
        out_dh_dy[out_idx] = dz_dy;
        out_d2h_dxx[out_idx] = d2z_dxx;
        out_d2h_dyy[out_idx] = d2z_dyy;
    }
}

// =============================================================================
// Backward Kernel - Input Gradients
// Computes gradients w.r.t. input tensors (h, dh_dx, dh_dy, d2h_dxx, d2h_dyy)
// =============================================================================

template <typename scalar_t>
__global__ void hermite_layer_backward_input_kernel(
    // Gradients from output
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_dh_dx,
    const scalar_t* __restrict__ grad_dh_dy,
    const scalar_t* __restrict__ grad_d2h_dxx,
    const scalar_t* __restrict__ grad_d2h_dyy,
    // Saved intermediates
    const scalar_t* __restrict__ save_z,
    const scalar_t* __restrict__ save_dz_dx,
    const scalar_t* __restrict__ save_dz_dy,
    const scalar_t* __restrict__ save_d2z_dxx,
    const scalar_t* __restrict__ save_d2z_dyy,
    // Weight
    const scalar_t* __restrict__ weight,
    // Output gradients w.r.t. inputs
    scalar_t* __restrict__ grad_h_in,
    scalar_t* __restrict__ grad_dh_dx_in,
    scalar_t* __restrict__ grad_dh_dy_in,
    scalar_t* __restrict__ grad_d2h_dxx_in,
    scalar_t* __restrict__ grad_d2h_dyy_in,
    const int N,
    const int D_in,
    const int D_out,
    const scalar_t omega,
    const bool has_activation
) {
    // Each thread computes gradient for one input element
    // grad_h_in[row, col] = sum over j of (grad_z[row, j] * weight[j, col])

    __shared__ scalar_t tile_grad_z[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_grad_dz_dx[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_grad_dz_dy[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_grad_d2z_dxx[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_grad_d2z_dyy[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_w[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // batch index
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // input dim index

    scalar_t sum_h = 0;
    scalar_t sum_dx = 0;
    scalar_t sum_dy = 0;
    scalar_t sum_dxx = 0;
    scalar_t sum_dyy = 0;

    const scalar_t omega2 = omega * omega;
    const scalar_t omega3 = omega2 * omega;

    const int num_tiles = (D_out + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_j = t * TILE_SIZE + threadIdx.x;  // output dim for loading grads
        const int tile_j_w = t * TILE_SIZE + threadIdx.y;  // output dim for loading weights

        // Load gradients and compute grad_z from chain rule
        if (row < N && tile_j < D_out) {
            const int idx = row * D_out + tile_j;

            if (has_activation) {
                scalar_t z = save_z[idx];
                scalar_t dz_dx_val = save_dz_dx[idx];
                scalar_t dz_dy_val = save_dz_dy[idx];
                scalar_t d2z_dxx_val = save_d2z_dxx[idx];
                scalar_t d2z_dyy_val = save_d2z_dyy[idx];

                scalar_t omega_z = omega * z;
                scalar_t sin_val = sin(omega_z);
                scalar_t cos_val = cos(omega_z);

                scalar_t h_p = omega * cos_val;
                scalar_t h_pp = -omega2 * sin_val;
                scalar_t h_ppp = -omega3 * cos_val;

                // grad_z from out_h: d(sin(ωz))/dz = ω*cos(ωz)
                scalar_t g_z = grad_h[idx] * h_p;

                // grad_z from out_dh_dx: d(h_p * dz_dx)/dz = h_pp * dz_dx
                g_z += grad_dh_dx[idx] * h_pp * dz_dx_val;

                // grad_z from out_dh_dy
                g_z += grad_dh_dy[idx] * h_pp * dz_dy_val;

                // grad_z from out_d2h_dxx: d(h_pp * dz_dx^2 + h_p * d2z_dxx)/dz
                // = h_ppp * dz_dx^2 + h_pp * d2z_dxx
                g_z += grad_d2h_dxx[idx] * (h_ppp * dz_dx_val * dz_dx_val + h_pp * d2z_dxx_val);

                // grad_z from out_d2h_dyy
                g_z += grad_d2h_dyy[idx] * (h_ppp * dz_dy_val * dz_dy_val + h_pp * d2z_dyy_val);

                tile_grad_z[threadIdx.y][threadIdx.x] = g_z;

                // grad_dz_dx from out_dh_dx and out_d2h_dxx
                scalar_t g_dz_dx = grad_dh_dx[idx] * h_p;
                g_dz_dx += grad_d2h_dxx[idx] * 2 * h_pp * dz_dx_val;
                tile_grad_dz_dx[threadIdx.y][threadIdx.x] = g_dz_dx;

                // grad_dz_dy
                scalar_t g_dz_dy = grad_dh_dy[idx] * h_p;
                g_dz_dy += grad_d2h_dyy[idx] * 2 * h_pp * dz_dy_val;
                tile_grad_dz_dy[threadIdx.y][threadIdx.x] = g_dz_dy;

                // grad_d2z_dxx
                tile_grad_d2z_dxx[threadIdx.y][threadIdx.x] = grad_d2h_dxx[idx] * h_p;

                // grad_d2z_dyy
                tile_grad_d2z_dyy[threadIdx.y][threadIdx.x] = grad_d2h_dyy[idx] * h_p;
            } else {
                // Linear layer - direct pass through
                tile_grad_z[threadIdx.y][threadIdx.x] = grad_h[idx];
                tile_grad_dz_dx[threadIdx.y][threadIdx.x] = grad_dh_dx[idx];
                tile_grad_dz_dy[threadIdx.y][threadIdx.x] = grad_dh_dy[idx];
                tile_grad_d2z_dxx[threadIdx.y][threadIdx.x] = grad_d2h_dxx[idx];
                tile_grad_d2z_dyy[threadIdx.y][threadIdx.x] = grad_d2h_dyy[idx];
            }
        } else {
            tile_grad_z[threadIdx.y][threadIdx.x] = 0;
            tile_grad_dz_dx[threadIdx.y][threadIdx.x] = 0;
            tile_grad_dz_dy[threadIdx.y][threadIdx.x] = 0;
            tile_grad_d2z_dxx[threadIdx.y][threadIdx.x] = 0;
            tile_grad_d2z_dyy[threadIdx.y][threadIdx.x] = 0;
        }

        // Load weights: weight[j, col] for j in tile
        if (tile_j_w < D_out && col < D_in) {
            tile_w[threadIdx.y][threadIdx.x] = weight[tile_j_w * D_in + col];
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Accumulate: grad_h_in = sum_j(grad_z[j] * weight[j, col])
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            scalar_t w = tile_w[k][threadIdx.x];
            sum_h += tile_grad_z[threadIdx.y][k] * w;
            sum_dx += tile_grad_dz_dx[threadIdx.y][k] * w;
            sum_dy += tile_grad_dz_dy[threadIdx.y][k] * w;
            sum_dxx += tile_grad_d2z_dxx[threadIdx.y][k] * w;
            sum_dyy += tile_grad_d2z_dyy[threadIdx.y][k] * w;
        }

        __syncthreads();
    }

    if (row < N && col < D_in) {
        const int out_idx = row * D_in + col;
        grad_h_in[out_idx] = sum_h;
        grad_dh_dx_in[out_idx] = sum_dx;
        grad_dh_dy_in[out_idx] = sum_dy;
        grad_d2h_dxx_in[out_idx] = sum_dxx;
        grad_d2h_dyy_in[out_idx] = sum_dyy;
    }
}

// =============================================================================
// Backward Kernel - Weight Gradients
// Uses parallel reduction for weight gradients
// =============================================================================

template <typename scalar_t>
__global__ void hermite_layer_backward_weight_kernel(
    // Gradients from output
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_dh_dx,
    const scalar_t* __restrict__ grad_dh_dy,
    const scalar_t* __restrict__ grad_d2h_dxx,
    const scalar_t* __restrict__ grad_d2h_dyy,
    // Saved intermediates
    const scalar_t* __restrict__ save_z,
    const scalar_t* __restrict__ save_dz_dx,
    const scalar_t* __restrict__ save_dz_dy,
    const scalar_t* __restrict__ save_d2z_dxx,
    const scalar_t* __restrict__ save_d2z_dyy,
    // Input tensors
    const scalar_t* __restrict__ h_in,
    const scalar_t* __restrict__ dh_dx_in,
    const scalar_t* __restrict__ dh_dy_in,
    const scalar_t* __restrict__ d2h_dxx_in,
    const scalar_t* __restrict__ d2h_dyy_in,
    // Output
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    const int N,
    const int D_in,
    const int D_out,
    const scalar_t omega,
    const bool has_activation
) {
    // Each block handles one (out_idx, in_idx) pair
    // Threads within block reduce over N

    const int out_idx = blockIdx.x;
    const int in_idx = blockIdx.y;

    if (out_idx >= D_out || in_idx >= D_in) return;

    const scalar_t omega2 = omega * omega;
    const scalar_t omega3 = omega2 * omega;

    __shared__ scalar_t sdata[256];
    __shared__ scalar_t sdata_bias[256];

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    scalar_t sum_w = 0;
    scalar_t sum_b = 0;

    for (int n = tid; n < N; n += stride) {
        const int idx_out = n * D_out + out_idx;
        const int idx_in = n * D_in + in_idx;

        scalar_t g_z, g_dz_dx, g_dz_dy, g_d2z_dxx, g_d2z_dyy;

        if (has_activation) {
            scalar_t z = save_z[idx_out];
            scalar_t dz_dx_val = save_dz_dx[idx_out];
            scalar_t dz_dy_val = save_dz_dy[idx_out];
            scalar_t d2z_dxx_val = save_d2z_dxx[idx_out];
            scalar_t d2z_dyy_val = save_d2z_dyy[idx_out];

            scalar_t omega_z = omega * z;
            scalar_t sin_val = sin(omega_z);
            scalar_t cos_val = cos(omega_z);

            scalar_t h_p = omega * cos_val;
            scalar_t h_pp = -omega2 * sin_val;
            scalar_t h_ppp = -omega3 * cos_val;

            g_z = grad_h[idx_out] * h_p;
            g_z += grad_dh_dx[idx_out] * h_pp * dz_dx_val;
            g_z += grad_dh_dy[idx_out] * h_pp * dz_dy_val;
            g_z += grad_d2h_dxx[idx_out] * (h_ppp * dz_dx_val * dz_dx_val + h_pp * d2z_dxx_val);
            g_z += grad_d2h_dyy[idx_out] * (h_ppp * dz_dy_val * dz_dy_val + h_pp * d2z_dyy_val);

            g_dz_dx = grad_dh_dx[idx_out] * h_p + grad_d2h_dxx[idx_out] * 2 * h_pp * dz_dx_val;
            g_dz_dy = grad_dh_dy[idx_out] * h_p + grad_d2h_dyy[idx_out] * 2 * h_pp * dz_dy_val;
            g_d2z_dxx = grad_d2h_dxx[idx_out] * h_p;
            g_d2z_dyy = grad_d2h_dyy[idx_out] * h_p;
        } else {
            g_z = grad_h[idx_out];
            g_dz_dx = grad_dh_dx[idx_out];
            g_dz_dy = grad_dh_dy[idx_out];
            g_d2z_dxx = grad_d2h_dxx[idx_out];
            g_d2z_dyy = grad_d2h_dyy[idx_out];
        }

        // grad_weight[out, in] = sum_n(grad_z[n] * h_in[n, in] + ...)
        sum_w += g_z * h_in[idx_in];
        sum_w += g_dz_dx * dh_dx_in[idx_in];
        sum_w += g_dz_dy * dh_dy_in[idx_in];
        sum_w += g_d2z_dxx * d2h_dxx_in[idx_in];
        sum_w += g_d2z_dyy * d2h_dyy_in[idx_in];

        // Bias gradient (only from g_z, only compute once per out_idx)
        if (in_idx == 0) {
            sum_b += g_z;
        }
    }

    sdata[tid] = sum_w;
    sdata_bias[tid] = sum_b;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata_bias[tid] += sdata_bias[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_weight[out_idx * D_in + in_idx] = sdata[0];
        if (in_idx == 0) {
            grad_bias[out_idx] = sdata_bias[0];
        }
    }
}

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> hermite_layer_forward_v2_cuda(
    torch::Tensor h,
    torch::Tensor dh_dx,
    torch::Tensor dh_dy,
    torch::Tensor d2h_dxx,
    torch::Tensor d2h_dyy,
    torch::Tensor weight,
    torch::Tensor bias,
    float omega,
    bool apply_activation
) {
    CHECK_INPUT(h);
    CHECK_INPUT(dh_dx);
    CHECK_INPUT(dh_dy);
    CHECK_INPUT(d2h_dxx);
    CHECK_INPUT(d2h_dyy);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const int N = h.size(0);
    const int D_in = weight.size(1);
    const int D_out = weight.size(0);

    auto options = h.options();
    auto out_h = torch::empty({N, D_out}, options);
    auto out_dx = torch::empty({N, D_out}, options);
    auto out_dy = torch::empty({N, D_out}, options);
    auto out_dxx = torch::empty({N, D_out}, options);
    auto out_dyy = torch::empty({N, D_out}, options);

    // Saved for backward
    auto save_z = torch::empty({N, D_out}, options);
    auto save_dz_dx = torch::empty({N, D_out}, options);
    auto save_dz_dy = torch::empty({N, D_out}, options);
    auto save_d2z_dxx = torch::empty({N, D_out}, options);
    auto save_d2z_dyy = torch::empty({N, D_out}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((D_out + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hermite_layer_forward_v2", ([&] {
        hermite_layer_forward_v2_kernel<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            dh_dx.data_ptr<scalar_t>(),
            dh_dy.data_ptr<scalar_t>(),
            d2h_dxx.data_ptr<scalar_t>(),
            d2h_dyy.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out_h.data_ptr<scalar_t>(),
            out_dx.data_ptr<scalar_t>(),
            out_dy.data_ptr<scalar_t>(),
            out_dxx.data_ptr<scalar_t>(),
            out_dyy.data_ptr<scalar_t>(),
            save_z.data_ptr<scalar_t>(),
            save_dz_dx.data_ptr<scalar_t>(),
            save_dz_dy.data_ptr<scalar_t>(),
            save_d2z_dxx.data_ptr<scalar_t>(),
            save_d2z_dyy.data_ptr<scalar_t>(),
            N, D_in, D_out,
            static_cast<scalar_t>(omega),
            apply_activation
        );
    }));

    return {out_h, out_dx, out_dy, out_dxx, out_dyy, save_z, save_dz_dx, save_dz_dy, save_d2z_dxx, save_d2z_dyy};
}

std::vector<torch::Tensor> hermite_layer_backward_v2_cuda(
    torch::Tensor grad_h,
    torch::Tensor grad_dh_dx,
    torch::Tensor grad_dh_dy,
    torch::Tensor grad_d2h_dxx,
    torch::Tensor grad_d2h_dyy,
    torch::Tensor save_z,
    torch::Tensor save_dz_dx,
    torch::Tensor save_dz_dy,
    torch::Tensor save_d2z_dxx,
    torch::Tensor save_d2z_dyy,
    torch::Tensor h_in,
    torch::Tensor dh_dx_in,
    torch::Tensor dh_dy_in,
    torch::Tensor d2h_dxx_in,
    torch::Tensor d2h_dyy_in,
    torch::Tensor weight,
    float omega,
    bool has_activation
) {
    CHECK_INPUT(grad_h);
    CHECK_INPUT(save_z);
    CHECK_INPUT(h_in);
    CHECK_INPUT(weight);

    const int N = grad_h.size(0);
    const int D_out = grad_h.size(1);
    const int D_in = weight.size(1);

    auto options = grad_h.options();

    // Gradients w.r.t. inputs
    auto grad_h_in = torch::empty({N, D_in}, options);
    auto grad_dh_dx_in = torch::empty({N, D_in}, options);
    auto grad_dh_dy_in = torch::empty({N, D_in}, options);
    auto grad_d2h_dxx_in = torch::empty({N, D_in}, options);
    auto grad_d2h_dyy_in = torch::empty({N, D_in}, options);

    // Gradients w.r.t. weight and bias
    auto grad_weight = torch::zeros({D_out, D_in}, options);
    auto grad_bias = torch::zeros({D_out}, options);

    // Launch input gradient kernel
    dim3 threads_in(TILE_SIZE, TILE_SIZE);
    dim3 blocks_in((D_in + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "hermite_layer_backward_input", ([&] {
        hermite_layer_backward_input_kernel<scalar_t><<<blocks_in, threads_in>>>(
            grad_h.data_ptr<scalar_t>(),
            grad_dh_dx.data_ptr<scalar_t>(),
            grad_dh_dy.data_ptr<scalar_t>(),
            grad_d2h_dxx.data_ptr<scalar_t>(),
            grad_d2h_dyy.data_ptr<scalar_t>(),
            save_z.data_ptr<scalar_t>(),
            save_dz_dx.data_ptr<scalar_t>(),
            save_dz_dy.data_ptr<scalar_t>(),
            save_d2z_dxx.data_ptr<scalar_t>(),
            save_d2z_dyy.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_h_in.data_ptr<scalar_t>(),
            grad_dh_dx_in.data_ptr<scalar_t>(),
            grad_dh_dy_in.data_ptr<scalar_t>(),
            grad_d2h_dxx_in.data_ptr<scalar_t>(),
            grad_d2h_dyy_in.data_ptr<scalar_t>(),
            N, D_in, D_out,
            static_cast<scalar_t>(omega),
            has_activation
        );
    }));

    // Launch weight gradient kernel
    dim3 blocks_w(D_out, D_in);
    int threads_w = min(256, N);

    AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "hermite_layer_backward_weight", ([&] {
        hermite_layer_backward_weight_kernel<scalar_t><<<blocks_w, threads_w>>>(
            grad_h.data_ptr<scalar_t>(),
            grad_dh_dx.data_ptr<scalar_t>(),
            grad_dh_dy.data_ptr<scalar_t>(),
            grad_d2h_dxx.data_ptr<scalar_t>(),
            grad_d2h_dyy.data_ptr<scalar_t>(),
            save_z.data_ptr<scalar_t>(),
            save_dz_dx.data_ptr<scalar_t>(),
            save_dz_dy.data_ptr<scalar_t>(),
            save_d2z_dxx.data_ptr<scalar_t>(),
            save_d2z_dyy.data_ptr<scalar_t>(),
            h_in.data_ptr<scalar_t>(),
            dh_dx_in.data_ptr<scalar_t>(),
            dh_dy_in.data_ptr<scalar_t>(),
            d2h_dxx_in.data_ptr<scalar_t>(),
            d2h_dyy_in.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            N, D_in, D_out,
            static_cast<scalar_t>(omega),
            has_activation
        );
    }));

    return {grad_h_in, grad_dh_dx_in, grad_dh_dy_in, grad_d2h_dxx_in, grad_d2h_dyy_in, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hermite_layer_forward_v2_cuda, "Hermite layer forward V2 (CUDA)");
    m.def("backward", &hermite_layer_backward_v2_cuda, "Hermite layer backward V2 (CUDA)");
}
