/*
 * CUDA Kernel for Hermite Hash Encoding
 *
 * Fused implementation for maximum performance.
 * Computes: value, first derivatives (dx, dy), and Laplacian in one kernel.
 *
 * This is a NEW file - does not modify original PyTorch implementation.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constants
#define WARP_SIZE 32
#define MAX_LEVELS 16
#define MAX_FEATURES 8

// Hash function (same as PyTorch version)
__device__ __forceinline__ int hash_coords(int x, int y, int hashmap_size) {
    const unsigned int prime1 = 1u;
    const unsigned int prime2 = 2654435761u;
    return ((unsigned int)x * prime1 ^ (unsigned int)y * prime2) % hashmap_size;
}

// Hermite basis functions - computed in registers for speed
__device__ __forceinline__ void hermite_basis(
    float t,
    float& h0, float& h1, float& h2, float& h3,      // H(t)
    float& dh0, float& dh1, float& dh2, float& dh3,  // H'(t)
    float& ddh0, float& ddh1, float& ddh2, float& ddh3  // H''(t)
) {
    float t2 = t * t;
    float t3 = t2 * t;

    // Value basis H(t)
    h0 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    h1 = t3 - 2.0f * t2 + t;
    h2 = -2.0f * t3 + 3.0f * t2;
    h3 = t3 - t2;

    // First derivative H'(t)
    dh0 = 6.0f * t2 - 6.0f * t;
    dh1 = 3.0f * t2 - 4.0f * t + 1.0f;
    dh2 = -6.0f * t2 + 6.0f * t;
    dh3 = 3.0f * t2 - 2.0f * t;

    // Second derivative H''(t)
    ddh0 = 12.0f * t - 6.0f;
    ddh1 = 6.0f * t - 4.0f;
    ddh2 = -12.0f * t + 6.0f;
    ddh3 = 6.0f * t - 2.0f;
}

/*
 * Forward kernel: computes encoding only (no derivatives)
 *
 * Args:
 *   x: input coordinates [N, 2]
 *   hash_table: [L, hashmap_size, F*4]
 *   output: [N, L*F]
 *   resolutions: [L]
 */
__global__ void hermite_encoding_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hash_table_1,
    const float* __restrict__ hash_table_2,
    const float* __restrict__ hash_table_3,
    float* __restrict__ output,
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 2];
    float py = x[idx * 2 + 1];

    // Process each level
    for (int level = 0; level < L; level++) {
        float res = resolutions[level];

        // Scale coordinates
        float sx = px * res;
        float sy = py * res;

        // Grid coordinates
        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);

        // Local coordinates [0, 1]
        float tx = sx - (float)ix;
        float ty = sy - (float)iy;

        // Hermite basis functions
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);

        // Hash corners
        int idx00_1 = hash_coords(ix, iy, hashmap_size_1);
        int idx10_1 = hash_coords(ix + 1, iy, hashmap_size_1);
        int idx01_1 = hash_coords(ix, iy + 1, hashmap_size_1);
        int idx11_1 = hash_coords(ix + 1, iy + 1, hashmap_size_1);

        int idx00_2 = hash_coords(ix, iy, hashmap_size_2);
        int idx10_2 = hash_coords(ix + 1, iy, hashmap_size_2);
        int idx01_2 = hash_coords(ix, iy + 1, hashmap_size_2);
        int idx11_2 = hash_coords(ix + 1, iy + 1, hashmap_size_2);

        int idx00_3 = hash_coords(ix, iy, hashmap_size_3);
        int idx10_3 = hash_coords(ix + 1, iy, hashmap_size_3);
        int idx01_3 = hash_coords(ix, iy + 1, hashmap_size_3);
        int idx11_3 = hash_coords(ix + 1, iy + 1, hashmap_size_3);

        // Base offset in hash table for this level
        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 2;
        int level_offset_3 = level * hashmap_size_3 * F;

        // Process each feature
        for (int f = 0; f < F; f++) {
            // Gather 4 corners x 4 values (f, fx, fy, fxy)
            int feat_offset = f;

            float f00   = hash_table_1[level_offset_1 + idx00_1 * F + feat_offset];
            float fx00  = hash_table_2[level_offset_2 + idx00_2 * F * 2 + feat_offset];
            float fy00  = hash_table_2[level_offset_2 + idx00_2 * F * 2 + F + feat_offset];
            float fxy00 = hash_table_3[level_offset_3 + idx00_3 * F + feat_offset];

            float f10   = hash_table_1[level_offset_1 + idx10_1 * F + feat_offset];
            float fx10  = hash_table_2[level_offset_2 + idx10_2 * F * 2 + feat_offset];
            float fy10  = hash_table_2[level_offset_2 + idx10_2 * F * 2 + F + feat_offset];
            float fxy10 = hash_table_3[level_offset_3 + idx10_3 * F + feat_offset];

            float f01   = hash_table_1[level_offset_1 + idx01_1 * F + feat_offset];
            float fx01  = hash_table_2[level_offset_2 + idx01_2 * F * 2 + feat_offset];
            float fy01  = hash_table_2[level_offset_2 + idx01_2 * F * 2 + F + feat_offset];
            float fxy01 = hash_table_3[level_offset_3 + idx01_3 * F + feat_offset];

            float f11   = hash_table_1[level_offset_1 + idx11_1 * F + feat_offset];
            float fx11  = hash_table_2[level_offset_2 + idx11_2 * F * 2 + feat_offset];
            float fy11  = hash_table_2[level_offset_2 + idx11_2 * F * 2 + F + feat_offset];
            float fxy11 = hash_table_3[level_offset_3 + idx11_3 * F + feat_offset];
            // Hermite interpolation (16 terms)
            float value =
                f00 * hx0 * hy0 + f10 * hx2 * hy0 + f01 * hx0 * hy2 + f11 * hx2 * hy2 +
                fx00 * hx1 * hy0 + fx10 * hx3 * hy0 + fx01 * hx1 * hy2 + fx11 * hx3 * hy2 +
                fy00 * hx0 * hy1 + fy10 * hx2 * hy1 + fy01 * hx0 * hy3 + fy11 * hx2 * hy3 +
                fxy00 * hx1 * hy1 + fxy10 * hx3 * hy1 + fxy01 * hx1 * hy3 + fxy11 * hx3 * hy3;

            // Write output
            output[idx * L * F + level * F + f] = value;
        }
    }
}

/*
 * Forward with Laplacian kernel: computes encoding + derivatives + second derivatives
 *
 * Args:
 *   x: input coordinates [N, 2]
 *   hash_table: [L, hashmap_size, F*4]
 *   output: [N, L*F]
 *   output_dx: [N, L*F]
 *   output_dy: [N, L*F]
 *   output_dxx: [N, L*F] - d²enc/dx²
 *   output_dyy: [N, L*F] - d²enc/dy²
 *   resolutions: [L]
 */
__global__ void hermite_encoding_with_laplacian_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hash_table_1,
    const float* __restrict__ hash_table_2,
    const float* __restrict__ hash_table_3,
    float* __restrict__ output,
    float* __restrict__ output_dx,
    float* __restrict__ output_dy,
    float* __restrict__ output_dxx,
    float* __restrict__ output_dyy,
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 2];
    float py = x[idx * 2 + 1];

    // Process each level
    for (int level = 0; level < L; level++) {
        float res = resolutions[level];
        float res2 = res * res;

        // Scale coordinates
        float sx = px * res;
        float sy = py * res;

        // Grid coordinates
        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);

        // Local coordinates [0, 1]
        float tx = sx - (float)ix;
        float ty = sy - (float)iy;

        // Hermite basis functions
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);

        // Hash corners
        int idx00_1 = hash_coords(ix, iy, hashmap_size_1);
        int idx10_1 = hash_coords(ix + 1, iy, hashmap_size_1);
        int idx01_1 = hash_coords(ix, iy + 1, hashmap_size_1);
        int idx11_1 = hash_coords(ix + 1, iy + 1, hashmap_size_1);

        int idx00_2 = hash_coords(ix, iy, hashmap_size_2);
        int idx10_2 = hash_coords(ix + 1, iy, hashmap_size_2);
        int idx01_2 = hash_coords(ix, iy + 1, hashmap_size_2);
        int idx11_2 = hash_coords(ix + 1, iy + 1, hashmap_size_2);

        int idx00_3 = hash_coords(ix, iy, hashmap_size_3);
        int idx10_3 = hash_coords(ix + 1, iy, hashmap_size_3);
        int idx01_3 = hash_coords(ix, iy + 1, hashmap_size_3);
        int idx11_3 = hash_coords(ix + 1, iy + 1, hashmap_size_3);

        // Base offset in hash table for this level
        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 2;
        int level_offset_3 = level * hashmap_size_3 * F;

        // Process each feature
        for (int f = 0; f < F; f++) {
            int feat_offset = f;

            // Gather all 16 values from 4 corners
            float f00   = hash_table_1[level_offset_1 + idx00_1 * F + feat_offset];
            float fx00  = hash_table_2[level_offset_2 + idx00_2 * F * 2 + feat_offset];
            float fy00  = hash_table_2[level_offset_2 + idx00_2 * F * 2 + F + feat_offset];
            float fxy00 = hash_table_3[level_offset_3 + idx00_3 * F + feat_offset];

            float f10   = hash_table_1[level_offset_1 + idx10_1 * F + feat_offset];
            float fx10  = hash_table_2[level_offset_2 + idx10_2 * F * 2 + feat_offset];
            float fy10  = hash_table_2[level_offset_2 + idx10_2 * F * 2 + F + feat_offset];
            float fxy10 = hash_table_3[level_offset_3 + idx10_3 * F + feat_offset];

            float f01   = hash_table_1[level_offset_1 + idx01_1 * F + feat_offset];
            float fx01  = hash_table_2[level_offset_2 + idx01_2 * F * 2 + feat_offset];
            float fy01  = hash_table_2[level_offset_2 + idx01_2 * F * 2 + F + feat_offset];
            float fxy01 = hash_table_3[level_offset_3 + idx01_3 * F + feat_offset];

            float f11   = hash_table_1[level_offset_1 + idx11_1 * F + feat_offset];
            float fx11  = hash_table_2[level_offset_2 + idx11_2 * F * 2 + feat_offset];
            float fy11  = hash_table_2[level_offset_2 + idx11_2 * F * 2 + F + feat_offset];
            float fxy11 = hash_table_3[level_offset_3 + idx11_3 * F + feat_offset];

            // Value (16 terms)
            float value =
                f00 * hx0 * hy0 + f10 * hx2 * hy0 + f01 * hx0 * hy2 + f11 * hx2 * hy2 +
                fx00 * hx1 * hy0 + fx10 * hx3 * hy0 + fx01 * hx1 * hy2 + fx11 * hx3 * hy2 +
                fy00 * hx0 * hy1 + fy10 * hx2 * hy1 + fy01 * hx0 * hy3 + fy11 * hx2 * hy3 +
                fxy00 * hx1 * hy1 + fxy10 * hx3 * hy1 + fxy01 * hx1 * hy3 + fxy11 * hx3 * hy3;

            // du/dx (H'(tx) * H(ty)) * resolution
            float dudx = (
                f00 * dhx0 * hy0 + f10 * dhx2 * hy0 + f01 * dhx0 * hy2 + f11 * dhx2 * hy2 +
                fx00 * dhx1 * hy0 + fx10 * dhx3 * hy0 + fx01 * dhx1 * hy2 + fx11 * dhx3 * hy2 +
                fy00 * dhx0 * hy1 + fy10 * dhx2 * hy1 + fy01 * dhx0 * hy3 + fy11 * dhx2 * hy3 +
                fxy00 * dhx1 * hy1 + fxy10 * dhx3 * hy1 + fxy01 * dhx1 * hy3 + fxy11 * dhx3 * hy3
            ) * res;

            // du/dy (H(tx) * H'(ty)) * resolution
            float dudy = (
                f00 * hx0 * dhy0 + f10 * hx2 * dhy0 + f01 * hx0 * dhy2 + f11 * hx2 * dhy2 +
                fx00 * hx1 * dhy0 + fx10 * hx3 * dhy0 + fx01 * hx1 * dhy2 + fx11 * hx3 * dhy2 +
                fy00 * hx0 * dhy1 + fy10 * hx2 * dhy1 + fy01 * hx0 * dhy3 + fy11 * hx2 * dhy3 +
                fxy00 * hx1 * dhy1 + fxy10 * hx3 * dhy1 + fxy01 * hx1 * dhy3 + fxy11 * hx3 * dhy3
            ) * res;

            // d²u/dx² (H''(tx) * H(ty)) * resolution²
            float d2udx2 = (
                f00 * ddx0 * hy0 + f10 * ddx2 * hy0 + f01 * ddx0 * hy2 + f11 * ddx2 * hy2 +
                fx00 * ddx1 * hy0 + fx10 * ddx3 * hy0 + fx01 * ddx1 * hy2 + fx11 * ddx3 * hy2 +
                fy00 * ddx0 * hy1 + fy10 * ddx2 * hy1 + fy01 * ddx0 * hy3 + fy11 * ddx2 * hy3 +
                fxy00 * ddx1 * hy1 + fxy10 * ddx3 * hy1 + fxy01 * ddx1 * hy3 + fxy11 * ddx3 * hy3
            ) * res2;

            // d²u/dy² (H(tx) * H''(ty)) * resolution²
            float d2udy2 = (
                f00 * hx0 * ddy0 + f10 * hx2 * ddy0 + f01 * hx0 * ddy2 + f11 * hx2 * ddy2 +
                fx00 * hx1 * ddy0 + fx10 * hx3 * ddy0 + fx01 * hx1 * ddy2 + fx11 * hx3 * ddy2 +
                fy00 * hx0 * ddy1 + fy10 * hx2 * ddy1 + fy01 * hx0 * ddy3 + fy11 * hx2 * ddy3 +
                fxy00 * hx1 * ddy1 + fxy10 * hx3 * ddy1 + fxy01 * hx1 * ddy3 + fxy11 * hx3 * ddy3
            ) * res2;

            // Write outputs (return dxx and dyy separately for accurate chain rule)
            int out_idx = idx * L * F + level * F + f;
            output[out_idx] = value;
            output_dx[out_idx] = dudx;
            output_dy[out_idx] = dudy;
            output_dxx[out_idx] = d2udx2;
            output_dyy[out_idx] = d2udy2;
        }
    }
}

/*
 * Backward kernel for gradient computation (basic - only enc output)
 * Computes gradients w.r.t. hash_table given grad_output
 */
__global__ void hermite_encoding_backward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_hash_table_1,
    float* __restrict__ grad_hash_table_2,
    float* __restrict__ grad_hash_table_3,
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 2];
    float py = x[idx * 2 + 1];

    for (int level = 0; level < L; level++) {
        float res = resolutions[level];

        float sx = px * res;
        float sy = py * res;

        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);

        float tx = sx - (float)ix;
        float ty = sy - (float)iy;

        // Hermite basis (only need H(t) for backward of value)
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);

        int idx00_1 = hash_coords(ix, iy, hashmap_size_1);
        int idx10_1 = hash_coords(ix + 1, iy, hashmap_size_1);
        int idx01_1 = hash_coords(ix, iy + 1, hashmap_size_1);
        int idx11_1 = hash_coords(ix + 1, iy + 1, hashmap_size_1);

        int idx00_2 = hash_coords(ix, iy, hashmap_size_2);
        int idx10_2 = hash_coords(ix + 1, iy, hashmap_size_2);
        int idx01_2 = hash_coords(ix, iy + 1, hashmap_size_2);
        int idx11_2 = hash_coords(ix + 1, iy + 1, hashmap_size_2);

        int idx00_3 = hash_coords(ix, iy, hashmap_size_3);
        int idx10_3 = hash_coords(ix + 1, iy, hashmap_size_3);
        int idx01_3 = hash_coords(ix, iy + 1, hashmap_size_3);
        int idx11_3 = hash_coords(ix + 1, iy + 1, hashmap_size_3);


        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 2;
        int level_offset_3 = level * hashmap_size_3 * F;

        for (int f = 0; f < F; f++) {
            float grad = grad_output[idx * L * F + level * F + f];

            // Accumulate gradients to hash table (atomic add for thread safety)
            // Corner (0,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + idx00_1 * F + f], grad * hx0 * hy0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx00_2 * F * 2 + f], grad * hx1 * hy0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx00_2 * F * 2 + F + f], grad * hx0 * hy1);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx00_3 * F + f], grad * hx1 * hy1);

            // Corner (1,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + idx10_1 * F + f], grad * hx2 * hy0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx10_2 * F * 2 + f], grad * hx3 * hy0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx10_2 * F * 2 + F + f], grad * hx2 * hy1);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx10_3 * F + f], grad * hx3 * hy1);

            // Corner (0,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + idx01_1 * F + f], grad * hx0 * hy2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx01_2 * F * 2 + f], grad * hx1 * hy2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx01_2 * F * 2 + F + f], grad * hx0 * hy3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx01_3 * F + f], grad * hx1 * hy3);

            // Corner (1,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + idx11_1 * F + f], grad * hx2 * hy2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx11_2 * F * 2 + f], grad * hx3 * hy2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx11_2 * F * 2 + F + f], grad * hx2 * hy3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx11_3 * F + f], grad * hx3 * hy3);
        }
    }
}

/*
 * FULL Backward kernel - handles gradients from ALL outputs:
 *   - enc (value)
 *   - d_enc/dx, d_enc/dy (first derivatives)
 *   - d²enc/dx², d²enc/dy² (second derivatives)
 *
 * This enables full CUDA training for PINN without PyTorch autograd overhead.
 *
 * Analytic gradients:
 *   d(enc)/d(f00) = H0(tx) * H0(ty)
 *   d(enc)/d(fx00) = H1(tx) * H0(ty)
 *   d(enc)/d(fy00) = H0(tx) * H1(ty)
 *   d(enc)/d(fxy00) = H1(tx) * H1(ty)
 *
 *   d(d_enc/dx)/d(f00) = H0'(tx) * H0(ty) * res
 *   d(d²enc/dx²)/d(f00) = H0''(tx) * H0(ty) * res²
 *   etc.
 */
__global__ void hermite_encoding_backward_full_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_enc,      // [N, L*F] gradient from enc output
    const float* __restrict__ grad_dx,       // [N, L*F] gradient from d_enc/dx output
    const float* __restrict__ grad_dy,       // [N, L*F] gradient from d_enc/dy output
    const float* __restrict__ grad_dxx,      // [N, L*F] gradient from d²enc/dx² output
    const float* __restrict__ grad_dyy,      // [N, L*F] gradient from d²enc/dy² output
    float* __restrict__ grad_hash_table_1,     // [L, hashmap_size, F*4] output gradients
    float* __restrict__ grad_hash_table_2,     // [L, hashmap_size, F*4] output gradients
    float* __restrict__ grad_hash_table_3,     // [L, hashmap_size, F*4] output gradients
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 2];
    float py = x[idx * 2 + 1];

    for (int level = 0; level < L; level++) {
        float res = resolutions[level];
        float res2 = res * res;

        float sx = px * res;
        float sy = py * res;

        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);

        float tx = sx - (float)ix;
        float ty = sy - (float)iy;

        // Compute ALL Hermite basis functions: H, H', H''
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);

        // Hash corners
        int idx00_1 = hash_coords(ix, iy, hashmap_size_1);
        int idx10_1 = hash_coords(ix + 1, iy, hashmap_size_1);
        int idx01_1 = hash_coords(ix, iy + 1, hashmap_size_1);
        int idx11_1 = hash_coords(ix + 1, iy + 1, hashmap_size_1);

        int idx00_2 = hash_coords(ix, iy, hashmap_size_2);
        int idx10_2 = hash_coords(ix + 1, iy, hashmap_size_2);
        int idx01_2 = hash_coords(ix, iy + 1, hashmap_size_2);
        int idx11_2 = hash_coords(ix + 1, iy + 1, hashmap_size_2);

        int idx00_3 = hash_coords(ix, iy, hashmap_size_3);
        int idx10_3 = hash_coords(ix + 1, iy, hashmap_size_3);
        int idx01_3 = hash_coords(ix, iy + 1, hashmap_size_3);
        int idx11_3 = hash_coords(ix + 1, iy + 1, hashmap_size_3);

        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 2;
        int level_offset_3 = level * hashmap_size_3 * F;

        for (int f = 0; f < F; f++) {
            int out_idx = idx * L * F + level * F + f;

            // Get upstream gradients for this feature
            float g_enc = grad_enc[out_idx];
            float g_dx = grad_dx[out_idx];
            float g_dy = grad_dy[out_idx];
            float g_dxx = grad_dxx[out_idx];
            float g_dyy = grad_dyy[out_idx];

            // ========== Corner (0,0) ==========
            // Gradient contributions from each output:
            // enc:  H0(tx)*H0(ty) for f, H1(tx)*H0(ty) for fx, etc.
            // dx:   H0'(tx)*H0(ty)*res for f, H1'(tx)*H0(ty)*res for fx, etc.
            // dy:   H0(tx)*H0'(ty)*res for f, etc.
            // dxx:  H0''(tx)*H0(ty)*res² for f, etc.
            // dyy:  H0(tx)*H0''(ty)*res² for f, etc.

            float grad_f00 = g_enc * hx0 * hy0
                           + g_dx * dhx0 * hy0 * res
                           + g_dy * hx0 * dhy0 * res
                           + g_dxx * ddx0 * hy0 * res2
                           + g_dyy * hx0 * ddy0 * res2;

            float grad_fx00 = g_enc * hx1 * hy0
                            + g_dx * dhx1 * hy0 * res
                            + g_dy * hx1 * dhy0 * res
                            + g_dxx * ddx1 * hy0 * res2
                            + g_dyy * hx1 * ddy0 * res2;

            float grad_fy00 = g_enc * hx0 * hy1
                            + g_dx * dhx0 * hy1 * res
                            + g_dy * hx0 * dhy1 * res
                            + g_dxx * ddx0 * hy1 * res2
                            + g_dyy * hx0 * ddy1 * res2;

            float grad_fxy00 = g_enc * hx1 * hy1
                             + g_dx * dhx1 * hy1 * res
                             + g_dy * hx1 * dhy1 * res
                             + g_dxx * ddx1 * hy1 * res2
                             + g_dyy * hx1 * ddy1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx00_1 * F + f], grad_f00);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx00_2 * F * 2 + f], grad_fx00);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx00_2 * F * 2 + F + f], grad_fy00);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx00_3 * F + f], grad_fxy00);

            // ========== Corner (1,0) ==========
            float grad_f10 = g_enc * hx2 * hy0
                           + g_dx * dhx2 * hy0 * res
                           + g_dy * hx2 * dhy0 * res
                           + g_dxx * ddx2 * hy0 * res2
                           + g_dyy * hx2 * ddy0 * res2;

            float grad_fx10 = g_enc * hx3 * hy0
                            + g_dx * dhx3 * hy0 * res
                            + g_dy * hx3 * dhy0 * res
                            + g_dxx * ddx3 * hy0 * res2
                            + g_dyy * hx3 * ddy0 * res2;

            float grad_fy10 = g_enc * hx2 * hy1
                            + g_dx * dhx2 * hy1 * res
                            + g_dy * hx2 * dhy1 * res
                            + g_dxx * ddx2 * hy1 * res2
                            + g_dyy * hx2 * ddy1 * res2;

            float grad_fxy10 = g_enc * hx3 * hy1
                             + g_dx * dhx3 * hy1 * res
                             + g_dy * hx3 * dhy1 * res
                             + g_dxx * ddx3 * hy1 * res2
                             + g_dyy * hx3 * ddy1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx10_1 * F + f], grad_f10);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx10_2 * F * 2 + f], grad_fx10);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx10_2 * F * 2 + F + f], grad_fy10);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx10_3 * F + f], grad_fxy10);

            // ========== Corner (0,1) ==========
            float grad_f01 = g_enc * hx0 * hy2
                           + g_dx * dhx0 * hy2 * res
                           + g_dy * hx0 * dhy2 * res
                           + g_dxx * ddx0 * hy2 * res2
                           + g_dyy * hx0 * ddy2 * res2;

            float grad_fx01 = g_enc * hx1 * hy2
                            + g_dx * dhx1 * hy2 * res
                            + g_dy * hx1 * dhy2 * res
                            + g_dxx * ddx1 * hy2 * res2
                            + g_dyy * hx1 * ddy2 * res2;

            float grad_fy01 = g_enc * hx0 * hy3
                            + g_dx * dhx0 * hy3 * res
                            + g_dy * hx0 * dhy3 * res
                            + g_dxx * ddx0 * hy3 * res2
                            + g_dyy * hx0 * ddy3 * res2;

            float grad_fxy01 = g_enc * hx1 * hy3
                             + g_dx * dhx1 * hy3 * res
                             + g_dy * hx1 * dhy3 * res
                             + g_dxx * ddx1 * hy3 * res2
                             + g_dyy * hx1 * ddy3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx01_1 * F + f], grad_f01);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx01_2 * F * 2 + f], grad_fx01);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx01_2 * F * 2 + F + f], grad_fy01);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx01_3 * F + f], grad_fxy01);

            // ========== Corner (1,1) ==========
            float grad_f11 = g_enc * hx2 * hy2
                           + g_dx * dhx2 * hy2 * res
                           + g_dy * hx2 * dhy2 * res
                           + g_dxx * ddx2 * hy2 * res2
                           + g_dyy * hx2 * ddy2 * res2;

            float grad_fx11 = g_enc * hx3 * hy2
                            + g_dx * dhx3 * hy2 * res
                            + g_dy * hx3 * dhy2 * res
                            + g_dxx * ddx3 * hy2 * res2
                            + g_dyy * hx3 * ddy2 * res2;

            float grad_fy11 = g_enc * hx2 * hy3
                            + g_dx * dhx2 * hy3 * res
                            + g_dy * hx2 * dhy3 * res
                            + g_dxx * ddx2 * hy3 * res2
                            + g_dyy * hx2 * ddy3 * res2;

            float grad_fxy11 = g_enc * hx3 * hy3
                             + g_dx * dhx3 * hy3 * res
                             + g_dy * hx3 * dhy3 * res
                             + g_dxx * ddx3 * hy3 * res2
                             + g_dyy * hx3 * ddy3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx11_1 * F + f], grad_f11);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx11_2 * F * 2 + f], grad_fx11);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx11_2 * F * 2 + F + f], grad_fy11);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx11_3 * F + f], grad_fxy11);
        }
    }
}

// C++ interface functions

torch::Tensor hermite_encoding_forward_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto output = torch::zeros({N, L * F}, x.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        hash_table_1.data_ptr<float>(),
        hash_table_2.data_ptr<float>(),
        hash_table_3.data_ptr<float>(),
        output.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2, hashmap_size_3
    );

    return output;
}

std::vector<torch::Tensor> hermite_encoding_with_laplacian_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto output = torch::zeros({N, L * F}, x.options());
    auto output_dx = torch::zeros({N, L * F}, x.options());
    auto output_dy = torch::zeros({N, L * F}, x.options());
    auto output_dxx = torch::zeros({N, L * F}, x.options());
    auto output_dyy = torch::zeros({N, L * F}, x.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_with_laplacian_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        hash_table_1.data_ptr<float>(),
        hash_table_2.data_ptr<float>(),
        hash_table_3.data_ptr<float>(),
        output.data_ptr<float>(),
        output_dx.data_ptr<float>(),
        output_dy.data_ptr<float>(),
        output_dxx.data_ptr<float>(),
        output_dyy.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1,hashmap_size_2,hashmap_size_3
    );

    // Return: output, dx, dy, dxx, dyy
    return {output, output_dx, output_dy, output_dxx, output_dyy};
}

std::vector<torch::Tensor> hermite_encoding_backward_cuda(
    torch::Tensor x,
    torch::Tensor grad_output,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto grad_hash_table_1 = torch::zeros_like(hash_table_1);
    auto grad_hash_table_2 = torch::zeros_like(hash_table_2);
    auto grad_hash_table_3 = torch::zeros_like(hash_table_3);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_backward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_hash_table_1.data_ptr<float>(),
        grad_hash_table_2.data_ptr<float>(),
        grad_hash_table_3.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2, hashmap_size_3 
    );

    return {grad_hash_table_1, grad_hash_table_2, grad_hash_table_3};
}

std::vector<torch::Tensor> hermite_encoding_backward_full_cuda(
    torch::Tensor x,
    torch::Tensor grad_enc,
    torch::Tensor grad_dx,
    torch::Tensor grad_dy,
    torch::Tensor grad_dxx,
    torch::Tensor grad_dyy,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto grad_hash_table_1 = torch::zeros_like(hash_table_1);
    auto grad_hash_table_2 = torch::zeros_like(hash_table_2);
    auto grad_hash_table_3 = torch::zeros_like(hash_table_3);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_backward_full_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad_enc.data_ptr<float>(),
        grad_dx.data_ptr<float>(),
        grad_dy.data_ptr<float>(),
        grad_dxx.data_ptr<float>(),
        grad_dyy.data_ptr<float>(),
        grad_hash_table_1.data_ptr<float>(),
        grad_hash_table_2.data_ptr<float>(),
        grad_hash_table_3.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2, hashmap_size_3
    );

    return {grad_hash_table_1, grad_hash_table_2, grad_hash_table_3};
}
