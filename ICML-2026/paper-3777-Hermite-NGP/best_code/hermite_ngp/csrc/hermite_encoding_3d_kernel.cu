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
__device__ __forceinline__ int hash_coords(int x, int y, int z, int hashmap_size) {
    const unsigned int prime1 = 1u;
    const unsigned int prime2 = 2654435761u;
    const unsigned int prime3 = 805459861u;
    return ((unsigned int)x * prime1
          ^ (unsigned int)y * prime2
          ^ (unsigned int)z * prime3) % hashmap_size;
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
    const float* __restrict__ hash_table_4,
    float* __restrict__ output,
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3, int hashmap_size_4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 3];
    float py = x[idx * 3 + 1];
    float pz = x[idx * 3 + 2];

    // Process each level
    for (int level = 0; level < L; level++) {
        float res = resolutions[level];

        // Scale coordinates
        float sx = px * res;
        float sy = py * res;
        float sz = pz * res;

        // Grid coordinates
        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);
        int iz = (int)floorf(sz);

        // Local coordinates [0, 1]
        float tx = sx - (float)ix;
        float ty = sy - (float)iy;
        float tz = sz - (float)iz;

        // Hermite basis functions
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        float hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);
        hermite_basis(tz, hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3);

        // Hash corners
        int idx000_1 = hash_coords(ix,     iy,     iz,     hashmap_size_1);
        int idx100_1 = hash_coords(ix + 1, iy,     iz,     hashmap_size_1);
        int idx010_1 = hash_coords(ix,     iy + 1, iz,     hashmap_size_1);
        int idx110_1 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_1);
        int idx001_1 = hash_coords(ix,     iy,     iz + 1, hashmap_size_1);
        int idx101_1 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_1);
        int idx011_1 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_1);
        int idx111_1 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_1);

        // ---------- hash_table_2 ----------
        int idx000_2 = hash_coords(ix,     iy,     iz,     hashmap_size_2);
        int idx100_2 = hash_coords(ix + 1, iy,     iz,     hashmap_size_2);
        int idx010_2 = hash_coords(ix,     iy + 1, iz,     hashmap_size_2);
        int idx110_2 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_2);
        int idx001_2 = hash_coords(ix,     iy,     iz + 1, hashmap_size_2);
        int idx101_2 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_2);
        int idx011_2 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_2);
        int idx111_2 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_2);

        // ---------- hash_table_3 ----------
        int idx000_3 = hash_coords(ix,     iy,     iz,     hashmap_size_3);
        int idx100_3 = hash_coords(ix + 1, iy,     iz,     hashmap_size_3);
        int idx010_3 = hash_coords(ix,     iy + 1, iz,     hashmap_size_3);
        int idx110_3 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_3);
        int idx001_3 = hash_coords(ix,     iy,     iz + 1, hashmap_size_3);
        int idx101_3 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_3);
        int idx011_3 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_3);
        int idx111_3 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_3);

        // ---------- hash_table_4 ----------
        int idx000_4 = hash_coords(ix,     iy,     iz,     hashmap_size_4);
        int idx100_4 = hash_coords(ix + 1, iy,     iz,     hashmap_size_4);
        int idx010_4 = hash_coords(ix,     iy + 1, iz,     hashmap_size_4);
        int idx110_4 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_4);
        int idx001_4 = hash_coords(ix,     iy,     iz + 1, hashmap_size_4);
        int idx101_4 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_4);
        int idx011_4 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_4);
        int idx111_4 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_4);

        // Base offset in hash table for this level
        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 3;
        int level_offset_3 = level * hashmap_size_3 * F * 3;
        int level_offset_4 = level * hashmap_size_4 * F;

        // Process each feature
        for (int f = 0; f < F; f++) {
            int feat_offset = f;

            // ---------- c000 ----------
            float f000   = hash_table_1[level_offset_1 + idx000_1 * F + feat_offset];
            float fx000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + feat_offset];
            float fy000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + F + feat_offset];
            float fz000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + 2*F + feat_offset];
            float fxy000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + feat_offset];
            float fyz000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + F + feat_offset];
            float fzx000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + 2*F + feat_offset];
            float fxyz000= hash_table_4[level_offset_4 + idx000_4 * F + feat_offset];

            // ---------- c100 ----------
            float f100   = hash_table_1[level_offset_1 + idx100_1 * F + feat_offset];
            float fx100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + feat_offset];
            float fy100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + F + feat_offset];
            float fz100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + 2*F + feat_offset];
            float fxy100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + feat_offset];
            float fyz100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + F + feat_offset];
            float fzx100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + 2*F + feat_offset];
            float fxyz100= hash_table_4[level_offset_4 + idx100_4 * F + feat_offset];

            // ---------- c010 ----------
            float f010   = hash_table_1[level_offset_1 + idx010_1 * F + feat_offset];
            float fx010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + feat_offset];
            float fy010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + F + feat_offset];
            float fz010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + 2*F + feat_offset];
            float fxy010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + feat_offset];
            float fyz010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + F + feat_offset];
            float fzx010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + 2*F + feat_offset];
            float fxyz010= hash_table_4[level_offset_4 + idx010_4 * F + feat_offset];

            // ---------- c110 ----------
            float f110   = hash_table_1[level_offset_1 + idx110_1 * F + feat_offset];
            float fx110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + feat_offset];
            float fy110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + F + feat_offset];
            float fz110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + 2*F + feat_offset];
            float fxy110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + feat_offset];
            float fyz110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + F + feat_offset];
            float fzx110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + 2*F + feat_offset];
            float fxyz110= hash_table_4[level_offset_4 + idx110_4 * F + feat_offset];

            // ---------- c001 ----------
            float f001   = hash_table_1[level_offset_1 + idx001_1 * F + feat_offset];
            float fx001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + feat_offset];
            float fy001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + F + feat_offset];
            float fz001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + 2*F + feat_offset];
            float fxy001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + feat_offset];
            float fyz001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + F + feat_offset];
            float fzx001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + 2*F + feat_offset];
            float fxyz001= hash_table_4[level_offset_4 + idx001_4 * F + feat_offset];

            // ---------- c101 ----------
            float f101   = hash_table_1[level_offset_1 + idx101_1 * F + feat_offset];
            float fx101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + feat_offset];
            float fy101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + F + feat_offset];
            float fz101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + 2*F + feat_offset];
            float fxy101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + feat_offset];
            float fyz101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + F + feat_offset];
            float fzx101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + 2*F + feat_offset];
            float fxyz101= hash_table_4[level_offset_4 + idx101_4 * F + feat_offset];

            // ---------- c011 ----------
            float f011   = hash_table_1[level_offset_1 + idx011_1 * F + feat_offset];
            float fx011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + feat_offset];
            float fy011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + F + feat_offset];
            float fz011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + 2*F + feat_offset];
            float fxy011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + feat_offset];
            float fyz011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + F + feat_offset];
            float fzx011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + 2*F + feat_offset];
            float fxyz011= hash_table_4[level_offset_4 + idx011_4 * F + feat_offset];

            // ---------- c111 ----------
            float f111   = hash_table_1[level_offset_1 + idx111_1 * F + feat_offset];
            float fx111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + feat_offset];
            float fy111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + F + feat_offset];
            float fz111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + 2*F + feat_offset];
            float fxy111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + feat_offset];
            float fyz111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + F + feat_offset];
            float fzx111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + 2*F + feat_offset];
            float fxyz111= hash_table_4[level_offset_4 + idx111_4 * F + feat_offset];

            // ---------- tricubic Hermite interpolation (64 terms) ----------
            float value =
                // f
                f000 * hx0 * hy0 * hz0 + f100 * hx2 * hy0 * hz0 +
                f010 * hx0 * hy2 * hz0 + f110 * hx2 * hy2 * hz0 +
                f001 * hx0 * hy0 * hz2 + f101 * hx2 * hy0 * hz2 +
                f011 * hx0 * hy2 * hz2 + f111 * hx2 * hy2 * hz2 +

                // fx
                fx000 * hx1 * hy0 * hz0 + fx100 * hx3 * hy0 * hz0 +
                fx010 * hx1 * hy2 * hz0 + fx110 * hx3 * hy2 * hz0 +
                fx001 * hx1 * hy0 * hz2 + fx101 * hx3 * hy0 * hz2 +
                fx011 * hx1 * hy2 * hz2 + fx111 * hx3 * hy2 * hz2 +

                // fy
                fy000 * hx0 * hy1 * hz0 + fy100 * hx2 * hy1 * hz0 +
                fy010 * hx0 * hy3 * hz0 + fy110 * hx2 * hy3 * hz0 +
                fy001 * hx0 * hy1 * hz2 + fy101 * hx2 * hy1 * hz2 +
                fy011 * hx0 * hy3 * hz2 + fy111 * hx2 * hy3 * hz2 +

                // fz
                fz000 * hx0 * hy0 * hz1 + fz100 * hx2 * hy0 * hz1 +
                fz010 * hx0 * hy2 * hz1 + fz110 * hx2 * hy2 * hz1 +
                fz001 * hx0 * hy0 * hz3 + fz101 * hx2 * hy0 * hz3 +
                fz011 * hx0 * hy2 * hz3 + fz111 * hx2 * hy2 * hz3 +

                // fxy
                fxy000 * hx1 * hy1 * hz0 + fxy100 * hx3 * hy1 * hz0 +
                fxy010 * hx1 * hy3 * hz0 + fxy110 * hx3 * hy3 * hz0 +
                fxy001 * hx1 * hy1 * hz2 + fxy101 * hx3 * hy1 * hz2 +
                fxy011 * hx1 * hy3 * hz2 + fxy111 * hx3 * hy3 * hz2 +

                // fyz
                fyz000 * hx0 * hy1 * hz1 + fyz100 * hx2 * hy1 * hz1 +
                fyz010 * hx0 * hy3 * hz1 + fyz110 * hx2 * hy3 * hz1 +
                fyz001 * hx0 * hy1 * hz3 + fyz101 * hx2 * hy1 * hz3 +
                fyz011 * hx0 * hy3 * hz3 + fyz111 * hx2 * hy3 * hz3 +

                // fzx
                fzx000 * hx1 * hy0 * hz1 + fzx100 * hx3 * hy0 * hz1 +
                fzx010 * hx1 * hy2 * hz1 + fzx110 * hx3 * hy2 * hz1 +
                fzx001 * hx1 * hy0 * hz3 + fzx101 * hx3 * hy0 * hz3 +
                fzx011 * hx1 * hy2 * hz3 + fzx111 * hx3 * hy2 * hz3 +

                // fxyz
                fxyz000 * hx1 * hy1 * hz1 + fxyz100 * hx3 * hy1 * hz1 +
                fxyz010 * hx1 * hy3 * hz1 + fxyz110 * hx3 * hy3 * hz1 +
                fxyz001 * hx1 * hy1 * hz3 + fxyz101 * hx3 * hy1 * hz3 +
                fxyz011 * hx1 * hy3 * hz3 + fxyz111 * hx3 * hy3 * hz3;

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
    const float* __restrict__ hash_table_4,
    float* __restrict__ output,
    float* __restrict__ output_dx,
    float* __restrict__ output_dy,
    float* __restrict__ output_dz,
    float* __restrict__ output_dxx,
    float* __restrict__ output_dyy,
    float* __restrict__ output_dzz,
    const float* __restrict__ resolutions,
    int N, int L, int F, 
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3, int hashmap_size_4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 3];
    float py = x[idx * 3 + 1];
    float pz = x[idx * 3 + 2];

    // Process each level
    for (int level = 0; level < L; level++) {
        float res = resolutions[level];
        float res2 = res * res;

        // Scale coordinates
        float sx = px * res;
        float sy = py * res;
        float sz = pz * res;

        // Grid coordinates
        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);
        int iz = (int)floorf(sz);

        // Local coordinates [0, 1]
        float tx = sx - (float)ix;
        float ty = sy - (float)iy;
        float tz = sz - (float)iz;

        // Hermite basis functions
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        float hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);
        hermite_basis(tz, hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3);

// Hash corners
        int idx000_1 = hash_coords(ix,     iy,     iz,     hashmap_size_1);
        int idx100_1 = hash_coords(ix + 1, iy,     iz,     hashmap_size_1);
        int idx010_1 = hash_coords(ix,     iy + 1, iz,     hashmap_size_1);
        int idx110_1 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_1);
        int idx001_1 = hash_coords(ix,     iy,     iz + 1, hashmap_size_1);
        int idx101_1 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_1);
        int idx011_1 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_1);
        int idx111_1 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_1);

        // ---------- hash_table_2 ----------
        int idx000_2 = hash_coords(ix,     iy,     iz,     hashmap_size_2);
        int idx100_2 = hash_coords(ix + 1, iy,     iz,     hashmap_size_2);
        int idx010_2 = hash_coords(ix,     iy + 1, iz,     hashmap_size_2);
        int idx110_2 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_2);
        int idx001_2 = hash_coords(ix,     iy,     iz + 1, hashmap_size_2);
        int idx101_2 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_2);
        int idx011_2 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_2);
        int idx111_2 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_2);

        // ---------- hash_table_3 ----------
        int idx000_3 = hash_coords(ix,     iy,     iz,     hashmap_size_3);
        int idx100_3 = hash_coords(ix + 1, iy,     iz,     hashmap_size_3);
        int idx010_3 = hash_coords(ix,     iy + 1, iz,     hashmap_size_3);
        int idx110_3 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_3);
        int idx001_3 = hash_coords(ix,     iy,     iz + 1, hashmap_size_3);
        int idx101_3 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_3);
        int idx011_3 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_3);
        int idx111_3 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_3);

        // ---------- hash_table_4 ----------
        int idx000_4 = hash_coords(ix,     iy,     iz,     hashmap_size_4);
        int idx100_4 = hash_coords(ix + 1, iy,     iz,     hashmap_size_4);
        int idx010_4 = hash_coords(ix,     iy + 1, iz,     hashmap_size_4);
        int idx110_4 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_4);
        int idx001_4 = hash_coords(ix,     iy,     iz + 1, hashmap_size_4);
        int idx101_4 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_4);
        int idx011_4 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_4);
        int idx111_4 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_4);

        // Base offset in hash table for this level
        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 3;
        int level_offset_3 = level * hashmap_size_3 * F * 3;
        int level_offset_4 = level * hashmap_size_4 * F;

        // Process each feature
        for (int f = 0; f < F; f++) {
            int feat_offset = f;

            // ---------- c000 ----------
            float f000   = hash_table_1[level_offset_1 + idx000_1 * F + feat_offset];
            float fx000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + feat_offset];
            float fy000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + F + feat_offset];
            float fz000  = hash_table_2[level_offset_2 + idx000_2 * F * 3 + 2*F + feat_offset];
            float fxy000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + feat_offset];
            float fyz000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + F + feat_offset];
            float fzx000 = hash_table_3[level_offset_3 + idx000_3 * F * 3 + 2*F + feat_offset];
            float fxyz000= hash_table_4[level_offset_4 + idx000_4 * F + feat_offset];

            // ---------- c100 ----------
            float f100   = hash_table_1[level_offset_1 + idx100_1 * F + feat_offset];
            float fx100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + feat_offset];
            float fy100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + F + feat_offset];
            float fz100  = hash_table_2[level_offset_2 + idx100_2 * F * 3 + 2*F + feat_offset];
            float fxy100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + feat_offset];
            float fyz100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + F + feat_offset];
            float fzx100 = hash_table_3[level_offset_3 + idx100_3 * F * 3 + 2*F + feat_offset];
            float fxyz100= hash_table_4[level_offset_4 + idx100_4 * F + feat_offset];

            // ---------- c010 ----------
            float f010   = hash_table_1[level_offset_1 + idx010_1 * F + feat_offset];
            float fx010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + feat_offset];
            float fy010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + F + feat_offset];
            float fz010  = hash_table_2[level_offset_2 + idx010_2 * F * 3 + 2*F + feat_offset];
            float fxy010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + feat_offset];
            float fyz010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + F + feat_offset];
            float fzx010 = hash_table_3[level_offset_3 + idx010_3 * F * 3 + 2*F + feat_offset];
            float fxyz010= hash_table_4[level_offset_4 + idx010_4 * F + feat_offset];

            // ---------- c110 ----------
            float f110   = hash_table_1[level_offset_1 + idx110_1 * F + feat_offset];
            float fx110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + feat_offset];
            float fy110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + F + feat_offset];
            float fz110  = hash_table_2[level_offset_2 + idx110_2 * F * 3 + 2*F + feat_offset];
            float fxy110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + feat_offset];
            float fyz110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + F + feat_offset];
            float fzx110 = hash_table_3[level_offset_3 + idx110_3 * F * 3 + 2*F + feat_offset];
            float fxyz110= hash_table_4[level_offset_4 + idx110_4 * F + feat_offset];

            // ---------- c001 ----------
            float f001   = hash_table_1[level_offset_1 + idx001_1 * F + feat_offset];
            float fx001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + feat_offset];
            float fy001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + F + feat_offset];
            float fz001  = hash_table_2[level_offset_2 + idx001_2 * F * 3 + 2*F + feat_offset];
            float fxy001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + feat_offset];
            float fyz001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + F + feat_offset];
            float fzx001 = hash_table_3[level_offset_3 + idx001_3 * F * 3 + 2*F + feat_offset];
            float fxyz001= hash_table_4[level_offset_4 + idx001_4 * F + feat_offset];

            // ---------- c101 ----------
            float f101   = hash_table_1[level_offset_1 + idx101_1 * F + feat_offset];
            float fx101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + feat_offset];
            float fy101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + F + feat_offset];
            float fz101  = hash_table_2[level_offset_2 + idx101_2 * F * 3 + 2*F + feat_offset];
            float fxy101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + feat_offset];
            float fyz101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + F + feat_offset];
            float fzx101 = hash_table_3[level_offset_3 + idx101_3 * F * 3 + 2*F + feat_offset];
            float fxyz101= hash_table_4[level_offset_4 + idx101_4 * F + feat_offset];

            // ---------- c011 ----------
            float f011   = hash_table_1[level_offset_1 + idx011_1 * F + feat_offset];
            float fx011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + feat_offset];
            float fy011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + F + feat_offset];
            float fz011  = hash_table_2[level_offset_2 + idx011_2 * F * 3 + 2*F + feat_offset];
            float fxy011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + feat_offset];
            float fyz011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + F + feat_offset];
            float fzx011 = hash_table_3[level_offset_3 + idx011_3 * F * 3 + 2*F + feat_offset];
            float fxyz011= hash_table_4[level_offset_4 + idx011_4 * F + feat_offset];

            // ---------- c111 ----------
            float f111   = hash_table_1[level_offset_1 + idx111_1 * F + feat_offset];
            float fx111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + feat_offset];
            float fy111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + F + feat_offset];
            float fz111  = hash_table_2[level_offset_2 + idx111_2 * F * 3 + 2*F + feat_offset];
            float fxy111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + feat_offset];
            float fyz111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + F + feat_offset];
            float fzx111 = hash_table_3[level_offset_3 + idx111_3 * F * 3 + 2*F + feat_offset];
            float fxyz111= hash_table_4[level_offset_4 + idx111_4 * F + feat_offset];

            // ---------- tricubic Hermite interpolation (64 terms) ----------
            float value =
                // f
                f000 * hx0 * hy0 * hz0 + f100 * hx2 * hy0 * hz0 +
                f010 * hx0 * hy2 * hz0 + f110 * hx2 * hy2 * hz0 +
                f001 * hx0 * hy0 * hz2 + f101 * hx2 * hy0 * hz2 +
                f011 * hx0 * hy2 * hz2 + f111 * hx2 * hy2 * hz2 +

                // fx
                fx000 * hx1 * hy0 * hz0 + fx100 * hx3 * hy0 * hz0 +
                fx010 * hx1 * hy2 * hz0 + fx110 * hx3 * hy2 * hz0 +
                fx001 * hx1 * hy0 * hz2 + fx101 * hx3 * hy0 * hz2 +
                fx011 * hx1 * hy2 * hz2 + fx111 * hx3 * hy2 * hz2 +

                // fy
                fy000 * hx0 * hy1 * hz0 + fy100 * hx2 * hy1 * hz0 +
                fy010 * hx0 * hy3 * hz0 + fy110 * hx2 * hy3 * hz0 +
                fy001 * hx0 * hy1 * hz2 + fy101 * hx2 * hy1 * hz2 +
                fy011 * hx0 * hy3 * hz2 + fy111 * hx2 * hy3 * hz2 +

                // fz
                fz000 * hx0 * hy0 * hz1 + fz100 * hx2 * hy0 * hz1 +
                fz010 * hx0 * hy2 * hz1 + fz110 * hx2 * hy2 * hz1 +
                fz001 * hx0 * hy0 * hz3 + fz101 * hx2 * hy0 * hz3 +
                fz011 * hx0 * hy2 * hz3 + fz111 * hx2 * hy2 * hz3 +

                // fxy
                fxy000 * hx1 * hy1 * hz0 + fxy100 * hx3 * hy1 * hz0 +
                fxy010 * hx1 * hy3 * hz0 + fxy110 * hx3 * hy3 * hz0 +
                fxy001 * hx1 * hy1 * hz2 + fxy101 * hx3 * hy1 * hz2 +
                fxy011 * hx1 * hy3 * hz2 + fxy111 * hx3 * hy3 * hz2 +

                // fyz
                fyz000 * hx0 * hy1 * hz1 + fyz100 * hx2 * hy1 * hz1 +
                fyz010 * hx0 * hy3 * hz1 + fyz110 * hx2 * hy3 * hz1 +
                fyz001 * hx0 * hy1 * hz3 + fyz101 * hx2 * hy1 * hz3 +
                fyz011 * hx0 * hy3 * hz3 + fyz111 * hx2 * hy3 * hz3 +

                // fzx
                fzx000 * hx1 * hy0 * hz1 + fzx100 * hx3 * hy0 * hz1 +
                fzx010 * hx1 * hy2 * hz1 + fzx110 * hx3 * hy2 * hz1 +
                fzx001 * hx1 * hy0 * hz3 + fzx101 * hx3 * hy0 * hz3 +
                fzx011 * hx1 * hy2 * hz3 + fzx111 * hx3 * hy2 * hz3 +

                // fxyz
                fxyz000 * hx1 * hy1 * hz1 + fxyz100 * hx3 * hy1 * hz1 +
                fxyz010 * hx1 * hy3 * hz1 + fxyz110 * hx3 * hy3 * hz1 +
                fxyz001 * hx1 * hy1 * hz3 + fxyz101 * hx3 * hy1 * hz3 +
                fxyz011 * hx1 * hy3 * hz3 + fxyz111 * hx3 * hy3 * hz3;
            
            float dudx = (
                // f
                f000 * dhx0 * hy0 * hz0 + f100 * dhx2 * hy0 * hz0 +
                f010 * dhx0 * hy2 * hz0 + f110 * dhx2 * hy2 * hz0 +
                f001 * dhx0 * hy0 * hz2 + f101 * dhx2 * hy0 * hz2 +
                f011 * dhx0 * hy2 * hz2 + f111 * dhx2 * hy2 * hz2 +

                // fx
                fx000 * dhx1 * hy0 * hz0 + fx100 * dhx3 * hy0 * hz0 +
                fx010 * dhx1 * hy2 * hz0 + fx110 * dhx3 * hy2 * hz0 +
                fx001 * dhx1 * hy0 * hz2 + fx101 * dhx3 * hy0 * hz2 +
                fx011 * dhx1 * hy2 * hz2 + fx111 * dhx3 * hy2 * hz2 +

                // fy
                fy000 * dhx0 * hy1 * hz0 + fy100 * dhx2 * hy1 * hz0 +
                fy010 * dhx0 * hy3 * hz0 + fy110 * dhx2 * hy3 * hz0 +
                fy001 * dhx0 * hy1 * hz2 + fy101 * dhx2 * hy1 * hz2 +
                fy011 * dhx0 * hy3 * hz2 + fy111 * dhx2 * hy3 * hz2 +

                // fz
                fz000 * dhx0 * hy0 * hz1 + fz100 * dhx2 * hy0 * hz1 +
                fz010 * dhx0 * hy2 * hz1 + fz110 * dhx2 * hy2 * hz1 +
                fz001 * dhx0 * hy0 * hz3 + fz101 * dhx2 * hy0 * hz3 +
                fz011 * dhx0 * hy2 * hz3 + fz111 * dhx2 * hy2 * hz3 +

                // fxy
                fxy000 * dhx1 * hy1 * hz0 + fxy100 * dhx3 * hy1 * hz0 +
                fxy010 * dhx1 * hy3 * hz0 + fxy110 * dhx3 * hy3 * hz0 +
                fxy001 * dhx1 * hy1 * hz2 + fxy101 * dhx3 * hy1 * hz2 +
                fxy011 * dhx1 * hy3 * hz2 + fxy111 * dhx3 * hy3 * hz2 +

                // fyz
                fyz000 * dhx0 * hy1 * hz1 + fyz100 * dhx2 * hy1 * hz1 +
                fyz010 * dhx0 * hy3 * hz1 + fyz110 * dhx2 * hy3 * hz1 +
                fyz001 * dhx0 * hy1 * hz3 + fyz101 * dhx2 * hy1 * hz3 +
                fyz011 * dhx0 * hy3 * hz3 + fyz111 * dhx2 * hy3 * hz3 +

                // fzx
                fzx000 * dhx1 * hy0 * hz1 + fzx100 * dhx3 * hy0 * hz1 +
                fzx010 * dhx1 * hy2 * hz1 + fzx110 * dhx3 * hy2 * hz1 +
                fzx001 * dhx1 * hy0 * hz3 + fzx101 * dhx3 * hy0 * hz3 +
                fzx011 * dhx1 * hy2 * hz3 + fzx111 * dhx3 * hy2 * hz3 +

                // fxyz
                fxyz000 * dhx1 * hy1 * hz1 + fxyz100 * dhx3 * hy1 * hz1 +
                fxyz010 * dhx1 * hy3 * hz1 + fxyz110 * dhx3 * hy3 * hz1 +
                fxyz001 * dhx1 * hy1 * hz3 + fxyz101 * dhx3 * hy1 * hz3 +
                fxyz011 * dhx1 * hy3 * hz3 + fxyz111 * dhx3 * hy3 * hz3) * res;
            
            float dudy = (
                // f
                f000 * hx0 * dhy0 * hz0 + f100 * hx2 * dhy0 * hz0 +
                f010 * hx0 * dhy2 * hz0 + f110 * hx2 * dhy2 * hz0 +
                f001 * hx0 * dhy0 * hz2 + f101 * hx2 * dhy0 * hz2 +
                f011 * hx0 * dhy2 * hz2 + f111 * hx2 * dhy2 * hz2 +

                // fx
                fx000 * hx1 * dhy0 * hz0 + fx100 * hx3 * dhy0 * hz0 +
                fx010 * hx1 * dhy2 * hz0 + fx110 * hx3 * dhy2 * hz0 +
                fx001 * hx1 * dhy0 * hz2 + fx101 * hx3 * dhy0 * hz2 +
                fx011 * hx1 * dhy2 * hz2 + fx111 * hx3 * dhy2 * hz2 +

                // fy
                fy000 * hx0 * dhy1 * hz0 + fy100 * hx2 * dhy1 * hz0 +
                fy010 * hx0 * dhy3 * hz0 + fy110 * hx2 * dhy3 * hz0 +
                fy001 * hx0 * dhy1 * hz2 + fy101 * hx2 * dhy1 * hz2 +
                fy011 * hx0 * dhy3 * hz2 + fy111 * hx2 * dhy3 * hz2 +

                // fz
                fz000 * hx0 * dhy0 * hz1 + fz100 * hx2 * dhy0 * hz1 +
                fz010 * hx0 * dhy2 * hz1 + fz110 * hx2 * dhy2 * hz1 +
                fz001 * hx0 * dhy0 * hz3 + fz101 * hx2 * dhy0 * hz3 +
                fz011 * hx0 * dhy2 * hz3 + fz111 * hx2 * dhy2 * hz3 +

                // fxy
                fxy000 * hx1 * dhy1 * hz0 + fxy100 * hx3 * dhy1 * hz0 +
                fxy010 * hx1 * dhy3 * hz0 + fxy110 * hx3 * dhy3 * hz0 +
                fxy001 * hx1 * dhy1 * hz2 + fxy101 * hx3 * dhy1 * hz2 +
                fxy011 * hx1 * dhy3 * hz2 + fxy111 * hx3 * dhy3 * hz2 +

                // fyz
                fyz000 * hx0 * dhy1 * hz1 + fyz100 * hx2 * dhy1 * hz1 +
                fyz010 * hx0 * dhy3 * hz1 + fyz110 * hx2 * dhy3 * hz1 +
                fyz001 * hx0 * dhy1 * hz3 + fyz101 * hx2 * dhy1 * hz3 +
                fyz011 * hx0 * dhy3 * hz3 + fyz111 * hx2 * dhy3 * hz3 +

                // fzx
                fzx000 * hx1 * dhy0 * hz1 + fzx100 * hx3 * dhy0 * hz1 +
                fzx010 * hx1 * dhy2 * hz1 + fzx110 * hx3 * dhy2 * hz1 +
                fzx001 * hx1 * dhy0 * hz3 + fzx101 * hx3 * dhy0 * hz3 +
                fzx011 * hx1 * dhy2 * hz3 + fzx111 * hx3 * dhy2 * hz3 +

                // fxyz
                fxyz000 * hx1 * dhy1 * hz1 + fxyz100 * hx3 * dhy1 * hz1 +
                fxyz010 * hx1 * dhy3 * hz1 + fxyz110 * hx3 * dhy3 * hz1 +
                fxyz001 * hx1 * dhy1 * hz3 + fxyz101 * hx3 * dhy1 * hz3 +
                fxyz011 * hx1 * dhy3 * hz3 + fxyz111 * hx3 * dhy3 * hz3) * res;
            
            float dudz =(
                // f
                f000 * hx0 * hy0 * dhz0 + f100 * hx2 * hy0 * dhz0 +
                f010 * hx0 * hy2 * dhz0 + f110 * hx2 * hy2 * dhz0 +
                f001 * hx0 * hy0 * dhz2 + f101 * hx2 * hy0 * dhz2 +
                f011 * hx0 * hy2 * dhz2 + f111 * hx2 * hy2 * dhz2 +

                // fx
                fx000 * hx1 * hy0 * dhz0 + fx100 * hx3 * hy0 * dhz0 +
                fx010 * hx1 * hy2 * dhz0 + fx110 * hx3 * hy2 * dhz0 +
                fx001 * hx1 * hy0 * dhz2 + fx101 * hx3 * hy0 * dhz2 +
                fx011 * hx1 * hy2 * dhz2 + fx111 * hx3 * hy2 * dhz2 +

                // fy
                fy000 * hx0 * hy1 * dhz0 + fy100 * hx2 * hy1 * dhz0 +
                fy010 * hx0 * hy3 * dhz0 + fy110 * hx2 * hy3 * dhz0 +
                fy001 * hx0 * hy1 * dhz2 + fy101 * hx2 * hy1 * dhz2 +
                fy011 * hx0 * hy3 * dhz2 + fy111 * hx2 * hy3 * dhz2 +

                // fz
                fz000 * hx0 * hy0 * dhz1 + fz100 * hx2 * hy0 * dhz1 +
                fz010 * hx0 * hy2 * dhz1 + fz110 * hx2 * hy2 * dhz1 +
                fz001 * hx0 * hy0 * dhz3 + fz101 * hx2 * hy0 * dhz3 +
                fz011 * hx0 * hy2 * dhz3 + fz111 * hx2 * hy2 * dhz3 +

                // fxy
                fxy000 * hx1 * hy1 * dhz0 + fxy100 * hx3 * hy1 * dhz0 +
                fxy010 * hx1 * hy3 * dhz0 + fxy110 * hx3 * hy3 * dhz0 +
                fxy001 * hx1 * hy1 * dhz2 + fxy101 * hx3 * hy1 * dhz2 +
                fxy011 * hx1 * hy3 * dhz2 + fxy111 * hx3 * hy3 * dhz2 +

                // fyz
                fyz000 * hx0 * hy1 * dhz1 + fyz100 * hx2 * hy1 * dhz1 +
                fyz010 * hx0 * hy3 * dhz1 + fyz110 * hx2 * hy3 * dhz1 +
                fyz001 * hx0 * hy1 * dhz3 + fyz101 * hx2 * hy1 * dhz3 +
                fyz011 * hx0 * hy3 * dhz3 + fyz111 * hx2 * hy3 * dhz3 +

                // fzx
                fzx000 * hx1 * hy0 * dhz1 + fzx100 * hx3 * hy0 * dhz1 +
                fzx010 * hx1 * hy2 * dhz1 + fzx110 * hx3 * hy2 * dhz1 +
                fzx001 * hx1 * hy0 * dhz3 + fzx101 * hx3 * hy0 * dhz3 +
                fzx011 * hx1 * hy2 * dhz3 + fzx111 * hx3 * hy2 * dhz3 +

                // fxyz
                fxyz000 * hx1 * hy1 * dhz1 + fxyz100 * hx3 * hy1 * dhz1 +
                fxyz010 * hx1 * hy3 * dhz1 + fxyz110 * hx3 * hy3 * dhz1 +
                fxyz001 * hx1 * hy1 * dhz3 + fxyz101 * hx3 * hy1 * dhz3 +
                fxyz011 * hx1 * hy3 * dhz3 + fxyz111 * hx3 * hy3 * dhz3) * res;

            float d2udx2 = (
                // f
                f000 * ddx0 * hy0 * hz0 + f100 * ddx2 * hy0 * hz0 +
                f010 * ddx0 * hy2 * hz0 + f110 * ddx2 * hy2 * hz0 +
                f001 * ddx0 * hy0 * hz2 + f101 * ddx2 * hy0 * hz2 +
                f011 * ddx0 * hy2 * hz2 + f111 * ddx2 * hy2 * hz2 +

                // fx
                fx000 * ddx1 * hy0 * hz0 + fx100 * ddx3 * hy0 * hz0 +
                fx010 * ddx1 * hy2 * hz0 + fx110 * ddx3 * hy2 * hz0 +
                fx001 * ddx1 * hy0 * hz2 + fx101 * ddx3 * hy0 * hz2 +
                fx011 * ddx1 * hy2 * hz2 + fx111 * ddx3 * hy2 * hz2 +

                // fy
                fy000 * ddx0 * hy1 * hz0 + fy100 * ddx2 * hy1 * hz0 +
                fy010 * ddx0 * hy3 * hz0 + fy110 * ddx2 * hy3 * hz0 +
                fy001 * ddx0 * hy1 * hz2 + fy101 * ddx2 * hy1 * hz2 +
                fy011 * ddx0 * hy3 * hz2 + fy111 * ddx2 * hy3 * hz2 +

                // fz
                fz000 * ddx0 * hy0 * hz1 + fz100 * ddx2 * hy0 * hz1 +
                fz010 * ddx0 * hy2 * hz1 + fz110 * ddx2 * hy2 * hz1 +
                fz001 * ddx0 * hy0 * hz3 + fz101 * ddx2 * hy0 * hz3 +
                fz011 * ddx0 * hy2 * hz3 + fz111 * ddx2 * hy2 * hz3 +

                // fxy
                fxy000 * ddx1 * hy1 * hz0 + fxy100 * ddx3 * hy1 * hz0 +
                fxy010 * ddx1 * hy3 * hz0 + fxy110 * ddx3 * hy3 * hz0 +
                fxy001 * ddx1 * hy1 * hz2 + fxy101 * ddx3 * hy1 * hz2 +
                fxy011 * ddx1 * hy3 * hz2 + fxy111 * ddx3 * hy3 * hz2 +

                // fyz
                fyz000 * ddx0 * hy1 * hz1 + fyz100 * ddx2 * hy1 * hz1 +
                fyz010 * ddx0 * hy3 * hz1 + fyz110 * ddx2 * hy3 * hz1 +
                fyz001 * ddx0 * hy1 * hz3 + fyz101 * ddx2 * hy1 * hz3 +
                fyz011 * ddx0 * hy3 * hz3 + fyz111 * ddx2 * hy3 * hz3 +

                // fzx
                fzx000 * ddx1 * hy0 * hz1 + fzx100 * ddx3 * hy0 * hz1 +
                fzx010 * ddx1 * hy2 * hz1 + fzx110 * ddx3 * hy2 * hz1 +
                fzx001 * ddx1 * hy0 * hz3 + fzx101 * ddx3 * hy0 * hz3 +
                fzx011 * ddx1 * hy2 * hz3 + fzx111 * ddx3 * hy2 * hz3 +

                // fxyz
                fxyz000 * ddx1 * hy1 * hz1 + fxyz100 * ddx3 * hy1 * hz1 +
                fxyz010 * ddx1 * hy3 * hz1 + fxyz110 * ddx3 * hy3 * hz1 +
                fxyz001 * ddx1 * hy1 * hz3 + fxyz101 * ddx3 * hy1 * hz3 +
                fxyz011 * ddx1 * hy3 * hz3 + fxyz111 * ddx3 * hy3 * hz3) * res2;
            
            float d2udy2 = (
                // f
                f000 * hx0 * ddy0 * hz0 + f100 * hx2 * ddy0 * hz0 +
                f010 * hx0 * ddy2 * hz0 + f110 * hx2 * ddy2 * hz0 +
                f001 * hx0 * ddy0 * hz2 + f101 * hx2 * ddy0 * hz2 +
                f011 * hx0 * ddy2 * hz2 + f111 * hx2 * ddy2 * hz2 +

                // fx
                fx000 * hx1 * ddy0 * hz0 + fx100 * hx3 * ddy0 * hz0 +
                fx010 * hx1 * ddy2 * hz0 + fx110 * hx3 * ddy2 * hz0 +
                fx001 * hx1 * ddy0 * hz2 + fx101 * hx3 * ddy0 * hz2 +
                fx011 * hx1 * ddy2 * hz2 + fx111 * hx3 * ddy2 * hz2 +

                // fy
                fy000 * hx0 * ddy1 * hz0 + fy100 * hx2 * ddy1 * hz0 +
                fy010 * hx0 * ddy3 * hz0 + fy110 * hx2 * ddy3 * hz0 +
                fy001 * hx0 * ddy1 * hz2 + fy101 * hx2 * ddy1 * hz2 +
                fy011 * hx0 * ddy3 * hz2 + fy111 * hx2 * ddy3 * hz2 +

                // fz
                fz000 * hx0 * ddy0 * hz1 + fz100 * hx2 * ddy0 * hz1 +
                fz010 * hx0 * ddy2 * hz1 + fz110 * hx2 * ddy2 * hz1 +
                fz001 * hx0 * ddy0 * hz3 + fz101 * hx2 * ddy0 * hz3 +
                fz011 * hx0 * ddy2 * hz3 + fz111 * hx2 * ddy2 * hz3 +

                // fxy
                fxy000 * hx1 * ddy1 * hz0 + fxy100 * hx3 * ddy1 * hz0 +
                fxy010 * hx1 * ddy3 * hz0 + fxy110 * hx3 * ddy3 * hz0 +
                fxy001 * hx1 * ddy1 * hz2 + fxy101 * hx3 * ddy1 * hz2 +
                fxy011 * hx1 * ddy3 * hz2 + fxy111 * hx3 * ddy3 * hz2 +

                // fyz
                fyz000 * hx0 * ddy1 * hz1 + fyz100 * hx2 * ddy1 * hz1 +
                fyz010 * hx0 * ddy3 * hz1 + fyz110 * hx2 * ddy3 * hz1 +
                fyz001 * hx0 * ddy1 * hz3 + fyz101 * hx2 * ddy1 * hz3 +
                fyz011 * hx0 * ddy3 * hz3 + fyz111 * hx2 * ddy3 * hz3 +

                // fzx
                fzx000 * hx1 * ddy0 * hz1 + fzx100 * hx3 * ddy0 * hz1 +
                fzx010 * hx1 * ddy2 * hz1 + fzx110 * hx3 * ddy2 * hz1 +
                fzx001 * hx1 * ddy0 * hz3 + fzx101 * hx3 * ddy0 * hz3 +
                fzx011 * hx1 * ddy2 * hz3 + fzx111 * hx3 * ddy2 * hz3 +

                // fxyz
                fxyz000 * hx1 * ddy1 * hz1 + fxyz100 * hx3 * ddy1 * hz1 +
                fxyz010 * hx1 * ddy3 * hz1 + fxyz110 * hx3 * ddy3 * hz1 +
                fxyz001 * hx1 * ddy1 * hz3 + fxyz101 * hx3 * ddy1 * hz3 +
                fxyz011 * hx1 * ddy3 * hz3 + fxyz111 * hx3 * ddy3 * hz3) * res2;
            
            float d2udz2 =(
                // f
                f000 * hx0 * hy0 * ddz0 + f100 * hx2 * hy0 * ddz0 +
                f010 * hx0 * hy2 * ddz0 + f110 * hx2 * hy2 * ddz0 +
                f001 * hx0 * hy0 * ddz2 + f101 * hx2 * hy0 * ddz2 +
                f011 * hx0 * hy2 * ddz2 + f111 * hx2 * hy2 * ddz2 +

                // fx
                fx000 * hx1 * hy0 * ddz0 + fx100 * hx3 * hy0 * ddz0 +
                fx010 * hx1 * hy2 * ddz0 + fx110 * hx3 * hy2 * ddz0 +
                fx001 * hx1 * hy0 * ddz2 + fx101 * hx3 * hy0 * ddz2 +
                fx011 * hx1 * hy2 * ddz2 + fx111 * hx3 * hy2 * ddz2 +

                // fy
                fy000 * hx0 * hy1 * ddz0 + fy100 * hx2 * hy1 * ddz0 +
                fy010 * hx0 * hy3 * ddz0 + fy110 * hx2 * hy3 * ddz0 +
                fy001 * hx0 * hy1 * ddz2 + fy101 * hx2 * hy1 * ddz2 +
                fy011 * hx0 * hy3 * ddz2 + fy111 * hx2 * hy3 * ddz2 +

                // fz
                fz000 * hx0 * hy0 * ddz1 + fz100 * hx2 * hy0 * ddz1 +
                fz010 * hx0 * hy2 * ddz1 + fz110 * hx2 * hy2 * ddz1 +
                fz001 * hx0 * hy0 * ddz3 + fz101 * hx2 * hy0 * ddz3 +
                fz011 * hx0 * hy2 * ddz3 + fz111 * hx2 * hy2 * ddz3 +

                // fxy
                fxy000 * hx1 * hy1 * ddz0 + fxy100 * hx3 * hy1 * ddz0 +
                fxy010 * hx1 * hy3 * ddz0 + fxy110 * hx3 * hy3 * ddz0 +
                fxy001 * hx1 * hy1 * ddz2 + fxy101 * hx3 * hy1 * ddz2 +
                fxy011 * hx1 * hy3 * ddz2 + fxy111 * hx3 * hy3 * ddz2 +

                // fyz
                fyz000 * hx0 * hy1 * ddz1 + fyz100 * hx2 * hy1 * ddz1 +
                fyz010 * hx0 * hy3 * ddz1 + fyz110 * hx2 * hy3 * ddz1 +
                fyz001 * hx0 * hy1 * ddz3 + fyz101 * hx2 * hy1 * ddz3 +
                fyz011 * hx0 * hy3 * ddz3 + fyz111 * hx2 * hy3 * ddz3 +

                // fzx
                fzx000 * hx1 * hy0 * ddz1 + fzx100 * hx3 * hy0 * ddz1 +
                fzx010 * hx1 * hy2 * ddz1 + fzx110 * hx3 * hy2 * ddz1 +
                fzx001 * hx1 * hy0 * ddz3 + fzx101 * hx3 * hy0 * ddz3 +
                fzx011 * hx1 * hy2 * ddz3 + fzx111 * hx3 * hy2 * ddz3 +

                // fxyz
                fxyz000 * hx1 * hy1 * ddz1 + fxyz100 * hx3 * hy1 * ddz1 +
                fxyz010 * hx1 * hy3 * ddz1 + fxyz110 * hx3 * hy3 * ddz1 +
                fxyz001 * hx1 * hy1 * ddz3 + fxyz101 * hx3 * hy1 * ddz3 +
                fxyz011 * hx1 * hy3 * ddz3 + fxyz111 * hx3 * hy3 * ddz3) * res2;

            // Write outputs (return dxx and dyy separately for accurate chain rule)
            int out_idx = idx * L * F + level * F + f;
            output[out_idx] = value;
            output_dx[out_idx] = dudx;
            output_dy[out_idx] = dudy;
            output_dz[out_idx] = dudz;
            output_dxx[out_idx] = d2udx2;
            output_dyy[out_idx] = d2udy2;
            output_dzz[out_idx] = d2udz2;
        }
    }
}

/*
 * Backward kernel for gradient computation (basic - only enc output)
 * Computes gradients w.r.t. hash_table given grad_output
 */
__global__ void hermite_encoding_backward_kernel(
    const float* __restrict__ x,                // [N,3]
    const float* __restrict__ grad_output,      // [N, L, F]
    float* __restrict__ grad_hash_table_1,      // f
    float* __restrict__ grad_hash_table_2,      // fx,fy,fz  (packed: [idx][3*F])
    float* __restrict__ grad_hash_table_3,      // fxy,fyz,fzx (packed: [idx][3*F])
    float* __restrict__ grad_hash_table_4,      // fxyz
    const float* __restrict__ resolutions,
    int N, int L, int F,
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3, int hashmap_size_4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 3 + 0];
    float py = x[idx * 3 + 1];
    float pz = x[idx * 3 + 2];

    for (int level = 0; level < L; level++) {
        float res = resolutions[level];

        float sx = px * res;
        float sy = py * res;
        float sz = pz * res;

        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);
        int iz = (int)floorf(sz);

        float tx = sx - (float)ix;
        float ty = sy - (float)iy;
        float tz = sz - (float)iz;

        // Hermite basis
        float hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3;
        float hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3;
        float hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3;
        hermite_basis(tx, hx0, hx1, hx2, hx3, dhx0, dhx1, dhx2, dhx3, ddx0, ddx1, ddx2, ddx3);
        hermite_basis(ty, hy0, hy1, hy2, hy3, dhy0, dhy1, dhy2, dhy3, ddy0, ddy1, ddy2, ddy3);
        hermite_basis(tz, hz0, hz1, hz2, hz3, dhz0, dhz1, dhz2, dhz3, ddz0, ddz1, ddz2, ddz3);

        // 8 corners hash indices (for each table size)
        int i000_1 = hash_coords(ix,     iy,     iz,     hashmap_size_1);
        int i100_1 = hash_coords(ix + 1, iy,     iz,     hashmap_size_1);
        int i010_1 = hash_coords(ix,     iy + 1, iz,     hashmap_size_1);
        int i110_1 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_1);
        int i001_1 = hash_coords(ix,     iy,     iz + 1, hashmap_size_1);
        int i101_1 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_1);
        int i011_1 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_1);
        int i111_1 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_1);

        int i000_2 = hash_coords(ix,     iy,     iz,     hashmap_size_2);
        int i100_2 = hash_coords(ix + 1, iy,     iz,     hashmap_size_2);
        int i010_2 = hash_coords(ix,     iy + 1, iz,     hashmap_size_2);
        int i110_2 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_2);
        int i001_2 = hash_coords(ix,     iy,     iz + 1, hashmap_size_2);
        int i101_2 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_2);
        int i011_2 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_2);
        int i111_2 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_2);

        int i000_3 = hash_coords(ix,     iy,     iz,     hashmap_size_3);
        int i100_3 = hash_coords(ix + 1, iy,     iz,     hashmap_size_3);
        int i010_3 = hash_coords(ix,     iy + 1, iz,     hashmap_size_3);
        int i110_3 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_3);
        int i001_3 = hash_coords(ix,     iy,     iz + 1, hashmap_size_3);
        int i101_3 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_3);
        int i011_3 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_3);
        int i111_3 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_3);

        int i000_4 = hash_coords(ix,     iy,     iz,     hashmap_size_4);
        int i100_4 = hash_coords(ix + 1, iy,     iz,     hashmap_size_4);
        int i010_4 = hash_coords(ix,     iy + 1, iz,     hashmap_size_4);
        int i110_4 = hash_coords(ix + 1, iy + 1, iz,     hashmap_size_4);
        int i001_4 = hash_coords(ix,     iy,     iz + 1, hashmap_size_4);
        int i101_4 = hash_coords(ix + 1, iy,     iz + 1, hashmap_size_4);
        int i011_4 = hash_coords(ix,     iy + 1, iz + 1, hashmap_size_4);
        int i111_4 = hash_coords(ix + 1, iy + 1, iz + 1, hashmap_size_4);

        // Offsets match your forward
        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 3;
        int level_offset_3 = level * hashmap_size_3 * F * 3;
        int level_offset_4 = level * hashmap_size_4 * F;

        for (int f = 0; f < F; f++) {
            float g = grad_output[idx * L * F + level * F + f];
            // --- corner 000 (0,0,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i000_1 * F + f], g * hx0 * hy0 * hz0);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i000_2 * F * 3 + 0*F + f], g * hx1 * hy0 * hz0); // fx
            atomicAdd(&grad_hash_table_2[level_offset_2 + i000_2 * F * 3 + 1*F + f], g * hx0 * hy1 * hz0); // fy
            atomicAdd(&grad_hash_table_2[level_offset_2 + i000_2 * F * 3 + 2*F + f], g * hx0 * hy0 * hz1); // fz

            atomicAdd(&grad_hash_table_3[level_offset_3 + i000_3 * F * 3 + 0*F + f], g * hx1 * hy1 * hz0); // fxy
            atomicAdd(&grad_hash_table_3[level_offset_3 + i000_3 * F * 3 + 1*F + f], g * hx0 * hy1 * hz1); // fyz
            atomicAdd(&grad_hash_table_3[level_offset_3 + i000_3 * F * 3 + 2*F + f], g * hx1 * hy0 * hz1); // fzx

            atomicAdd(&grad_hash_table_4[level_offset_4 + i000_4 * F + f], g * hx1 * hy1 * hz1); // fxyz

            // --- corner 100 (1,0,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i100_1 * F + f], g * hx2 * hy0 * hz0);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i100_2 * F * 3 + 0*F + f], g * hx3 * hy0 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i100_2 * F * 3 + 1*F + f], g * hx2 * hy1 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i100_2 * F * 3 + 2*F + f], g * hx2 * hy0 * hz1);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i100_3 * F * 3 + 0*F + f], g * hx3 * hy1 * hz0);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i100_3 * F * 3 + 1*F + f], g * hx2 * hy1 * hz1);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i100_3 * F * 3 + 2*F + f], g * hx3 * hy0 * hz1);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i100_4 * F + f], g * hx3 * hy1 * hz1);

            // --- corner 010 (0,1,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i010_1 * F + f], g * hx0 * hy2 * hz0);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i010_2 * F * 3 + 0*F + f], g * hx1 * hy2 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i010_2 * F * 3 + 1*F + f], g * hx0 * hy3 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i010_2 * F * 3 + 2*F + f], g * hx0 * hy2 * hz1);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i010_3 * F * 3 + 0*F + f], g * hx1 * hy3 * hz0);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i010_3 * F * 3 + 1*F + f], g * hx0 * hy3 * hz1);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i010_3 * F * 3 + 2*F + f], g * hx1 * hy2 * hz1);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i010_4 * F + f], g * hx1 * hy3 * hz1);

            // --- corner 110 (1,1,0)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i110_1 * F + f], g * hx2 * hy2 * hz0);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i110_2 * F * 3 + 0*F + f], g * hx3 * hy2 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i110_2 * F * 3 + 1*F + f], g * hx2 * hy3 * hz0);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i110_2 * F * 3 + 2*F + f], g * hx2 * hy2 * hz1);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i110_3 * F * 3 + 0*F + f], g * hx3 * hy3 * hz0);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i110_3 * F * 3 + 1*F + f], g * hx2 * hy3 * hz1);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i110_3 * F * 3 + 2*F + f], g * hx3 * hy2 * hz1);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i110_4 * F + f], g * hx3 * hy3 * hz1);

            // --- corner 001 (0,0,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i001_1 * F + f], g * hx0 * hy0 * hz2);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i001_2 * F * 3 + 0*F + f], g * hx1 * hy0 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i001_2 * F * 3 + 1*F + f], g * hx0 * hy1 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i001_2 * F * 3 + 2*F + f], g * hx0 * hy0 * hz3);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i001_3 * F * 3 + 0*F + f], g * hx1 * hy1 * hz2);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i001_3 * F * 3 + 1*F + f], g * hx0 * hy1 * hz3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i001_3 * F * 3 + 2*F + f], g * hx1 * hy0 * hz3);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i001_4 * F + f], g * hx1 * hy1 * hz3);

            // --- corner 101 (1,0,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i101_1 * F + f], g * hx2 * hy0 * hz2);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i101_2 * F * 3 + 0*F + f], g * hx3 * hy0 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i101_2 * F * 3 + 1*F + f], g * hx2 * hy1 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i101_2 * F * 3 + 2*F + f], g * hx2 * hy0 * hz3);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i101_3 * F * 3 + 0*F + f], g * hx3 * hy1 * hz2);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i101_3 * F * 3 + 1*F + f], g * hx2 * hy1 * hz3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i101_3 * F * 3 + 2*F + f], g * hx3 * hy0 * hz3);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i101_4 * F + f], g * hx3 * hy1 * hz3);

            // --- corner 011 (0,1,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i011_1 * F + f], g * hx0 * hy2 * hz2);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i011_2 * F * 3 + 0*F + f], g * hx1 * hy2 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i011_2 * F * 3 + 1*F + f], g * hx0 * hy3 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i011_2 * F * 3 + 2*F + f], g * hx0 * hy2 * hz3);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i011_3 * F * 3 + 0*F + f], g * hx1 * hy3 * hz2);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i011_3 * F * 3 + 1*F + f], g * hx0 * hy3 * hz3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i011_3 * F * 3 + 2*F + f], g * hx1 * hy2 * hz3);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i011_4 * F + f], g * hx1 * hy3 * hz3);

            // --- corner 111 (1,1,1)
            atomicAdd(&grad_hash_table_1[level_offset_1 + i111_1 * F + f], g * hx2 * hy2 * hz2);

            atomicAdd(&grad_hash_table_2[level_offset_2 + i111_2 * F * 3 + 0*F + f], g * hx3 * hy2 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i111_2 * F * 3 + 1*F + f], g * hx2 * hy3 * hz2);
            atomicAdd(&grad_hash_table_2[level_offset_2 + i111_2 * F * 3 + 2*F + f], g * hx2 * hy2 * hz3);

            atomicAdd(&grad_hash_table_3[level_offset_3 + i111_3 * F * 3 + 0*F + f], g * hx3 * hy3 * hz2);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i111_3 * F * 3 + 1*F + f], g * hx2 * hy3 * hz3);
            atomicAdd(&grad_hash_table_3[level_offset_3 + i111_3 * F * 3 + 2*F + f], g * hx3 * hy2 * hz3);

            atomicAdd(&grad_hash_table_4[level_offset_4 + i111_4 * F + f], g * hx3 * hy3 * hz3);

        }
    }
}

__global__ void hermite_encoding_backward_full_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_enc,      // [N, L*F]
    const float* __restrict__ grad_dx,       // [N, L*F]
    const float* __restrict__ grad_dy,       // [N, L*F]
    const float* __restrict__ grad_dz,       // [N, L*F]
    const float* __restrict__ grad_dxx,      // [N, L*F]
    const float* __restrict__ grad_dyy,      // [N, L*F]
    const float* __restrict__ grad_dzz,      // [N, L*F]
    float* __restrict__ grad_hash_table_1,   // f
    float* __restrict__ grad_hash_table_2,   // fx,fy,fz
    float* __restrict__ grad_hash_table_3,   // fxy,fyz,fzx
    float* __restrict__ grad_hash_table_4,   // fxyz
    const float* __restrict__ resolutions,
    int N, int L, int F,
    int hashmap_size_1, int hashmap_size_2, int hashmap_size_3, int hashmap_size_4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = x[idx * 3 + 0];
    float py = x[idx * 3 + 1];
    float pz = x[idx * 3 + 2];

    for (int level = 0; level < L; level++) {
        float res  = resolutions[level];
        float res2 = res * res;

        float sx = px * res;
        float sy = py * res;
        float sz = pz * res;

        int ix = (int)floorf(sx);
        int iy = (int)floorf(sy);
        int iz = (int)floorf(sz);

        float tx = sx - (float)ix;
        float ty = sy - (float)iy;
        float tz = sz - (float)iz;

        // Hermite basis: H, H', H''
        float hx0,hx1,hx2,hx3, dhx0,dhx1,dhx2,dhx3, ddx0,ddx1,ddx2,ddx3;
        float hy0,hy1,hy2,hy3, dhy0,dhy1,dhy2,dhy3, ddy0,ddy1,ddy2,ddy3;
        float hz0,hz1,hz2,hz3, dhz0,dhz1,dhz2,dhz3, ddz0,ddz1,ddz2,ddz3;

        hermite_basis(tx, hx0,hx1,hx2,hx3, dhx0,dhx1,dhx2,dhx3, ddx0,ddx1,ddx2,ddx3);
        hermite_basis(ty, hy0,hy1,hy2,hy3, dhy0,dhy1,dhy2,dhy3, ddy0,ddy1,ddy2,ddy3);
        hermite_basis(tz, hz0,hz1,hz2,hz3, dhz0,dhz1,dhz2,dhz3, ddz0,ddz1,ddz2,ddz3);

        // Hash corners
        int idx000_1 = hash_coords(ix,   iy,   iz,   hashmap_size_1);
        int idx100_1 = hash_coords(ix+1, iy,   iz,   hashmap_size_1);
        int idx010_1 = hash_coords(ix,   iy+1, iz,   hashmap_size_1);
        int idx110_1 = hash_coords(ix+1, iy+1, iz,   hashmap_size_1);
        int idx001_1 = hash_coords(ix,   iy,   iz+1, hashmap_size_1);
        int idx101_1 = hash_coords(ix+1, iy,   iz+1, hashmap_size_1);
        int idx011_1 = hash_coords(ix,   iy+1, iz+1, hashmap_size_1);
        int idx111_1 = hash_coords(ix+1, iy+1, iz+1, hashmap_size_1);

        int idx000_2 = hash_coords(ix,   iy,   iz,   hashmap_size_2);
        int idx100_2 = hash_coords(ix+1, iy,   iz,   hashmap_size_2);
        int idx010_2 = hash_coords(ix,   iy+1, iz,   hashmap_size_2);
        int idx110_2 = hash_coords(ix+1, iy+1, iz,   hashmap_size_2);
        int idx001_2 = hash_coords(ix,   iy,   iz+1, hashmap_size_2);
        int idx101_2 = hash_coords(ix+1, iy,   iz+1, hashmap_size_2);
        int idx011_2 = hash_coords(ix,   iy+1, iz+1, hashmap_size_2);
        int idx111_2 = hash_coords(ix+1, iy+1, iz+1, hashmap_size_2);

        int idx000_3 = hash_coords(ix,   iy,   iz,   hashmap_size_3);
        int idx100_3 = hash_coords(ix+1, iy,   iz,   hashmap_size_3);
        int idx010_3 = hash_coords(ix,   iy+1, iz,   hashmap_size_3);
        int idx110_3 = hash_coords(ix+1, iy+1, iz,   hashmap_size_3);
        int idx001_3 = hash_coords(ix,   iy,   iz+1, hashmap_size_3);
        int idx101_3 = hash_coords(ix+1, iy,   iz+1, hashmap_size_3);
        int idx011_3 = hash_coords(ix,   iy+1, iz+1, hashmap_size_3);
        int idx111_3 = hash_coords(ix+1, iy+1, iz+1, hashmap_size_3);

        int idx000_4 = hash_coords(ix,   iy,   iz,   hashmap_size_4);
        int idx100_4 = hash_coords(ix+1, iy,   iz,   hashmap_size_4);
        int idx010_4 = hash_coords(ix,   iy+1, iz,   hashmap_size_4);
        int idx110_4 = hash_coords(ix+1, iy+1, iz,   hashmap_size_4);
        int idx001_4 = hash_coords(ix,   iy,   iz+1, hashmap_size_4);
        int idx101_4 = hash_coords(ix+1, iy,   iz+1, hashmap_size_4);
        int idx011_4 = hash_coords(ix,   iy+1, iz+1, hashmap_size_4);
        int idx111_4 = hash_coords(ix+1, iy+1, iz+1, hashmap_size_4);

        int level_offset_1 = level * hashmap_size_1 * F;
        int level_offset_2 = level * hashmap_size_2 * F * 3;
        int level_offset_3 = level * hashmap_size_3 * F * 3;
        int level_offset_4 = level * hashmap_size_4 * F;

        for (int f = 0; f < F; f++) {
            int out_idx = idx * L * F + level * F + f;

            float g_enc = grad_enc[out_idx];
            float g_dx  = grad_dx[out_idx];
            float g_dy  = grad_dy[out_idx];
            float g_dz  = grad_dz[out_idx];
            float g_dxx = grad_dxx[out_idx];
            float g_dyy = grad_dyy[out_idx];
            float g_dzz = grad_dzz[out_idx];

            // ============================================================
            // ========== Corner (0,0,0) ==========
            float grad_f000 =
                g_enc * hx0 * hy0 * hz0 +
                g_dx  * dhx0 * hy0 * hz0 * res +
                g_dy  * hx0 * dhy0 * hz0 * res +
                g_dz  * hx0 * hy0 * dhz0 * res +
                g_dxx * ddx0 * hy0 * hz0 * res2 +
                g_dyy * hx0 * ddy0 * hz0 * res2 +
                g_dzz * hx0 * hy0 * ddz0 * res2;

            float grad_fx000 =
                g_enc * hx1 * hy0 * hz0 +
                g_dx  * dhx1 * hy0 * hz0 * res +
                g_dy  * hx1 * dhy0 * hz0 * res +
                g_dz  * hx1 * hy0 * dhz0 * res +
                g_dxx * ddx1 * hy0 * hz0 * res2 +
                g_dyy * hx1 * ddy0 * hz0 * res2 +
                g_dzz * hx1 * hy0 * ddz0 * res2;

            float grad_fy000 =
                g_enc * hx0 * hy1 * hz0 +
                g_dx  * dhx0 * hy1 * hz0 * res +
                g_dy  * hx0 * dhy1 * hz0 * res +
                g_dz  * hx0 * hy1 * dhz0 * res +
                g_dxx * ddx0 * hy1 * hz0 * res2 +
                g_dyy * hx0 * ddy1 * hz0 * res2 +
                g_dzz * hx0 * hy1 * ddz0 * res2;

            float grad_fz000 =
                g_enc * hx0 * hy0 * hz1 +
                g_dx  * dhx0 * hy0 * hz1 * res +
                g_dy  * hx0 * dhy0 * hz1 * res +
                g_dz  * hx0 * hy0 * dhz1 * res +
                g_dxx * ddx0 * hy0 * hz1 * res2 +
                g_dyy * hx0 * ddy0 * hz1 * res2 +
                g_dzz * hx0 * hy0 * ddz1 * res2;

            float grad_fxy000 =
                g_enc * hx1 * hy1 * hz0 +
                g_dx  * dhx1 * hy1 * hz0 * res +
                g_dy  * hx1 * dhy1 * hz0 * res +
                g_dz  * hx1 * hy1 * dhz0 * res +
                g_dxx * ddx1 * hy1 * hz0 * res2 +
                g_dyy * hx1 * ddy1 * hz0 * res2 +
                g_dzz * hx1 * hy1 * ddz0 * res2;

            float grad_fyz000 =
                g_enc * hx0 * hy1 * hz1 +
                g_dx  * dhx0 * hy1 * hz1 * res +
                g_dy  * hx0 * dhy1 * hz1 * res +
                g_dz  * hx0 * hy1 * dhz1 * res +
                g_dxx * ddx0 * hy1 * hz1 * res2 +
                g_dyy * hx0 * ddy1 * hz1 * res2 +
                g_dzz * hx0 * hy1 * ddz1 * res2;

            float grad_fzx000 =
                g_enc * hx1 * hy0 * hz1 +
                g_dx  * dhx1 * hy0 * hz1 * res +
                g_dy  * hx1 * dhy0 * hz1 * res +
                g_dz  * hx1 * hy0 * dhz1 * res +
                g_dxx * ddx1 * hy0 * hz1 * res2 +
                g_dyy * hx1 * ddy0 * hz1 * res2 +
                g_dzz * hx1 * hy0 * ddz1 * res2;

            float grad_fxyz000 =
                g_enc * hx1 * hy1 * hz1 +
                g_dx  * dhx1 * hy1 * hz1 * res +
                g_dy  * hx1 * dhy1 * hz1 * res +
                g_dz  * hx1 * hy1 * dhz1 * res +
                g_dxx * ddx1 * hy1 * hz1 * res2 +
                g_dyy * hx1 * ddy1 * hz1 * res2 +
                g_dzz * hx1 * hy1 * ddz1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx000_1 * F + f], grad_f000);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx000_2 * F * 3 + 0*F + f], grad_fx000);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx000_2 * F * 3 + 1*F + f], grad_fy000);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx000_2 * F * 3 + 2*F + f], grad_fz000);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx000_3 * F * 3 + 0*F + f], grad_fxy000);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx000_3 * F * 3 + 1*F + f], grad_fyz000);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx000_3 * F * 3 + 2*F + f], grad_fzx000);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx000_4 * F + f], grad_fxyz000);

            // ============================================================
            // ========== Corner (1,0,0) ==========
            float grad_f100 =
                g_enc * hx2 * hy0 * hz0 +
                g_dx  * dhx2 * hy0 * hz0 * res +
                g_dy  * hx2 * dhy0 * hz0 * res +
                g_dz  * hx2 * hy0 * dhz0 * res +
                g_dxx * ddx2 * hy0 * hz0 * res2 +
                g_dyy * hx2 * ddy0 * hz0 * res2 +
                g_dzz * hx2 * hy0 * ddz0 * res2;

            float grad_fx100 =
                g_enc * hx3 * hy0 * hz0 +
                g_dx  * dhx3 * hy0 * hz0 * res +
                g_dy  * hx3 * dhy0 * hz0 * res +
                g_dz  * hx3 * hy0 * dhz0 * res +
                g_dxx * ddx3 * hy0 * hz0 * res2 +
                g_dyy * hx3 * ddy0 * hz0 * res2 +
                g_dzz * hx3 * hy0 * ddz0 * res2;

            float grad_fy100 =
                g_enc * hx2 * hy1 * hz0 +
                g_dx  * dhx2 * hy1 * hz0 * res +
                g_dy  * hx2 * dhy1 * hz0 * res +
                g_dz  * hx2 * hy1 * dhz0 * res +
                g_dxx * ddx2 * hy1 * hz0 * res2 +
                g_dyy * hx2 * ddy1 * hz0 * res2 +
                g_dzz * hx2 * hy1 * ddz0 * res2;

            float grad_fz100 =
                g_enc * hx2 * hy0 * hz1 +
                g_dx  * dhx2 * hy0 * hz1 * res +
                g_dy  * hx2 * dhy0 * hz1 * res +
                g_dz  * hx2 * hy0 * dhz1 * res +
                g_dxx * ddx2 * hy0 * hz1 * res2 +
                g_dyy * hx2 * ddy0 * hz1 * res2 +
                g_dzz * hx2 * hy0 * ddz1 * res2;

            float grad_fxy100 =
                g_enc * hx3 * hy1 * hz0 +
                g_dx  * dhx3 * hy1 * hz0 * res +
                g_dy  * hx3 * dhy1 * hz0 * res +
                g_dz  * hx3 * hy1 * dhz0 * res +
                g_dxx * ddx3 * hy1 * hz0 * res2 +
                g_dyy * hx3 * ddy1 * hz0 * res2 +
                g_dzz * hx3 * hy1 * ddz0 * res2;

            float grad_fyz100 =
                g_enc * hx2 * hy1 * hz1 +
                g_dx  * dhx2 * hy1 * hz1 * res +
                g_dy  * hx2 * dhy1 * hz1 * res +
                g_dz  * hx2 * hy1 * dhz1 * res +
                g_dxx * ddx2 * hy1 * hz1 * res2 +
                g_dyy * hx2 * ddy1 * hz1 * res2 +
                g_dzz * hx2 * hy1 * ddz1 * res2;

            float grad_fzx100 =
                g_enc * hx3 * hy0 * hz1 +
                g_dx  * dhx3 * hy0 * hz1 * res +
                g_dy  * hx3 * dhy0 * hz1 * res +
                g_dz  * hx3 * hy0 * dhz1 * res +
                g_dxx * ddx3 * hy0 * hz1 * res2 +
                g_dyy * hx3 * ddy0 * hz1 * res2 +
                g_dzz * hx3 * hy0 * ddz1 * res2;

            float grad_fxyz100 =
                g_enc * hx3 * hy1 * hz1 +
                g_dx  * dhx3 * hy1 * hz1 * res +
                g_dy  * hx3 * dhy1 * hz1 * res +
                g_dz  * hx3 * hy1 * dhz1 * res +
                g_dxx * ddx3 * hy1 * hz1 * res2 +
                g_dyy * hx3 * ddy1 * hz1 * res2 +
                g_dzz * hx3 * hy1 * ddz1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx100_1 * F + f], grad_f100);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx100_2 * F * 3 + 0*F + f], grad_fx100);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx100_2 * F * 3 + 1*F + f], grad_fy100);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx100_2 * F * 3 + 2*F + f], grad_fz100);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx100_3 * F * 3 + 0*F + f], grad_fxy100);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx100_3 * F * 3 + 1*F + f], grad_fyz100);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx100_3 * F * 3 + 2*F + f], grad_fzx100);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx100_4 * F + f], grad_fxyz100);

            // ============================================================
            // ========== Corner (0,1,0) ==========
            float grad_f010 =
                g_enc * hx0 * hy2 * hz0 +
                g_dx  * dhx0 * hy2 * hz0 * res +
                g_dy  * hx0 * dhy2 * hz0 * res +
                g_dz  * hx0 * hy2 * dhz0 * res +
                g_dxx * ddx0 * hy2 * hz0 * res2 +
                g_dyy * hx0 * ddy2 * hz0 * res2 +
                g_dzz * hx0 * hy2 * ddz0 * res2;

            float grad_fx010 =
                g_enc * hx1 * hy2 * hz0 +
                g_dx  * dhx1 * hy2 * hz0 * res +
                g_dy  * hx1 * dhy2 * hz0 * res +
                g_dz  * hx1 * hy2 * dhz0 * res +
                g_dxx * ddx1 * hy2 * hz0 * res2 +
                g_dyy * hx1 * ddy2 * hz0 * res2 +
                g_dzz * hx1 * hy2 * ddz0 * res2;

            float grad_fy010 =
                g_enc * hx0 * hy3 * hz0 +
                g_dx  * dhx0 * hy3 * hz0 * res +
                g_dy  * hx0 * dhy3 * hz0 * res +
                g_dz  * hx0 * hy3 * dhz0 * res +
                g_dxx * ddx0 * hy3 * hz0 * res2 +
                g_dyy * hx0 * ddy3 * hz0 * res2 +
                g_dzz * hx0 * hy3 * ddz0 * res2;

            float grad_fz010 =
                g_enc * hx0 * hy2 * hz1 +
                g_dx  * dhx0 * hy2 * hz1 * res +
                g_dy  * hx0 * dhy2 * hz1 * res +
                g_dz  * hx0 * hy2 * dhz1 * res +
                g_dxx * ddx0 * hy2 * hz1 * res2 +
                g_dyy * hx0 * ddy2 * hz1 * res2 +
                g_dzz * hx0 * hy2 * ddz1 * res2;

            float grad_fxy010 =
                g_enc * hx1 * hy3 * hz0 +
                g_dx  * dhx1 * hy3 * hz0 * res +
                g_dy  * hx1 * dhy3 * hz0 * res +
                g_dz  * hx1 * hy3 * dhz0 * res +
                g_dxx * ddx1 * hy3 * hz0 * res2 +
                g_dyy * hx1 * ddy3 * hz0 * res2 +
                g_dzz * hx1 * hy3 * ddz0 * res2;

            float grad_fyz010 =
                g_enc * hx0 * hy3 * hz1 +
                g_dx  * dhx0 * hy3 * hz1 * res +
                g_dy  * hx0 * dhy3 * hz1 * res +
                g_dz  * hx0 * hy3 * dhz1 * res +
                g_dxx * ddx0 * hy3 * hz1 * res2 +
                g_dyy * hx0 * ddy3 * hz1 * res2 +
                g_dzz * hx0 * hy3 * ddz1 * res2;

            float grad_fzx010 =
                g_enc * hx1 * hy2 * hz1 +
                g_dx  * dhx1 * hy2 * hz1 * res +
                g_dy  * hx1 * dhy2 * hz1 * res +
                g_dz  * hx1 * hy2 * dhz1 * res +
                g_dxx * ddx1 * hy2 * hz1 * res2 +
                g_dyy * hx1 * ddy2 * hz1 * res2 +
                g_dzz * hx1 * hy2 * ddz1 * res2;

            float grad_fxyz010 =
                g_enc * hx1 * hy3 * hz1 +
                g_dx  * dhx1 * hy3 * hz1 * res +
                g_dy  * hx1 * dhy3 * hz1 * res +
                g_dz  * hx1 * hy3 * dhz1 * res +
                g_dxx * ddx1 * hy3 * hz1 * res2 +
                g_dyy * hx1 * ddy3 * hz1 * res2 +
                g_dzz * hx1 * hy3 * ddz1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx010_1 * F + f], grad_f010);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx010_2 * F * 3 + 0*F + f], grad_fx010);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx010_2 * F * 3 + 1*F + f], grad_fy010);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx010_2 * F * 3 + 2*F + f], grad_fz010);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx010_3 * F * 3 + 0*F + f], grad_fxy010);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx010_3 * F * 3 + 1*F + f], grad_fyz010);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx010_3 * F * 3 + 2*F + f], grad_fzx010);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx010_4 * F + f], grad_fxyz010);

            // ============================================================
            // ========== Corner (1,1,0) ==========
            float grad_f110 =
                g_enc * hx2 * hy2 * hz0 +
                g_dx  * dhx2 * hy2 * hz0 * res +
                g_dy  * hx2 * dhy2 * hz0 * res +
                g_dz  * hx2 * hy2 * dhz0 * res +
                g_dxx * ddx2 * hy2 * hz0 * res2 +
                g_dyy * hx2 * ddy2 * hz0 * res2 +
                g_dzz * hx2 * hy2 * ddz0 * res2;

            float grad_fx110 =
                g_enc * hx3 * hy2 * hz0 +
                g_dx  * dhx3 * hy2 * hz0 * res +
                g_dy  * hx3 * dhy2 * hz0 * res +
                g_dz  * hx3 * hy2 * dhz0 * res +
                g_dxx * ddx3 * hy2 * hz0 * res2 +
                g_dyy * hx3 * ddy2 * hz0 * res2 +
                g_dzz * hx3 * hy2 * ddz0 * res2;

            float grad_fy110 =
                g_enc * hx2 * hy3 * hz0 +
                g_dx  * dhx2 * hy3 * hz0 * res +
                g_dy  * hx2 * dhy3 * hz0 * res +
                g_dz  * hx2 * hy3 * dhz0 * res +
                g_dxx * ddx2 * hy3 * hz0 * res2 +
                g_dyy * hx2 * ddy3 * hz0 * res2 +
                g_dzz * hx2 * hy3 * ddz0 * res2;

            float grad_fz110 =
                g_enc * hx2 * hy2 * hz1 +
                g_dx  * dhx2 * hy2 * hz1 * res +
                g_dy  * hx2 * dhy2 * hz1 * res +
                g_dz  * hx2 * hy2 * dhz1 * res +
                g_dxx * ddx2 * hy2 * hz1 * res2 +
                g_dyy * hx2 * ddy2 * hz1 * res2 +
                g_dzz * hx2 * hy2 * ddz1 * res2;

            float grad_fxy110 =
                g_enc * hx3 * hy3 * hz0 +
                g_dx  * dhx3 * hy3 * hz0 * res +
                g_dy  * hx3 * dhy3 * hz0 * res +
                g_dz  * hx3 * hy3 * dhz0 * res +
                g_dxx * ddx3 * hy3 * hz0 * res2 +
                g_dyy * hx3 * ddy3 * hz0 * res2 +
                g_dzz * hx3 * hy3 * ddz0 * res2;

            float grad_fyz110 =
                g_enc * hx2 * hy3 * hz1 +
                g_dx  * dhx2 * hy3 * hz1 * res +
                g_dy  * hx2 * dhy3 * hz1 * res +
                g_dz  * hx2 * hy3 * dhz1 * res +
                g_dxx * ddx2 * hy3 * hz1 * res2 +
                g_dyy * hx2 * ddy3 * hz1 * res2 +
                g_dzz * hx2 * hy3 * ddz1 * res2;

            float grad_fzx110 =
                g_enc * hx3 * hy2 * hz1 +
                g_dx  * dhx3 * hy2 * hz1 * res +
                g_dy  * hx3 * dhy2 * hz1 * res +
                g_dz  * hx3 * hy2 * dhz1 * res +
                g_dxx * ddx3 * hy2 * hz1 * res2 +
                g_dyy * hx3 * ddy2 * hz1 * res2 +
                g_dzz * hx3 * hy2 * ddz1 * res2;

            float grad_fxyz110 =
                g_enc * hx3 * hy3 * hz1 +
                g_dx  * dhx3 * hy3 * hz1 * res +
                g_dy  * hx3 * dhy3 * hz1 * res +
                g_dz  * hx3 * hy3 * dhz1 * res +
                g_dxx * ddx3 * hy3 * hz1 * res2 +
                g_dyy * hx3 * ddy3 * hz1 * res2 +
                g_dzz * hx3 * hy3 * ddz1 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx110_1 * F + f], grad_f110);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx110_2 * F * 3 + 0*F + f], grad_fx110);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx110_2 * F * 3 + 1*F + f], grad_fy110);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx110_2 * F * 3 + 2*F + f], grad_fz110);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx110_3 * F * 3 + 0*F + f], grad_fxy110);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx110_3 * F * 3 + 1*F + f], grad_fyz110);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx110_3 * F * 3 + 2*F + f], grad_fzx110);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx110_4 * F + f], grad_fxyz110);

            // ============================================================
            // ========== Corner (0,0,1) ==========
            float grad_f001 =
                g_enc * hx0 * hy0 * hz2 +
                g_dx  * dhx0 * hy0 * hz2 * res +
                g_dy  * hx0 * dhy0 * hz2 * res +
                g_dz  * hx0 * hy0 * dhz2 * res +
                g_dxx * ddx0 * hy0 * hz2 * res2 +
                g_dyy * hx0 * ddy0 * hz2 * res2 +
                g_dzz * hx0 * hy0 * ddz2 * res2;

            float grad_fx001 =
                g_enc * hx1 * hy0 * hz2 +
                g_dx  * dhx1 * hy0 * hz2 * res +
                g_dy  * hx1 * dhy0 * hz2 * res +
                g_dz  * hx1 * hy0 * dhz2 * res +
                g_dxx * ddx1 * hy0 * hz2 * res2 +
                g_dyy * hx1 * ddy0 * hz2 * res2 +
                g_dzz * hx1 * hy0 * ddz2 * res2;

            float grad_fy001 =
                g_enc * hx0 * hy1 * hz2 +
                g_dx  * dhx0 * hy1 * hz2 * res +
                g_dy  * hx0 * dhy1 * hz2 * res +
                g_dz  * hx0 * hy1 * dhz2 * res +
                g_dxx * ddx0 * hy1 * hz2 * res2 +
                g_dyy * hx0 * ddy1 * hz2 * res2 +
                g_dzz * hx0 * hy1 * ddz2 * res2;

            float grad_fz001 =
                g_enc * hx0 * hy0 * hz3 +
                g_dx  * dhx0 * hy0 * hz3 * res +
                g_dy  * hx0 * dhy0 * hz3 * res +
                g_dz  * hx0 * hy0 * dhz3 * res +
                g_dxx * ddx0 * hy0 * hz3 * res2 +
                g_dyy * hx0 * ddy0 * hz3 * res2 +
                g_dzz * hx0 * hy0 * ddz3 * res2;

            float grad_fxy001 =
                g_enc * hx1 * hy1 * hz2 +
                g_dx  * dhx1 * hy1 * hz2 * res +
                g_dy  * hx1 * dhy1 * hz2 * res +
                g_dz  * hx1 * hy1 * dhz2 * res +
                g_dxx * ddx1 * hy1 * hz2 * res2 +
                g_dyy * hx1 * ddy1 * hz2 * res2 +
                g_dzz * hx1 * hy1 * ddz2 * res2;

            float grad_fyz001 =
                g_enc * hx0 * hy1 * hz3 +
                g_dx  * dhx0 * hy1 * hz3 * res +
                g_dy  * hx0 * dhy1 * hz3 * res +
                g_dz  * hx0 * hy1 * dhz3 * res +
                g_dxx * ddx0 * hy1 * hz3 * res2 +
                g_dyy * hx0 * ddy1 * hz3 * res2 +
                g_dzz * hx0 * hy1 * ddz3 * res2;

            float grad_fzx001 =
                g_enc * hx1 * hy0 * hz3 +
                g_dx  * dhx1 * hy0 * hz3 * res +
                g_dy  * hx1 * dhy0 * hz3 * res +
                g_dz  * hx1 * hy0 * dhz3 * res +
                g_dxx * ddx1 * hy0 * hz3 * res2 +
                g_dyy * hx1 * ddy0 * hz3 * res2 +
                g_dzz * hx1 * hy0 * ddz3 * res2;

            float grad_fxyz001 =
                g_enc * hx1 * hy1 * hz3 +
                g_dx  * dhx1 * hy1 * hz3 * res +
                g_dy  * hx1 * dhy1 * hz3 * res +
                g_dz  * hx1 * hy1 * dhz3 * res +
                g_dxx * ddx1 * hy1 * hz3 * res2 +
                g_dyy * hx1 * ddy1 * hz3 * res2 +
                g_dzz * hx1 * hy1 * ddz3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx001_1 * F + f], grad_f001);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx001_2 * F * 3 + 0*F + f], grad_fx001);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx001_2 * F * 3 + 1*F + f], grad_fy001);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx001_2 * F * 3 + 2*F + f], grad_fz001);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx001_3 * F * 3 + 0*F + f], grad_fxy001);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx001_3 * F * 3 + 1*F + f], grad_fyz001);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx001_3 * F * 3 + 2*F + f], grad_fzx001);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx001_4 * F + f], grad_fxyz001);

            // ============================================================
            // ========== Corner (1,0,1) ==========
            float grad_f101 =
                g_enc * hx2 * hy0 * hz2 +
                g_dx  * dhx2 * hy0 * hz2 * res +
                g_dy  * hx2 * dhy0 * hz2 * res +
                g_dz  * hx2 * hy0 * dhz2 * res +
                g_dxx * ddx2 * hy0 * hz2 * res2 +
                g_dyy * hx2 * ddy0 * hz2 * res2 +
                g_dzz * hx2 * hy0 * ddz2 * res2;

            float grad_fx101 =
                g_enc * hx3 * hy0 * hz2 +
                g_dx  * dhx3 * hy0 * hz2 * res +
                g_dy  * hx3 * dhy0 * hz2 * res +
                g_dz  * hx3 * hy0 * dhz2 * res +
                g_dxx * ddx3 * hy0 * hz2 * res2 +
                g_dyy * hx3 * ddy0 * hz2 * res2 +
                g_dzz * hx3 * hy0 * ddz2 * res2;

            float grad_fy101 =
                g_enc * hx2 * hy1 * hz2 +
                g_dx  * dhx2 * hy1 * hz2 * res +
                g_dy  * hx2 * dhy1 * hz2 * res +
                g_dz  * hx2 * hy1 * dhz2 * res +
                g_dxx * ddx2 * hy1 * hz2 * res2 +
                g_dyy * hx2 * ddy1 * hz2 * res2 +
                g_dzz * hx2 * hy1 * ddz2 * res2;

            float grad_fz101 =
                g_enc * hx2 * hy0 * hz3 +
                g_dx  * dhx2 * hy0 * hz3 * res +
                g_dy  * hx2 * dhy0 * hz3 * res +
                g_dz  * hx2 * hy0 * dhz3 * res +
                g_dxx * ddx2 * hy0 * hz3 * res2 +
                g_dyy * hx2 * ddy0 * hz3 * res2 +
                g_dzz * hx2 * hy0 * ddz3 * res2;

            float grad_fxy101 =
                g_enc * hx3 * hy1 * hz2 +
                g_dx  * dhx3 * hy1 * hz2 * res +
                g_dy  * hx3 * dhy1 * hz2 * res +
                g_dz  * hx3 * hy1 * dhz2 * res +
                g_dxx * ddx3 * hy1 * hz2 * res2 +
                g_dyy * hx3 * ddy1 * hz2 * res2 +
                g_dzz * hx3 * hy1 * ddz2 * res2;

            float grad_fyz101 =
                g_enc * hx2 * hy1 * hz3 +
                g_dx  * dhx2 * hy1 * hz3 * res +
                g_dy  * hx2 * dhy1 * hz3 * res +
                g_dz  * hx2 * hy1 * dhz3 * res +
                g_dxx * ddx2 * hy1 * hz3 * res2 +
                g_dyy * hx2 * ddy1 * hz3 * res2 +
                g_dzz * hx2 * hy1 * ddz3 * res2;

            float grad_fzx101 =
                g_enc * hx3 * hy0 * hz3 +
                g_dx  * dhx3 * hy0 * hz3 * res +
                g_dy  * hx3 * dhy0 * hz3 * res +
                g_dz  * hx3 * hy0 * dhz3 * res +
                g_dxx * ddx3 * hy0 * hz3 * res2 +
                g_dyy * hx3 * ddy0 * hz3 * res2 +
                g_dzz * hx3 * hy0 * ddz3 * res2;

            float grad_fxyz101 =
                g_enc * hx3 * hy1 * hz3 +
                g_dx  * dhx3 * hy1 * hz3 * res +
                g_dy  * hx3 * dhy1 * hz3 * res +
                g_dz  * hx3 * hy1 * dhz3 * res +
                g_dxx * ddx3 * hy1 * hz3 * res2 +
                g_dyy * hx3 * ddy1 * hz3 * res2 +
                g_dzz * hx3 * hy1 * ddz3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx101_1 * F + f], grad_f101);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx101_2 * F * 3 + 0*F + f], grad_fx101);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx101_2 * F * 3 + 1*F + f], grad_fy101);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx101_2 * F * 3 + 2*F + f], grad_fz101);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx101_3 * F * 3 + 0*F + f], grad_fxy101);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx101_3 * F * 3 + 1*F + f], grad_fyz101);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx101_3 * F * 3 + 2*F + f], grad_fzx101);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx101_4 * F + f], grad_fxyz101);

            // ============================================================
            // ========== Corner (0,1,1) ==========
            float grad_f011 =
                g_enc * hx0 * hy2 * hz2 +
                g_dx  * dhx0 * hy2 * hz2 * res +
                g_dy  * hx0 * dhy2 * hz2 * res +
                g_dz  * hx0 * hy2 * dhz2 * res +
                g_dxx * ddx0 * hy2 * hz2 * res2 +
                g_dyy * hx0 * ddy2 * hz2 * res2 +
                g_dzz * hx0 * hy2 * ddz2 * res2;

            float grad_fx011 =
                g_enc * hx1 * hy2 * hz2 +
                g_dx  * dhx1 * hy2 * hz2 * res +
                g_dy  * hx1 * dhy2 * hz2 * res +
                g_dz  * hx1 * hy2 * dhz2 * res +
                g_dxx * ddx1 * hy2 * hz2 * res2 +
                g_dyy * hx1 * ddy2 * hz2 * res2 +
                g_dzz * hx1 * hy2 * ddz2 * res2;

            float grad_fy011 =
                g_enc * hx0 * hy3 * hz2 +
                g_dx  * dhx0 * hy3 * hz2 * res +
                g_dy  * hx0 * dhy3 * hz2 * res +
                g_dz  * hx0 * hy3 * dhz2 * res +
                g_dxx * ddx0 * hy3 * hz2 * res2 +
                g_dyy * hx0 * ddy3 * hz2 * res2 +
                g_dzz * hx0 * hy3 * ddz2 * res2;

            float grad_fz011 =
                g_enc * hx0 * hy2 * hz3 +
                g_dx  * dhx0 * hy2 * hz3 * res +
                g_dy  * hx0 * dhy2 * hz3 * res +
                g_dz  * hx0 * hy2 * dhz3 * res +
                g_dxx * ddx0 * hy2 * hz3 * res2 +
                g_dyy * hx0 * ddy2 * hz3 * res2 +
                g_dzz * hx0 * hy2 * ddz3 * res2;

            float grad_fxy011 =
                g_enc * hx1 * hy3 * hz2 +
                g_dx  * dhx1 * hy3 * hz2 * res +
                g_dy  * hx1 * dhy3 * hz2 * res +
                g_dz  * hx1 * hy3 * dhz2 * res +
                g_dxx * ddx1 * hy3 * hz2 * res2 +
                g_dyy * hx1 * ddy3 * hz2 * res2 +
                g_dzz * hx1 * hy3 * ddz2 * res2;

            float grad_fyz011 =
                g_enc * hx0 * hy3 * hz3 +
                g_dx  * dhx0 * hy3 * hz3 * res +
                g_dy  * hx0 * dhy3 * hz3 * res +
                g_dz  * hx0 * hy3 * dhz3 * res +
                g_dxx * ddx0 * hy3 * hz3 * res2 +
                g_dyy * hx0 * ddy3 * hz3 * res2 +
                g_dzz * hx0 * hy3 * ddz3 * res2;

            float grad_fzx011 =
                g_enc * hx1 * hy2 * hz3 +
                g_dx  * dhx1 * hy2 * hz3 * res +
                g_dy  * hx1 * dhy2 * hz3 * res +
                g_dz  * hx1 * hy2 * dhz3 * res +
                g_dxx * ddx1 * hy2 * hz3 * res2 +
                g_dyy * hx1 * ddy2 * hz3 * res2 +
                g_dzz * hx1 * hy2 * ddz3 * res2;

            float grad_fxyz011 =
                g_enc * hx1 * hy3 * hz3 +
                g_dx  * dhx1 * hy3 * hz3 * res +
                g_dy  * hx1 * dhy3 * hz3 * res +
                g_dz  * hx1 * hy3 * dhz3 * res +
                g_dxx * ddx1 * hy3 * hz3 * res2 +
                g_dyy * hx1 * ddy3 * hz3 * res2 +
                g_dzz * hx1 * hy3 * ddz3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx011_1 * F + f], grad_f011);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx011_2 * F * 3 + 0*F + f], grad_fx011);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx011_2 * F * 3 + 1*F + f], grad_fy011);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx011_2 * F * 3 + 2*F + f], grad_fz011);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx011_3 * F * 3 + 0*F + f], grad_fxy011);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx011_3 * F * 3 + 1*F + f], grad_fyz011);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx011_3 * F * 3 + 2*F + f], grad_fzx011);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx011_4 * F + f], grad_fxyz011);

            // ============================================================
            // ========== Corner (1,1,1) ==========
            float grad_f111 =
                g_enc * hx2 * hy2 * hz2 +
                g_dx  * dhx2 * hy2 * hz2 * res +
                g_dy  * hx2 * dhy2 * hz2 * res +
                g_dz  * hx2 * hy2 * dhz2 * res +
                g_dxx * ddx2 * hy2 * hz2 * res2 +
                g_dyy * hx2 * ddy2 * hz2 * res2 +
                g_dzz * hx2 * hy2 * ddz2 * res2;

            float grad_fx111 =
                g_enc * hx3 * hy2 * hz2 +
                g_dx  * dhx3 * hy2 * hz2 * res +
                g_dy  * hx3 * dhy2 * hz2 * res +
                g_dz  * hx3 * hy2 * dhz2 * res +
                g_dxx * ddx3 * hy2 * hz2 * res2 +
                g_dyy * hx3 * ddy2 * hz2 * res2 +
                g_dzz * hx3 * hy2 * ddz2 * res2;

            float grad_fy111 =
                g_enc * hx2 * hy3 * hz2 +
                g_dx  * dhx2 * hy3 * hz2 * res +
                g_dy  * hx2 * dhy3 * hz2 * res +
                g_dz  * hx2 * hy3 * dhz2 * res +
                g_dxx * ddx2 * hy3 * hz2 * res2 +
                g_dyy * hx2 * ddy3 * hz2 * res2 +
                g_dzz * hx2 * hy3 * ddz2 * res2;

            float grad_fz111 =
                g_enc * hx2 * hy2 * hz3 +
                g_dx  * dhx2 * hy2 * hz3 * res +
                g_dy  * hx2 * dhy2 * hz3 * res +
                g_dz  * hx2 * hy2 * dhz3 * res +
                g_dxx * ddx2 * hy2 * hz3 * res2 +
                g_dyy * hx2 * ddy2 * hz3 * res2 +
                g_dzz * hx2 * hy2 * ddz3 * res2;

            float grad_fxy111 =
                g_enc * hx3 * hy3 * hz2 +
                g_dx  * dhx3 * hy3 * hz2 * res +
                g_dy  * hx3 * dhy3 * hz2 * res +
                g_dz  * hx3 * hy3 * dhz2 * res +
                g_dxx * ddx3 * hy3 * hz2 * res2 +
                g_dyy * hx3 * ddy3 * hz2 * res2 +
                g_dzz * hx3 * hy3 * ddz2 * res2;

            float grad_fyz111 =
                g_enc * hx2 * hy3 * hz3 +
                g_dx  * dhx2 * hy3 * hz3 * res +
                g_dy  * hx2 * dhy3 * hz3 * res +
                g_dz  * hx2 * hy3 * dhz3 * res +
                g_dxx * ddx2 * hy3 * hz3 * res2 +
                g_dyy * hx2 * ddy3 * hz3 * res2 +
                g_dzz * hx2 * hy3 * ddz3 * res2;

            float grad_fzx111 =
                g_enc * hx3 * hy2 * hz3 +
                g_dx  * dhx3 * hy2 * hz3 * res +
                g_dy  * hx3 * dhy2 * hz3 * res +
                g_dz  * hx3 * hy2 * dhz3 * res +
                g_dxx * ddx3 * hy2 * hz3 * res2 +
                g_dyy * hx3 * ddy2 * hz3 * res2 +
                g_dzz * hx3 * hy2 * ddz3 * res2;

            float grad_fxyz111 =
                g_enc * hx3 * hy3 * hz3 +
                g_dx  * dhx3 * hy3 * hz3 * res +
                g_dy  * hx3 * dhy3 * hz3 * res +
                g_dz  * hx3 * hy3 * dhz3 * res +
                g_dxx * ddx3 * hy3 * hz3 * res2 +
                g_dyy * hx3 * ddy3 * hz3 * res2 +
                g_dzz * hx3 * hy3 * ddz3 * res2;

            atomicAdd(&grad_hash_table_1[level_offset_1 + idx111_1 * F + f], grad_f111);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx111_2 * F * 3 + 0*F + f], grad_fx111);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx111_2 * F * 3 + 1*F + f], grad_fy111);
            atomicAdd(&grad_hash_table_2[level_offset_2 + idx111_2 * F * 3 + 2*F + f], grad_fz111);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx111_3 * F * 3 + 0*F + f], grad_fxy111);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx111_3 * F * 3 + 1*F + f], grad_fyz111);
            atomicAdd(&grad_hash_table_3[level_offset_3 + idx111_3 * F * 3 + 2*F + f], grad_fzx111);
            atomicAdd(&grad_hash_table_4[level_offset_4 + idx111_4 * F + f], grad_fxyz111);
        }
    }
}

// C++ interface functions

torch::Tensor hermite_encoding_forward_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int hashmap_size_4 = hash_table_4.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto output = torch::zeros({N, L * F}, x.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        hash_table_1.data_ptr<float>(),
        hash_table_2.data_ptr<float>(),
        hash_table_3.data_ptr<float>(),
        hash_table_4.data_ptr<float>(),
        output.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2,
        hashmap_size_3, hashmap_size_4
    );

    return output;
}

std::vector<torch::Tensor> hermite_encoding_with_laplacian_cuda(
    torch::Tensor x,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int hashmap_size_4 = hash_table_4.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto output = torch::zeros({N, L * F}, x.options());
    auto output_dx = torch::zeros({N, L * F}, x.options());
    auto output_dy = torch::zeros({N, L * F}, x.options());
    auto output_dz = torch::zeros({N, L * F}, x.options());
    auto output_dxx = torch::zeros({N, L * F}, x.options());
    auto output_dyy = torch::zeros({N, L * F}, x.options());
    auto output_dzz = torch::zeros({N, L * F}, x.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_with_laplacian_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        hash_table_1.data_ptr<float>(),
        hash_table_2.data_ptr<float>(),
        hash_table_3.data_ptr<float>(),
        hash_table_4.data_ptr<float>(),
        output.data_ptr<float>(),
        output_dx.data_ptr<float>(),
        output_dy.data_ptr<float>(),
        output_dz.data_ptr<float>(),
        output_dxx.data_ptr<float>(),
        output_dyy.data_ptr<float>(),
        output_dzz.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1,hashmap_size_2,
        hashmap_size_3, hashmap_size_4
    );

    // Return: output, dx, dy, dxx, dyy
    return {output, output_dx, output_dy, output_dz, output_dxx, output_dyy, output_dzz};
}

std::vector<torch::Tensor> hermite_encoding_backward_cuda(
    torch::Tensor x,
    torch::Tensor grad_output,
    torch::Tensor hash_table_1,
    torch::Tensor hash_table_2,
    torch::Tensor hash_table_3,
    torch::Tensor hash_table_4,
    torch::Tensor resolutions
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int hashmap_size_4 = hash_table_4.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto grad_hash_table_1 = torch::zeros_like(hash_table_1);
    auto grad_hash_table_2 = torch::zeros_like(hash_table_2);
    auto grad_hash_table_3 = torch::zeros_like(hash_table_3);
    auto grad_hash_table_4 = torch::zeros_like(hash_table_4);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_backward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_hash_table_1.data_ptr<float>(),
        grad_hash_table_2.data_ptr<float>(),
        grad_hash_table_3.data_ptr<float>(),
        grad_hash_table_4.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2, 
        hashmap_size_3,hashmap_size_4
    );

    return {grad_hash_table_1, grad_hash_table_2, grad_hash_table_3, grad_hash_table_4};
}

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
) {
    const int N = x.size(0);
    const int L = hash_table_1.size(0);
    const int hashmap_size_1 = hash_table_1.size(1);
    const int hashmap_size_2 = hash_table_2.size(1);
    const int hashmap_size_3 = hash_table_3.size(1);
    const int hashmap_size_4 = hash_table_4.size(1);
    const int F = hash_table_1.size(2);// / 4;

    auto grad_hash_table_1 = torch::zeros_like(hash_table_1);
    auto grad_hash_table_2 = torch::zeros_like(hash_table_2);
    auto grad_hash_table_3 = torch::zeros_like(hash_table_3);
    auto grad_hash_table_4 = torch::zeros_like(hash_table_4);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hermite_encoding_backward_full_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        grad_enc.data_ptr<float>(),
        grad_dx.data_ptr<float>(),
        grad_dy.data_ptr<float>(),
        grad_dz.data_ptr<float>(),
        grad_dxx.data_ptr<float>(),
        grad_dyy.data_ptr<float>(),
        grad_dzz.data_ptr<float>(),
        grad_hash_table_1.data_ptr<float>(),
        grad_hash_table_2.data_ptr<float>(),
        grad_hash_table_3.data_ptr<float>(),
        grad_hash_table_4.data_ptr<float>(),
        resolutions.data_ptr<float>(),
        N, L, F, hashmap_size_1, hashmap_size_2, 
        hashmap_size_3, hashmap_size_4
    );

    return {grad_hash_table_1, grad_hash_table_2, grad_hash_table_3, grad_hash_table_4};
}
