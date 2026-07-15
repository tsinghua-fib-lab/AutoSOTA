#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Utility functions
#include "utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Define block dimensions (32x8 = 256 threads)
// BLOCK_DIM_X is the warp size 32, facilitating fast
// intra-warp reduction using __shfl_down_sync()
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define BLOCK_DIM 256
#define MAX_NUM_BLOCK_1D 256

// Helper function: intra-warp reduction
__device__ __forceinline__ double warp_reduce_sum(double val)
{
    const unsigned int mask = 0xffffffff;
    for (int s = warpSize / 2; s > 0; s >>= 1)
    {
        val += __shfl_down_sync(mask, val, s);
    }
    return val;
}

// Helper function: intra-block reduction
// Input: local sum in the current thread
// Output: when threadIdx.x == 0, return the block sum
// Do not call this function for other threads
__device__ __forceinline__ double block_reduce_sum(double val)
{
    // Reduction within the warp
    double sum = warp_reduce_sum(val);

    // Use shared memory for intra-block reduction
    // Assuming max blockDim.x is 1024 and warpSize is 32, max 32 slots are needed
    static __shared__ double shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // These threads (lane == 0) contain the warp sums
    if (lane == 0)
    {
        // shared[0]: warp 0 sum
        // shared[1]: warp 1 sum
        // ...
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp aggregate results from all warps
    // Only warp 0 does this
    // This time results are stored in shared memory
    if (warp_id == 0)
    {
        // (blockDim.x + warpSize - 1) / warpSize means how many warps are actually used
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0;
        sum = warp_reduce_sum(sum);
    }

    return sum;
}

// Combined computations for line search
// 1. gamma = gamma_prev + step * direc, gamma = (alpha, beta)
// 2. Compute Trowsums and Tcolsums as in T_fused_kernel()
// 3. Compute objfn and grad as in obj_grad_kernel()
// 4. Compute <grad, direc>

// This kernel computes
//     gamma = gamma_prev + step * direc
// and initializes work = (Trowsums, Tcolsums) to zero
//
// In: gamma_prev     [n+m]
// In: direc          [n+m-1]
// In: step           [1]
// Out: gamma         [n+m]
// Out: work          [n+m]
__global__ void line_search_fused_kernel1(
    const double* __restrict__ gamma_prev,
    const double* __restrict__ direc,
    double step,
    int n,
    int m,
    double* __restrict__ gamma,
    double* __restrict__ work
)
{
    // Indices
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int size = n + m;
    const int last = size - 1;

    // Grid-stride loop
    for (int i = idx; i < size; i += stride)
    {
        // gamma = gamma_prev + step * direc
        // gamma[n+m-1] is set to zero
        if (i < last)
        {
            gamma[i] = gamma_prev[i] + step * direc[i];
        }
        else
        {
            gamma[i] = 0.0;
        }

        // Set work space to be zero
        work[i] = 0.0;
    }
}

// This kernel computes Trowsums and Tcolsums
// Same as T_fused_kernel()
//
// In: alpha          [n]
// In: beta           [m]
// In: M              [n*m]
// Out: Trowsums      [n]    assuming initialized to zero
// Out: Tcolsums      [m]    assuming initialized to zero
// Out: Tvalues       [n*(m-1)]
// Out: Tflatind      [n*(m-1)]
__global__ void line_search_fused_kernel2(
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    const double* __restrict__ M,
    double reg,
    int n,
    int m,
    bool write_values_and_indices,
    double* __restrict__ Trowsums,
    double* __restrict__ Tcolsums,
    double* __restrict__ Tvalues,
    int* __restrict__ Tflatind
)
{
    // Overall structure:
    // 1. We use a number of fixed-sized thread blocks to cover
    //    the whole M/T matrix.
    // 2. Each thread block is of size BLOCK_DIM_X * BLOCK_DIM_Y.
    // 3. We allocate a grid of blocks for parallel processing.
    // 4. On the x direction (column), the grid contains (m/BLOCK_DIM_X)
    //    blocks, so the grid has the same number of columns as M/T.
    // 5. On the y direction (row), the grid contains gridDim.y blocks,
    //    which can be an arbitrary number. This means that we will do
    //    grid-stride loop on the y direction.

    // Shared memory for partial column sums
    // Each block contains an array of size BLOCK_DIM_X
    // s_col_sum[tx] collects the sum of the tx-th column in this block
    // Each block column is of length BLOCK_DIM_Y
    __shared__ double s_col_sum[BLOCK_DIM_X];

    // Indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Global column index
    const int j = blockIdx.x * blockDim.x + tx;
    // Grid-stride loop on i
    const int start_i = blockIdx.y * blockDim.y + ty;
    const int stride_i = gridDim.y * blockDim.y;

    // Initialize shared memory
    // Only one thread per column needs to do this
    // Let the first row handle this
    if (ty == 0)
    {
        s_col_sum[tx] = 0.0;
    }
    __syncthreads();

    // This variable collects T_ij values on different strides
    double Tij_stride_sum = 0.0;
    // Quantities that do not vary with i
    const bool j_lt_m = (j < m);
    const bool write_val_ind = write_values_and_indices && (j < m - 1);
    // const int lane_id = tx % warpSize;
    const bool is_lane0 = (tx % warpSize == 0);
    const int flat_idx_Hsl_head = (n + j) * (n + m - 1);
    const double betaj = beta[j];
    // Grid-stride loop on i, with boundary check on j
    for (int i = start_i; i < n; i += stride_i)
    {
        // Default to be zero, so that padding threads can
        // join reduction but do not affect the result
        double T_ij = 0.0;

        // if (j < m)
        if (j_lt_m)
        {
            // Flattened index reading M[i, j]
            const int flat_idx_M = i * m + j;

            // Compute T[i, j]
            T_ij = exp((alpha[i] + betaj - M[flat_idx_M]) / reg);

            // Fill Tflatind and Tvalues only when j < m-1,
            // excluding the last row of T' (last column of T)
            // if (write_values_and_indices && j < m - 1)
            if (write_val_ind)
            {
                // We read T_t by row, and write to Tvalues [n*(m-1)]
                // T_t[i, j] -> Tvalues[i * (m-1) + j]
                const int flat_idx_Tvalues = flat_idx_M - i;  // == i * (m - 1) + j;
                // Index of T_t[i, j] in flattened Hsl matrix
                // const int flat_idx_Hsl = (n + j) * (n + m - 1) + i;
                const int flat_idx_Hsl = flat_idx_Hsl_head + i;

                Tflatind[flat_idx_Tvalues] = flat_idx_Hsl;
                Tvalues[flat_idx_Tvalues] = T_ij;
            }

            // Accumulate T_ij values across strides
            Tij_stride_sum += T_ij;
        }

        // Fast reduction within the warp
        double warp_row_sum = warp_reduce_sum(T_ij);

        // Write warp partial sums back to global memory
        // The warp leader (lane_id=0) writes back the row sums
        // to the global memory
        // Note that we allow multiple warps in each row of the block,
        // since BLOCK_DIM_X can be a multiple of warpSize
        if (is_lane0)
        {
            atomicAdd(&Trowsums[i], warp_row_sum);
        }
    }

    // Accumulate Tij_stride_sum to shared memory
    // Multiple rows within the block (with different ty)
    // will visit the same s_col_sum[tx], but it should be fast
    atomicAdd(&s_col_sum[tx], Tij_stride_sum);

    // Make sure s_col_sum has finished
    __syncthreads();

    // Write column sums within the block to the global memory
    // The first row of threads (ty=0) in each block does this
    // Different blocks on the y direction will visit the same Tcolsums[j]
    if (ty == 0 && j_lt_m)
    {
        atomicAdd(&Tcolsums[j], s_col_sum[tx]);
    }
}

// This kernel computes objective function value (objfn), gradient (grad), and <grad, direc>
//
// f = reg * sum(T) - <alpha, a> - <beta, b> = reg * <Trowsums, 1> - <gamma, ab>
// g = (Trowsums - a, Tcolsums_t - b_t)
//
// In: ab            [n+m]       ab = (a, b)
// In: gamma         [n+m]       gamma = (alpha, beta)
// In: direc         [n+m-1]
// In: Trowsums      [n]
// In: Tcolsums      [m]
// Out: grad         [n+m-1]
// Out: block_objfn  [MAX_NUM_BLOCK_1D]
// Out: block_dg     [MAX_NUM_BLOCK_1D]
__global__ void line_search_fused_kernel3(
    const double* __restrict__ ab,
    const double* __restrict__ gamma,
    const double* __restrict__ direc,
    const double* __restrict__ Trowsums,
    const double* __restrict__ Tcolsums,
    double reg,
    int n,
    int m,
    double* __restrict__ grad,
    double* __restrict__ block_objfn,
    double* __restrict__ block_dg
)
{
    // Indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int total_len = n + m;
    const int grad_len = total_len - 1;

    // Local variable to accumulate Trowsums
    double local_Tsum = 0.0;
    // Local variable to accumulate the dot product <gamma, ab> = <alpha, a> + <beta, b>
    double local_dotprod = 0.0;
    // Similarly for dg
    double local_dg = 0.0;

    // Grid-stride loop instead of `if (idx < total_len)`
    for (int i = idx; i < total_len; i += stride)
    {
        // Accumulate dot product
        double val_ab = ab[i];
        double val_gamma = gamma[i];
        local_dotprod += val_ab * val_gamma;

        // Calculate gradient
        // Note that grad is only [n + m - 1]
        if (i < grad_len)
        {
            double val_grad = 0.0;

            // First n elements of grad: Trowsums - a
            // Also accumulate Trowsums
            if (i < n)
            {
                // Accumulate Trowsums
                double val_Tr = Trowsums[i];
                local_Tsum += val_Tr;
                // Compute gradient
                val_grad = val_Tr - val_ab;
            }
            else
            {
                // Next m-1 elements of grad: Tcolsums_t - b_t
                // Current i corresponds to index (i - n) in b
                val_grad = Tcolsums[i - n] - val_ab;
            }
            grad[i] = val_grad;

            // Accumulate dg
            double val_direc = direc[i];
            local_dg += val_grad * val_direc;
        }

        // Note that when i == n + m - 1 (i.e., the last element of b),
        // we do not write to grad, but still compute <gamma, ab>
    }

    // Parallel reductions
    local_Tsum = block_reduce_sum(local_Tsum);
    local_dotprod = block_reduce_sum(local_dotprod);
    local_dg = block_reduce_sum(local_dg);

    // Write to block results
    if (tid == 0)
    {
        block_objfn[bid] = reg * local_Tsum - local_dotprod;
        block_dg[bid] = local_dg;
    }
}

// Helper function to launch line search computations
//
// In: d_gamma_prev     [n+m]
// In: d_direc          [n+m-1]
// In: step             [1]
// In: d_M              [n*m]
// In: d_ab             [n+m]
// In: reg              [1]
// Out: d_gamma         [n+m]
// Out: d_grad          [n+m-1]
// Out: objfn           [1]
// Out: dg              [1]
// Work: d_work         [n+m+n*(m-1)]
// Work: h_pinned       [2*MAX_NUM_BLOCK_1D]
void launch_line_search_computation(
    const double* d_gamma_prev,
    const double* d_direc,
    double step,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m,
    double* d_gamma,
    double* d_grad,
    double& objfn,
    double& dg,
    double* d_work, int* d_iwork, double* h_pinned,
    bool write_values_and_indices = true,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Pointer aliases
    double* d_alpha = d_gamma;
    double* d_beta = d_gamma + n;
    double* d_Trowsums = d_work;
    double* d_Tcolsums = d_work + n;
    double* d_Tvalues = d_work + (n + m);
    int* d_Tflatind = d_iwork;

    // Compute new gamma, gamma = gamma_prev + step * direc,
    // and initializes (Trowsums, Tcolsums) to be zero
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (n + m + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to MAX_NUM_BLOCK_1D (256)
    // The grid-stride loop in line_search_fused_kernel1() will handle larger sizes
    numBlocks = std::min(numBlocks, MAX_NUM_BLOCK_1D);

    line_search_fused_kernel1<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_gamma_prev, d_direc, step,
        n, m,
        d_gamma, d_work
    );

    // Compute Trowsums and Tcolsums
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    // The grid must cover all the columns, meaning that gridDim.x
    // must be m/BLOCK_DIM_X. But gridDim.y can be arbitrary, since
    // a grid-stride loop is implemented on the y direction
    // Then we use heuristics to adjust the total number of blocks
    gridDim.y = heuristic_num_blocks(gridDim.x, gridDim.y);

    line_search_fused_kernel2<<<gridDim, blockDim, 0, stream>>>(
        d_alpha, d_beta, d_M,
        reg, n, m,
        write_values_and_indices,
        d_Trowsums, d_Tcolsums,
        d_Tvalues, d_Tflatind
    );

    // Compute objfn, grad, and <grad, direc>
    // Limit number of blocks to MAX_NUM_BLOCK_1D (256)
    // The grid-stride loop in line_search_fused_kernel3() will handle larger sizes
    double* d_block_result;
    cudaHostGetDevicePointer(&d_block_result, h_pinned, 0);

    const double* h_objfn = h_pinned;
    const double* h_dg = h_pinned + MAX_NUM_BLOCK_1D;
    double* d_block_objfn = d_block_result;
    double* d_block_dg = d_block_result + MAX_NUM_BLOCK_1D;

    line_search_fused_kernel3<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_ab, d_gamma, d_direc,
        d_Trowsums, d_Tcolsums,
        reg, n, m,
        d_grad, d_block_objfn, d_block_dg
    );
    cudaStreamSynchronize(stream);

    // Compute final result on CPU
    objfn = dg = 0.0;
    for (int i = 0; i < numBlocks; i++)
    {
        objfn += h_objfn[i];
        dg += h_dg[i];
    }
}
