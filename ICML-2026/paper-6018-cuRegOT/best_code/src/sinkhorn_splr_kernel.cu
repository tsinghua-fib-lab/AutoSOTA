#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUB headers
#include <cub/cub.cuh>

// Thrust headers
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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



// Background: we want to solve the dual problem of regularized optimal transport
//
// The dual variables are x = (alpha, beta_t), where beta_t means beta[:-1],
// and we always set beta = (beta_t, 0)
// alpha [n], beta [m], beta_t [m-1]
//
// The objective function (f), gradient (g), and Hessian (H) have closed-form expressions
// Let T[i, j] = exp((alpha[i] + beta[j] - M[i, j]) / reg), T_t = T[:, :-1]
// f = reg * sum(T) - <alpha, a> - <beta, b>
// g = (rowsum(T) - a, colsum(T_t) - b_t) = (Trowsums - a, Tcolsums_t - b_t)
// H = [diag(Trowsums)  T_t             ] / reg
//     [(T_t)'          diag(Tcolsums_t)]
//
// In the SPLR algorithm, we will sparsify T into a sparse matrix Tsp, so
// Hsp = [diag(Trowsums)  Tsp_t           ] / reg
//       [(Tsp_t)'        diag(Tcolsums_t)]
// Then we need to compute the CSR representation of the Hsp matrix
// For sparse Cholesky decomposition, we only need the lower triangular part, so
// it suffices to find the scaled and lower-triangular sparse matrix
// Hsl = [diag(Trowsums)  0               ]
//       [(Tsp_t)'        diag(Tcolsums_t)]
//
// To get the CSR representation of Hsl, we need to know the values of the
// nonzero elements and their locations in the matrix
// One way to do this is flattening the matrix and obtain two pointers:
//     Hvalues = [h0, h1, ..., hs],
//     Hflatind = [i0, i1, ..., is],
// where Hflatind contains the flattened indices of Hvalues, and we assume that
// Hflatind is in ascending order
// Then the CSR representation of Hsl is
//     val = [h0, h1, ..., hs],
//     colind = [i0 % N, i1 % N, ..., is % N],
//     rowptr = [0, p[1], p[2], ..., p[N]], p[1] = #{i / N == 0},
//                                          p[2] - p[1] = #{i / N == 1},
//                                          p[3] - p[2] = #{i / N == 2}, ...,
// where N = n+m-1 is the size of Hsl
//
// Moreover, we know that Hvalues can be divided into three parts:
// 1. Trowsums = r = [r0, r1, ..., r[n-1]]
// 2. Tcolsums_t = c_t = [c0, c1, ..., c[m-2]]
// 3. Values of (Tsp_t)', Tvalues = [t0, t1, ..., t[K-1]]
// The corresponding indices are:
// 1. r indices: [i * N + i] = [i * (N + 1)]], i = 0, ..., n-1
// 2. c_t indices: [(n + j) * N + n + j] = [(n + j) * (N + 1)], j = 0, ..., m-2
// 3. Indices of (Tsp_t)', with the mapping
//        (i, j) in T -> (j, i) in T'/(T_t)' -> (n+j, i) in Hsl -> (n+j)*N+i for flattened indices

// To finish this task, we construct three core functions that mainly run on GPU
// 1. A fused CUDA kernel that focuses on the computation on T:
//    (a) Trowsums and Tcolsums
//    (b) Flattened T' values (Tvalues) and indices (Tflatind),
//        which are used for downstream sparsification
//        We need to exclude the last row of T' in the output Tvalues
// 2. A function preparing the data for (Tsp_t)':
//    (a) Find the top-K elements in the output Tvalues array and the corresponding indices Tflatind
//    (b) Put these K (Tvalues, Tflatind)-pairs at the front of (Hvalues, Hflatind) pointers
//    (c) Append (r, r indices) and (c_t, c_t indices) to the (Hvalues, Hflatind) pointers
//    (d) Sort these (N+K) (val, ind)-pairs according to ind, so that the reordered val pointer
//        is exactly the CSR value pointer of Hsl
// 3. A function to compute the CSR pointers from flattened values (Hvalues) and indices (Hflatind)
//
// Overall procedure:
//     alpha, beta, M, a, b, reg  ==>  T  ==>  f, g, Tvalues, Tflatind  ==>  Hvalues, Hflatind  ==>  Hsl



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

// CUDA kernel to transform the gamma=(alpha, beta) vector to gamma'=(alpha', beta_t', 0)
// alpha += beta[m-1], beta -= beta[m-1]
// The last zero will be set in the helper function that calls this kernel
__global__ void shift_gamma_kernel(double* d_gamma, int n, int m)
{
    // Get shift
    const int shift_index = n + m - 1;
    const double shift = d_gamma[shift_index];

    // Grid-stride loop
    int i_start = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = i_start; i < shift_index; i += stride)
    {
        d_gamma[i] += ((i < n) ? shift : -shift);
    }
}

// Helper function to transform the gamma=(alpha, beta) vector to gamma'=(alpha', beta_t', 0)
// alpha += beta[m-1], beta -= beta[m-1]
void shift_gamma(double* d_gamma, int n, int m, cudaStream_t stream = cudaStreamPerThread)
{
    if (n == 0 && m == 0)
    {
        return;
    }

    // Only work on the first (n + m - 1) elements
    int work_size = n + m - 1;
    if (work_size > 0)
    {
        dim3 threadsPerBlock(BLOCK_DIM);
        int numBlocks = (work_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
        // Limit number of blocks to 256
        // The grid-stride loop in shift_gamma_kernel() will handle larger sizes
        numBlocks = std::min(numBlocks, 256);

        shift_gamma_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_gamma, n, m
        );
    }

    // Set gamma[n+m-1]=0
    cudaMemsetAsync(d_gamma + (n + m - 1), 0, sizeof(double), stream);
}

// Fused CUDA kernel for computation on T
// 1. Compute T[i, j] = exp(...)
// 2. Perform parallel reduction for row sums and column sums using shared memory
// 3. Write (T_t)' (T' excluding the last row) to Tvalues
// 4. Compute the flat indices in the Hsl matrix coordinates for the subsequent Top-K selection
//
// The option write_values_and_indices controls whether do steps 3 and 4
//
// In: alpha          [n]
// In: beta           [m]
// In: M              [n*m]
// Out: Trowsums      [n]    assuming initialized to zero
// Out: Tcolsums      [m]    assuming initialized to zero
// Out: Tvalues       [n*(m-1)]
// Out: Tflatind      [n*(m-1)]
__global__ void T_fused_kernel(
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

            // 1. Compute T[i, j]
            // T_ij = exp((alpha[i] + beta[j] - M[flat_idx_M]) / reg);
            T_ij = exp((alpha[i] + betaj - M[flat_idx_M]) / reg);

            // 2. Fill Tflatind and Tvalues only when j < m-1,
            //    excluding the last row of T' (last column of T)
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

            // 3. Accumulate T_ij values across strides
            Tij_stride_sum += T_ij;
        }

        // Fast reduction within the warp
        double warp_row_sum = warp_reduce_sum(T_ij);

        // 4. Write warp partial sums back to global memory
        // The warp leader (lane_id=0) writes back the row sums
        // to the global memory
        // Note that we allow multiple warps in each row of the block,
        // since BLOCK_DIM_X can be a multiple of warpSize
        if (is_lane0)
        {
            atomicAdd(&Trowsums[i], warp_row_sum);
        }
    }

    // 5. Accumulate Tij_stride_sum to shared memory
    //    Multiple rows within the block (with different ty)
    //    will visit the same s_col_sum[tx], but it should be fast
    atomicAdd(&s_col_sum[tx], Tij_stride_sum);

    // Make sure s_col_sum has finished
    __syncthreads();

    // 6. Write column sums within the block to the global memory
    // The first row of threads (ty=0) in each block does this
    // Different blocks on the y direction will visit the same Tcolsums[j]
    if (ty == 0 && j_lt_m)
    {
        atomicAdd(&Tcolsums[j], s_col_sum[tx]);
    }
}

// CUDA kernel to compute objective function value (f) and gradient (g)
//
// f = reg * sum(T) - <alpha, a> - <beta, b> = reg * <Trowsums, 1> - <gamma, ab>
// g = (Trowsums - a, Tcolsums_t - b_t)
//
// If compute_squared_grad = true, also compute blockwise sum of squared gradients
//
// In: ab            [n+m]       ab = (a, b)
// In: gamma         [n+m]       gamma = (alpha, beta)
// In: Trowsums      [n]
// In: Tcolsums      [m]
// Out: grad         [n+m-1]
// Out: block_objfn  [MAX_NUM_BLOCK_1D]
// Out: block_g2     [MAX_NUM_BLOCK_1D]
__global__ void obj_grad_kernel(
    const double* __restrict__ ab,
    const double* __restrict__ gamma,
    const double* __restrict__ Trowsums,
    const double* __restrict__ Tcolsums,
    double reg,
    int n,
    int m,
    bool compute_squared_grad,
    double* __restrict__ grad,
    double* __restrict__ block_objfn,
    double* __restrict__ block_g2
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
    // Local variable to accumulate squared gradient
    double local_g2 = 0.0;

    // Grid-stride loop instead of `if (idx < total_len)`
    for (int i = idx; i < total_len; i += stride)
    {
        // Task 1: Accumulate dot product
        double val_ab = ab[i];
        double val_gamma = gamma[i];
        local_dotprod += val_ab * val_gamma;

        // Task 2: Calculate gradient
        // Note that grad is only [n + m - 1]
        if (i < grad_len)
        {
            double val_grad = 0.0;

            // First n elements of grad: Trowsums - a
            // Task 3: Also accumulate Trowsums
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

            // Task 4: Accumulate squared gradient
            if (compute_squared_grad)
            {
                local_g2 += val_grad * val_grad;
            }
        }

        // Note that when i == n + m - 1 (i.e., the last element of b),
        // we do not write to grad, but still compute <gamma, ab>
    }

    // Parallel reductions
    local_Tsum = block_reduce_sum(local_Tsum);
    local_dotprod = block_reduce_sum(local_dotprod);
    if (compute_squared_grad)
    {
        local_g2 = block_reduce_sum(local_g2);
    }

    // Write to block results
    if (tid == 0)
    {
        block_objfn[bid] = reg * local_Tsum - local_dotprod;
        if (compute_squared_grad)
        {
            block_g2[bid] = local_g2;
        }
    }
}

// Helper function to launch CUDA computations on T
//
// Given gamma = (alpha, beta), M, ab = (a, b) (all on device), and reg:
// 1. Compute T matrix
// 2. Compute row/column sums of T
// 3. Compute objective function value objfn and gradient grad
//
// In: d_gamma      [n+m]        d_gamma = (d_alpha, d_beta)
// In: d_M          [n*m]
// In: d_ab         [n+m]
// Out: d_grad      [n+m-1]
// Out: objfn       [1]
// Out: gnorm       [1]
// Out: d_work      [n+m+n*(m-1)]    (d_Trowsums, d_Tcolsums, d_Tvalues)
// OUt: d_iwork     [n*(m-1)]        d_Tflatind
// Work: h_pinned   [2*MAX_NUM_BLOCK_1D]
void launch_T_objfn_grad(
    const double* d_gamma,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m,
    bool write_values_and_indices,
    bool compute_grad_norm,
    double* d_grad,
    double& objfn, double& gnorm,
    double* d_work, int* d_iwork,
    double* h_pinned,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Pointer aliases
    const double* d_alpha = d_gamma;
    const double* d_beta = d_gamma + n;
    double* d_Trowsums = d_work;
    double* d_Tcolsums = d_work + n;
    double* d_Tvalues = d_work + (n + m);
    int* d_Tflatind = d_iwork;

    // Initialize work space d_work = (d_Trowsums, d_Tcolsums) to be zero
    CUDA_CHECK(cudaMemsetAsync(d_work, 0, (n + m) * sizeof(double), stream));

    // Compute dimensions
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    // The grid must cover all the columns, meaning that gridDim.x
    // must be m/BLOCK_DIM_X. But gridDim.y can be arbitrary, since
    // a grid-stride loop is implemented on the y direction
    // Then we use heuristics to adjust the total number of blocks
    gridDim.y = heuristic_num_blocks(gridDim.x, gridDim.y);

    // Launch the fused kernel
    T_fused_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_alpha, d_beta, d_M,
        reg, n, m,
        write_values_and_indices,
        d_Trowsums, d_Tcolsums,
        d_Tvalues, d_Tflatind
    );

    // Compute objfn, grad, and <grad, grad>
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (n + m + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to MAX_NUM_BLOCK_1D (256)
    // The grid-stride loop in obj_grad_kernel() will handle larger sizes
    numBlocks = std::min(numBlocks, MAX_NUM_BLOCK_1D);

    // Get device pointers to pinned memory
    double* d_block_result;
    cudaHostGetDevicePointer(&d_block_result, h_pinned, 0);

    const double* h_objfn = h_pinned;
    const double* h_g2 = h_pinned + MAX_NUM_BLOCK_1D;
    double* d_block_objfn = d_block_result;
    double* d_block_g2 = d_block_result + MAX_NUM_BLOCK_1D;

    obj_grad_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_ab, d_gamma, d_Trowsums, d_Tcolsums,
        reg, n, m,
        compute_grad_norm,
        d_grad, d_block_objfn, d_block_g2
    );
    cudaStreamSynchronize(stream);

    // Compute final result on CPU
    objfn = 0.0;
    double g2 = 0.0;
    for (int i = 0; i < numBlocks; i++)
    {
        objfn += h_objfn[i];
        if (compute_grad_norm)
        {
            g2 += h_g2[i];
        }
    }
    if (compute_grad_norm)
    {
        gnorm = std::sqrt(g2);
    }
}

// CUDA kernel to write diagonal elements of Hsl to flattened value and index pointers
//
// In: Trowsums  [n]
// In: Tcolsums  [m]
// Out: values   [n+m-1]
// OUt: indices  [n+m-1]
__global__ void write_diagonal_kernel(
    const double* __restrict__ Trowsums,
    const double* __restrict__ Tcolsums,
    double shift,
    int n,
    int m,
    double* __restrict__ values,
    int* __restrict__ indices
)
{
    // Indices
    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Write Trowsums[0...(n-1)] + shift to values[0...(n-1)]
    // Write i * (N + 1), i=0, ..., n-1 to indices[0...(n-1)]
    //
    // Write Tcolsums[0...(m-2)] + shift to (values+n)[0...(m-2)]
    // Write (n + j) * (N + 1), j=0, ..., m-2 to (indices+n)[0...(m-2)]
    int Hsize = n + m - 1;

    // Grid-stride loop instead of `if (idx < Hsize)`
    for (int idx = start; idx < Hsize; idx += stride)
    {
        values[idx] = (idx < n) ? (Trowsums[idx] + shift) : (Tcolsums[idx - n] + shift);
        indices[idx] = idx * (Hsize + 1);
    }
}

// Helper function to compute sparse representation of H
//
// (continued from launch_T_computation())
// 4. The largest K elements of (T_t)' are written into the first K elements of d_Hvalues
// 5. The corresponding (flattened) indices are written into d_Hflatind
// 6. Add diagonal elements of Hsl matrix (plus a shift) to d_Hvalues and corresponding indices to d_Hflatind
//    (overwrite d_Hvalues[K:] and d_Hflatind[K:])
// 7. The first (K+Hsize) elements in d_Hflatind are in ascending order, Hsize = n+m-1
//
// In: d_Tvalues    [n*(m-1)]
// In: d_Tflatind   [n*(m-1)]
// Out: d_Hvalues   [nnz] = [K+Hsize], reserve [Kmax+Hsize]
// Out: d_Hflatind  [nnz], reserve [Kmax+Hsize]
void launch_H_sparsification(
    const double* d_Tvalues,
    const int* d_Tflatind,
    const double* d_Trowsums,
    const double* d_Tcolsums,
    int n, int m, int K,
    double shift,
    double* d_Hvalues,
    int* d_Hflatind,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Total number of T elements
    size_t Te = n * (m - 1);
    size_t Hsize = n + m - 1;

    // Bound check
    size_t Ks = max(K, 1);
    Ks = min(Ks, Te);

    // We use cub::DeviceTopK::MaxPairs to find the largest K values in d_Tvalues,
    // and get the corresponding elements in d_Tflatind
    // The results are written to d_Hvalues and d_Hflatind, respectively

    // Prepare CUDA stream
    cuda::stream_ref stream_ref{stream};

    // Set required environment
    auto requirements = cuda::execution::require(
        cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted
    );

    // Create the environment with the stream and requirements
    auto env = cuda::std::execution::env{stream_ref, requirements};

    // Get memory size for DeviceTopK
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceTopK::MaxPairs(
        d_temp_storage, temp_storage_bytes,
        d_Tvalues, d_Hvalues,
        d_Tflatind, d_Hflatind,
        Te, Ks,
        env
    ));
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run DeviceTopK
    CUDA_CHECK(cub::DeviceTopK::MaxPairs(
        d_temp_storage, temp_storage_bytes,
        d_Tvalues, d_Hvalues,
        d_Tflatind, d_Hflatind,
        Te, Ks,
        env
    ));
    // Free memory
    CUDA_CHECK(cudaFree(d_temp_storage));

    // Add diagonal elements of Hsl to (d_Hvalues, d_Hflatind)
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks_write_diagonal((Hsize + threadsPerBlock.x - 1) / threadsPerBlock.x);
    write_diagonal_kernel<<<numBlocks_write_diagonal, threadsPerBlock, 0, stream>>>(
        d_Trowsums, d_Tcolsums, shift, n, m, d_Hvalues + Ks, d_Hflatind + Ks
    );

    // Wrap raw device pointers with thrust::device_ptr
    thrust::device_ptr<double> d_Hvalues_ptr = thrust::device_pointer_cast(d_Hvalues);
    thrust::device_ptr<int> d_Hflatind_ptr = thrust::device_pointer_cast(d_Hflatind);

    // Call thrust::sort_by_key to sort d_Hflatind in ascending order
    auto policy = thrust::cuda::par.on(stream);
    thrust::sort_by_key(
        policy,
        d_Hflatind_ptr,
        d_Hflatind_ptr + (Ks + Hsize),
        d_Hvalues_ptr
    );
}

// CUDA kernel to recompute nonzero elements according to existing sparsity pattern
//
// In: Hflatind  [K+Hsize]
// Out: Hvalues  [K+Hsize]
__global__ void recompute_nonzero_values_kernel(
    const int* __restrict__ Hflatind,
    const double* __restrict__ Trowsums,
    const double* __restrict__ Tcolsums,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    const double* __restrict__ M,
    double reg,
    double shift,
    int n,
    int m,
    int K,
    double* __restrict__ Hvalues
)
{
    // Indices
    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    int Hsize = n + m - 1;
    int total_len = K + Hsize;

    // Grid-stride loop instead of `if (idx < K + Hsize)`
    for (int idx = start; idx < total_len; idx += stride)
    {
        // Index is based on Hsl (flattened)
        int flat_ind_Hsl = Hflatind[idx];

        // Convert to (i, j) in Hsl
        int i = flat_ind_Hsl / Hsize;
        int j = flat_ind_Hsl % Hsize;

        double Hval = 0.0;
        if (i < n)
        {
            // Case 1: i < n -- diagonal elements of Hsl[:n, :n]
            //                  Hsl[i, i] = Trowsums[i]
            Hval = Trowsums[i] + shift;
        }
        else if (j < n)
        {
            // Case 2: i >= n, j < n -- Hsl[i, j] = T'[i - n, j] = T[j, i - n]
            const int Mind = j * m + (i - n);
            Hval = exp((alpha[j] + beta[i - n] - M[Mind]) / reg);
        }
        else
        {
            // Case 3: j >= n -- diagonal elements of Hsl[n:, n:]
            Hval = Tcolsums[j - n] + shift;
        }
        Hvalues[idx] = Hval;
    }
}

// CUDA kernel to extract column indices and count the number of elements per row
//
// In: Hflatind     [nnz]
// Out: Hcolind     [nnz]
// Out: row_counts  [cols]
__global__ void extract_columns_and_count_kernel(
    const int* __restrict__ Hflatind,
    int* __restrict__ Hcolind,
    int* __restrict__ row_counts,
    int nnz, int cols
)
{
    // Indices
    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop instead of `if (idx < nnz)`
    for (int idx = start; idx < nnz; idx += stride)
    {
        // Convert flattened index to row and column
        int flat_idx = Hflatind[idx];
        int row = flat_idx / cols;
        int col = flat_idx % cols;

        // Set column index
        Hcolind[idx] = col;

        // Increment row count atomically
        atomicAdd(&row_counts[row], 1);
    }
}

// Helper function to convert sorted flat indices to CSR format
//
// In: d_Hflatind               [nnz] = [K+Hsize] = [K+n+m-1]
// Out: d_Hcolind               [nnz]
// Out: d_Hrowptr               [Hsize + 1] = [n+m]
// Working space: d_row_counts  [Hsize] = [n+m-1]
void launch_csr_conversion(
    const int* d_Hflatind,
    int* d_Hcolind,
    int* d_Hrowptr,
    int* d_row_counts,
    int K, int n, int m,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Initialize d_row_counts to zero vector
    int Hsize = n + m - 1;
    int nnz = K + Hsize;
    CUDA_CHECK(cudaMemsetAsync(d_row_counts, 0, Hsize * sizeof(int), stream));

    // Extract column indices and count the number of elements per row
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks((nnz + threadsPerBlock.x - 1) / threadsPerBlock.x);
    // Adjust number of blocks using heuristics
    // The grid-stride loop in extract_columns_and_count_kernel()
    // will handle larger sizes
    numBlocks.x = heuristic_num_blocks(numBlocks.x);

    extract_columns_and_count_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_Hflatind, d_Hcolind, d_row_counts, nnz, Hsize
    );

    // Compute row pointers using inclusive sum
    // [a, b, c] -> [a, a + b, a + b + c]

    // The first element of row pointer is always zero
    CUDA_CHECK(cudaMemsetAsync(d_Hrowptr, 0, sizeof(int), stream));
    // Get memory size for InclusiveSum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_row_counts, d_Hrowptr + 1, Hsize, stream
    ));
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run InclusiveSum
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_row_counts, d_Hrowptr + 1, Hsize, stream
    ));
    // Free memory
    CUDA_CHECK(cudaFree(d_temp_storage));
}

// Helper function to compute sparsified Hessian (plus a shift) in CSR form
//
// In: d_gamma       [n+m]  d_gamma = (d_alpha, d_beta)
// In: d_M           [n*m]
// In: d_work        [n+m+n*(m-1)]  assuming it contains (d_Trowsums, d_Tcolsums, d_Tvalues)
// In/Work: d_iwork  [max(n*(m-1), n+m-1)]  assuming it contains d_Tflatind when passed in
//                                          will be overwritten on exit
// Out: d_Hvalues    [nnz] = [K+Hsize], reserve [Kmax+Hsize]
// Out: d_Hflatind   [nnz], reserve [Kmax+Hsize]
// Out: d_Hcolind    [nnz], reserve [Kmax+Hsize]
// Out: d_Hrowptr    [Hsize + 1] = [n+m]
//
// If fixed_indices = true, it means:
// 1. The first (K+Hsize) elements of d_Hflatind contain the sparsify pattern of Hsl
// 2. We keep d_Hflatind unchanged, and compute the corresponding Hsl values
// 3. Write these Hsl values to d_Hvalues
// 4. Continue computing d_Hcolind and d_Hrowptr
void launch_sphess(
    const double* d_gamma,
    const double* d_M,
    const double* d_work,
    int* d_iwork,
    double reg, double shift,
    int n, int m, int K,
    double* d_Hvalues, int* d_Hflatind, int* d_Hcolind, int* d_Hrowptr,
    bool fixed_indices = false,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Pointer aliases
    const double* d_alpha = d_gamma;
    const double* d_beta = d_gamma + n;
    const double* d_Trowsums = d_work;
    const double* d_Tcolsums = d_work + n;
    const double* d_Tvalues = d_work + (n + m);
    const int* d_Tflatind = d_iwork;

    // Dimensions
    size_t Te = n * (m - 1);
    size_t Hsize = n + m - 1;
    size_t Ks = std::max(K, 1);
    Ks = std::min(Ks, Te);
    size_t KHsize = Ks + Hsize;

    // Get d_Hvalues and d_Hflatind
    if (fixed_indices)
    {
        // If indices are fixed, we directly recompute d_Hvalues according to d_Hflatind

        // The first (K+Hsize) elements of d_Hflatind are already given
        // Now write elements of Hsl to d_Hvalues
        dim3 threadsPerBlock(BLOCK_DIM);
        int numBlocks = (KHsize + threadsPerBlock.x - 1) / threadsPerBlock.x;
        // Adjust number of blocks using heuristics
        // The grid-stride loop in recompute_nonzero_values_kernel()
        // will handle larger sizes
        numBlocks = heuristic_num_blocks(numBlocks);
        dim3 numBlocks_recompute_nonzero_values(numBlocks);
        recompute_nonzero_values_kernel<<<numBlocks_recompute_nonzero_values, threadsPerBlock, 0, stream>>>(
            d_Hflatind, d_Trowsums, d_Tcolsums, d_alpha, d_beta, d_M,
            reg, shift, n, m, Ks, d_Hvalues
        );
    }
    else
    {
        // Otherwise, call launch_H_sparsification to get
        // d_Hvalues and d_Hflatind
        launch_H_sparsification(
            d_Tvalues, d_Tflatind,
            d_Trowsums, d_Tcolsums,
            n, m, Ks, shift,
            d_Hvalues, d_Hflatind,
            stream
        );
    }

    // Finally, call launch_csr_conversion to compute d_Hcolind and d_Hrowptr

    // Now d_Tflatind is no longer used, so we can reuse
    // d_iwork for the working space
    int* d_row_counts = d_iwork;
    launch_csr_conversion(
        d_Hflatind, d_Hcolind, d_Hrowptr, d_row_counts, Ks, n, m, stream
    );
}

// Host function, mainly to test T computation and CSR conversion
void T_computation_sparsify_host(
    int nrun,
    const double* alpha,
    const double* beta,
    const double* M,
    const double* a,
    const double* b,
    double reg, double shift,
    int n, int m, int K,
    double* Trowsums, double* Tcolsums,
    double* objfn, double* grad, double* gnorm,
    double* Tvalues, int* Tflatind,
    double* csr_val, int* csr_rowptr, int* csr_colind
)
{
    // Total number of elements of M and T_t
    size_t Me = n * m;
    size_t Te = n * (m - 1);

    // Bound check for K
    size_t Ks = std::max(K, 1);
    Ks = std::min(Ks, Te);

    // Size of Hsl
    size_t Hsize = n + m - 1;

    // Number of nonzero elements in Hsl
    size_t nnz = Ks + Hsize;

    // Allocate device memory
    double *d_gamma, *d_M, *d_ab;
    double *d_work;
    double *d_grad;
    double *d_Hvalues;
    int *d_Tflatind, *d_Hflatind, *d_csr_rowptr, *d_csr_colind, *d_row_counts;

    CUDA_CHECK(cudaMalloc(&d_gamma, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_M, Me * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ab, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work, (n + m + Te) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hvalues, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tflatind, Te * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Hflatind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_colind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_rowptr, (Hsize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_counts, Hsize * sizeof(int)));

    // Allocate pinned memory
    double* h_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_pinned, 512 * sizeof(double), cudaHostAllocMapped));

    // Pointer aliases
    double* d_alpha = d_gamma;
    double* d_beta = d_gamma + n;
    double* d_a = d_ab;
    double* d_b = d_ab + n;
    double *d_Trowsums = d_work;
    double *d_Tcolsums = d_work + n;
    double *d_Tvalues = d_work + (n + m);

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M, Me * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice));

    // Launch computation
    // Multiple runs for benchmarking
    for (int i = 0; i < nrun; i++)
    {
        launch_T_objfn_grad(
            d_gamma, d_M, d_ab,
            reg, n, m,
            true, true,
            d_grad,
            *objfn, *gnorm,
            d_work, d_Tflatind,
            h_pinned
        );
        launch_H_sparsification(
            d_Tvalues, d_Tflatind,
            d_Trowsums, d_Tcolsums,
            n, m, Ks, shift,
            d_Hvalues, d_Hflatind
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    // Convert flattened elements to CSR format
    launch_csr_conversion(
        d_Hflatind, d_csr_colind, d_csr_rowptr, d_row_counts, Ks, n, m
    );

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(Trowsums, d_Trowsums, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tcolsums, d_Tcolsums, m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad, d_grad, Hsize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tvalues, d_Tvalues, Te * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tflatind, d_Tflatind, Te * sizeof(int), cudaMemcpyDeviceToHost));

    // Copy CSR results
    CUDA_CHECK(cudaMemcpy(csr_val, d_Hvalues, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_colind, d_csr_colind, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_rowptr, d_csr_rowptr, (Hsize + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_Hvalues));
    CUDA_CHECK(cudaFree(d_Tflatind));
    CUDA_CHECK(cudaFree(d_Hflatind));
    CUDA_CHECK(cudaFree(d_csr_colind));
    CUDA_CHECK(cudaFree(d_csr_rowptr));
    CUDA_CHECK(cudaFree(d_row_counts));

    // Free pinned memory
    CUDA_CHECK(cudaFreeHost(h_pinned));
}
