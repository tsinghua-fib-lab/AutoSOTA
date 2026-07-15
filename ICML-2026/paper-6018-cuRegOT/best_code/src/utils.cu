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

// Define block dimensions (16x16 = 256 threads)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 256

// Intra-warp reduction
__device__ __forceinline__ double warp_reduce_sum(double val)
{
    const unsigned int mask = 0xffffffff;
    for (int s = warpSize / 2; s > 0; s >>= 1)
    {
        val += __shfl_down_sync(mask, val, s);
    }
    return val;
}

// Intra-block reduction
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

// CUDA kernel for computing sum of squares of vector difference
__global__ void compute_squared_l2_distance_kernel(
    const double* __restrict__ vec1,
    const double* __restrict__ vec2,
    double* __restrict__ result,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double squared_diff = 0.0;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    for (int i = idx; i < size; i += stride)
    {
        double diff = vec1[i] - vec2[i];
        squared_diff += diff * diff;
    }

    // Intra-block reduce
    // Sums up results within this block
    // Stored in the first thread of the block
    squared_diff = block_reduce_sum(squared_diff);

    // The first thread of the block adds the block's sum to the global memory
    if (tid == 0)
    {
        // result must be initialized to zero
        atomicAdd(result, squared_diff);
    }
}

// Helper function to compute the l2 distance between vectors on device
double compute_l2_distance_cuda(const double* d_vec1, const double* d_vec2, int size, cudaStream_t stream)
{
    // Initialize result to zero
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), stream));

    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_squared_l2_distance_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_squared_l2_distance_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_vec1, d_vec2, d_result, size
    );

    // Copy back to host
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return std::sqrt(result);
}

// CUDA kernel for computing the squared l2 norm of a vector
__global__ void compute_squared_l2_norm_kernel(
    const double* __restrict__ vec,
    double* __restrict__ result,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum_of_squares = 0.0;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    for (int i = idx; i < size; i += stride)
    {
        double val = vec[i];
        sum_of_squares += val * val;
    }

    // Intra-block reduce
    // Sums up results within this block
    // Stored in the first thread of the block
    sum_of_squares = block_reduce_sum(sum_of_squares);

    // The first thread of the block adds the block's sum to the global memory
    if (tid == 0)
    {
        // result must be initialized to zero
        atomicAdd(result, sum_of_squares);
    }
}

// Helper function to compute the l2 norm of a vector on device
double compute_l2_norm_cuda(const double* d_vec, int size, cudaStream_t stream)
{
    // Initialize result to zero
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), stream));

    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_squared_l2_norm_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_squared_l2_norm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_vec, d_result, size
    );

    // Copy back to host
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return std::sqrt(result);
}

// CUDA kernel for computing the squared l2 norm of a vector
// Only stores blockwise result
__global__ void compute_block_squared_l2_norm_kernel(
    const double* __restrict__ vec,
    double* __restrict__ block_result,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum_of_squares = 0.0;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    for (int i = idx; i < size; i += stride)
    {
        double val = vec[i];
        sum_of_squares += val * val;
    }

    // Intra-block reduce
    // Sums up results within this block
    // Stored in the first thread of the block
    sum_of_squares = block_reduce_sum(sum_of_squares);

    // The first thread of the block adds the block's sum to the global memory
    if (tid == 0)
    {
        block_result[bid] = sum_of_squares;
    }
}

// Helper function to compute the l2 norm of a vector on device,
// using pinned memory containing at least 256 elements
double compute_l2_norm_cuda(const double* d_vec, double* h_pinned, int size, cudaStream_t stream)
{
    // Get device pointer to the pinned memory
    double* d_block_result;
    cudaHostGetDevicePointer(&d_block_result, h_pinned, 0);

    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_squared_l2_norm_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_block_squared_l2_norm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_vec, d_block_result, size
    );
    cudaStreamSynchronize(stream);

    // Compute final result on CPU
    double result = 0.0;
    for (int i = 0; i < numBlocks; i++)
    {
        result += h_pinned[i];
    }

    return std::sqrt(result);
}

// CUDA kernel to compute elementwise logarithm using grid-stride loop
__global__ void compute_log_vector_kernel(
    const double* __restrict__ x,
    double* __restrict__ logx,
    int size
)
{
    // Indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    // Even if size > blockDim * gridDim, the loop will cover all elements
    for (int i = idx; i < size; i += stride)
    {
        logx[i] = log(x[i]);
    }
}

// Helper function to compute elementwise logarithm of a vector on device
void compute_log_vector_cuda(const double* d_x, double* d_logx, int size, cudaStream_t stream)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_log_vector_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_log_vector_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_x, d_logx, size
    );
}

// CUDA kernel for computing x <- a * x
__global__ void compute_scaling_inplace_kernel(
    double a,
    double* __restrict__ x,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    // Even if size > blockDim * gridDim, the loop will cover all elements
    for (int i = idx; i < size; i += stride)
    {
        x[i] *= a;
    }
}

// Helper function to compute x <- a * x on device
void compute_scaling_inplace_cuda(double a, double* d_x, int size, cudaStream_t stream)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_scaling_inplace_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_scaling_inplace_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        a, d_x, size
    );
}

// CUDA kernel for computing z = a * x + y
__global__ void compute_axpy_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double a,
    double* __restrict__ z,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    // Even if size > blockDim * gridDim, the loop will cover all elements
    for (int i = idx; i < size; i += stride)
    {
        z[i] = a * x[i] + y[i];
    }
}

// Helper function to compute z = a * x + y on device
void compute_axpy_cuda(const double* d_x, const double* d_y, double a, double* d_z, int size, cudaStream_t stream)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_axpy_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_axpy_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_x, d_y, a, d_z, size
    );
}

// CUDA kernel for computing inner product x'y
__global__ void compute_dot_prod_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ result,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double dot_prod = 0.0;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    for (int i = idx; i < size; i += stride)
    {
        dot_prod += x[i] * y[i];
    }

    // Intra-block reduce
    // Sums up results within this block
    // Stored in the first thread of the block
    dot_prod = block_reduce_sum(dot_prod);

    // The first thread of the block adds the block's sum to the global memory
    if (tid == 0)
    {
        // result must be initialized to zero
        atomicAdd(result, dot_prod);
    }
}

// Helper function to compute inner product x'y on device
double compute_dot_prod_cuda(const double* d_x, const double* d_y, int size, cudaStream_t stream)
{
    // Initialize result to zero
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), stream));

    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_dot_prod_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_dot_prod_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_x, d_y, d_result, size
    );

    // Copy back to host
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

// CUDA kernel for computing inner product x'y
// Only stores blockwise result
__global__ void compute_block_dot_prod_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ block_result,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double dot_prod = 0.0;

    // Grid-stride loop
    // Similar to "if (idx < size)" but handles arbitrary vector size
    for (int i = idx; i < size; i += stride)
    {
        dot_prod += x[i] * y[i];
    }

    // Intra-block reduce
    // Sums up results within this block
    // Stored in the first thread of the block
    dot_prod = block_reduce_sum(dot_prod);

    // The first thread of the block adds the block's sum to the global memory
    if (tid == 0)
    {
        block_result[bid] = dot_prod;
    }
}

// Helper function to compute inner product x'y on device
// using pinned memory containing at least 256 elements
double compute_dot_prod_cuda(const double* d_x, const double* d_y, double* h_pinned, int size, cudaStream_t stream)
{
    // Get device pointer to the pinned memory
    double* d_block_result;
    cudaHostGetDevicePointer(&d_block_result, h_pinned, 0);

    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in compute_dot_prod_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);

    compute_block_dot_prod_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_x, d_y, d_block_result, size
    );
    cudaStreamSynchronize(stream);

    // Compute final result on CPU
    double result = 0.0;
    for (int i = 0; i < numBlocks; i++)
    {
        result += h_pinned[i];
    }

    return result;
}
