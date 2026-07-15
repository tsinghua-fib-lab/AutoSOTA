#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Utility functions
#include "utils.h"
#include "sinkhorn.h"

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

// CUDA kernel for computing optimal beta given alpha
__global__ void optimal_beta_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ logb,
    double* __restrict__ beta,
    double reg,
    int n, int m
)
{
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // Grid stride (number of blocks)
    int stride = gridDim.x;

    // Grid-stride loop instead of `if (j < m)`
    // Each block processes multiple columns
    for (int j = blockIdx.x; j < m; j += stride)
    {
        // First pass: find maximum in column j
        double local_max = -INFINITY;

        // Each thread processes multiple elements of the column
        for (int i = tid; i < n; i += block_size)
        {
            double D_ij = (alpha[i] - M[i * m + j]) / reg;
            local_max = max(local_max, D_ij);
        }

        // Find global maximum across all threads in the block
        shared_data[tid] = local_max;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }

        double global_max = shared_data[0];
        __syncthreads();

        // Second pass: compute sum of exp(D_ij - global_max)
        double local_sum = 0.0;

        // Each thread processes multiple elements again
        for (int i = tid; i < n; i += block_size)
        {
            double D_ij = (alpha[i] - M[i * m + j]) / reg;
            local_sum += exp(D_ij - global_max);
        }

        // Reduce sum across all threads in the block
        shared_data[tid] = local_sum;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        double global_sum = shared_data[0];

        // First thread writes the result
        if (tid == 0)
        {
            double log_sum = global_max + log(global_sum);
            beta[j] = reg * (logb[j] - log_sum);
        }
    }
}

// CUDA kernel for computing optimal alpha given beta
__global__ void optimal_alpha_kernel(
    const double* __restrict__ M,
    const double* __restrict__ beta,
    const double* __restrict__ loga,
    double* __restrict__ alpha,
    double reg,
    int n, int m
)
{
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // Grid stride (number of blocks)
    int stride = gridDim.x;

    // Grid-stride loop instead of `if (i < n)`
    // Each block processes multiple rows
    for (int i = blockIdx.x; i < n; i += stride)
    {
        // First pass: find maximum in row i
        double local_max = -INFINITY;

        // Each thread processes multiple elements of the row
        for (int j = tid; j < m; j += block_size)
        {
            double D_ij = (beta[j] - M[i * m + j]) / reg;
            local_max = max(local_max, D_ij);
        }

        // Find global maximum across all threads in the block
        shared_data[tid] = local_max;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }

        double global_max = shared_data[0];
        __syncthreads();

        // Second pass: compute sum of exp(D_ij - global_max)
        double local_sum = 0.0;

        // Each thread processes multiple elements again
        for (int j = tid; j < m; j += block_size)
        {
            double D_ij = (beta[j] - M[i * m + j]) / reg;
            local_sum += exp(D_ij - global_max);
        }

        // Reduce sum across all threads in the block
        shared_data[tid] = local_sum;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        double global_sum = shared_data[0];

        // First thread writes the result
        if (tid == 0)
        {
            double log_sum = global_max + log(global_sum);
            alpha[i] = reg * (loga[i] - log_sum);
        }
    }
}

// Helper function to compute optimal beta given alpha
void compute_optimal_beta(
    const double* d_M, const double* d_alpha, const double* d_logb,
    double* d_beta, double reg, int n, int m,
    cudaStream_t stream
)
{
    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_DIM);
    // Use heuristics to set the total number of blocks
    // Target total number of blocks
    int target_num_blocks = heuristic_num_blocks();
    // Limit number of blocks to target_num_blocks to avoid excessive kernel launch overhead
    // Each block will process multiple rows/columns via the
    // grid-stride loop in optimal_alpha_kernel()/optimal_beta_kernel()
    // dim3 numBlocks_alpha(std::min(n, target_num_blocks));
    dim3 numBlocks_beta(std::min(m, target_num_blocks));

    // Calculate shared memory size for each kernel (for reduction operations)
    // size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);  // For reduction in optimal_alpha_kernel
    size_t sharedMemory_beta = threadsPerBlock.x * sizeof(double);   // For reduction in optimal_beta_kernel

    // Optimal beta given alpha
    optimal_beta_kernel<<<numBlocks_beta, threadsPerBlock, sharedMemory_beta, stream>>>(
        d_M, d_alpha, d_logb, d_beta, reg, n, m
    );
}

// Helper function to compute optimal alpha given beta
void compute_optimal_alpha(
    const double* d_M, const double* d_beta, const double* d_loga,
    double* d_alpha, double reg, int n, int m,
    cudaStream_t stream
)
{
    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_DIM);
    // Use heuristics to set the total number of blocks
    // Target total number of blocks
    int target_num_blocks = heuristic_num_blocks();
    // Limit number of blocks to target_num_blocks to avoid excessive kernel launch overhead
    // Each block will process multiple rows/columns via the
    // grid-stride loop in optimal_alpha_kernel()/optimal_beta_kernel()
    dim3 numBlocks_alpha(std::min(n, target_num_blocks));
    // dim3 numBlocks_beta(std::min(m, target_num_blocks));

    // Calculate shared memory size for each kernel (for reduction operations)
    size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);  // For reduction in optimal_alpha_kernel
    // size_t sharedMemory_beta = threadsPerBlock.x * sizeof(double);   // For reduction in optimal_beta_kernel

    // Optimal alpha given beta
    optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha, stream>>>(
        d_M, d_beta, d_loga, d_alpha, reg, n, m
    );
}

// Functor to compute squared difference with a scalar shift.
// Calculates: ((val1 - val2) + shift)^2
struct SquaredDiffWithShift
{
    const double m_shift;

    __host__ __device__
    SquaredDiffWithShift(double s):
        m_shift(s)
    {}

    __host__ __device__
    double operator()(const thrust::tuple<double, double>& t) const
    {
        double val1 = thrust::get<0>(t);
        double val2 = thrust::get<1>(t);
        double diff = (val1 - val2) + m_shift;
        return diff * diff;
    }
};

// Compute difference between (alpha, beta) and (alpha_prev, beta_prev)
// Note that (alpha, beta) is equivalent to (alpha + c * 1, beta - c * 1) for any c,
// so we need to first standardize vectors before comparing
// Let (alpha', beta') = (alpha + c * 1, beta - c * 1) such that sum(alpha') = sum(beta'),
// then c = (sum(beta) - sum(alpha)) / (m + n)
// Similarly, let (alpha_prev', beta_prev') = (alpha_prev + d * 1, beta_prev - d * 1)
// with d = (sum(beta_prev) - sum(alpha_prev)) / (m + n)
// Define K = c - d, then alpha' - alpha_prev' = alpha - alpha_prev + K * 1
//                        beta' - beta_prev' = beta - beta_prev - K * 1
double compute_iter_difference(
    const double* d_alpha, const double* d_beta,
    const double* d_alpha_prev, const double* d_beta_prev,
    int n, int m
)
{
    // Wrap raw pointers with thrust device pointers
    thrust::device_ptr<const double> d_alpha_ptr(d_alpha);
    thrust::device_ptr<const double> d_beta_ptr(d_beta);
    thrust::device_ptr<const double> d_alpha_prev_ptr(d_alpha_prev);
    thrust::device_ptr<const double> d_beta_prev_ptr(d_beta_prev);

    // Compute sums
    double sum_alpha = thrust::reduce(d_alpha_ptr, d_alpha_ptr + n, 0.0, thrust::plus<double>());
    double sum_beta  = thrust::reduce(d_beta_ptr, d_beta_ptr + m, 0.0, thrust::plus<double>());
    double sum_alpha_prev = thrust::reduce(d_alpha_prev_ptr, d_alpha_prev_ptr + n, 0.0, thrust::plus<double>());
    double sum_beta_prev  = thrust::reduce(d_beta_prev_ptr, d_beta_prev_ptr + m, 0.0, thrust::plus<double>());

    // Compupte shifts
    double c = (sum_beta - sum_alpha) / (n + m);
    double d = (sum_beta_prev - sum_alpha_prev) / (n + m);
    double K = c - d;

    // Compute Euclidean distances
    // (alpha[i] - alpha_prev[i] + K)^2
    auto alpha_iter_begin = thrust::make_zip_iterator(thrust::make_tuple(d_alpha_ptr, d_alpha_prev_ptr));
    auto alpha_iter_end = thrust::make_zip_iterator(thrust::make_tuple(d_alpha_ptr + n, d_alpha_prev_ptr + n));
    double alpha_diff = thrust::transform_reduce(
        alpha_iter_begin, 
        alpha_iter_end,
        SquaredDiffWithShift(K),
        0.0,
        thrust::plus<double>()
    );
    // (beta[j] - beta_prev[j] - K)^2
    auto beta_iter_begin = thrust::make_zip_iterator(thrust::make_tuple(d_beta_ptr, d_beta_prev_ptr));
    auto beta_iter_end = thrust::make_zip_iterator(thrust::make_tuple(d_beta_ptr + m, d_beta_prev_ptr + m));
    double beta_diff = thrust::transform_reduce(
        beta_iter_begin,
        beta_iter_end,
        SquaredDiffWithShift(-K),
        0.0,
        thrust::plus<double>()
    );

    return std::sqrt(alpha_diff + beta_diff);
}

// CUDA kernel for computing marginal a (row sums of transport plan)
__global__ void compute_marginal_a_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ marginal_a,
    double reg,
    int n, int m
)
{
    // Grid-stride loop instead of `if (i < n)`
    // Each thread processes multiple rows
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        double sum = 0.0;
        const double alpha_i = alpha[i];
        const double* M_i = M + i * m;
        for (int j = 0; j < m; j++)
        {
            // exp((alpha_i + beta_j - M_ij) / reg)
            sum += exp((alpha_i + beta[j] - M_i[j]) / reg);
        }
        marginal_a[i] = sum;
    }
}

// CUDA kernel for computing marginal b (column sums of transport plan)
__global__ void compute_marginal_b_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ marginal_b,
    double reg,
    int n, int m
)
{
    // Grid-stride loop instead of `if (j < m)`
    // Each thread processes multiple columns
    int stride = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < m; j += stride)
    {
        double sum = 0.0;
        const double beta_j = beta[j];
        const double* Mj = M + j;
        for (int i = 0; i < n; i++, Mj += m)
        {
            // exp((alpha_i + beta_j - M_ij) / reg)
            sum += exp((alpha[i] + beta_j - *Mj) / reg);
        }
        marginal_b[j] = sum;
    }
}

// CUDA kernel for computing final transport plan P
__global__ void compute_transport_plan_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ P,
    double reg,
    int n, int m
)
{
    // Grid-stride loop instead of `if (i < n && j < m)` for 2D grid
    int stride_i = gridDim.y * blockDim.y;
    int stride_j = gridDim.x * blockDim.x;

    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += stride_i)
    {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < m; j += stride_j)
        {
            P[i * m + j] = exp((alpha[i] + beta[j] - M[i * m + j]) / reg);
        }
    }
}

// Helper function to compute final transport plan P
void compute_transport_plan(
    const double* d_M, const double* d_alpha, const double* d_beta,
    double* d_P, double reg, int n, int m
)
{
    // Compute final transport plan
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    int gridDim_x = (m + blockDim.x - 1) / blockDim.x;
    int gridDim_y = (n + blockDim.y - 1) / blockDim.y;
    // Limit number of blocks
    // The grid-stride loop in compute_transport_plan_kernel()
    // will handle larger sizes
    gridDim_x = std::min(gridDim_x, 64);
    gridDim_y = std::min(gridDim_y, 64);
    dim3 gridDim(gridDim_x, gridDim_y);
    compute_transport_plan_kernel<<<gridDim, blockDim>>>(
        d_M, d_alpha, d_beta, d_P, reg, n, m
    );
}

// CUDA implementation of BCD algorithm for entropic-regularized OT
// input_on_device = true means that M, a, b, and x0 (if not nullptr) are device pointers
// output_on_device = true means that P and dual (if not nullptr) are device pointers
void cuda_sinkhorn_bcd(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    const double* x0, double* dual,
    bool input_on_device, bool output_on_device
)
{
    // Pointers to input data on the device
    const double *d_M, *d_a, *d_b;
    // Pointer to output data
    double* d_P;

    // If M, a, and b are already on the device,
    // then directly assign them to d_* variables
    // Otherwise, allocate device memory and copy from host pointers
    double *d_M_storage = nullptr, *d_a_storage = nullptr, *d_b_storage = nullptr;
    if (input_on_device)
    {
        d_M = M;
        d_a = a;
        d_b = b;
    }
    else
    {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_M_storage, n * m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_a_storage, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b_storage, m * sizeof(double)));
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_M_storage, M, n * m * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_storage, a, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_storage, b, m * sizeof(double), cudaMemcpyHostToDevice));
        // Set input data pointers
        d_M = d_M_storage;
        d_a = d_a_storage;
        d_b = d_b_storage;
    }

    // If P is already on the device, directly set d_P = P
    // Otherwise, allocate device memory
    double *d_P_storage = nullptr;
    if (output_on_device)
    {
        d_P = P;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_P_storage, n * m * sizeof(double)));
        d_P = d_P_storage;
    }

    // Allocate device memory for intermediate vectors
    double *d_alpha, *d_beta, *d_loga, *d_logb;
    double *d_alpha_prev, *d_beta_prev, *d_marginal;
    CUDA_CHECK(cudaMalloc(&d_alpha, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loga, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_logb, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha_prev, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta_prev, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_marginal, max(n, m) * sizeof(double)));

    // Compute log vectors on device
    compute_log_vector_cuda(d_a, d_loga, n);
    compute_log_vector_cuda(d_b, d_logb, m);

    // Initialize alpha and beta
    if (x0 != nullptr)
    {
        // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
        CUDA_CHECK(cudaMemcpy(
            d_alpha, x0, n * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_beta, x0 + n, m * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));
    }
    else
    {
        // Initialize to zero if no initial values provided
        CUDA_CHECK(cudaMemset(d_alpha, 0, n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_beta, 0, m * sizeof(double)));
    }
    CUDA_CHECK(cudaMemset(d_alpha_prev, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_beta_prev, 0, m * sizeof(double)));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_DIM);
    // Use heuristics to set the total number of blocks
    // Target total number of blocks
    int target_num_blocks = heuristic_num_blocks();
    // Limit number of blocks to target_num_blocks to avoid excessive kernel launch overhead
    // Each block will process multiple rows/columns via the
    // grid-stride loop in optimal_alpha_kernel()/optimal_beta_kernel()
    dim3 numBlocks_alpha(std::min(n, target_num_blocks));
    dim3 numBlocks_beta(std::min(m, target_num_blocks));

    // Calculate shared memory size for each kernel (for reduction operations)
    size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);  // For reduction in optimal_alpha_kernel
    size_t sharedMemory_beta = threadsPerBlock.x * sizeof(double);   // For reduction in optimal_beta_kernel

    // BCD iterations with convergence checking
    // First use iterates difference error, and then switch to marginal error
    bool use_marginal_error = false;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Store previous values for error checking
        CUDA_CHECK(cudaMemcpy(d_alpha_prev, d_alpha, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta_prev, d_beta, m * sizeof(double), cudaMemcpyDeviceToDevice));

        // Optimal alpha given beta
        optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha>>>(
            d_M, d_beta, d_loga, d_alpha, reg, n, m
        );

        // Optimal beta given alpha
        optimal_beta_kernel<<<numBlocks_beta, threadsPerBlock, sharedMemory_beta>>>(
            d_M, d_alpha, d_logb, d_beta, reg, n, m
        );

        // Check convergence using dual variable differences -- easier to compute
        if (!use_marginal_error)
        {
            // double alpha_diff = compute_l2_distance_cuda(d_alpha, d_alpha_prev, n);
            // double beta_diff = compute_l2_distance_cuda(d_beta, d_beta_prev, m);
            // double diff_error = std::hypot(alpha_diff, beta_diff);

            double diff_error = compute_iter_difference(
                d_alpha, d_beta, d_alpha_prev, d_beta_prev, n, m
            );

            if (diff_error < tol * 10)
            {
                use_marginal_error = true;
            }
        }
        else
        {
            // Use marginal error
            // After computing optimal beta, b_tilde=colsum(P) is always equal to b
            // So we use ||a_tilde - a|| as the convergence criterion,
            // where a_tilde=rowsum(P)
            // Compute a_tilde based on current alpha and beta
            int numBlocks_marginal_a = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
            // Limit number of blocks to 256
            // The grid-stride loop in compute_marginal_a_kernel() will handle larger sizes
            numBlocks_marginal_a = std::min(numBlocks_marginal_a, 256);
            compute_marginal_a_kernel<<<numBlocks_marginal_a, threadsPerBlock>>>(
                d_M, d_alpha, d_beta, d_marginal, reg, n, m
            );

            double marginal_error = compute_l2_distance_cuda(d_marginal, d_a, n);

            if (marginal_error < tol)
            {
                *niter = iter + 1;
                break;
            }
        }

        *niter = iter + 1;
    }

    // Compute final transport plan
    compute_transport_plan(d_M, d_alpha, d_beta, d_P, reg, n, m);

    // Copy result back to host if needed
    if (!output_on_device)
    {
        CUDA_CHECK(cudaMemcpy(P, d_P, n * m * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Copy dual variables back to host/device if requested
    if (dual != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(
            dual, d_alpha, n * sizeof(double),
            output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(
            dual + n, d_beta, m * sizeof(double),
            output_on_device ? cudaMemcpyDeviceToDevice: cudaMemcpyDeviceToHost
        ));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_loga));
    CUDA_CHECK(cudaFree(d_logb));
    CUDA_CHECK(cudaFree(d_alpha_prev));
    CUDA_CHECK(cudaFree(d_beta_prev));
    CUDA_CHECK(cudaFree(d_marginal));
    if (!input_on_device)
    {
        CUDA_CHECK(cudaFree(d_M_storage));
        CUDA_CHECK(cudaFree(d_a_storage));
        CUDA_CHECK(cudaFree(d_b_storage));
    }
    if (!output_on_device)
    {
        CUDA_CHECK(cudaFree(d_P_storage));
    }
}
