#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

// copied from vllm/csrc/layernorm_kernels.cu
namespace {

struct SumOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const {
    return a + b;
  }
};

// Template to handle different data types (float, half, bfloat16)
template <typename scalar_t>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
                                const int64_t input_stride, const scalar_t* __restrict__ weight, const float epsilon,
                                const int hidden_size) {
  __shared__ float s_variance;

  float variance = 0.0f;
  // 1. compute RMS (Root Mean of Squares)
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    variance += x * x;
  }

  // Use a fixed block size for the CUB reduction template argument
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, SumOp{});

  if (threadIdx.x == 0) {
    s_variance = 1.0f / sqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // 2. multiply x * w / RMS
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    out[blockIdx.x * hidden_size + idx] = static_cast<scalar_t>(x * s_variance) * weight[idx];
  }
}

// Fused kernel: first adds residual, then applies RMSNorm in-place
template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(scalar_t* __restrict__ input, const int64_t input_stride,
                                          scalar_t* __restrict__ residual, const scalar_t* __restrict__ weight,
                                          const float epsilon, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  // 1. Compute variance on float-promoted sum
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    float r = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
    float sum = x + r;
    variance += sum * sum;
  }

  // Use a fixed block size for the CUB reduction template argument
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, SumOp{});

  if (threadIdx.x == 0) {
    s_variance = 1.0f / sqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // 2. Update residual and compute output
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    float r = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
    float sum = x + r;

    // Update residual with cast-back sum
    residual[blockIdx.x * hidden_size + idx] = static_cast<scalar_t>(sum);

    // Compute output using float sum and cast back
    input[blockIdx.x * input_stride + idx] = static_cast<scalar_t>(sum * s_variance) * weight[idx];
  }
}

} // anonymous namespace

// Launcher for standard RMSNorm
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  // Non-contiguous inputs are handled by passing the stride to the kernel
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);

  dim3 grid(num_tokens);
  // CUB reduction requires the block size to be known at compile time,
  // so we use a fixed max size here.
  dim3 block(std::min(hidden_size, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Use the AT_DISPATCH macro that includes bfloat16 support
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "rms_norm_kernel", [&] {
    // Launch the templated kernel for the dispatched scalar_t type
    rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                                          input_stride, weight.data_ptr<scalar_t>(),
                                                          static_cast<float>(epsilon), hidden_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launcher for Fused Add + RMSNorm
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "Residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  // Non-contiguous inputs are handled by passing the stride
  TORCH_CHECK(residual.is_contiguous(), "Residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);

  dim3 grid(num_tokens);
  // CUB reduction requires the block size to be known at compile time
  dim3 block(std::min(hidden_size, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Use the AT_DISPATCH macro that includes bfloat16 support
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "fused_add_rms_norm_kernel", [&] {
    // Launch the templated kernel for the dispatched scalar_t type
    fused_add_rms_norm_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(), input_stride, residual.data_ptr<scalar_t>(),
                                     weight.data_ptr<scalar_t>(), static_cast<float>(epsilon), hidden_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
