// SPDX-License-Identifier: Apache-2.0
/**
 * 对比测试：
 * 1. Tensorwide Scaling (标量 scale) - 应该在所有 FP8 GPU 上工作
 * 2. Outer Vector Scaling (向量 scale) - 仅在 SM 9.0 (Hopper) 上工作
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <cstdio>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CUBLASLT_CHECK(status)                                           \
    do {                                                                 \
        cublasStatus_t err = status;                                     \
        if (err != CUBLAS_STATUS_SUCCESS) {                              \
            printf("cuBLASLt error %d at %s:%d\n", err, __FILE__, __LINE__); \
            TORCH_CHECK(false, "cuBLASLt error: ", err);                 \
        }                                                                \
    } while (0)

static cublasLtHandle_t g_handle = nullptr;
static std::mutex g_mutex;

static cublasLtHandle_t get_handle() {
    if (g_handle == nullptr) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_handle == nullptr) {
            CUBLASLT_CHECK(cublasLtCreate(&g_handle));
        }
    }
    return g_handle;
}

/**
 * 方案 1: Tensorwide Scaling (标量 scale)
 * 
 * D = alpha * (A @ B^T)
 * 然后手动应用 per-token/per-channel scale
 */
torch::Tensor fp8_gemm_tensorwide(
    torch::Tensor A,      // [M, K] FP8
    torch::Tensor B,      // [N, K] FP8
    float scale_a,        // 标量
    float scale_b,        // 标量
    torch::Dtype out_dtype
) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch");
    
    printf("=== Tensorwide Scaling (scalar) ===\n");
    printf("M=%ld, N=%ld, K=%ld, scale_a=%.4f, scale_b=%.4f\n", M, N, K, scale_a, scale_b);
    
    auto options = torch::TensorOptions()
        .dtype(out_dtype)
        .device(A.device());
    torch::Tensor D = torch::empty({M, N}, options);
    
    cudaDataType_t a_type = CUDA_R_8F_E4M3;
    cudaDataType_t b_type = CUDA_R_8F_E4M3;
    cudaDataType_t d_type;
    if (out_dtype == torch::kBFloat16) {
        d_type = CUDA_R_16BF;
    } else if (out_dtype == torch::kFloat16) {
        d_type = CUDA_R_16F;
    } else {
        d_type = CUDA_R_32F;
    }
    
    cublasLtHandle_t handle = get_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 矩阵 layout
    cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, N, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, d_type, M, N, N));
    
    // Matmul 描述符
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // 默认使用 SCALAR_32F (Tensorwide Scaling)
    // 不需要额外设置，这是默认值
    
    // 算法搜索
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspaceSize = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    printf("Searching for algorithm...\n");
    cublasStatus_t algStatus = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
        preference, 1, &heuristicResult, &returnedResults);
    
    if (algStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        printf("Algorithm search failed! status=%d\n", algStatus);
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutD);
        TORCH_CHECK(false, "No algorithm found");
    }
    
    printf("Found algorithm!\n");
    
    void* workspace = nullptr;
    if (heuristicResult.workspaceSize > 0) {
        cudaMalloc(&workspace, heuristicResult.workspaceSize);
    }
    
    // alpha = scale_a * scale_b
    float alpha = scale_a * scale_b;
    float beta = 0.0f;
    
    printf("Executing matmul with alpha=%.4f...\n", alpha);
    cublasStatus_t status = cublasLtMatmul(
        handle, matmulDesc, &alpha,
        A.data_ptr(), layoutA,
        B.data_ptr(), layoutB,
        &beta,
        D.data_ptr(), layoutD,
        D.data_ptr(), layoutD,
        &heuristicResult.algo,
        workspace, heuristicResult.workspaceSize,
        stream
    );
    
    cudaStreamSynchronize(stream);
    
    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutD);
    
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", status);
    
    printf("Success!\n");
    return D;
}

/**
 * 方案 2: Outer Vector Scaling (向量 scale)
 * 只在 Hopper (SM 9.0) 上支持
 */
torch::Tensor fp8_gemm_outer_vec(
    torch::Tensor A,      // [M, K] FP8
    torch::Tensor B,      // [N, K] FP8
    torch::Tensor scale_A,// [M] FP32
    torch::Tensor scale_B,// [N] FP32
    torch::Dtype out_dtype
) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CUDA(scale_A);
    CHECK_CUDA(scale_B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_CONTIGUOUS(scale_A);
    CHECK_CONTIGUOUS(scale_B);
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    printf("=== Outer Vector Scaling ===\n");
    printf("M=%ld, N=%ld, K=%ld\n", M, N, K);
    
    auto options = torch::TensorOptions()
        .dtype(out_dtype)
        .device(A.device());
    torch::Tensor D = torch::empty({M, N}, options);
    
    cudaDataType_t a_type = CUDA_R_8F_E4M3;
    cudaDataType_t b_type = CUDA_R_8F_E4M3;
    cudaDataType_t d_type;
    if (out_dtype == torch::kBFloat16) {
        d_type = CUDA_R_16BF;
    } else if (out_dtype == torch::kFloat16) {
        d_type = CUDA_R_16F;
    } else {
        d_type = CUDA_R_32F;
    }
    
    cublasLtHandle_t handle = get_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, N, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, d_type, M, N, N));
    
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // 设置 Outer Vector Scaling
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    
    const float* scale_A_ptr = scale_A.data_ptr<float>();
    const float* scale_B_ptr = scale_B.data_ptr<float>();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_A_ptr, sizeof(scale_A_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_B_ptr, sizeof(scale_B_ptr)));
    
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspaceSize = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    printf("Searching for algorithm...\n");
    cublasStatus_t algStatus = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
        preference, 1, &heuristicResult, &returnedResults);
    
    // 清理
    auto cleanup = [&]() {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutD);
    };
    
    if (algStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        printf("Algorithm search failed! status=%d (Outer Vector Scaling not supported)\n", algStatus);
        cleanup();
        TORCH_CHECK(false, "Outer Vector Scaling not supported on this GPU");
    }
    
    printf("Found algorithm!\n");
    
    void* workspace = nullptr;
    if (heuristicResult.workspaceSize > 0) {
        cudaMalloc(&workspace, heuristicResult.workspaceSize);
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    printf("Executing matmul...\n");
    cublasStatus_t status = cublasLtMatmul(
        handle, matmulDesc, &alpha,
        A.data_ptr(), layoutA,
        B.data_ptr(), layoutB,
        &beta,
        D.data_ptr(), layoutD,
        D.data_ptr(), layoutD,
        &heuristicResult.algo,
        workspace, heuristicResult.workspaceSize,
        stream
    );
    
    cudaStreamSynchronize(stream);
    
    if (workspace) cudaFree(workspace);
    cleanup();
    
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", status);
    
    printf("Success!\n");
    return D;
}

/**
 * 检查 GPU 是否支持 Outer Vector Scaling
 */
bool check_outer_vec_support() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int sm_version = prop.major * 10 + prop.minor;
    printf("GPU: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);
    
    // 根据文档，Outer Vector Scaling 需要 SM 9.0 (Hopper)
    // 但 SM 12.0 (Blackwell) 似乎不支持
    bool supported = (sm_version == 90);  // 只有 Hopper
    
    printf("Outer Vector Scaling supported: %s\n", supported ? "YES" : "NO");
    return supported;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm_tensorwide", &fp8_gemm_tensorwide,
          "FP8 GEMM with Tensorwide Scaling (scalar scale)");
    m.def("fp8_gemm_outer_vec", &fp8_gemm_outer_vec,
          "FP8 GEMM with Outer Vector Scaling (only Hopper)");
    m.def("check_outer_vec_support", &check_outer_vec_support,
          "Check if GPU supports Outer Vector Scaling");
}
