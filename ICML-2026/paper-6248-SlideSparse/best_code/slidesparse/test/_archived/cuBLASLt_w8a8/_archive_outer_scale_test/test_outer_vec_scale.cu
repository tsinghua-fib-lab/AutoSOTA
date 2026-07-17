// SPDX-License-Identifier: Apache-2.0
/**
 * 最简洁的 cuBLASLt FP8 + Outer Vector Scaling 测试
 * 
 * 目的: 验证 Outer Vector Scaling 在特定 GPU 上是否可用
 * 
 * 计算: D[M,N] = scale_A[M] * scale_B[N] * (A[M,K] @ B[K,N]^T)
 * 
 * 布局: TN 格式 (A: row-major, B: row-major, D: row-major)
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <cstdio>

// 错误检查宏
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

// 全局 handle
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
 * 简单的 FP8 GEMM + Outer Vector Scaling
 * 
 * D[M,N] = scale_A[M] * scale_B[N] * (A[M,K] @ B[N,K]^T)
 * 
 * 参数:
 *   A: [M, K] FP8 E4M3, row-major
 *   B: [N, K] FP8 E4M3, row-major (会被转置)
 *   scale_A: [M] FP32
 *   scale_B: [N] FP32
 *   out_dtype: 输出类型 (BF16/FP16/FP32)
 */
torch::Tensor simple_fp8_gemm_outer_vec(
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
    
    // 维度
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch");
    TORCH_CHECK(scale_A.numel() == M, "scale_A size mismatch");
    TORCH_CHECK(scale_B.numel() == N, "scale_B size mismatch");
    
    printf("=== FP8 GEMM Outer Vector Scaling ===\n");
    printf("M=%ld, N=%ld, K=%ld\n", M, N, K);
    
    // 创建输出
    auto options = torch::TensorOptions()
        .dtype(out_dtype)
        .device(A.device());
    torch::Tensor D = torch::empty({M, N}, options);
    
    // cuBLASLt 数据类型
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
    
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    
    cublasLtHandle_t handle = get_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // ========== 创建矩阵描述符 ==========
    // TN 格式: op(A) = A, op(B) = B^T
    // A: [M, K] row-major -> lda = K
    // B: [N, K] row-major, transposed -> ldb = K
    // D: [M, N] row-major -> ldd = N
    
    cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
    
    // A: [M, K], row-major, no transpose
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, M, K, K));
    
    // B: [N, K], row-major, will be transposed
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, N, K, K));
    
    // D: [M, N], row-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, d_type, M, N, N));
    
    // ========== 创建 Matmul 描述符 ==========
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, compute_type, scale_type));
    
    // 设置转置: A 不转置, B 转置
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // ========== 设置 Outer Vector Scaling ==========
    // scale_A 对应矩阵 A 的行 (M 维度)
    // scale_B 对应矩阵 B 的列 (N 维度，因为 B 被转置)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    
    printf("Setting Outer Vector Scaling mode...\n");
    
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    
    // 设置 scale 指针
    const float* scale_A_ptr = scale_A.data_ptr<float>();
    const float* scale_B_ptr = scale_B.data_ptr<float>();
    
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_A_ptr, sizeof(scale_A_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_B_ptr, sizeof(scale_B_ptr)));
    
    // ========== 算法搜索 ==========
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    printf("Searching for algorithm...\n");
    cublasStatus_t algStatus = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
        preference, 1, &heuristicResult, &returnedResults);
    
    if (algStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        printf("Algorithm search failed! status=%d, returnedResults=%d\n", algStatus, returnedResults);
        
        // 清理
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutD);
        
        TORCH_CHECK(false, "No suitable algorithm found for FP8 Outer Vector Scaling");
    }
    
    printf("Found algorithm! workspaceSize=%zu\n", heuristicResult.workspaceSize);
    
    // ========== 分配 workspace ==========
    void* workspace = nullptr;
    if (heuristicResult.workspaceSize > 0) {
        cudaMalloc(&workspace, heuristicResult.workspaceSize);
    }
    
    // ========== 执行 matmul ==========
    float alpha = 1.0f;
    float beta = 0.0f;
    
    printf("Executing matmul...\n");
    cublasStatus_t status = cublasLtMatmul(
        handle,
        matmulDesc,
        &alpha,
        A.data_ptr(),
        layoutA,
        B.data_ptr(),
        layoutB,
        &beta,
        D.data_ptr(),
        layoutD,
        D.data_ptr(),
        layoutD,
        &heuristicResult.algo,
        workspace,
        heuristicResult.workspaceSize,
        stream
    );
    
    // 同步检查错误
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    
    // 清理
    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutD);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasLtMatmul failed with status %d\n", status);
        TORCH_CHECK(false, "cublasLtMatmul failed: ", status);
    }
    
    if (syncErr != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(syncErr));
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(syncErr));
    }
    
    printf("Success!\n");
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simple_fp8_gemm_outer_vec", &simple_fp8_gemm_outer_vec,
          "Simple FP8 GEMM with Outer Vector Scaling");
}
