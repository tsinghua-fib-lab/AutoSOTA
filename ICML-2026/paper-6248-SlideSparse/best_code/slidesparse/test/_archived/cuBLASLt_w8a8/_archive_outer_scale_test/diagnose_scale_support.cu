// SPDX-License-Identifier: Apache-2.0
/**
 * 诊断脚本：测试 SM120 (Blackwell) 上的 FP8 Scale 支持情况
 * 
 * 测试矩阵：
 * 1. 无 Scale (基线)
 * 2. Tensorwide Scaling (SCALAR_32F)
 * 3. Outer Vector Scaling (OUTER_VEC_32F) - 预计 SM 9.0 only
 * 
 * 同时测试不同布局组合
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#define CUBLASLT_CHECK(status)                                           \
    do {                                                                 \
        cublasStatus_t err = status;                                     \
        if (err != CUBLAS_STATUS_SUCCESS) {                              \
            return std::string("cuBLASLt error: ") + std::to_string(err); \
        }                                                                \
    } while (0)

static cublasLtHandle_t g_handle = nullptr;

static cublasLtHandle_t get_handle() {
    if (g_handle == nullptr) {
        cublasLtCreate(&g_handle);
    }
    return g_handle;
}

/**
 * 测试特定配置是否能找到算法
 */
std::string test_config(
    int M, int N, int K,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t d_type,
    cublasOperation_t opA,
    cublasOperation_t opB,
    int scale_mode,  // 0=none, 1=tensorwide, 2=outer_vec
    bool use_col_major
) {
    cublasLtHandle_t handle = get_handle();
    
    // 计算 leading dimensions
    // col-major: lda = rows, ldb = rows, ldd = rows
    // row-major: lda = cols, ldb = cols, ldd = cols
    int64_t lda, ldb, ldd;
    cublasLtOrder_t order = use_col_major ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    
    if (use_col_major) {
        // col-major
        if (opA == CUBLAS_OP_N) {
            lda = M;  // A is M x K
        } else {
            lda = K;  // A^T is K x M, stored as K x M
        }
        if (opB == CUBLAS_OP_N) {
            ldb = K;  // B is K x N
        } else {
            ldb = N;  // B^T is N x K, stored as N x K
        }
        ldd = M;
    } else {
        // row-major (TN format: A[M,K] @ B[N,K]^T)
        lda = K;  // A is M x K
        ldb = K;  // B is N x K
        ldd = N;
    }
    
    // 创建 layout
    cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
    
    if (use_col_major) {
        // Col-major: 逻辑维度 (rows, cols) = (M, K) for A
        if (opA == CUBLAS_OP_N) {
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, M, K, lda));
        } else {
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, K, M, lda));
        }
        if (opB == CUBLAS_OP_N) {
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, K, N, ldb));
        } else {
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, N, K, ldb));
        }
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, d_type, M, N, ldd));
    } else {
        // Row-major (TN format)
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, a_type, M, K, lda));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, b_type, N, K, ldb));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, d_type, M, N, ldd));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    }
    
    // 创建 matmul desc
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // 设置 scale mode
    if (scale_mode == 1) {
        // Tensorwide: SCALAR_32F (默认值，但明确设置)
        cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    } else if (scale_mode == 2) {
        // Outer Vector
        cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    }
    
    // 算法搜索
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspaceSize = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    cublasStatus_t algStatus = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
        preference, 1, &heuristicResult, &returnedResults);
    
    // 清理
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutD);
    
    if (algStatus != CUBLAS_STATUS_SUCCESS) {
        return std::string("FAIL: algo search error ") + std::to_string(algStatus);
    }
    if (returnedResults == 0) {
        return "FAIL: no algorithm found";
    }
    
    return "OK";
}

/**
 * 运行诊断测试
 */
std::vector<std::tuple<std::string, std::string>> run_diagnostics() {
    std::vector<std::tuple<std::string, std::string>> results;
    
    // 获取 GPU 信息
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    char gpu_info[256];
    snprintf(gpu_info, sizeof(gpu_info), "GPU: %s, SM %d.%d", 
             prop.name, prop.major, prop.minor);
    results.push_back({gpu_info, ""});
    
    int M = 64, N = 128, K = 256;
    
    // 测试配置列表
    struct TestCase {
        const char* name;
        cudaDataType_t a_type;
        cudaDataType_t b_type;
        cudaDataType_t d_type;
        cublasOperation_t opA;
        cublasOperation_t opB;
        int scale_mode;
        bool col_major;
    };
    
    std::vector<TestCase> tests = {
        // 基线测试：无 scale
        {"FP8->BF16 TN row-major, no scale", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_T, 0, false},
        {"FP8->FP32 TN row-major, no scale", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_T, 0, false},
        
        // Col-major 测试
        {"FP8->BF16 NN col-major, no scale", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_N, 0, true},
        {"FP8->FP32 NN col-major, no scale", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_N, 0, true},
         
        // Tensorwide Scale
        {"FP8->BF16 TN row-major, tensorwide", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_T, 1, false},
        {"FP8->FP32 TN row-major, tensorwide", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_T, 1, false},
        {"FP8->BF16 NN col-major, tensorwide", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_N, 1, true},
        {"FP8->FP32 NN col-major, tensorwide", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_N, 1, true},
         
        // Outer Vector Scale (预计在 SM 12.0 失败)
        {"FP8->BF16 TN row-major, outer_vec", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_T, 2, false},
        {"FP8->FP32 TN row-major, outer_vec", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_T, 2, false},
        {"FP8->BF16 NN col-major, outer_vec", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 
         CUBLAS_OP_N, CUBLAS_OP_N, 2, true},
        {"FP8->FP32 NN col-major, outer_vec", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, 
         CUBLAS_OP_N, CUBLAS_OP_N, 2, true},
    };
    
    for (const auto& tc : tests) {
        std::string result = test_config(
            M, N, K, tc.a_type, tc.b_type, tc.d_type,
            tc.opA, tc.opB, tc.scale_mode, tc.col_major);
        results.push_back({tc.name, result});
    }
    
    return results;
}

/**
 * 获取 GPU SM 版本
 */
std::string get_gpu_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    char buf[256];
    snprintf(buf, sizeof(buf), "%s (SM %d.%d)", prop.name, prop.major, prop.minor);
    return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_diagnostics", &run_diagnostics, "Run scale mode diagnostics");
    m.def("get_gpu_info", &get_gpu_info, "Get GPU info");
}
