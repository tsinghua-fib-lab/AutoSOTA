// cuBLASLt FP8 GEMM Benchmark Kernel
// 固定 Layout: TN + CC + Col (PyTorch 行主序输入，cuBLASLt 列主序处理)
// C[N,M]_col = W[N,K]^T_col * A[K,M]_col
// 输入: FP8E4M3, 计算: FP32, 输出: BF16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <vector>
#include <sstream>

namespace py = pybind11;

// ===== 错误检查宏 =====
#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CHECK_CUBLAS(expr) do { \
    cublasStatus_t st = (expr); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLASLt error: ") + cublasLtGetStatusString(st)); \
    } \
} while(0)

// ===== 单个算法结果 =====
struct AlgResult {
    int alg_id;
    float lat_us;
    float tflops;
    size_t workspace;
};

// ===== benchmark_fp8_gemm =====
// 输入: W[N,K], A[M,K] (PyTorch 行主序)
// 输出: dict{best_lat_us, best_tflops, best_alg_id, all_results}
py::dict benchmark_fp8_gemm(
    int64_t M, int64_t N, int64_t K,
    int warmup, int repeat, int max_algos
) {
    // 维度对齐检查 (FP8 需 16 对齐)
    auto check_align = [](int64_t v, const char* name) {
        if (v % 16 != 0) {
            throw std::invalid_argument(std::string(name) + " must be multiple of 16");
        }
    };
    check_align(M, "M");
    check_align(N, "N");
    check_align(K, "K");
    
    auto device = torch::kCUDA;
    auto stream = at::cuda::getDefaultCUDAStream();
    
    // 创建 FP8 输入张量
    // W[N,K] 行主序 = [K,N] 列主序
    // A[M,K] 行主序 = [K,M] 列主序
    auto W_fp16 = torch::randn({N, K}, torch::dtype(torch::kFloat16).device(device));
    auto A_fp16 = torch::randn({M, K}, torch::dtype(torch::kFloat16).device(device));
    auto W_fp8 = W_fp16.to(torch::kFloat8_e4m3fn).contiguous();
    auto A_fp8 = A_fp16.to(torch::kFloat8_e4m3fn).contiguous();
    
    // 输出: C[M,N] 行主序 (PyTorch 视角)
    // = [N,M] 列主序 (cuBLASLt 视角)
    auto C_out = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(device));
    
    // cuBLASLt 初始化
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));
    
    // 固定 Layout: T/N + Col/Col + Col
    // W^T * A = C
    // W[N,K] transposed means stored as [K,N]_col
    // A[M,K] means stored as [K,M]_col  
    // C[N,M]_col = [M,N] row-major
    cublasOperation_t opW = CUBLAS_OP_T;
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasLtOrder_t orderCol = CUBLASLT_ORDER_COL;
    
    // 矩阵维度 (列主序存储)
    // W: 存储 [K,N], ld=K
    // A: 存储 [K,M], ld=K  
    // C: 存储 [N,M], ld=N
    int64_t ldw = K;
    int64_t lda = K;
    int64_t ldc = N;
    
    // 创建 matmul 描述符
    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    
    // 创建矩阵布局
    cublasLtMatrixLayout_t layoutW, layoutA, layoutC;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutW, CUDA_R_8F_E4M3, K, N, ldw));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderCol, sizeof(orderCol)));
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, M, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderCol, sizeof(orderCol)));
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, ldc));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderCol, sizeof(orderCol)));
    
    // Workspace (512MB)
    size_t workspace_size = 512 * 1024 * 1024;
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    // 创建算法偏好
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                       &workspace_size, sizeof(workspace_size)));
    
    // 获取候选算法 (启发式搜索)
    std::vector<cublasLtMatmulHeuristicResult_t> heurResults(max_algos);
    int returnedCount = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutW, layoutA, layoutC, layoutC,
                                                 pref, max_algos, heurResults.data(), &returnedCount));
    
    if (returnedCount == 0) {
        cudaFree(workspace);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtDestroy(handle);
        throw std::runtime_error("No valid algorithms found for this configuration");
    }
    
    float alpha = 1.0f, beta = 0.0f;
    std::vector<AlgResult> results;
    
    // 测试每个算法
    for (int i = 0; i < returnedCount; ++i) {
        if (heurResults[i].state != CUBLAS_STATUS_SUCCESS) continue;
        
        const auto* algo = &heurResults[i].algo;
        size_t ws = heurResults[i].workspaceSize;
        
        // Warmup
        for (int w = 0; w < warmup; ++w) {
            cublasLtMatmul(handle, matmulDesc, &alpha,
                          W_fp8.data_ptr(), layoutW,
                          A_fp8.data_ptr(), layoutA,
                          &beta,
                          C_out.data_ptr(), layoutC,
                          C_out.data_ptr(), layoutC,
                          algo, workspace, ws, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // 计时
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int r = 0; r < repeat; ++r) {
            cublasLtMatmul(handle, matmulDesc, &alpha,
                          W_fp8.data_ptr(), layoutW,
                          A_fp8.data_ptr(), layoutA,
                          &beta,
                          C_out.data_ptr(), layoutC,
                          C_out.data_ptr(), layoutC,
                          algo, workspace, ws, stream);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float total_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        // 计算结果
        float lat_us = (total_ms * 1000.0f) / repeat;
        double flops = 2.0 * M * N * K;
        float tflops = flops / (lat_us * 1e6);
        
        int alg_id = 0;
        cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID, &alg_id, sizeof(alg_id), nullptr);
        
        results.push_back({alg_id, lat_us, tflops, ws});
    }
    
    // 清理
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    
    // 找最佳结果
    if (results.empty()) {
        throw std::runtime_error("All algorithms failed");
    }
    
    auto best = std::min_element(results.begin(), results.end(),
                                  [](const AlgResult& a, const AlgResult& b) { return a.lat_us < b.lat_us; });
    
    // 构造返回值
    py::dict out;
    out["best_lat_us"] = best->lat_us;
    out["best_tflops"] = best->tflops;
    out["best_alg_id"] = best->alg_id;
    out["num_algos_tested"] = static_cast<int>(results.size());
    
    // 所有结果
    py::list all_results;
    for (const auto& r : results) {
        py::dict d;
        d["alg_id"] = r.alg_id;
        d["lat_us"] = r.lat_us;
        d["tflops"] = r.tflops;
        all_results.append(d);
    }
    out["all_results"] = all_results;
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("benchmark_fp8_gemm", &benchmark_fp8_gemm,
          "Benchmark cuBLASLt FP8 GEMM with heuristic algorithm search",
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("warmup") = 25, py::arg("repeat") = 100, py::arg("max_algos") = 5);
}
