/**
 * 测试 INT8 和 FP8 下同一 algo_id 的配置差异
 */

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

void printAlgoConfig(cublasLtMatmulAlgo_t& algo, int index, size_t workspace, float wavesCount) {
    int algoId, tile, stages, splitK, reduction, swizzle, custom;
    
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom), nullptr);
    
    printf("  [%d] id=%d tile=%d stages=%d splitK=%d red=%d swz=%d cust=%d ws=%zu waves=%.2f\n",
           index, algoId, tile, stages, splitK, reduction, swizzle, custom, workspace, wavesCount);
}

float benchmarkAlgo(cublasLtHandle_t handle,
                   cublasLtMatmulDesc_t matmulDesc,
                   cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
                   const void* A, const void* B, void* C,
                   cublasLtMatmulAlgo_t& algo,
                   void* workspace, size_t workspaceSize,
                   const void* alpha, const void* beta,
                   int warmup = 5, int repeat = 20) {
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int i = 0; i < warmup; i++) {
        cublasLtMatmul(handle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                       &algo, workspace, workspaceSize, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        cublasLtMatmul(handle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                       &algo, workspace, workspaceSize, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / repeat;
}

void testDataType(const char* name, cudaDataType_t dtype, cudaDataType_t outType, cublasComputeType_t computeType) {
    printf("\n========== %s ==========\n", name);
    
    int M = 64, N = 3072, K = 896;
    
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));
    
    size_t elemSize = (dtype == CUDA_R_8I || dtype == CUDA_R_8F_E4M3) ? 1 : 2;
    size_t outElemSize = (outType == CUDA_R_32I || outType == CUDA_R_32F) ? 4 : 
                         (outType == CUDA_R_16F || outType == CUDA_R_16BF) ? 2 : 1;
    
    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)N * K * elemSize));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * M * elemSize));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)N * M * outElemSize));
    CHECK_CUDA(cudaMemset(d_A, 0, (size_t)N * K * elemSize));
    CHECK_CUDA(cudaMemset(d_B, 0, (size_t)K * M * elemSize));
    
    size_t workspaceSize = 4 * 1024 * 1024;
    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, dtype, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, dtype, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, outType, N, M, N));
    
    cublasLtMatmulDesc_t matmulDesc;
    cudaDataType_t scaleType = (computeType == CUBLAS_COMPUTE_32I) ? CUDA_R_32I : CUDA_R_32F;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
    
    cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));
    
    const int maxAlgos = 100;
    cublasLtMatmulHeuristicResult_t results[maxAlgos];
    int returnedAlgos = 0;
    
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                                 preference, maxAlgos, results, &returnedAlgos));
    
    printf("启发式返回 %d 个配置:\n", returnedAlgos);
    
    std::map<int, std::vector<int>> algoGroups;
    for (int i = 0; i < returnedAlgos; i++) {
        printAlgoConfig(results[i].algo, i, results[i].workspaceSize, results[i].wavesCount);
        int algoId;
        cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        algoGroups[algoId].push_back(i);
    }
    
    // 性能测试
    printf("\n性能测试:\n");
    int32_t alpha_i = 1, beta_i = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    const void* alpha = (computeType == CUBLAS_COMPUTE_32I) ? (void*)&alpha_i : (void*)&alpha_f;
    const void* beta = (computeType == CUBLAS_COMPUTE_32I) ? (void*)&beta_i : (void*)&beta_f;
    
    std::vector<std::pair<int, float>> timings;
    for (int i = 0; i < returnedAlgos; i++) {
        float t = benchmarkAlgo(handle, matmulDesc, Adesc, Bdesc, Cdesc, d_A, d_B, d_C,
                                results[i].algo, workspace, workspaceSize, alpha, beta);
        timings.push_back({i, t});
        
        int algoId, tile, stages;
        cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr);
        printf("  [%d] id=%d tile=%d stages=%d -> %.3f ms\n", i, algoId, tile, stages, t);
    }
    
    // 分析同一 algo_id 内的性能差异
    printf("\n同一 algo_id 内的性能差异:\n");
    for (auto& [algoId, indices] : algoGroups) {
        if (indices.size() > 1) {
            float minT = 1e9, maxT = 0;
            for (int idx : indices) {
                float t = timings[idx].second;
                minT = std::min(minT, t);
                maxT = std::max(maxT, t);
            }
            printf("  algo_id=%d: %zu 配置, 性能差异 %.1f%% (%.3f ~ %.3f ms)\n",
                   algoId, indices.size(), (maxT - minT) / minT * 100, minT, maxT);
        }
    }
    
    auto best = std::min_element(timings.begin(), timings.end(),
                                  [](auto& a, auto& b) { return a.second < b.second; });
    printf("\n全局最优: 配置[%d] = %.3f ms\n", best->first, best->second);
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(workspace));
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(handle);
}

int main() {
    printf("=== 不同数据类型下的配置差异测试 ===\n");
    
    testDataType("INT8 (输出 INT32)", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I);
    testDataType("FP8 E4M3 (输出 FP16)", CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F);
    
    return 0;
}
