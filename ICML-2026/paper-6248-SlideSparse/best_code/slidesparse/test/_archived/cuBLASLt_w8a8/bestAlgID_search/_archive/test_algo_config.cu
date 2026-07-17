/**
 * 测试：同一个 algo_id 下不同配置的性能差异
 * 
 * 目的：验证是否只需要缓存 algo_id，还是需要缓存更细粒度的配置
 */

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

// 打印算法的详细配置
void printAlgoConfig(cublasLtMatmulAlgo_t& algo, int index) {
    int algoId;
    int tile;
    int stages;
    int splitK;
    int reductionScheme;
    int swizzle;
    int customOption;
    
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr));
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), nullptr));
    
    printf("  [%d] algo_id=%d, tile=%d, stages=%d, splitK=%d, reduction=%d, swizzle=%d, custom=%d\n",
           index, algoId, tile, stages, splitK, reductionScheme, swizzle, customOption);
}

// 性能测试
float benchmarkAlgo(cublasLtHandle_t handle,
                   cublasLtMatmulDesc_t matmulDesc,
                   cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc,
                   const void* A, const void* B, void* C,
                   cublasLtMatmulAlgo_t& algo,
                   void* workspace, size_t workspaceSize,
                   float alpha, float beta,
                   int warmup = 5, int repeat = 20) {
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublasLtMatmul(handle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, C, Cdesc,
                       &algo, workspace, workspaceSize, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        cublasLtMatmul(handle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, C, Cdesc,
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

int main() {
    printf("=== 测试同一 algo_id 下不同配置的性能差异 ===\n\n");
    
    // 矩阵尺寸 (与实际使用场景一致)
    int M = 64;   // batch size
    int N = 3072; // hidden size
    int K = 896;  // intermediate
    
    printf("矩阵尺寸: M=%d, N=%d, K=%d\n", M, N, K);
    printf("GEMM: C[%d,%d] = A[%d,%d]^T * B[%d,%d]\n\n", N, M, N, K, K, M);
    
    // 初始化
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));
    
    // 使用 FP16 测试（因为 FP16 有更多配置变体）
    cudaDataType_t dtype = CUDA_R_16F;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
    
    // 分配内存
    size_t sizeA = (size_t)N * K * sizeof(__half);
    size_t sizeB = (size_t)K * M * sizeof(__half);
    size_t sizeC = (size_t)N * M * sizeof(__half);
    
    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    
    // 初始化数据
    CHECK_CUDA(cudaMemset(d_A, 0, sizeA));
    CHECK_CUDA(cudaMemset(d_B, 0, sizeB));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC));
    
    // Workspace
    size_t workspaceSize = 4 * 1024 * 1024;  // 4MB
    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    
    // 创建矩阵描述符 (Column Major, TN layout)
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    // A: [N, K] col-major -> ld = N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, dtype, K, N, K));  // transposed view
    // B: [K, M] col-major -> ld = K  
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, dtype, K, M, K));
    // C: [N, M] col-major -> ld = N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, dtype, N, M, N));
    
    // 创建 matmul 描述符
    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_16F));
    
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    // 获取所有启发式算法
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));
    
    const int maxAlgos = 100;
    cublasLtMatmulHeuristicResult_t heuristicResults[maxAlgos];
    int returnedAlgos = 0;
    
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                                 preference, maxAlgos, heuristicResults, &returnedAlgos));
    
    printf("启发式搜索返回 %d 个算法配置\n\n", returnedAlgos);
    
    // 按 algo_id 分组
    std::map<int, std::vector<int>> algoGroups;  // algo_id -> indices
    
    for (int i = 0; i < returnedAlgos; i++) {
        int algoId;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &heuristicResults[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
        algoGroups[algoId].push_back(i);
    }
    
    printf("按 algo_id 分组:\n");
    for (auto& [algoId, indices] : algoGroups) {
        printf("  algo_id=%d: %zu 个配置\n", algoId, indices.size());
    }
    printf("\n");
    
    // 详细打印每个配置
    printf("所有配置详情:\n");
    for (int i = 0; i < returnedAlgos; i++) {
        printAlgoConfig(heuristicResults[i].algo, i);
        printf("        workspace=%zu bytes, wavesCount=%.2f\n",
               heuristicResults[i].workspaceSize, heuristicResults[i].wavesCount);
    }
    printf("\n");
    
    // 性能测试：比较同一 algo_id 下不同配置的性能
    printf("=== 性能测试 ===\n\n");
    
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    
    struct Result {
        int index;
        int algoId;
        int tile;
        int stages;
        size_t workspace;
        float time_ms;
    };
    std::vector<Result> results;
    
    for (int i = 0; i < returnedAlgos; i++) {
        int algoId, tile, stages;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &heuristicResults[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &heuristicResults[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr));
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &heuristicResults[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr));
        
        float time_ms = benchmarkAlgo(handle, matmulDesc, Adesc, Bdesc, Cdesc,
                                       d_A, d_B, d_C, heuristicResults[i].algo,
                                       workspace, workspaceSize, alpha_f, beta_f);
        
        results.push_back({i, algoId, tile, stages, heuristicResults[i].workspaceSize, time_ms});
        printf("配置 [%d]: algo_id=%d, tile=%d, stages=%d, workspace=%zu -> %.3f ms\n",
               i, algoId, tile, stages, heuristicResults[i].workspaceSize, time_ms);
    }
    
    // 按 algo_id 分组分析性能差异
    printf("\n=== 同一 algo_id 内的性能差异分析 ===\n\n");
    
    for (auto& [algoId, indices] : algoGroups) {
        if (indices.size() > 1) {
            printf("algo_id=%d (%zu 个配置):\n", algoId, indices.size());
            
            float minTime = 1e9, maxTime = 0;
            int minIdx = -1, maxIdx = -1;
            
            for (int idx : indices) {
                float t = results[idx].time_ms;
                if (t < minTime) { minTime = t; minIdx = idx; }
                if (t > maxTime) { maxTime = t; maxIdx = idx; }
            }
            
            printf("  最快: 配置[%d] = %.3f ms (tile=%d, stages=%d, ws=%zu)\n",
                   minIdx, minTime, results[minIdx].tile, results[minIdx].stages, results[minIdx].workspace);
            printf("  最慢: 配置[%d] = %.3f ms (tile=%d, stages=%d, ws=%zu)\n",
                   maxIdx, maxTime, results[maxIdx].tile, results[maxIdx].stages, results[maxIdx].workspace);
            printf("  性能差异: %.1f%%\n\n", (maxTime - minTime) / minTime * 100);
        }
    }
    
    // 找出全局最优
    printf("=== 全局最优 ===\n");
    auto best = std::min_element(results.begin(), results.end(),
                                  [](const Result& a, const Result& b) { return a.time_ms < b.time_ms; });
    printf("最优配置 [%d]: algo_id=%d, tile=%d, stages=%d, workspace=%zu -> %.3f ms\n",
           best->index, best->algoId, best->tile, best->stages, best->workspace, best->time_ms);
    
    // 清理
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
    
    printf("\n=== 结论 ===\n");
    printf("如果同一 algo_id 下不同配置的性能差异显著 (>5%%)，则需要缓存完整配置。\n");
    printf("如果差异不大 (<5%%)，则只缓存 algo_id 即可。\n");
    
    return 0;
}
