// 测试 cuBLASLt 算法 ID 机制
// 编译: nvcc -O3 test_algo_ids.cu -o test_algo_ids -lcublasLt -lcublas
// 运行: ./test_algo_ids

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <set>

#define CHECK_CUDA(expr) do { \
    cudaError_t st = (expr); \
    if (st != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(st)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(expr) do { \
    cublasStatus_t st = (expr); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error at %s:%d: code %d\n", __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while(0)

void print_separator(const char* title) {
    printf("\n");
    printf("============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

// 方法1：使用 cublasLtMatmulAlgoGetIds() 获取所有有效算法 ID
void method1_get_algo_ids(cublasLtHandle_t handle, 
                          cublasComputeType_t computeType,
                          cudaDataType_t scaleType,
                          cudaDataType_t typeAB,
                          cudaDataType_t typeC) {
    print_separator("方法1: cublasLtMatmulAlgoGetIds()");
    
    const int maxAlgos = 256;
    int algoIds[maxAlgos];
    int returnedCount = 0;
    
    cublasStatus_t st = cublasLtMatmulAlgoGetIds(
        handle,
        computeType,
        scaleType,
        typeAB,   // Atype
        typeAB,   // Btype
        typeC,    // Ctype
        typeC,    // Dtype
        maxAlgos,
        algoIds,
        &returnedCount);
    
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("  cublasLtMatmulAlgoGetIds 失败: code %d\n", (int)st);
        return;
    }
    
    printf("  返回的算法数量: %d\n", returnedCount);
    
    // 排序以便分析
    std::vector<int> sortedIds(algoIds, algoIds + returnedCount);
    std::sort(sortedIds.begin(), sortedIds.end());
    
    printf("  算法 ID 列表 (已排序): ");
    for (int i = 0; i < returnedCount; ++i) {
        printf("%d", sortedIds[i]);
        if (i < returnedCount - 1) printf(", ");
    }
    printf("\n");
    
    // 分析连续性
    if (returnedCount > 1) {
        int minId = sortedIds[0];
        int maxId = sortedIds[returnedCount - 1];
        bool isContinuous = (maxId - minId + 1 == returnedCount);
        printf("  ID 范围: [%d, %d]\n", minId, maxId);
        printf("  是否连续: %s\n", isContinuous ? "是" : "否");
        
        if (!isContinuous) {
            printf("  缺失的 ID: ");
            std::set<int> idSet(sortedIds.begin(), sortedIds.end());
            for (int id = minId; id <= maxId; ++id) {
                if (idSet.find(id) == idSet.end()) {
                    printf("%d ", id);
                }
            }
            printf("\n");
        }
    }
}

// 方法2：使用 cublasLtMatmulAlgoGetHeuristic() 启发式搜索
void method2_heuristic(cublasLtHandle_t handle,
                       cublasLtMatmulDesc_t matmulDesc,
                       cublasLtMatrixLayout_t layoutA,
                       cublasLtMatrixLayout_t layoutB,
                       cublasLtMatrixLayout_t layoutC) {
    print_separator("方法2: cublasLtMatmulAlgoGetHeuristic()");
    
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspaceSize = 32 * 1024 * 1024;  // 32 MB
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));
    
    const int maxAlgos = 64;
    cublasLtMatmulHeuristicResult_t heuristicResults[maxAlgos];
    int returnedCount = 0;
    
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutA,
        layoutB,
        layoutC,
        layoutC,
        preference,
        maxAlgos,
        heuristicResults,
        &returnedCount);
    
    cublasLtMatmulPreferenceDestroy(preference);
    
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("  cublasLtMatmulAlgoGetHeuristic 失败: code %d\n", (int)st);
        return;
    }
    
    printf("  返回的算法数量: %d\n", returnedCount);
    
    // 提取并显示算法 ID
    printf("  算法详情:\n");
    std::vector<int> algoIds;
    for (int i = 0; i < returnedCount; ++i) {
        if (heuristicResults[i].state == CUBLAS_STATUS_SUCCESS) {
            // 从 algo 结构中提取 ID
            int algoId = -1;
            size_t sizeWritten = 0;
            cublasLtMatmulAlgoConfigGetAttribute(
                &heuristicResults[i].algo,
                CUBLASLT_ALGO_CONFIG_ID,
                &algoId,
                sizeof(algoId),
                &sizeWritten);
            
            algoIds.push_back(algoId);
            printf("    [%d] algo_id=%d, workspace=%zu bytes, wavesCount=%.2f\n",
                   i, algoId, heuristicResults[i].workspaceSize, 
                   heuristicResults[i].wavesCount);
        }
    }
    
    if (!algoIds.empty()) {
        std::sort(algoIds.begin(), algoIds.end());
        printf("  算法 ID (已排序): ");
        for (size_t i = 0; i < algoIds.size(); ++i) {
            printf("%d", algoIds[i]);
            if (i < algoIds.size() - 1) printf(", ");
        }
        printf("\n");
    }
}

// 方法3：暴力遍历 + cublasLtMatmulAlgoCheck()
void method3_brute_force_check(cublasLtHandle_t handle,
                               cublasLtMatmulDesc_t matmulDesc,
                               cublasLtMatrixLayout_t layoutA,
                               cublasLtMatrixLayout_t layoutB,
                               cublasLtMatrixLayout_t layoutC,
                               cublasComputeType_t computeType,
                               cudaDataType_t scaleType,
                               cudaDataType_t typeAB,
                               cudaDataType_t typeC) {
    print_separator("方法3: 暴力遍历 + cublasLtMatmulAlgoCheck()");
    
    // 首先获取所有可能的算法 ID
    const int maxAlgos = 256;
    int algoIds[maxAlgos];
    int totalCount = 0;
    
    cublasStatus_t st = cublasLtMatmulAlgoGetIds(
        handle, computeType, scaleType, typeAB, typeAB, typeC, typeC,
        maxAlgos, algoIds, &totalCount);
    
    if (st != CUBLAS_STATUS_SUCCESS || totalCount == 0) {
        printf("  无法获取算法 ID 列表\n");
        return;
    }
    
    printf("  总共有 %d 个算法 ID 可供测试\n", totalCount);
    
    std::vector<int> validIds;
    std::vector<int> invalidIds;
    
    for (int i = 0; i < totalCount; ++i) {
        int algoId = algoIds[i];
        
        // 初始化算法描述符
        cublasLtMatmulAlgo_t algo;
        cublasLtMatmulAlgoInit(handle, computeType, scaleType, 
                               typeAB, typeAB, typeC, typeC, algoId, &algo);
        
        // 使用 cublasLtMatmulAlgoCheck 验证
        cublasLtMatmulHeuristicResult_t result;
        cublasStatus_t checkStatus = cublasLtMatmulAlgoCheck(
            handle, matmulDesc, layoutA, layoutB, layoutC, layoutC, &algo, &result);
        
        if (checkStatus == CUBLAS_STATUS_SUCCESS && result.state == CUBLAS_STATUS_SUCCESS) {
            validIds.push_back(algoId);
        } else {
            invalidIds.push_back(algoId);
        }
    }
    
    printf("  有效算法数量: %zu\n", validIds.size());
    printf("  无效算法数量: %zu\n", invalidIds.size());
    
    if (!validIds.empty()) {
        std::sort(validIds.begin(), validIds.end());
        printf("  有效算法 ID (已排序): ");
        for (size_t i = 0; i < validIds.size(); ++i) {
            printf("%d", validIds[i]);
            if (i < validIds.size() - 1) printf(", ");
        }
        printf("\n");
    }
    
    if (!invalidIds.empty() && invalidIds.size() <= 20) {
        std::sort(invalidIds.begin(), invalidIds.end());
        printf("  无效算法 ID (已排序): ");
        for (size_t i = 0; i < invalidIds.size(); ++i) {
            printf("%d", invalidIds[i]);
            if (i < invalidIds.size() - 1) printf(", ");
        }
        printf("\n");
    }
}

void test_dtype(const char* dtypeName,
                cublasComputeType_t computeType,
                cudaDataType_t scaleType,
                cudaDataType_t typeAB,
                cudaDataType_t typeC) {
    printf("\n");
    printf("************************************************************\n");
    printf("*  测试数据类型: %s\n", dtypeName);
    printf("************************************************************\n");
    
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));
    
    // 测试矩阵尺寸 (满足对齐要求)
    const int N = 256;
    const int K = 256;
    const int M = 256;
    
    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
    
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // 创建矩阵布局 (Column Major)
    // A (W): 存储 [K, N], 转置后逻辑 [N, K]
    // B (A): 存储 [K, M], 逻辑 [K, M]
    // C: 存储 [N, M]
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, typeAB, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, typeAB, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, typeC, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // 执行三种方法的测试
    method1_get_algo_ids(handle, computeType, scaleType, typeAB, typeC);
    method2_heuristic(handle, matmulDesc, layoutA, layoutB, layoutC);
    method3_brute_force_check(handle, matmulDesc, layoutA, layoutB, layoutC,
                              computeType, scaleType, typeAB, typeC);
    
    // 清理
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
}

int main() {
    printf("=== cuBLASLt 算法 ID 机制测试 ===\n");
    
    // 获取 GPU 信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // 测试 INT8
    test_dtype("INT8 (CUDA_R_8I -> CUDA_R_32I)",
               CUBLAS_COMPUTE_32I,
               CUDA_R_32I,
               CUDA_R_8I,
               CUDA_R_32I);
    
    // 测试 FP8 E4M3
    test_dtype("FP8 E4M3 (CUDA_R_8F_E4M3 -> CUDA_R_32F)",
               CUBLAS_COMPUTE_32F,
               CUDA_R_32F,
               CUDA_R_8F_E4M3,
               CUDA_R_32F);
    
    // 测试 FP16
    test_dtype("FP16 (CUDA_R_16F -> CUDA_R_16F)",
               CUBLAS_COMPUTE_16F,
               CUDA_R_16F,
               CUDA_R_16F,
               CUDA_R_16F);
    
    // 测试 BF16
    test_dtype("BF16 (CUDA_R_16BF -> CUDA_R_16BF)",
               CUBLAS_COMPUTE_32F,
               CUDA_R_32F,
               CUDA_R_16BF,
               CUDA_R_16BF);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}
