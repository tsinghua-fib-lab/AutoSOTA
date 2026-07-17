// cuBLASLt 调试测试：验证 TN 布局和 Column Major 输出
// 编译: nvcc -O3 debug_cublaslt.cu -o debug_cublaslt -lcublasLt -lcublas
// 运行: ./debug_cublaslt

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

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

// 简单的矩阵打印（用于小矩阵调试）
void print_matrix_int8(const char* name, const int8_t* data, int rows, int cols, bool col_major) {
    printf("%s (%dx%d, %s):\n", name, rows, cols, col_major ? "ColMajor" : "RowMajor");
    for (int i = 0; i < std::min(rows, 8); ++i) {
        printf("  ");
        for (int j = 0; j < std::min(cols, 8); ++j) {
            int idx = col_major ? (j * rows + i) : (i * cols + j);
            printf("%4d ", (int)data[idx]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("  ...\n");
}

void print_matrix_int32(const char* name, const int32_t* data, int rows, int cols, bool col_major) {
    printf("%s (%dx%d, %s):\n", name, rows, cols, col_major ? "ColMajor" : "RowMajor");
    for (int i = 0; i < std::min(rows, 8); ++i) {
        printf("  ");
        for (int j = 0; j < std::min(cols, 8); ++j) {
            int idx = col_major ? (j * rows + i) : (i * cols + j);
            printf("%8d ", data[idx]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("  ...\n");
}

// CPU 参考实现: C = W^T * A
// W: [N, K], A: [K, M], C: [N, M]
// 注意：这里我们按逻辑维度计算
void cpu_gemm_ref(const int8_t* W, const int8_t* A, int32_t* C, int N, int K, int M) {
    // C[n, m] = sum_k W[n, k] * A[k, m]
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                // W 是 Row Major [N, K]
                // A 是 Row Major [K, M] 或等价于 Col Major [M, K]
                sum += (int32_t)W[n * K + k] * (int32_t)A[k * M + m];
            }
            C[n * M + m] = sum;  // Row Major [N, M]
        }
    }
}

int main() {
    // 小尺寸测试：便于人工验证
    // 满足 INT8 对齐要求（4 的倍数）
    const int N = 8;   // W 的行数
    const int K = 4;   // 共享维度
    const int M = 8;   // A 的列数 / batch size
    
    printf("=== cuBLASLt INT8 GEMM 调试测试 ===\n");
    printf("N=%d, K=%d, M=%d\n", N, K, M);
    printf("目标: C[N,M] = W[N,K]^T * A[K,M]  (输出 Column Major)\n\n");
    
    // 分配 Host 内存
    std::vector<int8_t> h_W(N * K);
    std::vector<int8_t> h_A(K * M);
    std::vector<int32_t> h_C(N * M, 0);
    std::vector<int32_t> h_C_ref(N * M, 0);
    
    // 初始化数据：使用简单值便于调试
    // W[N, K] Row Major
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            h_W[i * K + j] = (int8_t)(i + 1);  // 每行填充行号+1
        }
    }
    // A[K, M] Row Major（等价于 A[M, K] Col Major 的转置）
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            h_A[i * M + j] = (int8_t)(j + 1);  // 每列填充列号+1
        }
    }
    
    printf("=== 输入数据 ===\n");
    print_matrix_int8("W (Row Major)", h_W.data(), N, K, false);
    print_matrix_int8("A (Row Major)", h_A.data(), K, M, false);
    
    // CPU 参考计算
    cpu_gemm_ref(h_W.data(), h_A.data(), h_C_ref.data(), N, K, M);
    printf("\n=== CPU 参考结果 ===\n");
    print_matrix_int32("C_ref (Row Major)", h_C_ref.data(), N, M, false);
    
    // GPU 计算
    int8_t *d_W, *d_A;
    int32_t *d_C;
    CHECK_CUDA(cudaMalloc(&d_W, N * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_A, K * M * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_C, N * M * sizeof(int32_t)));
    
    // 拷贝数据：需要考虑布局转换
    // cuBLASLt TN 布局：
    //   - A (左矩阵，带转置): 存储为 [K, N] Col Major
    //   - B (右矩阵，不转置): 存储为 [K, M] Col Major
    //   - C (输出): 存储为 [N, M] Col Major
    
    // W[N,K] Row Major -> W[K,N] Col Major (转置)
    // PyTorch 的 Row Major [N,K] 数据，直接上传后被 cuBLAS 解释为 Col Major [K,N]
    // 这正好是我们需要的！
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), N * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    
    // A[K,M] Row Major -> A[K,M] Col Major
    // 这里有问题！Row Major [K,M] 与 Col Major [K,M] 不同
    // 需要转置，或者直接按 Col Major 方式填充
    // 为简化，我们创建 Col Major 版本的 A
    std::vector<int8_t> h_A_col(K * M);
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            // Col Major: index = m * K + k
            // 从 Row Major: index = k * M + m
            h_A_col[m * K + k] = h_A[k * M + m];
        }
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A_col.data(), K * M * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, N * M * sizeof(int32_t)));
    
    // 创建 cuBLASLt handle
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    
    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    
    // 设置转置属性
    cublasOperation_t opA = CUBLAS_OP_T;  // W 转置
    cublasOperation_t opB = CUBLAS_OP_N;  // A 不转置
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    
    // === 关键：创建矩阵布局 ===
    // cublasLtMatrixLayoutCreate 参数: (layout, type, rows, cols, ld)
    // 这里的 rows/cols 是【存储】的行列，不是逻辑矩阵的行列
    
    // W 矩阵：逻辑 [N, K]，op=T，所以存储应该是 [K, N] (转置前的形状)
    // 但实际上 cuBLASLt 需要的是：在转置【后】的逻辑矩阵形状
    // TN 格式：A^T * B = C
    //   A: 转置后逻辑 [M_gemm, K_gemm]
    //   B: 逻辑 [K_gemm, N_gemm]
    //   C: 逻辑 [M_gemm, N_gemm]
    // 这里 M_gemm = N, K_gemm = K, N_gemm = M
    
    // 对于 cublasLtMatrixLayoutCreate:
    //   - A (带转置): 存储维度 [K, N]，ld = K (Col Major)
    //   - B (不转置): 存储维度 [K, M]，ld = K (Col Major)
    //   - C: 存储维度 [N, M]，ld = N (Col Major)
    
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    
    // A (W): 存储 [K, N]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // B (A): 存储 [K, M]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // C: 存储 [N, M]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // 创建算法偏好
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size = 32 * 1024 * 1024;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                      &workspace_size, sizeof(workspace_size)));
    
    // 获取算法
    cublasLtMatmulHeuristicResult_t heuristicResult[32];
    int returnedAlgoCount = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
                                                 preference, 32, heuristicResult, &returnedAlgoCount));
    printf("\n=== cuBLASLt 算法搜索 ===\n");
    printf("返回算法数: %d\n", returnedAlgoCount);
    
    if (returnedAlgoCount == 0) {
        printf("ERROR: 没有找到可用算法！\n");
        return 1;
    }
    
    // 分配 workspace
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    
    // 使用第一个算法
    int32_t alpha = 1, beta = 0;
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, matmulDesc, &alpha, d_W, layoutA, d_A, layoutB,
                                &beta, d_C, layoutC, d_C, layoutC,
                                &heuristicResult[0].algo, d_workspace, workspace_size, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回 Host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, N * M * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    printf("\n=== GPU 结果 (Col Major 存储) ===\n");
    print_matrix_int32("C_gpu (Col Major)", h_C.data(), N, M, true);
    
    // 转换为 Row Major 比较
    std::vector<int32_t> h_C_row(N * M);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            h_C_row[n * M + m] = h_C[m * N + n];  // Col Major to Row Major
        }
    }
    printf("\n=== GPU 结果 (转换为 Row Major) ===\n");
    print_matrix_int32("C_gpu (Row Major)", h_C_row.data(), N, M, false);
    
    // 验证
    printf("\n=== 验证 ===\n");
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N * M; ++i) {
        float err = std::abs((float)h_C_row[i] - (float)h_C_ref[i]);
        max_err = std::max(max_err, err);
        if (err > 0.001f) {
            if (errors < 10) {
                printf("  错误 @ %d: GPU=%d, CPU=%d\n", i, h_C_row[i], h_C_ref[i]);
            }
            errors++;
        }
    }
    printf("最大误差: %f\n", max_err);
    printf("错误数: %d / %d\n", errors, N * M);
    printf("验证: %s\n", errors == 0 ? "PASS" : "FAIL");
    
    // 清理
    cudaFree(d_W);
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(ltHandle);
    
    return errors == 0 ? 0 : 1;
}
