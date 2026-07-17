// SPDX-License-Identifier: Apache-2.0
/**
 * @file layout_search_cublaslt.cu
 * @brief cuBLASLt 布局搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuBLASLt 布局配置搜索功能，测试 16 种布局组合：
 *   - 转置 : TT, TN, NT, NN (4种)
 *   - A/B 排列 : RowCol, ColCol (2种)
 *   - R 输出 : Col, Row (2种)
 *
 * 固定最优布局: T/N + Col/Col + Col (权重 W 在左)
 *
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      layout_search_cublaslt.cu -lcublasLt -lcublas \
 *      -o layout_search_cublaslt.so
 *
 * 主要接口:
 * ---------
 * - cublaslt_layout_search_single()  : 测试单个 (N,K,M) 的 8 种布局
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

// =============================================================================
// 常量与类型定义
// =============================================================================

// 最大 workspace 大小: 512 MB
static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;

// 16 种布局组合 (4 种转置 × 2 种 orderW × 2 种 orderA × 2 种输出顺序，但部分组合可能无效)
// 前8种为标准有效组合，后8种为非标准组合（可能在某些配置下有效）
static constexpr int NUM_LAYOUTS = 16;

// 布局组合枚举
struct LayoutConfig {
    cublasOperation_t transW;
    cublasOperation_t transA;
    cublasLtOrder_t orderW;
    cublasLtOrder_t orderA;
    cublasLtOrder_t orderR;  // R 输出格式
    const char* name;
};

// 所有 16 种布局配置
// 命名格式: {transW}{transA}_{orderW}{orderA}_{orderR}
static const LayoutConfig LAYOUT_CONFIGS[NUM_LAYOUTS] = {
    // transW,       transA,       orderW,             orderA,             orderR,             name
    // === 标准有效组合 (前8种) ===
    // R 输出为 ColMajor (前4种)
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "TN_CC_Col"},  // 推荐
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "NT_RR_Col"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "NN_RC_Col"},
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "TT_CR_Col"},
    // R 输出为 RowMajor (后4种)
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, "TN_CC_Row"},
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, "NT_RR_Row"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, "NN_RC_Row"},
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, "TT_CR_Row"},
    // === 非标准组合 (后8种，测试用) ===
    // R 输出为 ColMajor
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "TN_RR_Col"},
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "NT_CC_Col"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "NN_CR_Col"},
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "TT_RC_Col"},
    // R 输出为 RowMajor
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, "TN_RR_Row"},
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, "NT_CC_Row"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, "NN_CR_Row"},
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW, "TT_RC_Row"},
};

// =============================================================================
// 错误处理
// =============================================================================

thread_local std::string tls_last_error;

static void set_error(const char* msg) {
    tls_last_error = msg;
}

static void set_error(const std::string& msg) {
    tls_last_error = msg;
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "CUDA Error %d: %s at %s:%d", \
                 (int)err, cudaGetErrorString(err), __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "cuBLASLt Error %d at %s:%d", \
                 (int)err, __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

// =============================================================================
// 全局资源
// =============================================================================

static cublasLtHandle_t g_lt_handle = nullptr;
static std::mutex g_handle_mutex;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_lt_handle) {
        cublasStatus_t err = cublasLtCreate(&g_lt_handle);
        if (err != CUBLAS_STATUS_SUCCESS) {
            set_error("Failed to create cuBLASLt handle");
            return -1;
        }
    }
    return 0;
}

// =============================================================================
// 数据类型辅助
// =============================================================================

struct DtypeInfo {
    cudaDataType_t cuda_type;
    int elem_size;
    int alignment;
    bool is_valid;
};

static DtypeInfo get_dtype_info(const char* dtype) {
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return {CUDA_R_8I, 1, 16, true};
    } else if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "FP8") == 0) {
        return {CUDA_R_8F_E4M3, 1, 16, true};
    }
    return {{}, 0, 0, false};
}

static cudaDataType_t get_out_dtype(const char* outdtype) {
    if (strcmp(outdtype, "bf16") == 0 || strcmp(outdtype, "BF16") == 0) {
        return CUDA_R_16BF;
    } else if (strcmp(outdtype, "fp32") == 0 || strcmp(outdtype, "FP32") == 0) {
        return CUDA_R_32F;
    } else if (strcmp(outdtype, "int32") == 0 || strcmp(outdtype, "INT32") == 0) {
        return CUDA_R_32I;
    }
    return CUDA_R_16BF;  // 默认
}

static cublasComputeType_t get_compute_type(const char* dtype) {
    // FP8 使用 CUBLAS_COMPUTE_32F 支持 BF16/FP32 输出
    // INT8 使用 CUBLAS_COMPUTE_32I，只支持 INT32 输出（cuBLASLt 限制）
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return CUBLAS_COMPUTE_32I;
    }
    // FP8 使用 CUBLAS_COMPUTE_32F
    return CUBLAS_COMPUTE_32F;
}

static cudaDataType_t get_scale_type(const char* dtype) {
    // INT8 + COMPUTE_32I 需要 INT32 scale type
    // FP8 + COMPUTE_32F 需要 FP32 scale type
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return CUDA_R_32I;
    }
    return CUDA_R_32F;
}

// =============================================================================
// 布局搜索结果结构
// =============================================================================

struct LayoutResult {
    int layout_id;              // 布局索引 [0, 7]
    char layout_name[32];       // 布局名称
    float lat_us;               // 延迟 (微秒)
    float tops;                 // 吞吐量 (TOPS)
    int best_alg_id;            // 最佳算法 ID
    int64_t workspace;          // workspace 大小
    int split_k;                // split_k 参数
    uint8_t algo_data[64];      // 算法序列化数据
    uint8_t valid;              // 是否有效
};

// =============================================================================
// 单个布局测试
// =============================================================================

static int test_single_layout(
    const LayoutConfig& layout,
    int layout_id,
    void* W_ptr, void* A_ptr, void* R_ptr,
    int64_t N, int64_t K, int64_t M,  // 注意：参数顺序改为 N, K, M
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    void* workspace, size_t workspace_size,
    cudaStream_t stream,
    LayoutResult* result)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype");
        return -1;
    }
    
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cublasComputeType_t compute_type = get_compute_type(dtype);
    cudaDataType_t scale_type = get_scale_type(dtype);
    
    result->layout_id = layout_id;
    strncpy(result->layout_name, layout.name, 31);
    result->layout_name[31] = '\0';
    result->valid = 0;
    
    // =========================================================================
    // 维度计算
    // cuBLASLt GEMM: R = alpha * op(W) * op(A) + beta * C
    // 我们的约定: R = W * A (W 在左边)
    // 其中 W[N,K], A[K,M], R[N,M]
    // =========================================================================
    
    bool isW_transposed = (layout.transW == CUBLAS_OP_T);
    bool isA_transposed = (layout.transA == CUBLAS_OP_T);
    bool isW_rowmajor = (layout.orderW == CUBLASLT_ORDER_ROW);
    bool isA_rowmajor = (layout.orderA == CUBLASLT_ORDER_ROW);
    bool isR_rowmajor = (layout.orderR == CUBLASLT_ORDER_ROW);
    
    // W (矩阵 A in cuBLASLt): 逻辑维度 [N,K]
    // 如果 opW=T，存储为 [K,N]；如果 opW=N，存储为 [N,K]
    int64_t num_W_rows = isW_transposed ? K : N;
    int64_t num_W_cols = isW_transposed ? N : K;
    
    // A (矩阵 B in cuBLASLt): 逻辑维度 [K,M]
    // 如果 opA=T，存储为 [M,K]；如果 opA=N，存储为 [K,M]
    int64_t num_A_rows = isA_transposed ? M : K;
    int64_t num_A_cols = isA_transposed ? K : M;
    
    // R (矩阵 C/D): [N,M]
    int64_t num_R_rows = N;
    int64_t num_R_cols = M;
    
    // Leading dimensions
    int64_t ldw = isW_rowmajor ? num_W_cols : num_W_rows;
    int64_t lda = isA_rowmajor ? num_A_cols : num_A_rows;
    int64_t ldr = isR_rowmajor ? num_R_cols : num_R_rows;
    
    // 创建矩阵描述符
    cublasLtMatrixLayout_t layoutW = nullptr, layoutA_mat = nullptr, layoutR = nullptr;
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    
    // W 矩阵布局 (矩阵 A in cuBLASLt)
    cublasStatus_t status = cublasLtMatrixLayoutCreate(&layoutW, info.cuda_type, num_W_rows, num_W_cols, ldw);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 0;  // 不支持此布局
    }
    cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &layout.orderW, sizeof(layout.orderW));
    
    // A 矩阵布局 (矩阵 B in cuBLASLt)
    status = cublasLtMatrixLayoutCreate(&layoutA_mat, info.cuda_type, num_A_rows, num_A_cols, lda);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutW);
        return 0;
    }
    cublasLtMatrixLayoutSetAttribute(layoutA_mat, CUBLASLT_MATRIX_LAYOUT_ORDER, &layout.orderA, sizeof(layout.orderA));
    
    // R 矩阵布局 (矩阵 C/D)
    status = cublasLtMatrixLayoutCreate(&layoutR, out_type, num_R_rows, num_R_cols, ldr);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA_mat);
        return 0;
    }
    cublasLtMatrixLayoutSetAttribute(layoutR, CUBLASLT_MATRIX_LAYOUT_ORDER, &layout.orderR, sizeof(layout.orderR));
    
    // 创建矩阵乘法描述符 (INT8 用 INT32 scale_type, FP8 用 FP32)
    status = cublasLtMatmulDescCreate(&opDesc, compute_type, scale_type);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA_mat);
        cublasLtMatrixLayoutDestroy(layoutR);
        return 0;
    }
    
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &layout.transW, sizeof(layout.transW));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &layout.transA, sizeof(layout.transA));
    
    // 创建偏好
    status = cublasLtMatmulPreferenceCreate(&pref);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA_mat);
        cublasLtMatrixLayoutDestroy(layoutR);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
    
    // 获取启发式算法 (增加搜索数量以找到最优算法)
    const int max_algo_count = 128;
    cublasLtMatmulHeuristicResult_t heurResult[max_algo_count];
    int numResults = 0;
    
    status = cublasLtMatmulAlgoGetHeuristic(g_lt_handle, opDesc, layoutW, layoutA_mat, layoutR, layoutR,
                                            pref, max_algo_count, heurResult, &numResults);
    
    if (status != CUBLAS_STATUS_SUCCESS || numResults == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA_mat);
        cublasLtMatrixLayoutDestroy(layoutR);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;  // 此布局不支持
    }
    
    // alpha/beta 类型需要与 scale_type 匹配
    // INT8 GEMM: scale_type = CUDA_R_32I, 需要 int32 alpha/beta
    // FP8 GEMM: scale_type = CUDA_R_32F, 需要 float alpha/beta
    float alpha_f = 1.0f, beta_f = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    const void* alpha_ptr = (scale_type == CUDA_R_32I) ? (const void*)&alpha_i : (const void*)&alpha_f;
    const void* beta_ptr = (scale_type == CUDA_R_32I) ? (const void*)&beta_i : (const void*)&beta_f;
    
    // 遍历所有算法找到最优的
    float best_lat_us = 1e12f;
    int best_algo_idx = -1;
    int best_split_k = 1;
    size_t best_ws_size = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int alg_idx = 0; alg_idx < numResults; ++alg_idx) {
        if (heurResult[alg_idx].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        
        cublasLtMatmulAlgo_t& algo = heurResult[alg_idx].algo;
        size_t ws_size = heurResult[alg_idx].workspaceSize;
        
        if (ws_size > workspace_size) {
            continue;  // workspace 不足
        }
        
        // 验证算法可用性
        status = cublasLtMatmul(g_lt_handle, opDesc, alpha_ptr,
                               W_ptr, layoutW, A_ptr, layoutA_mat,
                               beta_ptr, R_ptr, layoutR, R_ptr, layoutR,
                               &algo, workspace, ws_size, stream);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        
        cudaStreamSynchronize(stream);
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            cublasLtMatmul(g_lt_handle, opDesc, alpha_ptr,
                          W_ptr, layoutW, A_ptr, layoutA_mat,
                          beta_ptr, R_ptr, layoutR, R_ptr, layoutR,
                          &algo, workspace, ws_size, stream);
        }
        cudaStreamSynchronize(stream);
        
        // Benchmark
        cudaEventRecord(start, stream);
        for (int i = 0; i < repeat; ++i) {
            cublasLtMatmul(g_lt_handle, opDesc, alpha_ptr,
                          W_ptr, layoutW, A_ptr, layoutA_mat,
                          beta_ptr, R_ptr, layoutR, R_ptr, layoutR,
                          &algo, workspace, ws_size, stream);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms_total = 0;
        cudaEventElapsedTime(&ms_total, start, stop);
        float lat_us = (ms_total * 1000.0f) / repeat;
        
        if (lat_us < best_lat_us) {
            best_lat_us = lat_us;
            best_algo_idx = alg_idx;
            best_ws_size = ws_size;
            // 获取 split_k 参数
            int splitK = 1;
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                  &splitK, sizeof(splitK), nullptr);
            best_split_k = splitK;
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 没有找到有效算法
    if (best_algo_idx < 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutW);
        cublasLtMatrixLayoutDestroy(layoutA_mat);
        cublasLtMatrixLayoutDestroy(layoutR);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    float tops = (2.0 * M * N * K / 1e12) / (best_lat_us / 1e6);
    
    // 提取最佳算法 ID
    int algo_id = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&heurResult[best_algo_idx].algo, CUBLASLT_ALGO_CONFIG_ID,
                                          &algo_id, sizeof(algo_id), nullptr);
    
    // 填充结果
    result->lat_us = best_lat_us;
    result->tops = tops;
    result->best_alg_id = algo_id;
    result->workspace = (int64_t)best_ws_size;
    result->split_k = best_split_k;
    memcpy(result->algo_data, &heurResult[best_algo_idx].algo, 
           sizeof(cublasLtMatmulAlgo_t) < 64 ? sizeof(cublasLtMatmulAlgo_t) : 64);
    result->valid = 1;
    
    // 清理
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA_mat);
    cublasLtMatrixLayoutDestroy(layoutR);
    cublasLtMatmulDescDestroy(opDesc);
    
    return 1;  // 成功
}

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 测试单个 (N,K,M) 的 16 种布局配置
 *
 * @param W_ptr        W 权重矩阵设备指针
 * @param A_ptr        A 激活矩阵设备指针
 * @param R_ptr        R 输出矩阵设备指针
 * @param N            N 维度 (输出行数)
 * @param K            K 维度 (共享维度)
 * @param M            M 维度 (batch size)
 * @param dtype        输入数据类型 ("int8" / "fp8e4m3")
 * @param outdtype     输出数据类型 ("bf16" / "fp32")
 * @param warmup       预热次数
 * @param repeat       计时重复次数
 *
 * 输出参数 (每个数组大小为 16):
 * @param out_layout_ids     布局 ID
 * @param out_layout_names   布局名称 (每个 32 字节)
 * @param out_lat_us         延迟 (微秒)
 * @param out_tops           吞吐量 (TOPS)
 * @param out_workspace      workspace 大小
 * @param out_best_alg_id    最佳算法 ID
 * @param out_split_k        split_k 参数
 * @param out_valid          是否有效
 * @param out_num_valid      有效布局数量
 * @param stream             CUDA 流 (可为 nullptr)
 *
 * @return 0 成功, -1 失败
 */
int cublaslt_layout_search_single(
    void* W_ptr, void* A_ptr, void* R_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    // 输出数组 (大小 = NUM_LAYOUTS = 16)
    int* out_layout_ids,
    char* out_layout_names,  // 16 * 32 = 512 bytes
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    int* out_best_alg_id,
    int* out_split_k,
    uint8_t* out_valid,
    int* out_num_valid,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    // 分配 workspace
    void* workspace = nullptr;
    size_t workspace_size = MAX_WORKSPACE_SIZE;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    int num_valid = 0;
    
    for (int i = 0; i < NUM_LAYOUTS; ++i) {
        LayoutResult result;
        memset(&result, 0, sizeof(result));
        
        int ret = test_single_layout(
            LAYOUT_CONFIGS[i], i,
            W_ptr, A_ptr, R_ptr,
            N, K, M,
            dtype, outdtype,
            warmup, repeat,
            workspace, workspace_size,
            cu_stream,
            &result);
        
        out_layout_ids[i] = result.layout_id;
        memcpy(out_layout_names + i * 32, result.layout_name, 32);
        out_lat_us[i] = result.lat_us;
        out_tops[i] = result.tops;
        out_workspace[i] = result.workspace;
        out_best_alg_id[i] = result.best_alg_id;
        out_split_k[i] = result.split_k;
        out_valid[i] = result.valid;
        
        if (result.valid) {
            num_valid++;
        }
    }
    
    *out_num_valid = num_valid;
    
    cudaFree(workspace);
    
    return 0;
}

/**
 * @brief 获取布局名称
 */
const char* cublaslt_layout_get_name(int layout_id) {
    if (layout_id < 0 || layout_id >= NUM_LAYOUTS) {
        return "INVALID";
    }
    return LAYOUT_CONFIGS[layout_id].name;
}

/**
 * @brief 获取布局数量
 */
int cublaslt_layout_get_count() {
    return NUM_LAYOUTS;
}

/**
 * @brief 检查 cuBLASLt 是否可用
 */
int cublaslt_layout_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 获取最后一条错误信息
 */
const char* cublaslt_layout_search_get_last_error() {
    return tls_last_error.c_str();
}

}  // extern "C"
