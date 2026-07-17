// SPDX-License-Identifier: Apache-2.0
/**
 * @file alg_search_cusparselt.cu
 * @brief cuSPARSELt 算法离线搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuSPARSELt 稀疏矩阵乘法的算法搜索功能，包含:
 *   - 2:4 稀疏模式 (prune_24)
 *   - Split-K 自适应倍增搜索
 *   - Segment-K 检测 (SM90+)
 *   - 官方 API 搜索对比
 *
 * 固定 Layout:
 * - T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
 *
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      alg_search_cusparselt.cu -lcusparseLt -lcusparse -lcublas \
 *      -o alg_search_cusparselt.so
 *
 * 主要接口:
 * ---------
 * - cusparselt_search_single_m()      : 搜索单个 (N,K,M) 的最优算法
 * - cusparselt_prune_24()             : 对矩阵进行 2:4 剪枝
 * - cusparselt_supports_segment_k()   : 检测是否支持 segment-k
 *
 * API 注意事项 (新版 cuSPARSELt):
 * - cusparseLtSpMMAPrune() 第二参数是 cusparseLtMatmulDescriptor_t*
 * - cusparseLtMatmulPlanInit() 只有 4 个参数 (移除了 workspace size)
 * - cusparseLtSpMMACompressedSize() 需要 cusparseLtMatmulPlan_t*
 * - cusparseLtSpMMACompress() 需要 cusparseLtMatmulPlan_t*
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

// =============================================================================
// 常量定义
// =============================================================================

// 最大 workspace 大小: 512 MB
static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;

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

#define CHECK_CUSPARSELT(call) do { \
    cusparseStatus_t err = (call); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "cuSPARSELt Error %d at %s:%d", \
                 (int)err, __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

// =============================================================================
// 全局资源
// =============================================================================

static cusparseLtHandle_t g_handle;
static bool g_handle_init = false;
static std::mutex g_handle_mutex;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_init) {
        cusparseStatus_t err = cusparseLtInit(&g_handle);
        if (err != CUSPARSE_STATUS_SUCCESS) {
            set_error("Failed to initialize cuSPARSELt handle");
            return -1;
        }
        g_handle_init = true;
    }
    return 0;
}

// =============================================================================
// 全局禁用标志 (跨调用保持)
// =============================================================================
// 使用静态变量，当发现硬件不支持某功能时全局禁用，避免后续重复尝试

static bool g_disable_segment_k = false;      // Segment-K 全局禁用
static bool g_disable_split_k_doubling = false;  // Split-K 倍增全局禁用

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
    return CUDA_R_16BF;
}

static int get_out_dtype_size(const char* outdtype) {
    if (strcmp(outdtype, "bf16") == 0 || strcmp(outdtype, "BF16") == 0) {
        return 2;
    } else if (strcmp(outdtype, "fp32") == 0 || strcmp(outdtype, "FP32") == 0) {
        return 4;
    } else if (strcmp(outdtype, "int32") == 0 || strcmp(outdtype, "INT32") == 0) {
        return 4;
    }
    return 2;
}

static cusparseComputeType get_compute_type(const char* dtype) {
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return CUSPARSE_COMPUTE_32I;
    }
    return CUSPARSE_COMPUTE_32F;
}

// =============================================================================
// Segment-K 检测
// =============================================================================

static bool check_segment_k_support() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Segment-K 需要 SM90+ (Hopper 及更高)
    return (prop.major >= 9);
}

// =============================================================================
// 算法搜索结构
// =============================================================================

struct AlgRecord {
    int alg_id;
    int split_k;
    float lat_us;
    float tops;
    int64_t workspace;
    bool valid;
    bool from_api_search;  // 是否来自官方 API 搜索
    float max_abs_err;     // 验证误差（如果启用验证）
    
    AlgRecord() : alg_id(-1), split_k(1), lat_us(0), tops(0), workspace(0), 
                  valid(false), from_api_search(false), max_abs_err(0) {}
};

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 对矩阵进行 2:4 剪枝
 * 
 * 重要: 新版 cuSPARSELt API 要求 prune 操作必须使用 matmulDescriptor，
 * 因此这里需要创建完整的 matmul context。
 * 
 * @param input   输入矩阵设备指针 [rows x cols, 列主序]
 * @param output  输出矩阵设备指针
 * @param rows    行数 (K)
 * @param cols    列数 (N)
 * @param dtype   数据类型
 * @param stream  CUDA 流
 * @return 0 成功, -1 失败
 */
int cusparselt_prune_24(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype for prune");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    // 使用一个虚拟 M 值创建完整的 matmul context
    int64_t dummy_M = 16;
    
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // W (稀疏, structured) - [rows, cols] = [K, N]
    cusparseLtMatDescriptor_t matW;
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        rows, cols, rows,  // ld = rows for col-major
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        rows, dummy_M, rows,
        16, info.cuda_type, order));
    
    // R (dense) - [N, M]
    cusparseLtMatDescriptor_t matR;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matR,
        cols, dummy_M, cols,
        16, CUDA_R_16BF, order));  // 使用 bf16 输出
    
    // 创建 matmul 描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR,
        compute_type));
    
    // 调用 prune (使用 matmulDescriptor)
    cusparseStatus_t prune_status = cusparseLtSpMMAPrune(
        &g_handle, &matmul,
        input, output,
        CUSPARSELT_PRUNE_SPMMA_TILE,
        cu_stream);
    
    // 清理
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    if (prune_status != CUSPARSE_STATUS_SUCCESS) {
        char buf[256];
        snprintf(buf, sizeof(buf), "cusparseLtSpMMAPrune failed with status %d", (int)prune_status);
        set_error(buf);
        return -1;
    }
    
    cudaStreamSynchronize(cu_stream);
    
    return 0;
}

/**
 * @brief 获取压缩后的大小
 * 
 * 注意: 新版 API 需要 plan，这里创建临时 plan 来获取大小
 */
int64_t cusparselt_get_compressed_size(
    int64_t rows, int64_t cols,
    const char* dtype)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    int64_t dummy_M = 16;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matR;
    cusparseStatus_t status;
    
    status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matW, rows, cols, rows, 16,
        info.cuda_type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matA, rows, dummy_M, rows, 16, info.cuda_type, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matR, cols, dummy_M, cols, 16, CUDA_R_16BF, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    // 创建 alg selection 和 plan
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    // 新版 API: 4 参数
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    // 获取压缩大小
    size_t compressed_size = 0, compressed_buffer_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
    
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

/**
 * @brief 压缩稀疏矩阵
 * 
 * 注意: 新版 API 需要 plan
 */
int64_t cusparselt_compress(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype for compress");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    int64_t dummy_M = 16;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matR;
    cusparseStatus_t status;
    
    status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matW, rows, cols, rows, 16,
        info.cuda_type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) { set_error("Failed to init matW"); return -1; }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matA, rows, dummy_M, rows, 16, info.cuda_type, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        set_error("Failed to init matA");
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matR, cols, dummy_M, cols, 16, CUDA_R_16BF, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        set_error("Failed to init matR");
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init matmul");
        return -1;
    }
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init alg_sel");
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init plan");
        return -1;
    }
    
    // 获取压缩大小
    size_t compressed_size = 0, compressed_buffer_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to get compressed size");
        return -1;
    }
    
    // 分配压缩 buffer
    void* compress_buffer = nullptr;
    if (compressed_buffer_size > 0) {
        cudaError_t alloc_err = cudaMalloc(&compress_buffer, compressed_buffer_size);
        if (alloc_err != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            cusparseLtMatDescriptorDestroy(&matW);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matR);
            set_error("Failed to allocate compress buffer");
            return -1;
        }
    }
    
    // 压缩
    status = cusparseLtSpMMACompress(&g_handle, &plan, input, output, compress_buffer, cu_stream);
    
    if (compress_buffer) cudaFree(compress_buffer);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        set_error("cusparseLtSpMMACompress failed");
        return -1;
    }
    
    cudaStreamSynchronize(cu_stream);
    
    return (int64_t)compressed_size;
}

/**
 * @brief 搜索单个 (N,K,M) 组合的最优算法
 *
 * 采用双层网格搜索策略:
 * - 外层遍历 alg_id
 * - 内层自适应调整 split_k_val (1, 2, 4, 8, ... 及 -1 表示 Segment-K)
 * - 每个 alg_id 单独压缩权重
 * - 可选的官方 API 搜索对比
 *
 * @param W_pruned_ptr  W 矩阵设备指针 (已剪枝但未压缩, K x N 列主序)
 * @param A_ptr         A 矩阵设备指针 (K x M 列主序)
 * @param R_ptr         R 矩阵设备指针 (N x M 列主序)
 * @param N             N 维度
 * @param K             K 维度
 * @param M             M 维度
 * @param dtype         输入数据类型 ("int8" / "fp8e4m3")
 * @param outdtype      输出数据类型 ("bf16" / "fp32")
 * @param warmup        预热次数
 * @param repeat        计时重复次数
 * @param topk          返回前 k 个结果
 * @param test_segment_k  是否测试 segment-k (-1)
 * @param do_api_search   是否执行官方 API 搜索对比
 *
 * 输出参数:
 * @param out_alg_ids      算法 ID 数组 (大小 = topk)
 * @param out_split_k      split-k 值数组 (大小 = topk)
 * @param out_lat_us       延迟数组 (大小 = topk)
 * @param out_tops         吞吐量数组 (大小 = topk)
 * @param out_workspace    workspace 数组 (大小 = topk)
 * @param out_valid        有效标记数组 (大小 = topk)
 * @param out_num_valid    有效结果数量
 * @param out_alg_count    算法总数 (max_alg_id)
 * @param out_config_count 实测配置数
 * @param out_api_alg_id   API 搜索结果算法 ID (-1 表示无效)
 * @param out_api_split_k  API 搜索结果 split_k
 * @param out_api_lat_us   API 搜索结果延迟
 * @param out_api_rank     API 搜索结果在总排名中的位置 (1-based, -1 表示无效)
 * @param stream           CUDA 流 (可为 nullptr)
 *
 * @return 0 成功, -1 失败
 */
int cusparselt_search_single_m(
    void* W_pruned_ptr, void* A_ptr, void* R_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    int topk,
    int test_segment_k,
    int do_api_search,
    // 输出
    int* out_alg_ids,
    int* out_split_k,
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    uint8_t* out_valid,
    int* out_num_valid,
    int* out_alg_count,
    int* out_config_count,
    int* out_api_alg_id,
    int* out_api_split_k,
    float* out_api_lat_us,
    int* out_api_rank,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // 初始化输出
    *out_num_valid = 0;
    *out_alg_count = 0;
    *out_config_count = 0;
    *out_api_alg_id = -1;
    *out_api_split_k = 1;
    *out_api_lat_us = 0.0f;
    *out_api_rank = -1;
    
    for (int i = 0; i < topk; ++i) {
        out_alg_ids[i] = -1;
        out_split_k[i] = 0;
        out_lat_us[i] = 0;
        out_tops[i] = 0;
        out_workspace[i] = 0;
        out_valid[i] = 0;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 创建基础矩阵描述符
    // R = W^T * A  (W: K x N 存储, sparse, A: K x M dense, R: N x M)
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matW, matA, matR;
    
    // W (稀疏, structured) - 存储为 [K, N], 转置后为 [N, K]
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        K, N, K,  // rows=K, cols=N, ld=K
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        K, M, K,
        16, info.cuda_type, order));
    
    // R (dense) - [N, M]
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matR,
        N, M, N,
        16, out_type, order));
    
    // 创建 matmul 描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR,
        compute_type));
    
    // 获取最大算法 ID
    cusparseLtMatmulAlgSelection_t alg_sel_tmp;
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel_tmp, &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT));
    
    int max_alg_id = 0;
    cusparseStatus_t attr_status = cusparseLtMatmulAlgGetAttribute(
        &g_handle, &alg_sel_tmp,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id, sizeof(max_alg_id));
    
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    
    if (attr_status != CUSPARSE_STATUS_SUCCESS || max_alg_id <= 0) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to get max algorithm ID");
        return -1;
    }
    
    *out_alg_count = max_alg_id;
    
    // 共享 workspace
    void* shared_workspace = nullptr;
    size_t current_workspace_size = MAX_WORKSPACE_SIZE;
    CHECK_CUDA(cudaMalloc(&shared_workspace, current_workspace_size));
    
    // 收集所有有效结果
    std::vector<AlgRecord> records;
    int config_count = 0;
    
    // API 搜索结果
    AlgRecord api_search_record;
    api_search_record.valid = false;
    
    // === 官方 API 搜索 (可选) ===
    if (do_api_search) {
        cusparseLtMatmulAlgSelection_t alg_sel_api;
        cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_api, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        
        if (sel_st == CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlan_t plan_api;
            cusparseStatus_t plan_st = cusparseLtMatmulPlanInit(
                &g_handle, &plan_api, &matmul, &alg_sel_api);
            
            if (plan_st == CUSPARSE_STATUS_SUCCESS) {
                // 获取压缩大小
                size_t comp_size_api = 0, comp_buf_size_api = 0;
                cusparseLtSpMMACompressedSize(&g_handle, &plan_api, &comp_size_api, &comp_buf_size_api);
                
                // 分配压缩缓冲区
                void* W_comp_api = nullptr;
                void* comp_buf_api = nullptr;
                cudaMalloc(&W_comp_api, comp_size_api);
                if (comp_buf_size_api > 0) cudaMalloc(&comp_buf_api, comp_buf_size_api);
                
                // 压缩
                cusparseStatus_t comp_st = cusparseLtSpMMACompress(
                    &g_handle, &plan_api, W_pruned_ptr, W_comp_api, comp_buf_api, cu_stream);
                
                if (comp_st == CUSPARSE_STATUS_SUCCESS) {
                    // 获取 workspace
                    size_t ws_api = 0;
                    cusparseLtMatmulGetWorkspace(&g_handle, &plan_api, &ws_api);
                    
                    // 调用官方搜索 API
                    cusparseStatus_t search_st = cusparseLtMatmulSearch(
                        &g_handle, &plan_api, &alpha, W_comp_api, A_ptr,
                        &beta, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                    
                    if (search_st == CUSPARSE_STATUS_SUCCESS) {
                        // 提取选中的算法参数
                        int api_alg_id = 0;
                        int api_split_k_val = 1;
                        
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                            &api_alg_id, sizeof(api_alg_id));
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_SPLIT_K,
                            &api_split_k_val, sizeof(api_split_k_val));
                        
                        // 用相同方法测量性能
                        bool success = true;
                        
                        // 预热
                        for (int i = 0; i < warmup && success; ++i) {
                            cusparseStatus_t st = cusparseLtMatmul(
                                &g_handle, &plan_api, &alpha, W_comp_api, A_ptr,
                                &beta, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                            if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                        }
                        cudaStreamSynchronize(cu_stream);
                        
                        if (success) {
                            cudaEvent_t start, stop;
                            cudaEventCreate(&start);
                            cudaEventCreate(&stop);
                            cudaEventRecord(start, cu_stream);
                            
                            for (int r = 0; r < repeat && success; ++r) {
                                cusparseStatus_t st = cusparseLtMatmul(
                                    &g_handle, &plan_api, &alpha, W_comp_api, A_ptr,
                                    &beta, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                                if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                            }
                            
                            cudaEventRecord(stop, cu_stream);
                            cudaEventSynchronize(stop);
                            float total_ms = 0;
                            cudaEventElapsedTime(&total_ms, start, stop);
                            cudaEventDestroy(start);
                            cudaEventDestroy(stop);
                            
                            if (success) {
                                api_search_record.alg_id = api_alg_id;
                                api_search_record.split_k = api_split_k_val;
                                api_search_record.lat_us = (total_ms * 1000.0f) / (float)repeat;
                                double ops = 2.0 * (double)M * (double)N * (double)K;
                                api_search_record.tops = (float)(ops / (api_search_record.lat_us / 1e6) / 1e12);
                                api_search_record.workspace = (int64_t)ws_api;
                                api_search_record.valid = true;
                                api_search_record.from_api_search = true;
                            }
                        }
                    }
                }
                
                if (W_comp_api) cudaFree(W_comp_api);
                if (comp_buf_api) cudaFree(comp_buf_api);
                cusparseLtMatmulPlanDestroy(&plan_api);
            }
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_api);
        }
    }
    
    // 如果官方搜索成功，先加入 records
    if (api_search_record.valid) {
        records.push_back(api_search_record);
    }
    
    // === 双层网格搜索：外层遍历 alg_id，内层自适应调整 split_k_val ===
    for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
        
        // === 为当前 alg_id 创建 plan 并压缩权重 ===
        cusparseLtMatmulAlgSelection_t alg_sel_compress;
        cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_compress, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (sel_st != CUSPARSE_STATUS_SUCCESS) continue;
        
        cusparseStatus_t set_st = cusparseLtMatmulAlgSetAttribute(
            &g_handle, &alg_sel_compress, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &alg_id, sizeof(alg_id));
        if (set_st != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        
        cusparseLtMatmulPlan_t plan_compress;
        cusparseStatus_t plan_st = cusparseLtMatmulPlanInit(
            &g_handle, &plan_compress, &matmul, &alg_sel_compress);
        if (plan_st != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        
        // 获取压缩所需大小
        size_t compressed_size = 0, compressed_buffer_size = 0;
        cusparseLtSpMMACompressedSize(&g_handle, &plan_compress, &compressed_size, &compressed_buffer_size);
        
        // 分配压缩权重存储
        void* W_compressed = nullptr;
        void* compress_buffer = nullptr;
        cudaError_t alloc_st = cudaMalloc(&W_compressed, compressed_size);
        if (alloc_st != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan_compress);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        if (compressed_buffer_size > 0) {
            alloc_st = cudaMalloc(&compress_buffer, compressed_buffer_size);
            if (alloc_st != cudaSuccess) {
                cudaFree(W_compressed);
                cusparseLtMatmulPlanDestroy(&plan_compress);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
                continue;
            }
        }
        
        // 执行压缩
        cusparseStatus_t compress_st = cusparseLtSpMMACompress(
            &g_handle, &plan_compress, W_pruned_ptr, W_compressed, compress_buffer, cu_stream);
        
        // 释放压缩 buffer（压缩完成后不再需要）
        if (compress_buffer) {
            cudaFree(compress_buffer);
            compress_buffer = nullptr;
        }
        
        // 释放用于压缩的 plan 和 alg_sel（后续 split_k 测试会创建新的）
        cusparseLtMatmulPlanDestroy(&plan_compress);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
        
        if (compress_st != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(W_compressed);
            continue;
        }
        
        // 用于跟踪当前 alg_id 下的最优延时（自适应倍增策略）
        float best_lat_us_for_doubling = -1.0f;
        
        // 构建 split_k 候选列表
        std::vector<int> split_k_candidates;
        split_k_candidates.push_back(1);  // 总是先测试不切分
        if (!g_disable_split_k_doubling) {
            for (int sk = 2; sk <= K; sk *= 2) {
                split_k_candidates.push_back(sk);
            }
        }
        // Segment-K：需要 SM90+ 且 N >= 400
        // 实测发现 N < 400 时 segment-k 可能卡死，与 M/K 无关
        constexpr int64_t SEGMENT_K_MIN_N = 400;
        bool hw_supports_segment_k = check_segment_k_support();
        bool shape_safe_for_segment_k = (N >= SEGMENT_K_MIN_N);
        if (test_segment_k && !g_disable_segment_k && hw_supports_segment_k && shape_safe_for_segment_k) {
            split_k_candidates.push_back(-1);  // Segment-K
        }
        
        bool stop_doubling = false;
        
        for (int split_k_val : split_k_candidates) {
            // 去重检查：跳过与官方 API 搜索结果相同的配置
            if (api_search_record.valid &&
                alg_id == api_search_record.alg_id &&
                split_k_val == api_search_record.split_k) {
                continue;
            }
            
            // 自适应倍增策略：当前 alg_id 内的局部停止
            if (stop_doubling && split_k_val > 1 && split_k_val != -1) {
                continue;
            }
            
            cusparseLtMatmulAlgSelection_t alg_sel;
            cusparseStatus_t sel_status = cusparseLtMatmulAlgSelectionInit(
                &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
            if (sel_status != CUSPARSE_STATUS_SUCCESS) break;
            
            // 设置算法 ID
            cusparseStatus_t set_status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id));
            if (set_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            // 设置 Split-K
            cusparseStatus_t split_k_status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K,
                &split_k_val, sizeof(split_k_val));
            if (split_k_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val > 1) stop_doubling = true;
                continue;
            }
            
            cusparseLtMatmulPlan_t plan;
            cusparseStatus_t plan_status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
            if (plan_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                // 如果是 segment-k 或 split-k 倍增失败，全局禁用
                if (split_k_val == -1) {
                    g_disable_segment_k = true;
                } else if (split_k_val > 1) {
                    stop_doubling = true;
                    g_disable_split_k_doubling = true;
                }
                continue;
            }
            
            size_t workspace_size = 0;
            cusparseLtMatmulGetWorkspace(&g_handle, &plan, &workspace_size);
            
            if (workspace_size > current_workspace_size) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            void* workspace = (workspace_size > 0) ? shared_workspace : nullptr;
            bool success = true;
            
            // 预热
            for (int i = 0; i < warmup && success; ++i) {
                cusparseStatus_t st = cusparseLtMatmul(
                    &g_handle, &plan, &alpha, W_compressed, A_ptr,
                    &beta, R_ptr, R_ptr, workspace, &cu_stream, 1);
                if (st != CUSPARSE_STATUS_SUCCESS) success = false;
            }
            cudaStreamSynchronize(cu_stream);
            
            // 如果预热失败，停止倍增
            if (!success) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                // 预热阶段失败说明硬件不支持此配置，全局禁用
                if (split_k_val == -1) {
                    g_disable_segment_k = true;
                } else if (split_k_val > 1) {
                    stop_doubling = true;
                    g_disable_split_k_doubling = true;
                }
                continue;
            }
            
            // 计时
            cudaEvent_t start = nullptr, stop = nullptr;
            float total_ms = 0.0f;
            
            if (success) {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, cu_stream);
                
                for (int r = 0; r < repeat && success; ++r) {
                    cusparseStatus_t st = cusparseLtMatmul(
                        &g_handle, &plan, &alpha, W_compressed, A_ptr,
                        &beta, R_ptr, R_ptr, workspace, &cu_stream, 1);
                    if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                }
                
                cudaEventRecord(stop, cu_stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&total_ms, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            
            if (success) {
                AlgRecord rec;
                rec.alg_id = alg_id;
                rec.split_k = split_k_val;
                rec.lat_us = (total_ms * 1000.0f) / (float)repeat;
                double ops = 2.0 * (double)M * (double)N * (double)K;
                rec.tops = (float)(ops / (rec.lat_us / 1e6) / 1e12);
                rec.workspace = (int64_t)workspace_size;
                rec.valid = true;
                rec.from_api_search = false;
                
                records.push_back(rec);
                ++config_count;
                
                // 自适应倍增策略
                if (split_k_val >= 1) {
                    if (best_lat_us_for_doubling < 0 || rec.lat_us < best_lat_us_for_doubling) {
                        best_lat_us_for_doubling = rec.lat_us;
                    } else if (rec.lat_us * 1.05f > best_lat_us_for_doubling && split_k_val > 1) {
                        stop_doubling = true;
                    }
                }
            }
            
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        }  // end split_k loop
        
        cudaFree(W_compressed);
    }  // end alg_id loop
    
    *out_config_count = config_count;
    
    // 按延迟排序
    std::sort(records.begin(), records.end(),
              [](const AlgRecord& a, const AlgRecord& b) {
                  return a.lat_us < b.lat_us;
              });
    
    // 记录 API 搜索结果的排名
    if (api_search_record.valid) {
        for (size_t i = 0; i < records.size(); ++i) {
            if (records[i].from_api_search) {
                *out_api_rank = (int)(i + 1);  // 1-based
                *out_api_alg_id = records[i].alg_id;
                *out_api_split_k = records[i].split_k;
                *out_api_lat_us = records[i].lat_us;
                break;
            }
        }
    }
    
    // 填充输出
    *out_num_valid = (int)records.size();
    
    for (int i = 0; i < topk; ++i) {
        if (i < (int)records.size()) {
            const auto& r = records[i];
            out_alg_ids[i] = r.alg_id;
            out_split_k[i] = r.split_k;
            out_lat_us[i] = r.lat_us;
            out_tops[i] = r.tops;
            out_workspace[i] = r.workspace;
            out_valid[i] = 1;
        } else {
            out_alg_ids[i] = -1;
            out_split_k[i] = 0;
            out_lat_us[i] = 0;
            out_tops[i] = 0;
            out_workspace[i] = 0;
            out_valid[i] = 0;
        }
    }
    
    // === 使用最优算法重新执行一次 matmul，以便 Python 端验证 ===
    // 这解决了 R_ptr 被多次覆盖导致验证失败的问题
    if (!records.empty()) {
        const AlgRecord& best = records[0];
        
        // 重新创建 plan 并压缩
        cusparseLtMatmulAlgSelection_t alg_sel_verify;
        cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_verify, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        
        if (sel_st == CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel_verify, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &best.alg_id, sizeof(best.alg_id));
            cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel_verify, CUSPARSELT_MATMUL_SPLIT_K,
                &best.split_k, sizeof(best.split_k));
            
            cusparseLtMatmulPlan_t plan_verify;
            cusparseStatus_t plan_st = cusparseLtMatmulPlanInit(
                &g_handle, &plan_verify, &matmul, &alg_sel_verify);
            
            if (plan_st == CUSPARSE_STATUS_SUCCESS) {
                // 压缩权重
                size_t comp_size = 0, comp_buf_size = 0;
                cusparseLtSpMMACompressedSize(&g_handle, &plan_verify, &comp_size, &comp_buf_size);
                
                void* W_comp_verify = nullptr;
                void* comp_buf_verify = nullptr;
                cudaMalloc(&W_comp_verify, comp_size);
                if (comp_buf_size > 0) cudaMalloc(&comp_buf_verify, comp_buf_size);
                
                cusparseStatus_t compress_st = cusparseLtSpMMACompress(
                    &g_handle, &plan_verify, W_pruned_ptr, W_comp_verify, comp_buf_verify, cu_stream);
                
                if (compress_st == CUSPARSE_STATUS_SUCCESS) {
                    // 执行一次 matmul 写入 R_ptr
                    cusparseLtMatmul(&g_handle, &plan_verify, &alpha, W_comp_verify, A_ptr,
                                     &beta, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                    cudaStreamSynchronize(cu_stream);
                }
                
                if (W_comp_verify) cudaFree(W_comp_verify);
                if (comp_buf_verify) cudaFree(comp_buf_verify);
                cusparseLtMatmulPlanDestroy(&plan_verify);
            }
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_verify);
        }
    }
    
    // 清理
    cudaFree(shared_workspace);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    return 0;
}

/**
 * @brief 检查是否支持 segment-k
 */
int cusparselt_supports_segment_k() {
    return check_segment_k_support() ? 1 : 0;
}

/**
 * @brief 检查 cuSPARSELt 是否可用
 */
int cusparselt_alg_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 获取最后一条错误信息
 */
const char* cusparselt_alg_search_get_last_error() {
    return tls_last_error.c_str();
}

/**
 * @brief 获取对齐要求
 */
int cusparselt_alg_search_get_alignment(const char* dtype) {
    return 16;
}

}  // extern "C"
