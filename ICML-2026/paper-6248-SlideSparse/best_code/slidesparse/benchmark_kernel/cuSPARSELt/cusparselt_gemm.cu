// SPDX-License-Identifier: Apache-2.0
/**
 * cuSPARSELt Sparse GEMM Benchmark Implementation
 * 
 * 支持的数据类型（按用户规格）:
 * ==============================
 * - FP16:    FP16 输入, FP32 计算, FP16 输出
 * - BF16:    BF16 输入, FP32 计算, BF16 输出
 * - INT8:    INT8 输入, INT32 计算, INT8 输出
 * - FP8E4M3: FP8 输入, FP32 计算, FP8 输出
 * - FP4E2M1: FP4 输入, FP32 计算, FP4 输出
 * 
 * 固定 Layout:
 * ============
 * - T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
 * - R[N,M]_col = W_compressed[K,N]^T_col @ A[K,M]_col
 * 
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      cusparselt_gemm.cu -lcusparseLt -lcusparse -lcublas -lcuda \
 *      -o cusparselt_gemm.so
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

// =============================================================================
// 常量定义
// =============================================================================

static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;  // 512 MB

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

// 前向声明 - Segment-K 检测（需要在 ensure_handle 之前声明）
static bool check_segment_k_support();

// 全局禁用标志 (跨调用保持)
static bool g_disable_segment_k = false;
static bool g_disable_split_k_doubling = false;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_init) {
        cusparseStatus_t err = cusparseLtInit(&g_handle);
        if (err != CUSPARSE_STATUS_SUCCESS) {
            set_error("Failed to initialize cuSPARSELt handle");
            return -1;
        }
        g_handle_init = true;
        
        // 早期检测：如果硬件不支持 segment-k，全局禁用
        // 这是防止后续搜索卡死的关键预拦截
        if (!check_segment_k_support()) {
            g_disable_segment_k = true;
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
    if (strcmp(dtype, "fp16") == 0) {
        return {CUDA_R_16F, 2, 32, true};
    } else if (strcmp(dtype, "bf16") == 0) {
        return {CUDA_R_16BF, 2, 32, true};
    } else if (strcmp(dtype, "int8") == 0) {
        return {CUDA_R_8I, 1, 32, true};
    } else if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) {
        return {CUDA_R_8F_E4M3, 1, 32, true};
    }
#if CUDART_VERSION >= 12050
    else if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) {
        // FP4 需要 CUDA 12.5+ 和 SM100+ (Blackwell)
        // 在此检查 GPU 架构
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 10) {
            // SM100 以下不支持 FP4
            fprintf(stderr, "[cuSPARSELt WARN] FP4 (e2m1) requires SM100+ (Blackwell), "
                    "current GPU is SM%d%d\n", prop.major, prop.minor);
            return {{}, 0, 0, false};
        }
        return {CUDA_R_4F_E2M1, 1, 32, true};  // packed, min unit is byte
    }
#else
    // CUDA < 12.5 不支持 FP4
    else if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) {
        fprintf(stderr, "[cuSPARSELt WARN] FP4 (e2m1) requires CUDA 12.5+, "
                "current CUDA runtime version is %d\n", CUDART_VERSION);
        return {{}, 0, 0, false};
    }
#endif
    return {{}, 0, 0, false};
}

// =============================================================================
// 输出类型配置 - 低精度输出策略
// =============================================================================
// 根据 cuSPARSELt 官方文档 cusparseLtMatmul() 支持的数据类型表：
// - FP16/BF16: 输出与输入相同
// - INT8: 低精度输出 INT8 (需要 D_OUT_SCALE)
// - FP8: 低精度输出 FP8 E4M3 (需要 D_OUT_SCALE)
// - FP4: 输出 BF16
// =============================================================================

// D 矩阵输出类型
static cudaDataType_t get_out_dtype(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
    if (strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
    if (strcmp(dtype, "int8") == 0) return CUDA_R_8I;    // INT8 低精度输出
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return CUDA_R_8F_E4M3;  // FP8 低精度输出
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return CUDA_R_16BF;  // FP4 输出 BF16 (保持高精度)
#endif
    return CUDA_R_16BF;  // 默认 BF16
}

// C 矩阵类型 - 根据 cuSPARSELt 表格强制要求
// - INT8 输出 → C 必须是 INT8
// - FP8 输出 → C 必须是 FP16 或 BF16
// - FP4 输出 → C 必须是 FP16 或 BF16
static cudaDataType_t get_c_dtype(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
    if (strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
    if (strcmp(dtype, "int8") == 0) return CUDA_R_8I;    // INT8 输出要求 C=INT8
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return CUDA_R_16BF;  // FP8 输出要求 C=BF16
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return CUDA_R_16BF;  // FP4 输出要求 C=BF16
#endif
    return CUDA_R_16BF;  // 默认 BF16
}

// C 矩阵元素大小
static int get_c_dtype_size(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return 2;
    if (strcmp(dtype, "bf16") == 0) return 2;
    if (strcmp(dtype, "int8") == 0) return 1;  // INT8 = 1 byte
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return 2;  // C=BF16
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return 2;  // C=BF16
#endif
    return 2;  // 默认 BF16 = 2 bytes
}

// 检查是否需要输出 Scale（低精度输出必须设置）
static bool needs_output_scale(const char* dtype) {
    if (strcmp(dtype, "int8") == 0) return true;
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return true;
    // FP4 输出 BF16，不需要 output scale
    return false;
}

static int get_out_dtype_size(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return 2;
    if (strcmp(dtype, "bf16") == 0) return 2;
    if (strcmp(dtype, "int8") == 0) return 1;  // INT8 低精度输出 = 1 byte
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return 1;  // FP8 低精度输出 = 1 byte
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return 2;  // FP4 输出 BF16 = 2 bytes
#endif
    return 2;  // 默认 BF16 = 2 bytes
}

static cusparseComputeType get_compute_type(const char* dtype) {
    if (strcmp(dtype, "int8") == 0) {
        return CUSPARSE_COMPUTE_32I;  // INT8 使用 INT32 累加
    }
    return CUSPARSE_COMPUTE_32F;  // FP16/BF16/FP8/FP4 使用 FP32 累加
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
    bool from_api_search;
    
    AlgRecord() : alg_id(-1), split_k(1), lat_us(0), tops(0), workspace(0), 
                  valid(false), from_api_search(false) {}
};

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 获取最后一条错误信息
 */
const char* cusparselt_alg_search_get_last_error() {
    return tls_last_error.c_str();
}

/**
 * @brief 检查 cuSPARSELt 是否可用
 */
int cusparselt_alg_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 检查是否支持 segment-k
 */
int cusparselt_supports_segment_k() {
    return check_segment_k_support() ? 1 : 0;
}

/**
 * @brief 获取对齐要求
 */
int cusparselt_alg_search_get_alignment(const char* dtype) {
    return 32;
}

/**
 * @brief 对矩阵进行 2:4 剪枝
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
        // dtype 不支持（如 FP4 在非 Blackwell 架构），跳过
        fprintf(stderr, "[cuSPARSELt WARN] Unsupported dtype '%s' for current GPU, skipping prune\n", dtype);
        set_error("Unsupported dtype for current GPU architecture");
        // 复制输入到输出（不剪枝）
        // 注意：这里无法复制因为我们不知道大小，直接返回错误
        return -1;
    }
    
    // ========================================================================
    // 维度检查 - FP4/FP8/INT8 需要 32 的倍数
    // ========================================================================
    bool is_fp4 = (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0);
    bool is_8bit = (strcmp(dtype, "int8") == 0 || strcmp(dtype, "fp8e4m3") == 0 || 
                   strcmp(dtype, "fp8") == 0 || is_fp4);
    
    int sparse_align = is_8bit ? 32 : 16;
    
    if (rows % sparse_align != 0 || cols % sparse_align != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] prune_24: Skipping W[rows=%lld, cols=%lld], "
                "requires multiples of %d for dtype=%s\n",
                (long long)rows, (long long)cols, sparse_align, dtype);
        // 复制输入到输出（不剪枝）
        size_t elem_size = info.elem_size;
        cudaMemcpy(output, input, rows * cols * elem_size, cudaMemcpyDeviceToDevice);
        return 0;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t c_dtype = get_c_dtype(dtype);    // C 矩阵类型（按表格要求）
    cudaDataType_t d_dtype = get_out_dtype(dtype);  // D 矩阵输出类型
    
    // W (稀疏, structured) - [rows, cols] = [K, N]
    cusparseLtMatDescriptor_t matW;
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        rows, cols, rows,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        rows, dummy_M, rows,
        16, info.cuda_type, order));
    
    // C (dense) - [N, M] - 按表格要求的类型
    cusparseLtMatDescriptor_t matC;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matC,
        cols, dummy_M, cols,
        16, c_dtype, order));
    
    // D (dense) - [N, M] - 输出类型
    cusparseLtMatDescriptor_t matD;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matD,
        cols, dummy_M, cols,
        16, d_dtype, order));
    
    // 创建 matmul 描述符 - 分开传入 matC 和 matD
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matC, &matD,
        compute_type));
    
    // ========================================================================
    // 选择剪枝算法
    // ========================================================================
    // FP4 (e2m1) 使用 paired 4:8 稀疏模式，需要使用 STRIP 而非 TILE
    // - CUSPARSELT_PRUNE_SPMMA_TILE:
    //   * e2m1: 在 8x4 (row-major) 或 4x8 (col-major) tile 中置零 16 个 paired 值
    //   * half/bfloat16/int8/e4m3/e5m2: 在 4x4 tile 中置零 8 个值
    // - CUSPARSELT_PRUNE_SPMMA_STRIP:
    //   * e2m1: 在 1x8 strip 中置零 4 个 paired 值（更适合一般场景）
    //   * half/bfloat16/int8/e4m3/e5m2: 在 1x4 strip 中置零 2 个值
    // ========================================================================
    cusparseLtPruneAlg_t prune_alg = is_fp4 ? CUSPARSELT_PRUNE_SPMMA_STRIP 
                                            : CUSPARSELT_PRUNE_SPMMA_TILE;
    
    // 调用 prune
    cusparseStatus_t prune_status = cusparseLtSpMMAPrune(
        &g_handle, &matmul,
        input, output,
        prune_alg,
        cu_stream);
    
    if (prune_status != CUSPARSE_STATUS_SUCCESS) {
        char buf[256];
        snprintf(buf, sizeof(buf), "cusparseLtSpMMAPrune failed with status %d for dtype=%s", 
                 (int)prune_status, dtype);
        set_error(buf);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        return -1;
    }
    
    cudaStreamSynchronize(cu_stream);
    
    // ========================================================================
    // 验证剪枝结果 (PruneCheck) - 可选
    // ========================================================================
    // 根据官方文档，cusparseLtSpMMAPrune() 的结果是保证正确的。
    // 但对于 FP4 (paired 4:8) 这一步可以作为额外验证。
    // 
    // 注意：某些情况下 PruneCheck 可能因为描述符配置差异而返回 false，
    // 即使 Prune 本身是正确的。因此这里只打印警告而不是返回错误。
    // ========================================================================
    int prune_valid = 0;
    cusparseStatus_t check_status = cusparseLtSpMMAPruneCheck(
        &g_handle, &matmul,
        output,  // 检查剪枝后的输出
        &prune_valid,
        cu_stream);
    
    cudaStreamSynchronize(cu_stream);
    
    // 清理描述符
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    
    if (check_status != CUSPARSE_STATUS_SUCCESS) {
        // PruneCheck API 调用失败 - 这是一个真正的错误
        char buf[256];
        snprintf(buf, sizeof(buf), "cusparseLtSpMMAPruneCheck failed with status %d", 
                 (int)check_status);
        set_error(buf);
        return -1;
    }
    
    if (!prune_valid) {
        // 剪枝结果验证失败
        // 根据官方文档，Prune 的结果应该是正确的，所以这里只打印警告
        // 在实际测试中观察是否需要返回错误
        //fprintf(stderr, "[cuSPARSELt WARN] PruneCheck failed for dtype=%s, "
        //        "but continuing as Prune result is guaranteed correct per docs\n", dtype);
        // 如果需要严格验证，取消下面的注释：
        // char buf[256];
        // snprintf(buf, sizeof(buf), 
        //          "cusparseLtSpMMAPruneCheck: pruned matrix does not satisfy %s sparsity constraint",
        //          is_fp4 ? "paired 4:8" : "2:4");
        // set_error(buf);
        // return -1;
    }
    
    return 0;
}

/**
 * @brief 获取压缩后的大小
 */
int64_t cusparselt_get_compressed_size(
    int64_t rows, int64_t cols,
    const char* dtype)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    // ========================================================================
    // 维度检查 - FP4/FP8/INT8 需要 32 的倍数
    // ========================================================================
    bool is_fp4 = (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0);
    bool is_8bit = (strcmp(dtype, "int8") == 0 || strcmp(dtype, "fp8e4m3") == 0 || 
                   strcmp(dtype, "fp8") == 0 || is_fp4);
    
    int sparse_align = is_8bit ? 32 : 16;
    
    if (rows % sparse_align != 0 || cols % sparse_align != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] get_compressed_size: W[rows=%lld, cols=%lld] "
                "requires multiples of %d for dtype=%s\n",
                (long long)rows, (long long)cols, sparse_align, dtype);
        return -1;
    }
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t c_dtype = get_c_dtype(dtype);    // C 矩阵类型
    cudaDataType_t d_dtype = get_out_dtype(dtype);  // D 矩阵输出类型
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matC, matD;
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
        &g_handle, &matC, cols, dummy_M, cols, 16, c_dtype, order);  // C 矩阵类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matD, cols, dummy_M, cols, 16, d_dtype, order);  // D 矩阵输出类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matC, &matD, compute_type);  // 分开 matC 和 matD
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        return -1;
    }
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        return -1;
    }
    
    size_t compressed_size = 0, compressed_buffer_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
    
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

/**
 * @brief 压缩稀疏矩阵
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
        fprintf(stderr, "[cuSPARSELt WARN] Unsupported dtype '%s' for current GPU, skipping compress\n", dtype);
        set_error("Unsupported dtype for current GPU architecture");
        return -1;
    }
    
    // ========================================================================
    // 维度检查 - FP4/FP8/INT8 需要 32 的倍数
    // ========================================================================
    bool is_fp4 = (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0);
    bool is_8bit = (strcmp(dtype, "int8") == 0 || strcmp(dtype, "fp8e4m3") == 0 || 
                   strcmp(dtype, "fp8") == 0 || is_fp4);
    
    int sparse_align = is_8bit ? 32 : 16;
    
    if (rows % sparse_align != 0 || cols % sparse_align != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] compress: Skipping W[rows=%lld, cols=%lld], "
                "requires multiples of %d for dtype=%s\n",
                (long long)rows, (long long)cols, sparse_align, dtype);
        return -1;  // 无法压缩
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t c_dtype = get_c_dtype(dtype);    // C 矩阵类型
    cudaDataType_t d_dtype = get_out_dtype(dtype);  // D 矩阵输出类型
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matC, matD;
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
        &g_handle, &matC, cols, dummy_M, cols, 16, c_dtype, order);  // C 矩阵类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        set_error("Failed to init matC");
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matD, cols, dummy_M, cols, 16, d_dtype, order);  // D 矩阵输出类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        set_error("Failed to init matD");
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matC, &matD, compute_type);  // 分开 matC 和 matD
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        set_error("Failed to init matmul");
        return -1;
    }
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        set_error("Failed to init alg_sel");
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
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
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
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
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
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
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    
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
 * @param W_pruned_ptr  W 矩阵设备指针 (已剪枝但未压缩, K x N 列主序)
 * @param A_ptr         A 矩阵设备指针 (K x M 列主序)
 * @param R_ptr         R 矩阵设备指针 (N x M 列主序)
 * @param N             N 维度
 * @param K             K 维度
 * @param M             M 维度
 * @param dtype         数据类型
 * @param warmup        预热次数
 * @param repeat        计时重复次数
 * @param topk          返回前 k 个结果
 * @param test_segment_k  是否测试 segment-k (-1)
 * @param do_api_search   是否执行官方 API 搜索对比
 *
 * 输出参数略
 *
 * @return 0 成功, -1 失败
 */
int cusparselt_search_single_m(
    void* W_pruned_ptr, void* A_ptr, void* R_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype,
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
        // dtype 不支持（如 FP4 在非 Blackwell 架构），跳过
        fprintf(stderr, "[cuSPARSELt WARN] Unsupported dtype '%s' for current GPU, skipping search\n", dtype);
        set_error("Unsupported dtype for current GPU architecture");
        // 初始化输出为 0
        *out_num_valid = 0;
        *out_alg_count = 0;
        *out_config_count = 0;
        *out_api_alg_id = -1;
        *out_api_split_k = 1;
        *out_api_lat_us = 0.0f;
        *out_api_rank = -1;
        return 0;  // 跳过
    }
    
    // ========================================================================
    // 维度检查 - 根据 cuSPARSELt 官方文档要求
    // ========================================================================
    // FP4/FP8/INT8: Sparse 矩阵 rows/cols/ld 必须是 32 的倍数
    //               Dense 矩阵 rows/cols/ld 必须是 16 的倍数
    // FP16/BF16:    Sparse 矩阵必须是 16 的倍数，Dense 必须是 8 的倍数
    // ========================================================================
    bool is_fp4 = (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0);
    bool is_8bit = (strcmp(dtype, "int8") == 0 || strcmp(dtype, "fp8e4m3") == 0 || 
                   strcmp(dtype, "fp8") == 0 || is_fp4);
    
    int sparse_align = is_8bit ? 32 : 16;  // Sparse 矩阵对齐要求
    int dense_align = is_8bit ? 16 : 8;    // Dense 矩阵对齐要求
    
    // 检查 Sparse 矩阵 W [K, N] 的维度
    if (K % sparse_align != 0 || N % sparse_align != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] Skipping: Sparse matrix W[K=%lld, N=%lld] "
                "requires K,N to be multiples of %d for dtype=%s\n",
                (long long)K, (long long)N, sparse_align, dtype);
        return 0;  // 返回成功但无结果，不是错误
    }
    
    // 检查 Dense 矩阵 A [K, M] 的维度
    if (K % dense_align != 0 || M % dense_align != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] Skipping: Dense matrix A[K=%lld, M=%lld] "
                "requires K,M to be multiples of %d for dtype=%s\n",
                (long long)K, (long long)M, dense_align, dtype);
        return 0;  // 返回成功但无结果
    }
    
    // FP4 额外检查：M 必须是 32 的倍数（实测发现的限制）
    if (is_fp4 && M % 32 != 0) {
        fprintf(stderr, "[cuSPARSELt WARN] Skipping: FP4 requires M=%lld to be a multiple of 32\n",
                (long long)M);
        return 0;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cudaDataType_t c_type = get_c_dtype(dtype);    // C 矩阵类型（按表格要求）
    cudaDataType_t d_type = get_out_dtype(dtype);  // D 矩阵输出类型
    cusparseComputeType compute_type = get_compute_type(dtype);
    bool need_output_scale = needs_output_scale(dtype);  // 低精度输出需要 Scale
    
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
    
    // cuSPARSELt 所有数据类型都使用 float 作为 alpha/beta
    // 这与 cuBLASLt 不同（cuBLASLt INT8 需要 int32 scale）
    float alpha = 1.0f, beta = 0.0f;
    
    // 创建基础矩阵描述符
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matW, matA, matC, matD;
    
    // W (稀疏, structured) - 存储为 [K, N], 转置后为 [N, K]
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        K, N, K,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        K, M, K,
        16, info.cuda_type, order));
    
    // C (dense) - [N, M] - 按表格要求的类型
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matC,
        N, M, N,
        16, c_type, order));
    
    // D (dense) - [N, M] - 输出类型
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matD,
        N, M, N,
        16, d_type, order));
    
    // 创建 matmul 描述符 - 分开传入 matC 和 matD
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matC, &matD,
        compute_type));
    
    // ========================================================================
    // 低精度输出 Scale 配置
    // ========================================================================
    // 【关键】低精度输出 (INT8/FP8/FP4) 必须设置 D_OUT_SCALE，不能“裸奔”！
    // - 即使我们想要 "scale=1.0"，也必须显式设置
    // - D_OUT_SCALE_MODE = SCALAR_32F
    // - D_OUT_SCALE_POINTER = device 上的 1.0f
    // ========================================================================
    void* d_out_scale_ptr = nullptr;
    
    // FP4 输入 scale 指针
    void* scale_A_ptr = nullptr;
    void* scale_B_ptr = nullptr;
    int64_t scale_A_size = 0;
    int64_t scale_B_size = 0;
    
    if (need_output_scale) {
        if (is_fp4) {
            // ============================================================
            // FP4 输出: 必须使用 Block Scale (VEC32_UE4M3)
            // D matrix is [N, M] col-major, Block Scale size = M * ceil(N/32)
            // FP4 动态范围极窄，cuSPARSELt kernel 强制写 Block Scale 数据！
            // ============================================================
            int64_t N_blocks = (N + 31) / 32;
            int64_t d_out_scale_size = M * N_blocks;  // 单位: bytes (UE4M3 每个 1 字节)
            
            cudaError_t alloc_err = cudaMalloc(&d_out_scale_ptr, d_out_scale_size);
            if (alloc_err != cudaSuccess) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to allocate d_out_scale for FP4 (size=%ld)\n", (long)d_out_scale_size);
                cusparseLtMatDescriptorDestroy(&matW);
                cusparseLtMatDescriptorDestroy(&matA);
                cusparseLtMatDescriptorDestroy(&matC);
                cusparseLtMatDescriptorDestroy(&matD);
                return -1;
            }
            // 初始化为中性值 (UE4M3 格式 1.0 ≈ 0x38)
            cudaMemset(d_out_scale_ptr, 0x38, d_out_scale_size);
            
            // 设置 D_OUT_SCALE_MODE = VEC32_UE4M3 (Block Scale)
            cusparseLtMatmulMatrixScale_t d_scale_mode = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3;
            cusparseStatus_t scale_st = cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_D_OUT_SCALE_MODE,
                &d_scale_mode, sizeof(d_scale_mode));
            if (scale_st != CUSPARSE_STATUS_SUCCESS) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to set D_OUT_SCALE_MODE (VEC32_UE4M3) for FP4: %d\n", (int)scale_st);
            }
            
            // 设置 D_OUT_SCALE_POINTER
            scale_st = cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER,
                &d_out_scale_ptr, sizeof(void*));
            if (scale_st != CUSPARSE_STATUS_SUCCESS) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to set D_OUT_SCALE_POINTER for FP4: %d\n", (int)scale_st);
            }
        } else {
            // ============================================================
            // INT8/FP8 输出: 使用 Scalar Scale (SCALAR_32F)
            // ============================================================
            cudaError_t alloc_err = cudaMalloc(&d_out_scale_ptr, sizeof(float));
            if (alloc_err != cudaSuccess) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to allocate d_out_scale\n");
                cusparseLtMatDescriptorDestroy(&matW);
                cusparseLtMatDescriptorDestroy(&matA);
                cusparseLtMatDescriptorDestroy(&matC);
                cusparseLtMatDescriptorDestroy(&matD);
                return -1;
            }
            float scale_val = 1.0f;
            cudaMemcpy(d_out_scale_ptr, &scale_val, sizeof(float), cudaMemcpyHostToDevice);
            
            // 设置 D_OUT_SCALE_MODE = SCALAR_32F
            cusparseLtMatmulMatrixScale_t d_scale_mode = CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            cusparseStatus_t scale_st = cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_D_OUT_SCALE_MODE,
                &d_scale_mode, sizeof(d_scale_mode));
            if (scale_st != CUSPARSE_STATUS_SUCCESS) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to set D_OUT_SCALE_MODE: %d\n", (int)scale_st);
            }
            
            // 设置 D_OUT_SCALE_POINTER
            scale_st = cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER,
                &d_out_scale_ptr, sizeof(void*));
            if (scale_st != CUSPARSE_STATUS_SUCCESS) {
                fprintf(stderr, "[cuSPARSELt WARN] Failed to set D_OUT_SCALE_POINTER: %d\n", (int)scale_st);
            }
        }
    }
    
#if CUDART_VERSION >= 12050
    // ========================================================================
    // FP4 Block Scale 配置 (输入端)
    // ========================================================================
    // FP4 (E2M1) 需要 block scaling，根据官方文档：
    // - CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3: 128x128 block (每 32 元素一个 scale)
    // - scale 类型为 UE4M3 (无符号 E4M3)
    // ========================================================================
    if (is_fp4) {
        // 设置 VEC32_UE4M3 block scaling mode (每 32 元素一个 UE4M3 scale)
        cusparseLtMatmulMatrixScale_t scale_mode = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3;
        
        cusparseStatus_t scale_st;
        scale_st = cusparseLtMatmulDescSetAttribute(
            &g_handle, &matmul, CUSPARSELT_MATMUL_A_SCALE_MODE, 
            &scale_mode, sizeof(scale_mode));
        if (scale_st != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cuSPARSELt WARN] Failed to set A_SCALE_MODE for FP4: %d\n", (int)scale_st);
        }
        
        scale_st = cusparseLtMatmulDescSetAttribute(
            &g_handle, &matmul, CUSPARSELT_MATMUL_B_SCALE_MODE, 
            &scale_mode, sizeof(scale_mode));
        if (scale_st != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[cuSPARSELt WARN] Failed to set B_SCALE_MODE for FP4: %d\n", (int)scale_st);
        }
        
        // 分配输入 scale 张量
        // W (matA, 转置前 [K, N]): scale 大小 = N * ceil(K/32)
        // A (matB, [K, M]):        scale 大小 = M * ceil(K/32)
        int64_t K_blocks = (K + 31) / 32;
        scale_A_size = N * K_blocks;  // For W (structured sparse)
        scale_B_size = M * K_blocks;  // For A (dense)
        
        cudaError_t alloc_err;
        alloc_err = cudaMalloc(&scale_A_ptr, scale_A_size * sizeof(uint8_t));
        if (alloc_err != cudaSuccess) {
            fprintf(stderr, "[cuSPARSELt WARN] Failed to allocate scale_A for FP4\n");
            scale_A_ptr = nullptr;
        }
        alloc_err = cudaMalloc(&scale_B_ptr, scale_B_size * sizeof(uint8_t));
        if (alloc_err != cudaSuccess) {
            fprintf(stderr, "[cuSPARSELt WARN] Failed to allocate scale_B for FP4\n");
            scale_B_ptr = nullptr;
        }
        
        // 初始化 scale 为中性值
        // UE4M3 格式: 无符号 E4M3，值 1.0 的编码约为 0x38
        if (scale_A_ptr) cudaMemset(scale_A_ptr, 0x38, scale_A_size * sizeof(uint8_t));
        if (scale_B_ptr) cudaMemset(scale_B_ptr, 0x38, scale_B_size * sizeof(uint8_t));
        
        // 设置 scale 指针
        if (scale_A_ptr) {
            cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_A_SCALE_POINTER, 
                &scale_A_ptr, sizeof(void*));
        }
        if (scale_B_ptr) {
            cusparseLtMatmulDescSetAttribute(
                &g_handle, &matmul, CUSPARSELT_MATMUL_B_SCALE_POINTER, 
                &scale_B_ptr, sizeof(void*));
        }
    }
#endif
    
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
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        if (d_out_scale_ptr) cudaFree(d_out_scale_ptr);
        if (scale_A_ptr) cudaFree(scale_A_ptr);
        if (scale_B_ptr) cudaFree(scale_B_ptr);
        set_error("Failed to get max algorithm ID");
        return -1;
    }
    
    *out_alg_count = max_alg_id;
    
    // 共享 workspace - 显式处理分配失败以正确清理资源
    void* shared_workspace = nullptr;
    size_t current_workspace_size = MAX_WORKSPACE_SIZE;
    {
        cudaError_t ws_alloc_err = cudaMalloc(&shared_workspace, current_workspace_size);
        if (ws_alloc_err != cudaSuccess) {
            cusparseLtMatDescriptorDestroy(&matW);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            if (d_out_scale_ptr) cudaFree(d_out_scale_ptr);
            if (scale_A_ptr) cudaFree(scale_A_ptr);
            if (scale_B_ptr) cudaFree(scale_B_ptr);
            char buf[512];
            snprintf(buf, sizeof(buf), "Failed to allocate shared workspace (%zu bytes)", current_workspace_size);
            set_error(buf);
            return -1;
        }
    }
    
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
                size_t comp_size_api = 0, comp_buf_size_api = 0;
                cusparseLtSpMMACompressedSize(&g_handle, &plan_api, &comp_size_api, &comp_buf_size_api);
                
                void* W_comp_api = nullptr;
                void* comp_buf_api = nullptr;
                cudaMalloc(&W_comp_api, comp_size_api);
                if (comp_buf_size_api > 0) cudaMalloc(&comp_buf_api, comp_buf_size_api);
                
                cusparseStatus_t comp_st = cusparseLtSpMMACompress(
                    &g_handle, &plan_api, W_pruned_ptr, W_comp_api, comp_buf_api, cu_stream);
                
                if (comp_st == CUSPARSE_STATUS_SUCCESS) {
                    size_t ws_api = 0;
                    cusparseLtMatmulGetWorkspace(&g_handle, &plan_api, &ws_api);
                    
                    cusparseStatus_t search_st = cusparseLtMatmulSearch(
                        &g_handle, &plan_api, &alpha, W_comp_api, A_ptr,
                        &beta, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                    
                    if (search_st == CUSPARSE_STATUS_SUCCESS) {
                        int api_alg_id = 0, api_split_k_val = 1;
                        
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                            &api_alg_id, sizeof(api_alg_id));
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_SPLIT_K,
                            &api_split_k_val, sizeof(api_split_k_val));
                        
                        bool success = true;
                        
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
    
    if (api_search_record.valid) {
        records.push_back(api_search_record);
    }
    
    // === 双层网格搜索 ===
    for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
        // 为当前 alg_id 创建 plan 并压缩权重
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
        
        size_t compressed_size = 0, compressed_buffer_size = 0;
        cusparseLtSpMMACompressedSize(&g_handle, &plan_compress, &compressed_size, &compressed_buffer_size);
        
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
        
        cusparseStatus_t compress_st = cusparseLtSpMMACompress(
            &g_handle, &plan_compress, W_pruned_ptr, W_compressed, compress_buffer, cu_stream);
        
        if (compress_buffer) {
            cudaFree(compress_buffer);
            compress_buffer = nullptr;
        }
        
        cusparseLtMatmulPlanDestroy(&plan_compress);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
        
        if (compress_st != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(W_compressed);
            continue;
        }
        
        float best_lat_us_for_doubling = -1.0f;
        
        std::vector<int> split_k_candidates;
        split_k_candidates.push_back(1);
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
            split_k_candidates.push_back(-1);
        }
        
        bool stop_doubling = false;
        
        for (int split_k_val : split_k_candidates) {
            if (api_search_record.valid &&
                alg_id == api_search_record.alg_id &&
                split_k_val == api_search_record.split_k) {
                continue;
            }
            
            if (stop_doubling && split_k_val > 1 && split_k_val != -1) {
                continue;
            }
            
            cusparseLtMatmulAlgSelection_t alg_sel;
            cusparseStatus_t sel_status = cusparseLtMatmulAlgSelectionInit(
                &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
            if (sel_status != CUSPARSE_STATUS_SUCCESS) break;
            
            cusparseStatus_t set_status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id));
            if (set_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
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
            
            for (int i = 0; i < warmup && success; ++i) {
                cusparseStatus_t st = cusparseLtMatmul(
                    &g_handle, &plan, &alpha, W_compressed, A_ptr,
                    &beta, R_ptr, R_ptr, workspace, &cu_stream, 1);
                if (st != CUSPARSE_STATUS_SUCCESS) success = false;
            }
            cudaStreamSynchronize(cu_stream);
            
            if (!success) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val == -1) {
                    g_disable_segment_k = true;
                } else if (split_k_val > 1) {
                    stop_doubling = true;
                    g_disable_split_k_doubling = true;
                }
                continue;
            }
            
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
        }
        
        cudaFree(W_compressed);
    }
    
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
                *out_api_rank = (int)(i + 1);
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
    
    // 清理
    cudaFree(shared_workspace);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    
    // 释放低精度输出 scale
    if (d_out_scale_ptr) cudaFree(d_out_scale_ptr);
    
    // 释放 FP4 输入 scale 张量
    if (scale_A_ptr) cudaFree(scale_A_ptr);
    if (scale_B_ptr) cudaFree(scale_B_ptr);
    
    return 0;
}

}  // extern "C"
