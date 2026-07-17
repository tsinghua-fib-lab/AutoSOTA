// SPDX-License-Identifier: Apache-2.0
/**
 * cuSPARSELt Sparse GEMM Implementation for SlideSparse (extern "C" 版本)
 * 
 * 设计目标：
 * =========
 * 1. 使用 cuSPARSELt API 实现 2:4 结构化稀疏矩阵乘法（不带 scale/bias 融合）
 * 2. 支持 FP8E4M3 和 INT8 输入
 * 3. FP8 支持 BF16/FP32 输出，INT8 支持 BF16/INT32 输出（不支持 FP32）
 * 4. Dequant + bias 由后续 Triton kernel 处理
 * 
 * 计算流程：
 * =========
 * D[N,M]_col = W_slide_compressed[K',N]^T_col @ A_slide[K',M]_col
 * 
 * 其中 W_compressed 是 cuSPARSELt 压缩后的权重（由 compress.py 预处理）
 * 
 * cuSPARSELt 配置（TN-CC 布局）：
 * ==============================
 * - 布局：TN + CCC（W 转置，A 不转置，全列主序）
 * - W[N,K] 行主序 → 声明列主序 [K,N] → opW=T → [N,K]
 * - A[M,K] 行主序 → 声明列主序 [K,M] → opA=N → [K,M]
 * - D[N,M] 列主序结果 → 按行主序读 = [M,N]
 * 
 * 支持的数据类型组合：
 * ===================
 * - FP8 输入: inner_dtype = "bf16" 或 "fp32"，compute_type = CUSPARSE_COMPUTE_32F
 * - INT8 输入: inner_dtype = "bf16" 或 "int32"，compute_type = CUSPARSE_COMPUTE_32I
 *   注意: INT8 不支持 FP32 输出（cuSPARSELt 限制）
 * 
 * 接口设计（extern "C"）：
 * =======================
 * - 所有函数返回 int（0=成功，-1=失败）
 * - 错误信息通过 cusparselt_gemm_get_last_error() 获取
 * - 调用方预分配输出 tensor，传入 data_ptr
 * - 权重必须是 cuSPARSELt 压缩后的格式
 * 
 * 与 cuBLASLt 版本的主要区别：
 * ==========================
 * 1. 使用 cusparseLt 而非 cublasLt API
 * 2. 权重是预压缩的 2:4 稀疏格式
 * 3. 需要缓存计划（plan）以避免重复创建开销
 * 4. W 使用 StructuredDescriptor，A 使用 DenseDescriptor
 */

#include <cusparseLt.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

// ============================================================================
// 错误检查宏
// ============================================================================

#define CHECK_CUSPARSE(expr)                                                   \
    do {                                                                       \
        cusparseStatus_t _status = (expr);                                     \
        if (_status != CUSPARSE_STATUS_SUCCESS) {                              \
            std::ostringstream _oss;                                           \
            _oss << "[cuSPARSELt Error] status=" << _status                    \
                 << " at " << __FILE__ << ":" << __LINE__;                     \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t _status = (expr);                                          \
        if (_status != cudaSuccess) {                                          \
            std::ostringstream _oss;                                           \
            _oss << "[CUDA Error] " << cudaGetErrorString(_status)             \
                 << " (code " << _status << ") at "                            \
                 << __FILE__ << ":" << __LINE__;                               \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

// ============================================================================
// 数据类型转换辅助函数
// ============================================================================

/**
 * 将字符串 input_dtype 转换为 CUDA 数据类型
 */
static cudaDataType_t get_cuda_input_dtype(const std::string& input_dtype) {
    if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
        return CUDA_R_8F_E4M3;
    } else if (input_dtype == "int8") {
        return CUDA_R_8I;
    } else {
        throw std::invalid_argument(
            "Unsupported input_dtype: " + input_dtype +
            ". Supported: fp8e4m3, int8");
    }
}

/**
 * 将字符串 inner_dtype 转换为 CUDA 数据类型
 * 
 * 支持的类型：
 * - bf16: BFloat16 (FP8/INT8 输入均支持)
 * - fp32: Float32 (仅 FP8 输入支持)
 * - int32: Int32 (仅 INT8 输入支持，cuSPARSELt 限制)
 */
static cudaDataType_t get_cuda_inner_dtype(const std::string& inner_dtype) {
    if (inner_dtype == "bf16") {
        return CUDA_R_16BF;
    } else if (inner_dtype == "fp32") {
        return CUDA_R_32F;
    } else if (inner_dtype == "int32") {
        return CUDA_R_32I;
    } else {
        throw std::invalid_argument(
            "Unsupported inner_dtype: " + inner_dtype +
            ". Supported: bf16, fp32, int32");
    }
}

/**
 * 根据 input_dtype 和 inner_dtype 获取计算类型
 * 
 * cuSPARSELt 支持的组合（经验证）：
 * =================================
 * FP8 输入:
 *   - compute_type = CUSPARSE_COMPUTE_32F
 *   - inner_dtype = bf16 或 fp32
 * 
 * INT8 输入:
 *   - compute_type = CUSPARSE_COMPUTE_32I（必须！不支持 32F）
 *   - inner_dtype = bf16 或 int32（不支持 fp32）
 * 
 * 注意: INT8 + COMPUTE_32F 或 INT8 + FP32 输出会返回 status=10 (不支持)
 */
static cusparseComputeType get_compute_type(const std::string& input_dtype,
                                             const std::string& inner_dtype) {
    if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
        // FP8 输入始终使用 COMPUTE_32F
        return CUSPARSE_COMPUTE_32F;
    } else if (input_dtype == "int8") {
        // INT8 输入必须使用 COMPUTE_32I
        // cuSPARSELt 不支持 INT8 + COMPUTE_32F 的组合
        return CUSPARSE_COMPUTE_32I;
    } else {
        throw std::invalid_argument("Unsupported input_dtype: " + input_dtype);
    }
}

/**
 * 获取数据类型 ID（用于计划缓存 key）
 * 
 * 编码方案：
 * - FP8 + BF16: 0
 * - FP8 + FP32: 1
 * - INT8 + BF16: 2
 * - INT8 + INT32: 3
 */
static int get_dtype_id(const std::string& input_dtype, const std::string& inner_dtype) {
    int base = 0;
    if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
        base = 0;
    } else if (input_dtype == "int8") {
        base = 2;
    }
    if (inner_dtype == "bf16") {
        return base;
    } else if (inner_dtype == "fp32" || inner_dtype == "int32") {
        return base + 1;
    }
    return -1;
}

// ============================================================================
// 全局 cuSPARSELt Handle 管理
// ============================================================================

static cusparseLtHandle_t g_cusparselt_handle;
static bool g_cusparselt_initialized = false;
static std::mutex g_cusparselt_init_mutex;

/**
 * 获取全局 cuSPARSELt 句柄（懒初始化，线程安全）
 */
static cusparseLtHandle_t get_cusparselt_handle() {
    if (!g_cusparselt_initialized) {
        std::lock_guard<std::mutex> lock(g_cusparselt_init_mutex);
        if (!g_cusparselt_initialized) {
            cusparseStatus_t status = cusparseLtInit(&g_cusparselt_handle);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "Failed to initialize cuSPARSELt handle, status=" +
                    std::to_string(static_cast<int>(status)));
            }
            g_cusparselt_initialized = true;
            
            // 注册程序退出时的清理函数
            std::atexit([]() {
                if (g_cusparselt_initialized) {
                    cusparseLtDestroy(&g_cusparselt_handle);
                    g_cusparselt_initialized = false;
                }
            });
        }
    }
    return g_cusparselt_handle;
}

// ============================================================================
// 计划缓存管理
// ============================================================================
// cuSPARSELt 计划创建开销很大，必须缓存以获得最佳性能
// 缓存 key 由 (M, N, K, dtype_id, alg_id, split_k) 组成

struct PlanKey {
    int64_t M;          // 激活行数
    int64_t N;          // 权重行数 (out_features)
    int64_t K;          // 内维度 (in_features, slide 后)
    int dtype_id;       // 数据类型组合 ID
    int alg_id;         // 算法 ID（-1 表示使用默认）
    int split_k;        // split_k 设置（-1 表示不设置）
    
    bool operator<(const PlanKey& other) const {
        return std::tie(M, N, K, dtype_id, alg_id, split_k) < 
               std::tie(other.M, other.N, other.K, other.dtype_id, other.alg_id, other.split_k);
    }
};

struct PlanContext {
    cusparseLtMatDescriptor_t matW;      // 稀疏权重描述符
    cusparseLtMatDescriptor_t matA;      // 稠密激活描述符
    cusparseLtMatDescriptor_t matD;      // 输出描述符
    cusparseLtMatmulDescriptor_t matmul; // 矩阵乘描述符
    cusparseLtMatmulAlgSelection_t alg;  // 算法选择
    cusparseLtMatmulPlan_t plan;         // 执行计划
    size_t workspace_size = 0;           // workspace 大小
    bool initialized = false;
};

static std::map<PlanKey, PlanContext> g_plan_cache;
static std::mutex g_plan_mutex;
static bool g_cleanup_registered = false;

static void destroy_plan(PlanContext& ctx) {
    if (!ctx.initialized) return;
    cusparseLtMatmulPlanDestroy(&ctx.plan);
    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
    cusparseLtMatDescriptorDestroy(&ctx.matW);
    cusparseLtMatDescriptorDestroy(&ctx.matA);
    cusparseLtMatDescriptorDestroy(&ctx.matD);
    ctx.workspace_size = 0;
    ctx.initialized = false;
}

static void cleanup_all_plans() {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    for (auto& kv : g_plan_cache) {
        destroy_plan(kv.second);
    }
    g_plan_cache.clear();
}

// ============================================================================
// Workspace 管理
// ============================================================================

// Workspace memory is now managed by the caller (Python side) via torch allocator.
// This is required for CUDAGraph compatibility.
// static constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32 MB
// static thread_local void* t_workspace = nullptr;
// static thread_local size_t t_workspace_size = 0;


// ============================================================================
// 计划获取或创建
// ============================================================================

/**
 * 获取或创建 cuSPARSELt 矩阵乘法计划
 * 
 * TN-CC 布局配置：
 * - W[N,K] 行主序 → 视为列主序 [K,N] → opW=T → 参与计算时为 [N,K]
 * - A[M,K] 行主序 → 视为列主序 [K,M] → opA=N → 参与计算时为 [K,M]
 * - D = W × A = [N,K] × [K,M] = [N,M] 列主序 → 按行主序读 = [M,N]
 * 
 * 描述符配置：
 * - matW: StructuredDescriptor, rows=K, cols=N, ld=K (列主序视角)
 * - matA: DenseDescriptor, rows=K, cols=M, ld=K (列主序视角)
 * - matD: DenseDescriptor, rows=N, cols=M, ld=N (列主序视角)
 * 
 * @param alg_id   算法 ID，-1 表示使用默认算法
 * @param split_k  split_k 设置，-1 表示不设置
 */
static PlanContext& get_or_create_plan(
    cusparseLtHandle_t handle,
    int64_t M, int64_t N, int64_t K,
    const std::string& input_dtype,
    const std::string& inner_dtype,
    int alg_id,
    int split_k)
{
    int dtype_id = get_dtype_id(input_dtype, inner_dtype);
    if (dtype_id < 0) {
        throw std::invalid_argument(
            "Invalid dtype combination: " + input_dtype + ", " + inner_dtype);
    }
    
    PlanKey key{M, N, K, dtype_id, alg_id, split_k};
    
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }
    
    // 注册退出清理
    if (!g_cleanup_registered) {
        std::atexit(cleanup_all_plans);
        g_cleanup_registered = true;
    }
    
    auto [iter, inserted] = g_plan_cache.try_emplace(key);
    PlanContext& ctx = iter->second;
    
    // 获取数据类型配置
    cudaDataType_t cuda_input_dtype = get_cuda_input_dtype(input_dtype);
    cudaDataType_t cuda_inner_dtype = get_cuda_inner_dtype(inner_dtype);
    cusparseComputeType compute_type = get_compute_type(input_dtype, inner_dtype);
    
    // TN-CC 布局配置
    const auto order_col = CUSPARSE_ORDER_COL;
    const auto opW = CUSPARSE_OPERATION_TRANSPOSE;
    const auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    
    // 描述符维度（列主序视角）
    // PyTorch Row-Major W[N,K] = Col-Major [K,N]
    const int64_t num_W_rows = K;
    const int64_t num_W_cols = N;
    const int64_t num_A_rows = K;
    const int64_t num_A_cols = M;
    const int64_t num_D_rows = N;
    const int64_t num_D_cols = M;
    
    // Leading dimensions（列主序 ld = rows）
    const int64_t ldW = K;
    const int64_t ldA = K;
    const int64_t ldD = N;
    const unsigned alignment = 16;
    
    // 维度检查：cuSPARSELt 对 INT8/FP8 稀疏矩阵要求 32 对齐
    if (N % 32 != 0 || K % 32 != 0) {
        std::ostringstream oss;
        oss << "Dimension alignment error: N=" << N << ", K=" << K
            << ". For INT8/FP8 sparse matrices, N and K must be multiples of 32.";
        throw std::invalid_argument(oss.str());
    }
    
    bool matW_ok = false, matA_ok = false, matD_ok = false;
    bool alg_ok = false, plan_ok = false;
    
    try {
        // W: 稀疏权重，Col-Major [K,N]，opW=T 后变成 [N,K]
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            &handle, &ctx.matW,
            num_W_rows, num_W_cols, ldW,
            alignment, cuda_input_dtype, order_col,
            CUSPARSELT_SPARSITY_50_PERCENT));
        matW_ok = true;
        
        // A: 稠密激活，Col-Major [K,M]
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matA,
            num_A_rows, num_A_cols, ldA,
            alignment, cuda_input_dtype, order_col));
        matA_ok = true;
        
        // D: 输出，Col-Major [N,M]
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matD,
            num_D_rows, num_D_cols, ldD,
            alignment, cuda_inner_dtype, order_col));
        matD_ok = true;
        
        // 矩阵乘描述符：D = op(W) × op(A) = W^T × A
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            &handle, &ctx.matmul,
            opW, opA,
            &ctx.matW, &ctx.matA, &ctx.matD, &ctx.matD,
            compute_type));
        
        // 算法选择（初始化为默认算法）
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            &handle, &ctx.alg, &ctx.matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT));
        
        // 设置算法配置
        // 有预搜索配置时（alg_id >= 0）：使用配置中的 alg_id 和 split_k
        // 无配置时（alg_id = -1）：fallback 到 alg_id = 0，split_k 保持默认
        if (alg_id >= 0) {
            // 使用预搜索的配置
            CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                &handle, &ctx.alg,
                CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id)));
            
            // split_k 也来自预搜索配置（split_k != 1 时才需要设置）
            if (split_k != 1) {
                CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                    &handle, &ctx.alg,
                    CUSPARSELT_MATMUL_SPLIT_K,
                    &split_k, sizeof(split_k)));
            }
        } else {
            // Fallback: 没有预搜索配置，使用 alg_id = 0, split_k 保持默认
            int fallback_alg_id = 0;
            CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                &handle, &ctx.alg,
                CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &fallback_alg_id, sizeof(fallback_alg_id)));
        }
        alg_ok = true;
        
        // 初始化计划
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
            &handle, &ctx.plan, &ctx.matmul, &ctx.alg));
        plan_ok = true;
        
        // 查询 workspace 大小
        ctx.workspace_size = 0;
        cusparseStatus_t ws_status = cusparseLtMatmulGetWorkspace(
            &handle, &ctx.plan, &ctx.workspace_size);
        if (ws_status != CUSPARSE_STATUS_SUCCESS &&
            ws_status != CUSPARSE_STATUS_NOT_SUPPORTED) {
            CHECK_CUSPARSE(ws_status);
        }
        
        ctx.initialized = true;
        return ctx;
        
    } catch (...) {
        // 异常时清理已创建的资源
        if (plan_ok) cusparseLtMatmulPlanDestroy(&ctx.plan);
        if (alg_ok) cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
        if (matD_ok) cusparseLtMatDescriptorDestroy(&ctx.matD);
        if (matA_ok) cusparseLtMatDescriptorDestroy(&ctx.matA);
        if (matW_ok) cusparseLtMatDescriptorDestroy(&ctx.matW);
        g_plan_cache.erase(iter);
        throw;
    }
}

// ============================================================================
// cuSPARSELt GEMM 核心实现
// ============================================================================

/**
 * cuSPARSELt 稀疏矩阵乘法内部实现
 * 
 * 计算：D[M,N] = A[M,K] @ W_compressed[N,K]^T
 * 
 * @param W_compressed_ptr  压缩后的稀疏权重指针（GPU 内存）
 * @param A_ptr             输入矩阵指针 [M, K]，FP8/INT8，行主序（GPU 内存）
 * @param D_ptr             输出矩阵指针 [M, N]，BF16/FP32，行主序（GPU 内存，调用方预分配）
 * @param M                 A 的行数
 * @param N                 W 的行数（输出列数）
 * @param K                 内维度（slide 后的 K'）
 * @param input_dtype       输入数据类型字符串："fp8e4m3" 或 "int8"
 * @param inner_dtype       输出数据类型字符串："bf16" 或 "fp32"
 * @param alg_id            算法 ID，-1 表示使用默认算法
 * @param split_k           split_k 设置，-1 表示不设置
 * @param stream            CUDA 流（可为 nullptr 使用默认流）
 */
static void cusparselt_mm_impl(
    const void* W_compressed_ptr,
    const void* A_ptr,
    void* D_ptr,
    int64_t M, int64_t N, int64_t K,
    const std::string& input_dtype,
    const std::string& inner_dtype,
    int alg_id,
    int split_k,
    void* workspace_ptr,    // Added
    size_t workspace_size,  // Added
    cudaStream_t stream)
{
    // 获取句柄和计划
    cusparseLtHandle_t handle = get_cusparselt_handle();
    PlanContext& ctx = get_or_create_plan(handle, M, N, K, input_dtype, inner_dtype,
                                          alg_id, split_k);
    
    // Check if provided workspace is sufficient
    if (workspace_ptr != nullptr && workspace_size < ctx.workspace_size) {
        std::ostringstream oss;
        oss << "Provided workspace size (" << workspace_size << " bytes) is too small. "
            << "Required: " << ctx.workspace_size << " bytes.";
        throw std::runtime_error(oss.str());
    }
    
    // 执行矩阵乘法
    // alpha = 1.0, beta = 0.0（纯矩阵乘法，scale 由后续 kernel 处理）
    // 注意：cuSPARSELt 要求 alpha/beta 是 float 类型
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cusparseLtMatmul 参数说明：
    // ...
    // - workspace: 临时工作区
    // ...
    CHECK_CUSPARSE(cusparseLtMatmul(
        &handle, &ctx.plan,
        &alpha,
        W_compressed_ptr,  // 稀疏权重（已压缩）
        A_ptr,             // 稠密激活
        &beta,
        D_ptr,             // 输出（beta=0 时不累加）
        D_ptr,             // 输出
        workspace_ptr,     // Use provided workspace
        stream ? &stream : nullptr,
        stream ? 1 : 0));
}

// =============================================================================
// C 导出接口 - 错误处理基础设施
// =============================================================================

static thread_local char g_last_error[1024] = "";

static void set_error(const char* msg) {
    std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

static void clear_error() {
    g_last_error[0] = '\0';
}

// =============================================================================
// C 导出接口
// =============================================================================

extern "C" {

/**
 * 获取最后一次错误信息
 * 
 * @return 错误信息字符串（线程安全）
 */
const char* cusparselt_gemm_get_last_error() {
    return g_last_error;
}

/**
 * cuSPARSELt FP8 稀疏矩阵乘法
 * 
 * 计算：D[M,N] = A[M,K] @ W_compressed[N,K]^T
 * 
 * @param W_compressed_ptr  压缩后的稀疏权重指针，由 compress.py 预处理
 * @param A_ptr             输入矩阵指针 [M, K]，FP8E4M3，行主序（GPU 内存）
 * @param D_ptr             输出矩阵指针 [M, N]，BF16/FP32，行主序（GPU 内存，调用方预分配）
 * @param M                 A 的行数
 * @param N                 W 的行数（输出列数）
 * @param K                 内维度（slide 后的 K'）
 * @param inner_dtype       输出数据类型字符串："bf16" 或 "fp32"
 * @param alg_id            算法 ID，-1 表示使用默认算法
 * @param split_k           split_k 设置，-1 表示不设置
 * @param algo_workspace    算法配置中指定的 workspace 大小（预留，当前未使用）
 * @param stream            CUDA 流（可为 NULL 使用默认流）
 * @return                  0 成功，-1 失败
 */
int cusparselt_fp8_mm(
    const void* W_compressed_ptr,
    const void* A_ptr,
    void* D_ptr,
    int64_t M, int64_t N, int64_t K,
    const char* inner_dtype,
    int alg_id,
    int split_k,
    void* workspace_ptr,
    size_t workspace_size,
    cudaStream_t stream
) {
    clear_error();
    try {
        if (!W_compressed_ptr || !A_ptr || !D_ptr) {
            set_error("Input/output pointers cannot be null");
            return -1;
        }
        if (!inner_dtype) {
            set_error("inner_dtype cannot be null");
            return -1;
        }
        if (M <= 0 || N <= 0 || K <= 0) {
            set_error("Matrix dimensions must be positive");
            return -1;
        }
        
        cusparselt_mm_impl(W_compressed_ptr, A_ptr, D_ptr, M, N, K,
                           "fp8e4m3", std::string(inner_dtype),
                           alg_id, split_k, workspace_ptr, workspace_size, stream);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cusparselt_fp8_mm");
        return -1;
    }
}

/**
 * cuSPARSELt INT8 稀疏矩阵乘法
 * 
 * 计算：D[M,N] = A[M,K] @ W_compressed[N,K]^T
 * 
 * 注意: INT8 使用 CUSPARSE_COMPUTE_32I，支持 BF16 或 INT32 输出
 *       不支持 FP32 输出（cuSPARSELt 限制）
 * 
 * @param W_compressed_ptr  压缩后的稀疏权重指针，由 compress.py 预处理
 * @param A_ptr             输入矩阵指针 [M, K]，INT8，行主序（GPU 内存）
 * @param D_ptr             输出矩阵指针 [M, N]，BF16/INT32，行主序（GPU 内存，调用方预分配）
 * @param M                 A 的行数
 * @param N                 W 的行数（输出列数）
 * @param K                 内维度（slide 后的 K'）
 * @param inner_dtype       输出数据类型字符串："bf16" 或 "int32"
 * @param alg_id            算法 ID，-1 表示使用默认算法
 * @param split_k           split_k 设置，-1 表示不设置
 * @param algo_workspace    算法配置中指定的 workspace 大小（预留，当前未使用）
 * @param stream            CUDA 流（可为 NULL 使用默认流）
 * @return                  0 成功，-1 失败
 */
int cusparselt_int8_mm(
    const void* W_compressed_ptr,
    const void* A_ptr,
    void* D_ptr,
    int64_t M, int64_t N, int64_t K,
    const char* inner_dtype,
    int alg_id,
    int split_k,
    void* workspace_ptr,
    size_t workspace_size,
    cudaStream_t stream
) {
    clear_error();
    try {
        if (!W_compressed_ptr || !A_ptr || !D_ptr) {
            set_error("Input/output pointers cannot be null");
            return -1;
        }
        if (!inner_dtype) {
            set_error("inner_dtype cannot be null");
            return -1;
        }
        if (M <= 0 || N <= 0 || K <= 0) {
            set_error("Matrix dimensions must be positive");
            return -1;
        }
        
        // INT8 cuSPARSELt only supports BF16 or INT32 output, not FP32
        std::string dtype_str(inner_dtype);
        if (dtype_str == "fp32") {
            set_error("cuSPARSELt INT8 GEMM does not support FP32 output. Use 'bf16' (default) or 'int32'.");
            return -1;
        }
        
        cusparselt_mm_impl(W_compressed_ptr, A_ptr, D_ptr, M, N, K,
                           "int8", dtype_str,
                           alg_id, split_k, workspace_ptr, workspace_size, stream);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cusparselt_int8_mm");
        return -1;
    }
}

/**
 * 查询指定维度 GEMM 所需的 workspace 大小
 * 
 * @param M                 A 的行数
 * @param N                 W 的行数
 * @param K                 内维度
 * @param input_dtype       输入数据类型："fp8e4m3" 或 "int8"
 * @param inner_dtype       输出数据类型："bf16" 或 "fp32"
 * @param alg_id            算法 ID，-1 表示使用默认算法
 * @param split_k           split_k 设置，-1 表示不设置
 * @param workspace_size    [out] workspace 大小（字节）
 * @return                  0 成功，-1 失败
 */
int cusparselt_get_workspace_size(
    int64_t M, int64_t N, int64_t K,
    const char* input_dtype,
    const char* inner_dtype,
    int alg_id,
    int split_k,
    size_t* workspace_size
) {
    clear_error();
    try {
        if (!input_dtype || !inner_dtype || !workspace_size) {
            set_error("Parameters cannot be null");
            return -1;
        }
        
        cusparseLtHandle_t handle = get_cusparselt_handle();
        PlanContext& ctx = get_or_create_plan(handle, M, N, K,
                                               std::string(input_dtype),
                                               std::string(inner_dtype),
                                               alg_id, split_k);
        *workspace_size = ctx.workspace_size;
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cusparselt_get_workspace_size");
        return -1;
    }
}

/**
 * 检查 cuSPARSELt 是否可用
 * 
 * @return 1 可用，0 不可用
 */
int cusparselt_is_available() {
    try {
        get_cusparselt_handle();
        return 1;
    } catch (...) {
        return 0;
    }
}

/**
 * 获取支持的数据类型
 * 
 * @return 逗号分隔的类型字符串
 */
const char* cusparselt_get_supported_dtypes() {
    return "fp8e4m3,int8";
}

}  // extern "C"
