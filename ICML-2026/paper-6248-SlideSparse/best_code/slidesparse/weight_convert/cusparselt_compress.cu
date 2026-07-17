/**
 * cuSPARSELt 2:4 结构化稀疏权重压缩库
 *
 * 功能：将满足 2:4 稀疏约束的权重压缩为 cuSPARSELt 硬件加速格式
 * 
 * ============================================================================
 * 支持的数据类型
 * ============================================================================
 * 
 * - INT8:     CUDA_R_8I,      计算类型 CUSPARSE_COMPUTE_32I
 * - FP8E4M3:  CUDA_R_8F_E4M3, 计算类型 CUSPARSE_COMPUTE_32F
 * 
 * 参考 cuBLASLt GEMM 实现，通过字符串参数 "int8" / "fp8e4m3" 选择数据类型。
 * 
 * ============================================================================
 * 布局说明（TN-CC 格式）
 * ============================================================================
 * 
 * 目标运算：D[N,M] = W[N,K]^T × A[K,M]
 * 
 * PyTorch 权重存储：
 *   - W.shape = (N, K)，行主序 (Row-Major)
 *   - 物理内存：W[0,0], W[0,1], ..., W[0,K-1], W[1,0], ...
 * 
 * 关键转换：
 *   Row-Major [N,K] ≡ Col-Major [K,N]  (同一块内存，不同视角)
 * 
 * cuSPARSELt 配置：
 *   - 告诉库这是 Col-Major [K,N] 矩阵（描述符: rows=K, cols=N, ld=K）
 *   - 设置 opW = TRANSPOSE
 *   - 库会将 [K,N] 转置为 [N,K] 参与计算，正好还原原始语义
 * 
 * 维度约束 (INT8/FP8):
 *   - 稀疏矩阵: rows, cols, ld 必须是 32 的倍数
 *   - 稠密矩阵: rows, cols, ld 必须是 16 的倍数
 *
 * ============================================================================
 */

#include <cusparseLt.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

// =============================================================================
// 调试工具
// =============================================================================

static bool debug_enabled() {
    static bool flag = []() {
        const char* env = std::getenv("CUSPARSELT_COMPRESS_DEBUG");
        return env && std::strcmp(env, "1") == 0;
    }();
    return flag;
}

#define DEBUG_LOG(msg)                                                         \
    do {                                                                       \
        if (debug_enabled()) {                                                 \
            std::ostringstream oss;                                            \
            oss << "[cusparselt-compress] " << msg << std::endl;               \
            std::cerr << oss.str();                                            \
        }                                                                      \
    } while (0)

// =============================================================================
// 错误检查宏
// =============================================================================

#define CHECK_CUSPARSE(func)                                                   \
    do {                                                                       \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            std::ostringstream oss;                                            \
            oss << "cuSPARSELt error at " << __FILE__ << ":" << __LINE__       \
                << " - " << #func << " returned " << static_cast<int>(status); \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

#define CHECK_CUDA(func)                                                       \
    do {                                                                       \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            std::ostringstream oss;                                            \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(status);                        \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

// =============================================================================
// 数据类型转换辅助函数（参考 cuBLASLt GEMM）
// =============================================================================

/**
 * 将字符串数据类型转换为 CUDA 数据类型
 * 
 * @param dtype  "int8" 或 "fp8e4m3" / "fp8"
 * @return       对应的 cudaDataType_t
 */
static cudaDataType_t get_cuda_dtype(const std::string& dtype) {
    if (dtype == "fp8e4m3" || dtype == "fp8") {
        return CUDA_R_8F_E4M3;
    } else if (dtype == "int8") {
        return CUDA_R_8I;
    } else {
        throw std::invalid_argument(
            "Unsupported dtype: " + dtype + ". Supported: int8, fp8e4m3");
    }
}

/**
 * 根据数据类型获取计算类型
 * 
 * @param dtype  "int8" 或 "fp8e4m3" / "fp8"
 * @return       对应的 cusparseComputeType
 */
static cusparseComputeType get_compute_type(const std::string& dtype) {
    if (dtype == "fp8e4m3" || dtype == "fp8") {
        return CUSPARSE_COMPUTE_32F;
    } else if (dtype == "int8") {
        return CUSPARSE_COMPUTE_32I;
    } else {
        throw std::invalid_argument(
            "Unsupported dtype: " + dtype + ". Supported: int8, fp8e4m3");
    }
}

/**
 * 根据数据类型获取输出数据类型
 * 
 * INT8 计算输出 INT32
 * FP8 计算输出 BF16 (或 FP32，这里用 BF16)
 */
static cudaDataType_t get_output_dtype(const std::string& dtype) {
    if (dtype == "fp8e4m3" || dtype == "fp8") {
        return CUDA_R_16BF;  // FP8 -> BF16 输出
    } else if (dtype == "int8") {
        return CUDA_R_32I;   // INT8 -> INT32 输出
    } else {
        throw std::invalid_argument("Unsupported dtype: " + dtype);
    }
}

/**
 * 获取数据类型标签（用于缓存 key）
 */
static int get_dtype_id(const std::string& dtype) {
    if (dtype == "fp8e4m3" || dtype == "fp8") {
        return 1;
    } else if (dtype == "int8") {
        return 0;
    }
    return -1;
}

// =============================================================================
// 全局 cuSPARSELt 句柄管理
// =============================================================================

static cusparseLtHandle_t g_handle;
static bool g_handle_initialized = false;
static std::mutex g_handle_mutex;

static cusparseLtHandle_t get_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_initialized) {
        CHECK_CUSPARSE(cusparseLtInit(&g_handle));
        g_handle_initialized = true;
        DEBUG_LOG("cuSPARSELt handle initialized");
        
        std::atexit([]() {
            if (g_handle_initialized) {
                cusparseLtDestroy(&g_handle);
                g_handle_initialized = false;
                DEBUG_LOG("cuSPARSELt handle destroyed");
            }
        });
    }
    return g_handle;
}

// =============================================================================
// 压缩计划缓存（按数据类型和维度缓存）
// =============================================================================

struct PlanKey {
    int M;       // 激活行数（用于构建计划，压缩时固定为 1024）
    int N;       // 权重行数 (out_features)
    int K;       // 权重列数 (in_features)
    int dtype;   // 数据类型：0=INT8, 1=FP8E4M3
    
    bool operator<(const PlanKey& other) const {
        return std::tie(M, N, K, dtype) < std::tie(other.M, other.N, other.K, other.dtype);
    }
};

struct PlanContext {
    cusparseLtMatDescriptor_t matW;      // 稀疏权重描述符
    cusparseLtMatDescriptor_t matA;      // 稠密激活描述符
    cusparseLtMatDescriptor_t matD;      // 输出描述符
    cusparseLtMatmulDescriptor_t matmul; // 矩阵乘描述符
    cusparseLtMatmulAlgSelection_t alg;  // 算法选择
    cusparseLtMatmulPlan_t plan;         // 执行计划
    bool initialized = false;
};

static std::map<PlanKey, PlanContext> g_plan_cache;
static std::mutex g_plan_mutex;

static void destroy_plan(PlanContext& ctx) {
    if (!ctx.initialized) return;
    cusparseLtMatmulPlanDestroy(&ctx.plan);
    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
    cusparseLtMatDescriptorDestroy(&ctx.matW);
    cusparseLtMatDescriptorDestroy(&ctx.matA);
    cusparseLtMatDescriptorDestroy(&ctx.matD);
    ctx.initialized = false;
}

static void cleanup_all_plans() {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    for (auto& kv : g_plan_cache) {
        destroy_plan(kv.second);
    }
    g_plan_cache.clear();
}

static bool g_cleanup_registered = false;

/**
 * 获取或创建压缩计划（支持多数据类型）
 * 
 * 参数含义（逻辑维度）：
 *   - N: 权重的 out_features (W.shape[0])
 *   - K: 权重的 in_features (W.shape[1])
 *   - M: 激活的 batch 维度（压缩时固定为 1024）
 *   - dtype_str: 数据类型字符串 "int8" 或 "fp8e4m3"
 * 
 * 描述符配置：
 *   PyTorch Row-Major W[N,K] 被视为 Col-Major [K,N]
 *   因此描述符填 (rows=K, cols=N, ld=K)
 */
static PlanContext& get_or_create_plan(cusparseLtHandle_t handle, 
                                        int M, int N, int K,
                                        const std::string& dtype_str) {
    int dtype_id = get_dtype_id(dtype_str);
    if (dtype_id < 0) {
        throw std::invalid_argument("Unknown dtype: " + dtype_str);
    }
    
    PlanKey key{M, N, K, dtype_id};
    
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        DEBUG_LOG("Plan cache hit: M=" << M << " N=" << N << " K=" << K << " dtype=" << dtype_str);
        return it->second;
    }
    
    // 注册退出清理
    if (!g_cleanup_registered) {
        std::atexit(cleanup_all_plans);
        g_cleanup_registered = true;
    }
    
    DEBUG_LOG("Creating plan: M=" << M << " N=" << N << " K=" << K << " dtype=" << dtype_str);
    
    auto [iter, inserted] = g_plan_cache.try_emplace(key);
    PlanContext& ctx = iter->second;
    
    // 获取数据类型配置
    cudaDataType_t cuda_dtype = get_cuda_dtype(dtype_str);
    cudaDataType_t output_dtype = get_output_dtype(dtype_str);
    cusparseComputeType compute_type = get_compute_type(dtype_str);
    
    // TN-CC 布局：opW=TRANSPOSE, opA=NON_TRANSPOSE, 全列主序
    const auto order_col = CUSPARSE_ORDER_COL;
    const auto opW = CUSPARSE_OPERATION_TRANSPOSE;
    const auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    
    // PyTorch Row-Major W[N,K] = Col-Major [K,N]
    // 描述符填写 Col-Major 视角的形状
    const int num_W_rows = K;
    const int num_W_cols = N;
    const int num_A_rows = K;
    const int num_A_cols = M;
    const int num_D_rows = N;
    const int num_D_cols = M;
    
    // 列主序 ld = rows
    const int ldW = K;
    const int ldA = K;
    const int ldD = N;
    const unsigned alignment = 16;
    
    // 维度检查：INT8/FP8 稀疏矩阵需要 32 对齐
    if (N % 32 != 0 || K % 32 != 0) {
        std::ostringstream oss;
        oss << "Dimension alignment error: N=" << N << ", K=" << K 
            << ". For INT8/FP8 sparse matrices, N and K must be multiples of 32. "
            << "Use slide.py with align_to=32.";
        throw std::invalid_argument(oss.str());
    }
    
    bool matW_ok = false, matA_ok = false, matD_ok = false;
    bool alg_ok = false, plan_ok = false;
    
    try {
        // W: Col-Major [K,N], opW=T 后变成 [N,K] 参与计算
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            &handle, &ctx.matW,
            num_W_rows, num_W_cols, ldW,
            alignment, cuda_dtype, order_col,
            CUSPARSELT_SPARSITY_50_PERCENT));
        matW_ok = true;
        DEBUG_LOG("  matW (sparse " << dtype_str << "): [" << num_W_rows << "," << num_W_cols << "] ld=" << ldW);
        
        // A: Col-Major [K,M]，与 W 相同的数据类型
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matA,
            num_A_rows, num_A_cols, ldA,
            alignment, cuda_dtype, order_col));
        matA_ok = true;
        DEBUG_LOG("  matA (dense " << dtype_str << "): [" << num_A_rows << "," << num_A_cols << "] ld=" << ldA);
        
        // D: Col-Major [N,M]，输出类型（INT8->INT32, FP8->BF16）
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matD,
            num_D_rows, num_D_cols, ldD,
            alignment, output_dtype, order_col));
        matD_ok = true;
        DEBUG_LOG("  matD (output): [" << num_D_rows << "," << num_D_cols << "] ld=" << ldD);
        
        // D = op(W) * op(A) = W^T * A
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            &handle, &ctx.matmul,
            opW, opA,
            &ctx.matW, &ctx.matA, &ctx.matD, &ctx.matD,
            compute_type));
        DEBUG_LOG("  matmul descriptor initialized with compute_type=" << static_cast<int>(compute_type));
        
        // 算法选择（使用默认算法 ID=0）
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            &handle, &ctx.alg, &ctx.matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT));
        
        int alg_id = 0;
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
            &handle, &ctx.alg,
            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &alg_id, sizeof(alg_id)));
        alg_ok = true;
        DEBUG_LOG("  algorithm selection initialized (alg_id=0)");
        
        // 初始化计划
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
            &handle, &ctx.plan, &ctx.matmul, &ctx.alg));
        plan_ok = true;
        DEBUG_LOG("  plan initialized");
        
        ctx.initialized = true;
        return ctx;
        
    } catch (...) {
        if (plan_ok) cusparseLtMatmulPlanDestroy(&ctx.plan);
        if (alg_ok) cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
        if (matW_ok) cusparseLtMatDescriptorDestroy(&ctx.matW);
        if (matA_ok) cusparseLtMatDescriptorDestroy(&ctx.matA);
        if (matD_ok) cusparseLtMatDescriptorDestroy(&ctx.matD);
        g_plan_cache.erase(iter);
        throw;
    }
}

}  // namespace

// =============================================================================
// C 导出接口 - 错误处理基础设施
// =============================================================================

// 线程安全的错误信息存储
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
const char* cusparselt_compress_get_last_error() {
    return g_last_error;
}

/**
 * 查询压缩所需的缓冲区大小（支持多数据类型）
 * 
 * @param N                  权重行数 (out_features)
 * @param K                  权重列数 (in_features)
 * @param dtype              数据类型字符串: "int8" 或 "fp8e4m3"
 * @param compressed_size    [out] 压缩后数据大小（字节）
 * @param temp_buffer_size   [out] 临时缓冲区大小（字节）
 * @return                   0 成功，-1 失败（调用 cusparselt_compress_get_last_error 获取错误信息）
 * 
 * 注意：M 固定为 1024，用于构建压缩计划
 */
int cusparselt_get_compress_sizes(
    int N, int K,
    const char* dtype,
    size_t* compressed_size,
    size_t* temp_buffer_size
) {
    clear_error();
    try {
        if (!compressed_size || !temp_buffer_size) {
            set_error("Output pointers cannot be null");
            return -1;
        }
        
        if (!dtype) {
            set_error("dtype cannot be null");
            return -1;
        }
        
        std::string dtype_str(dtype);
        const int M = 1024;  // 固定 M 用于构建计划
        
        cusparseLtHandle_t handle = get_handle();
        PlanContext& ctx = get_or_create_plan(handle, M, N, K, dtype_str);
        
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
            &handle, &ctx.plan,
            compressed_size,
            temp_buffer_size));
        
        DEBUG_LOG("Query sizes: N=" << N << " K=" << K << " dtype=" << dtype_str
                  << " compressed=" << *compressed_size 
                  << " temp=" << *temp_buffer_size);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cusparselt_get_compress_sizes");
        return -1;
    }
}

/**
 * 压缩 2:4 稀疏权重（支持多数据类型）
 * 
 * @param input_weight       输入权重指针（GPU 内存，必须满足 2:4 稀疏）
 * @param compressed_weight  输出压缩数据指针（GPU 内存）
 * @param temp_buffer        临时缓冲区指针（GPU 内存，可为 NULL 自动分配）
 * @param N                  权重行数 (out_features)
 * @param K                  权重列数 (in_features)
 * @param dtype              数据类型字符串: "int8" 或 "fp8e4m3"
 * @param stream             CUDA 流（可为 NULL 使用默认流）
 * @return                   0 成功，-1 失败（调用 cusparselt_compress_get_last_error 获取错误信息）
 * 
 * 注意：
 *   - 输入权重必须满足 2:4 结构化稀疏约束
 *   - 权重布局为行主序 [N, K]（cuSPARSELt 视为列主序 [K, N]）
 */
int cusparselt_compress_weight(
    const void* input_weight,
    void* compressed_weight,
    void* temp_buffer,
    int N, int K,
    const char* dtype,
    cudaStream_t stream
) {
    clear_error();
    try {
        if (!input_weight || !compressed_weight) {
            set_error("Input/output pointers cannot be null");
            return -1;
        }
        
        if (!dtype) {
            set_error("dtype cannot be null");
            return -1;
        }
        
        std::string dtype_str(dtype);
        const int M = 1024;  // 固定 M 用于构建计划
        
        DEBUG_LOG("Compress: N=" << N << " K=" << K << " dtype=" << dtype_str);
        
        cusparseLtHandle_t handle = get_handle();
        PlanContext& ctx = get_or_create_plan(handle, M, N, K, dtype_str);
        
        // 查询所需大小
        size_t compressed_size = 0;
        size_t temp_buffer_size = 0;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
            &handle, &ctx.plan,
            &compressed_size,
            &temp_buffer_size));
        
        // 处理临时缓冲区
        void* temp_ptr = temp_buffer;
        bool temp_allocated = false;
        if (temp_buffer_size > 0 && temp_ptr == nullptr) {
            CHECK_CUDA(cudaMalloc(&temp_ptr, temp_buffer_size));
            temp_allocated = true;
            DEBUG_LOG("  Allocated temp buffer: " << temp_buffer_size << " bytes");
        }
        
        // 分配稀疏性验证标志
        int* d_valid = nullptr;
        CHECK_CUDA(cudaMalloc(&d_valid, sizeof(int)));
        
        try {
            // Step 1: 验证 2:4 稀疏性
            CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(
                &handle, &ctx.matmul,
                input_weight, d_valid, stream));
            
            int h_valid = 0;
            CHECK_CUDA(cudaMemcpyAsync(&h_valid, d_valid, sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
            
            if (stream) {
                CHECK_CUDA(cudaStreamSynchronize(stream));
            } else {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            
            CHECK_CUDA(cudaFree(d_valid));
            d_valid = nullptr;
            
            if (h_valid != 0) {
                throw std::runtime_error("Input weight does not satisfy 2:4 sparsity constraint");
            }
            DEBUG_LOG("  2:4 sparsity verified");
            
            // Step 2: 执行压缩
            CHECK_CUSPARSE(cusparseLtSpMMACompress(
                &handle, &ctx.plan,
                input_weight,
                compressed_weight,
                temp_ptr,
                stream));
            
            if (stream) {
                CHECK_CUDA(cudaStreamSynchronize(stream));
            } else {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            DEBUG_LOG("  Compression completed");
            
        } catch (...) {
            if (d_valid) cudaFree(d_valid);
            if (temp_allocated && temp_ptr) cudaFree(temp_ptr);
            throw;  // 重新抛出，让外层 catch 处理
        }
        
        // 清理
        if (temp_allocated && temp_ptr) {
            CHECK_CUDA(cudaFree(temp_ptr));
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cusparselt_compress_weight");
        return -1;
    }
}

/**
 * 检查 cuSPARSELt 库是否可用
 * 
 * @return 1 表示可用，0 表示不可用
 */
int cusparselt_is_available() {
    try {
        get_handle();
        return 1;
    } catch (...) {
        return 0;
    }
}

/**
 * 获取支持的数据类型
 * 
 * 返回逗号分隔的支持类型字符串
 */
const char* cusparselt_get_supported_dtypes() {
    return "int8,fp8e4m3";
}

}  // extern "C"
