// SPDX-License-Identifier: Apache-2.0
/**
 * cuBLASLt GEMM Implementation for SlideSparse (extern "C" 版本)
 * 
 * 设计目标：
 * =========
 * 1. 使用 cuBLASLt API 实现纯矩阵乘法（不带 scale/bias 融合）
 * 2. 支持 FP8E4M3 和 INT8 输入
 * 3. FP8 支持 BF16/FP32 输出，INT8 仅支持 INT32 输出
 * 4. Dequant + bias 由后续 Triton kernel 处理
 * 
 * 计算流程：
 * =========
 * D[N,M]_col = W[K,N]^T_col @ A[K,M]_col
 * 
 * cuBLASLt 配置：
 * ==============
 * - 布局：TN + CCC（W 转置，A 不转置，全列主序）
 * - W[N,K] 行主序 → 声明列主序 [K,N] → opA=T → [N,K]
 * - A[M,K] 行主序 → 声明列主序 [K,M] → opB=N → [K,M]
 * - D[N,M] 列主序结果 → 按行主序读 = [M,N]
 * 
 * 支持的数据类型组合：
 * ===================
 * - FP8 输入:
 *     - compute_type = CUBLAS_COMPUTE_32F
 *     - scale_type = CUDA_R_32F (alpha/beta 为 float)
 *     - inner_dtype = "bf16"（默认）或 "fp32"
 * 
 * - INT8 输入:
 *     - compute_type = CUBLAS_COMPUTE_32I
 *     - scale_type = CUDA_R_32I (alpha/beta 为 int32)
 *     - inner_dtype 参数被忽略，强制使用 "int32"
 *     - cuBLASLt INT8 GEMM 硬件限制：不支持 BF16/FP32 输出
 * 
 * 接口设计（extern "C"）：
 * =======================
 * - 所有函数返回 int（0=成功，-1=失败）
 * - 错误信息通过 cublaslt_gemm_get_last_error() 获取
 * - 调用方预分配输出 tensor，传入 data_ptr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include <cstring>
#include <mutex>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

// 编译期检查：确保 cublasLtMatmulAlgo_t 不超过我们预期的大小（64 字节）
// 算法搜索脚本生成的 algo_data 缓冲区大小为 64 字节
static_assert(sizeof(cublasLtMatmulAlgo_t) <= 64,
              "cublasLtMatmulAlgo_t size exceeds 64 bytes");

// ============================================================================
// 错误检查宏
// ============================================================================

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t _status = (expr);                                              \
    if (_status != cudaSuccess) {                                              \
      std::ostringstream _oss;                                                 \
      _oss << "[CUDA Error] " << cudaGetErrorString(_status)                   \
           << " (code " << _status << ") at " << __FILE__ << ":" << __LINE__;  \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#define CHECK_CUBLASLT(expr)                                                   \
  do {                                                                         \
    cublasStatus_t _status = (expr);                                           \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream _oss;                                                 \
      _oss << "[cuBLASLt Error] " << cublasLtGetStatusString(_status)          \
           << " (code " << static_cast<int>(_status) << ") at "                \
           << __FILE__ << ":" << __LINE__;                                     \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// ============================================================================
// 全局 cuBLASLt Handle 管理
// ============================================================================
// 使用 mutex 保护初始化，避免多线程竞争（如 TP 场景）
// 句柄创建后可以跨线程安全使用

static cublasLtHandle_t g_cublaslt_handle = nullptr;
static bool g_cublaslt_initialized = false;
static std::mutex g_cublaslt_init_mutex;

/**
 * 获取全局 cuBLASLt 句柄（懒初始化，线程安全）
 */
static cublasLtHandle_t get_cublaslt_handle() {
  // 双检锁模式
  if (!g_cublaslt_initialized) {
    std::lock_guard<std::mutex> lock(g_cublaslt_init_mutex);
    if (!g_cublaslt_initialized) {
      cublasStatus_t status = cublasLtCreate(&g_cublaslt_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Failed to create cuBLASLt handle: " +
            std::string(cublasLtGetStatusString(status)));
      }
      g_cublaslt_initialized = true;

      // 注册程序退出时的清理函数
      std::atexit([]() {
        if (g_cublaslt_initialized && g_cublaslt_handle) {
          cublasLtDestroy(g_cublaslt_handle);
          g_cublaslt_handle = nullptr;
          g_cublaslt_initialized = false;
        }
      });
    }
  }
  return g_cublaslt_handle;
}

// ============================================================================
// Workspace 管理
// ============================================================================
// Workspace 由调用方（Python）分配并传入，避免在 C++ 中使用 cudaMalloc（影响 CUDAGraph）

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
 * - bf16: BFloat16 (仅 FP8 输入支持)
 * - fp32: Float32 (仅 FP8 输入支持)
 * - int32: Int32 (仅 INT8 输入支持)
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
 * 根据 input_dtype 获取计算类型
 * 
 * FP8 输入: 使用 CUBLAS_COMPUTE_32F，支持 BF16/FP32 输出
 * INT8 输入: 使用 CUBLAS_COMPUTE_32I，仅支持 INT32 输出
 */
static cublasComputeType_t get_compute_type(const std::string& input_dtype) {
  if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
    return CUBLAS_COMPUTE_32F;
  } else if (input_dtype == "int8") {
    return CUBLAS_COMPUTE_32I;  // INT8 必须使用 32I，仅支持 INT32 输出
  } else {
    throw std::invalid_argument(
        "Unsupported input_dtype: " + input_dtype);
  }
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 检查字节数组是否全为零
 */
static bool is_all_zeros(const uint8_t* data, size_t len) {
  if (data == nullptr) return true;
  for (size_t i = 0; i < len; ++i) {
    if (data[i] != 0) return false;
  }
  return true;
}

// ============================================================================
// cuBLASLt GEMM 核心实现（内部函数）
// ============================================================================

/**
 * cuBLASLt Matrix Multiplication 内部实现
 * 
 * 计算：D[N,M]_col = W[K,N]^T_col @ A[K,M]_col
 * 
 * @param W_ptr       权重矩阵指针 [N, K]，FP8/INT8，行主序（GPU 内存）
 * @param A_ptr       输入矩阵指针 [M, K]，FP8/INT8，行主序（GPU 内存）
 * @param D_ptr       输出矩阵指针 [M, N]，BF16/FP32，行主序（GPU 内存，调用方预分配）
 * @param M           A 的行数
 * @param N           W 的行数（输出列数）
 * @param K           内维度
 * @param input_dtype 输入数据类型字符串："fp8e4m3" 或 "int8"
 * @param inner_dtype   输出数据类型字符串："bf16" 或 "fp32"
 * @param algo_data     算法配置数据（64字节），NULL或全零表示使用默认启发式
 * @param workspace_ptr 外部 workspace 指针（必须在 GPU 上）
 * @param workspace_size 外部 workspace 大小（字节）
 * @param stream        CUDA 流（可为 nullptr 使用默认流）
 * 
 * 实现细节：
 * - W 放在 cuBLASLt 的 A 位置（左矩阵），使用 opA=CUBLAS_OP_T
 * - A 放在 cuBLASLt 的 B 位置（右矩阵），使用 opB=CUBLAS_OP_N
 * - 所有矩阵声明为列主序（实际是行主序内存，利用转置等价）
 * - alpha = 1.0, beta = 0.0（纯矩阵乘法）
 */
static void cublaslt_mm_impl(
    const void* W_ptr,
    const void* A_ptr,
    void* D_ptr,
    int64_t M, int64_t N, int64_t K,
    const std::string& input_dtype,
    const std::string& inner_dtype,
    const uint8_t* algo_data,
    void* workspace_ptr,
    size_t workspace_size,
    cudaStream_t stream)
{
  // ========== 获取数据类型配置 ==========
  cudaDataType_t cuda_input_dtype = get_cuda_input_dtype(input_dtype);
  cudaDataType_t cuda_inner_dtype = get_cuda_inner_dtype(inner_dtype);
  cublasComputeType_t compute_type = get_compute_type(input_dtype);
  
  // Scale 类型：FP8 使用 FP32，INT8 使用 INT32
  // cuBLASLt 要求 scale_type 与 compute_type 匹配
  cudaDataType_t scale_type = (input_dtype == "int8") ? CUDA_R_32I : CUDA_R_32F;
  
  // ========== 获取 cuBLASLt handle ==========
  cublasLtHandle_t handle = get_cublaslt_handle();
  
  // ========== 创建 MatmulDesc ==========
  cublasLtMatmulDesc_t matmulDesc = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulDescCreate(
      &matmulDesc, compute_type, scale_type));
  
  // 设置转置操作
  // W 在 cuBLASLt 的 A 位置，需要转置（因为我们用行主序声明为列主序）
  // A 在 cuBLASLt 的 B 位置，不转置
  cublasOperation_t opW = CUBLAS_OP_T;
  cublasOperation_t opA = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
  
  // 默认 epilogue（无 bias）
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  
  // ========== 创建矩阵布局描述符 ==========
  // 关键技巧：行主序矩阵声明为列主序，配合转置操作实现正确计算
  //
  // W[N,K] 行主序存储：
  //   - 物理内存：N 行，每行 K 个元素，stride = K
  //   - 声明为列主序 [K,N]：K 行，N 列，ld = K
  //   - opA=T 后：[N,K]
  //
  // A[M,K] 行主序存储：
  //   - 物理内存：M 行，每行 K 个元素，stride = K
  //   - 声明为列主序 [K,M]：K 行，M 列，ld = K
  //   - opB=N 后：[K,M]
  //
  // 输出 D：
  //   - cuBLASLt 输出 [N,M] 列主序
  //   - 我们创建 [M,N] 行主序 tensor，ld = N
  //   - 声明为列主序 [N,M]：N 行，M 列，ld = N
  //   - 存储到 [M,N] 行主序内存中
  
  cublasLtMatrixLayout_t layoutW = nullptr;
  cublasLtMatrixLayout_t layoutA = nullptr;
  cublasLtMatrixLayout_t layoutD = nullptr;
  
  // W 布局：声明为列主序 [K, N]，ld = K
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutW, cuda_input_dtype, K, N, K));
  
  // A 布局：声明为列主序 [K, M]，ld = K
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutA, cuda_input_dtype, K, M, K));
  
  // D 布局：声明为列主序 [N, M]，ld = N
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutD, cuda_inner_dtype, N, M, N));
  
  // 设置所有矩阵为列主序（这是默认值，但显式设置更清晰）
  cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  
  // ========== 算法选择 ==========
  cublasLtMatmulAlgo_t custom_algo;
  const cublasLtMatmulAlgo_t* algo = nullptr;
  size_t required_workspace_size = 0;
  
  // 检查是否有有效的自定义配置
  bool use_custom = (algo_data != nullptr && !is_all_zeros(algo_data, 64));
  
  if (use_custom) {
    // 从 algo_data 恢复算法配置
    std::memcpy(&custom_algo, algo_data, sizeof(custom_algo));
    algo = &custom_algo;
    // 注意：这里假设调用方保证提供的 workspace_size 足够
    // 我们只是将 available workspace size 传递给 cublasLtMatmul
  } else {
    // 使用启发式搜索（默认逻辑）
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    
    // 设置最大 workspace 大小（限制为提供的 workspace_size）
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));
    
    // 获取最优算法
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedAlgoCount = 0;
    
    cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutW,
        layoutA,
        layoutD,  // C 和 D 使用相同布局（in-place）
        layoutD,
        preference,
        1,  // 只请求 1 个算法
        &heuristicResult,
        &returnedAlgoCount);
    
    // 如果启发式搜索成功，使用返回的算法
    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returnedAlgoCount > 0) {
      custom_algo = heuristicResult.algo;
      algo = &custom_algo;
      required_workspace_size = heuristicResult.workspaceSize;
      
      if (required_workspace_size > workspace_size) {
         // 理论上不会发生，因为我们设置了 PREF_MAX_WORKSPACE_BYTES
         // 但为了安全起见检查一下
         // 如果不足，我们只能尝试运行（可能会失败）或者报错
         // 由于是 void 返回，这里仅做警告或继续
      }
    } else {
      // 启发式搜索失败，使用 algo=NULL（让 cuBLASLt 内部选择）
      algo = nullptr;
    }
    
    cublasLtMatmulPreferenceDestroy(preference);
  }
  
  // ========== 执行 Matmul ==========
  // alpha = 1, beta = 0（纯矩阵乘法，scale 由后续 kernel 处理）
  // 注意：INT8 + COMPUTE_32I 需要 int32 类型的 alpha/beta
  //       FP8 + COMPUTE_32F 需要 float 类型的 alpha/beta
  const void* alpha_ptr;
  const void* beta_ptr;
  int32_t alpha_i32 = 1, beta_i32 = 0;
  float alpha_f32 = 1.0f, beta_f32 = 0.0f;
  
  if (input_dtype == "int8") {
    alpha_ptr = &alpha_i32;
    beta_ptr = &beta_i32;
  } else {
    alpha_ptr = &alpha_f32;
    beta_ptr = &beta_f32;
  }
  
  CHECK_CUBLASLT(cublasLtMatmul(
      handle,
      matmulDesc,
      alpha_ptr,
      W_ptr,
      layoutW,
      A_ptr,
      layoutA,
      beta_ptr,
      D_ptr,    // C（用于累加，这里 beta=0 所以不使用）
      layoutD,
      D_ptr,    // D（输出）
      layoutD,
      algo,
      workspace_ptr,
      workspace_size,
      stream));
  
  // ========== 清理资源 ==========
  cublasLtMatrixLayoutDestroy(layoutW);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutD);
  cublasLtMatmulDescDestroy(matmulDesc);
}

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
const char* cublaslt_gemm_get_last_error() {
    return g_last_error;
}

/**
 * cuBLASLt FP8 Matrix Multiplication
 * 
 * 
 * @param W_ptr         权重矩阵指针 [N, K]，FP8E4M3，行主序（GPU 内存）
 * @param A_ptr         输入矩阵指针 [M, K]，FP8E4M3，行主序（GPU 内存）
 * @param D_ptr         输出矩阵指针 [M, N]，BF16/FP32，行主序（GPU 内存，调用方预分配）
 * @param M             A 的行数
 * @param N             W 的行数（输出列数）
 * @param K             内维度
 * @param inner_dtype   输出数据类型字符串："bf16" 或 "fp32"
 * @param algo_data     算法配置数据（64字节），NULL或全零表示使用默认启发式
 * @param workspace_ptr 外部 workspace 指针
 * @param workspace_size 外部 workspace 大小
 * @param stream        CUDA 流（可为 NULL 使用默认流）
 * @return              0 成功，-1 失败
 */
int cublaslt_fp8_mm(
    const void* W_ptr,
    const void* A_ptr,
    void* D_ptr,
    void* workspace_ptr,
    size_t workspace_size,
    int64_t M, int64_t N, int64_t K,
    const char* inner_dtype,
    const uint8_t* algo_data,
    cudaStream_t stream
) {
    clear_error();
    try {
        if (!W_ptr || !A_ptr || !D_ptr) {
            set_error("Input/output pointers cannot be null");
            return -1;
        }
        if (!inner_dtype) {
            set_error("inner_dtype cannot be null");
            return -1;
        }
        
        cublaslt_mm_impl(W_ptr, A_ptr, D_ptr, M, N, K, 
                         "fp8e4m3", std::string(inner_dtype),
                         algo_data, workspace_ptr, workspace_size, stream);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cublaslt_fp8_mm");
        return -1;
    }
}

// 用于打印一次性 INT8 警告信息
static bool g_int8_info_printed = false;
static std::mutex g_int8_info_mutex;

static void print_int8_info_once() {
    if (!g_int8_info_printed) {
        std::lock_guard<std::mutex> lock(g_int8_info_mutex);
        if (!g_int8_info_printed) {
            std::cerr << "[cuBLASLt] INFO: INT8 GEMM only supports INT32 output. "
                      << "inner_dtype parameter is ignored, always using int32.\n";
            g_int8_info_printed = true;
        }
    }
}

/**
 * cuBLASLt INT8 Matrix Multiplication
 * 
 * 
 * 注意: cuBLASLt INT8 GEMM 仅支持 INT32 输出（CUBLAS_COMPUTE_32I）
 *       inner_dtype 参数被忽略，总是使用 "int32"
 * 
 * @param W_ptr         权重矩阵指针 [N, K]，INT8，行主序（GPU 内存）
 * @param A_ptr         输入矩阵指针 [M, K]，INT8，行主序（GPU 内存）
 * @param D_ptr         输出矩阵指针 [M, N]，INT32，行主序（GPU 内存，调用方预分配）
 * @param workspace_ptr 外部 workspace 指针
 * @param workspace_size 外部 workspace 大小
 * @param M             A 的行数
 * @param N             W 的行数（输出列数）
 * @param K             内维度
 * @param inner_dtype   被忽略，总是使用 "int32"
 * @param algo_data     算法配置数据（64字节），NULL或全零表示使用默认启发式
 * @param stream        CUDA 流（可为 NULL 使用默认流）
 * @return              0 成功，-1 失败
 */
int cublaslt_int8_mm(
    const void* W_ptr,
    const void* A_ptr,
    void* D_ptr,
    void* workspace_ptr,
    size_t workspace_size,
    int64_t M, int64_t N, int64_t K,
    const char* inner_dtype,
    const uint8_t* algo_data,
    cudaStream_t stream
) {
    clear_error();
    try {
        if (!W_ptr || !A_ptr || !D_ptr) {
            set_error("Input/output pointers cannot be null");
            return -1;
        }
        
        // 打印一次性 INFO: cuBLASLt INT8 仅支持 INT32 输出
        print_int8_info_once();
        
        // 忽略 inner_dtype 参数，强制使用 int32
        cublaslt_mm_impl(W_ptr, A_ptr, D_ptr, M, N, K,
                         "int8", "int32",
                         algo_data, workspace_ptr, workspace_size, stream);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("Unknown error in cublaslt_int8_mm");
        return -1;
    }
}

/**
 * 检查 cuBLASLt 是否可用
 * 
 * @return 1 可用，0 不可用
 */
int cublaslt_is_available() {
    try {
        get_cublaslt_handle();
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
const char* cublaslt_get_supported_dtypes() {
    return "fp8e4m3,int8";
}

}  // extern "C"
