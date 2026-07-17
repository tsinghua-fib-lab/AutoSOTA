// SPDX-License-Identifier: Apache-2.0
/**
 * cuBLASLt Dense GEMM Benchmark Implementation
 * 
 * 支持的数据类型（按用户规格）:
 * ==============================
 * - FP16:    FP16 输入, FP32 计算, FP16 输出
 * - BF16:    BF16 输入, FP32 计算, BF16 输出
 * - INT8:    INT8 输入, INT32 计算, INT8 输出
 * - FP8E4M3: FP8 输入, FP32 计算, FP8 输出
 * - FP4E2M1: FP4 输入, FP32 计算, FP4 输出 (需要强制 scale)
 * 
 * 固定 Layout:
 * ============
 * - T/N + Col/Col + Col (权重 W 在左)
 * - R[N,M]_col = W[K,N]^T_col @ A[K,M]_col
 * 
 * 等价于 PyTorch 行主序:
 * - W[N,K] 行主序 → 声明列主序 [K,N] → opA=T → [N,K]
 * - A[M,K] 行主序 → 声明列主序 [K,M] → opB=N → [K,M]
 * - R[N,M] 列主序结果 → 按行主序读 = [M,N]
 * 
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      cublaslt_gemm.cu -lcublasLt -lcublas -lcuda \
 *      -o cublaslt_gemm.so
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

static cublasLtHandle_t g_cublaslt_handle = nullptr;
static bool g_cublaslt_initialized = false;
static std::mutex g_cublaslt_init_mutex;

static cublasLtHandle_t get_cublaslt_handle() {
  if (!g_cublaslt_initialized) {
    std::lock_guard<std::mutex> lock(g_cublaslt_init_mutex);
    if (!g_cublaslt_initialized) {
      cublasStatus_t status = cublasLtCreate(&g_cublaslt_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLASLt handle");
      }
      g_cublaslt_initialized = true;

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
// 数据类型转换
// ============================================================================

// 检查 FP4 硬件支持
static bool check_fp4_support() {
#if CUDART_VERSION >= 12050
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // FP4 需要 SM100+ (Blackwell)
    return (prop.major >= 10);
#else
    return false;
#endif
}

// 获取输入数据类型
// 返回 CUDA_R_32F 表示不支持的类型
static cudaDataType to_cuda_input_dtype(const char* dtype, bool* supported = nullptr) {
  if (supported) *supported = true;
  
  if (std::strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
  if (std::strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
  if (std::strcmp(dtype, "int8") == 0) return CUDA_R_8I;
  if (std::strcmp(dtype, "fp8e4m3") == 0 || std::strcmp(dtype, "fp8") == 0) return CUDA_R_8F_E4M3;
#if CUDART_VERSION >= 12050
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) {
    if (!check_fp4_support()) {
      std::fprintf(stderr, "[cuBLASLt WARN] FP4 (e2m1) requires SM100+ (Blackwell), skipping\n");
      if (supported) *supported = false;
      return CUDA_R_32F;
    }
    return CUDA_R_4F_E2M1;
  }
#else
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) {
    std::fprintf(stderr, "[cuBLASLt WARN] FP4 (e2m1) requires CUDA 12.5+, skipping\n");
    if (supported) *supported = false;
    return CUDA_R_32F;
  }
#endif
  // 未知类型
  if (supported) *supported = false;
  return CUDA_R_32F;
}

// 累加器类型 (C 矩阵)
// - INT8 + PEDANTIC: CType = INT8
// - FP8 + FP8 输出: CType = BF16
// - FP4: CType = BF16（根据 Table 4 CType 列）
// - FP16/BF16: CType = 与输出相同
static cudaDataType to_cuda_accumulator_dtype(const char* dtype) {
  // INT8 + PEDANTIC 模式：C 和 D 都可以是 INT8
  if (std::strcmp(dtype, "int8") == 0) return CUDA_R_8I;
  // FP8 输出时，CType 应该是 BF16
  if (std::strcmp(dtype, "fp8e4m3") == 0 || std::strcmp(dtype, "fp8") == 0) return CUDA_R_16BF;
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) return CUDA_R_16BF;  // FP4 累加用 BF16
  // FP16/BF16: C 和 D 可以相同类型
  if (std::strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
  if (std::strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
  return CUDA_R_32F;  // 默认 FP32
}

// 输出类型 (D 矩阵) - 低精度输出，与输入类型相同
// 这是最终写入显存的类型，实现低 IO 开销
static cudaDataType to_cuda_output_dtype(const char* dtype) {
  if (std::strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
  if (std::strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
  if (std::strcmp(dtype, "int8") == 0) return CUDA_R_8I;      // INT8 → INT8
  if (std::strcmp(dtype, "fp8e4m3") == 0 || std::strcmp(dtype, "fp8") == 0) return CUDA_R_8F_E4M3;  // FP8 → FP8
#if CUDART_VERSION >= 12050
  // 暂时测试：FP4 输出改为 BF16，验证其他配置是否正确
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) return CUDA_R_16BF;  // FP4 → BF16 (临时)
#endif
  return CUDA_R_16BF;  // 默认 BF16
}

// 获取计算类型
// 注意：INT8 输出需要 PEDANTIC 模式
static cublasComputeType_t get_compute_type(const char* dtype) {
  if (std::strcmp(dtype, "int8") == 0) {
    // INT8 输出需要 CUBLAS_COMPUTE_32I_PEDANTIC + FP32 scale
    return CUBLAS_COMPUTE_32I_PEDANTIC;
  }
  return CUBLAS_COMPUTE_32F;  // FP16/BF16/FP8/FP4 都使用 FP32 累加
}

// 获取 scale 类型
// 注意：INT8 输出需要 FP32 scale（不是 INT32）
static cudaDataType get_scale_type(const char* dtype) {
  // INT8 + PEDANTIC 需要 FP32 scale（根据 Table 2 第二行）
  // 其他类型也使用 FP32 scale
  return CUDA_R_32F;
}

// 获取输入元素大小 (字节)
static int get_elem_size(const char* dtype) {
  if (std::strcmp(dtype, "fp16") == 0 || std::strcmp(dtype, "bf16") == 0) return 2;
  if (std::strcmp(dtype, "int8") == 0 || std::strcmp(dtype, "fp8e4m3") == 0 || 
      std::strcmp(dtype, "fp8") == 0) return 1;
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) return 1;
  return 1;
}

// 获取输出元素大小 (字节) - 低精度输出
static int get_out_elem_size(const char* dtype) {
  if (std::strcmp(dtype, "fp16") == 0 || std::strcmp(dtype, "bf16") == 0) return 2;
  if (std::strcmp(dtype, "int8") == 0) return 1;  // INT8 输出
  if (std::strcmp(dtype, "fp8e4m3") == 0 || std::strcmp(dtype, "fp8") == 0) return 1;  // FP8 输出
  if (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0) return 1;  // FP4 打包输出 (2 值/字节，但按字节算)
  return 1;
}

// ============================================================================
// 搜索结果结构
// ============================================================================

// 编译期检查：确保 cublasLtMatmulAlgo_t 不超过 64 字节
static_assert(sizeof(cublasLtMatmulAlgo_t) <= 64,
              "cublasLtMatmulAlgo_t size exceeds 64 bytes");

struct AlgRecord {
  int alg_id{-1};
  float lat_us{0.f};
  float tops{0.f};
  int64_t workspace{0};
  float waves_count{0.f};
  bool valid{false};
  uint8_t algo_data[64];
  
  AlgRecord() {
    memset(algo_data, 0, sizeof(algo_data));
  }
};

// ============================================================================
// 错误处理
// ============================================================================

static thread_local char g_last_error[1024] = "";

static void set_error(const char* msg) {
  std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
  g_last_error[sizeof(g_last_error) - 1] = '\0';
}

static void clear_error() {
  g_last_error[0] = '\0';
}

// ============================================================================
// extern "C" 导出接口
// ============================================================================

extern "C" {

/**
 * 获取最后一次错误信息
 */
const char* cublaslt_alg_search_get_last_error() {
  return g_last_error;
}

/**
 * 检查 cuBLASLt 是否可用
 */
int cublaslt_alg_search_is_available() {
  try {
    get_cublaslt_handle();
    return 1;
  } catch (...) {
    return 0;
  }
}

/**
 * 获取支持的数据类型
 */
const char* cublaslt_alg_search_get_supported_dtypes() {
  return "fp16,bf16,int8,fp8e4m3,fp4e2m1";
}

/**
 * 获取对齐要求
 */
int cublaslt_alg_search_get_alignment(const char* dtype) {
  // 所有类型都要求 32 对齐
  return 32;
}

/**
 * 搜索单个 (N, K, M) 组合的最佳算法
 *
 * 固定布局: T/N + Col/Col + Col (权重 W 在左)
 * R[N,M]_col = W[K,N]^T_col @ A[K,M]_col
 *
 * @param W_ptr       权重矩阵指针 [K, N] 列主序
 * @param A_ptr       激活矩阵指针 [K, M] 列主序
 * @param R_ptr       输出矩阵指针 [N, M] 列主序
 * @param N           输出行数 / 权重列数
 * @param K           内维度
 * @param M           输出列数 / 激活列数
 * @param dtype       数据类型 ("fp16", "bf16", "int8", "fp8e4m3", "fp4e2m1")
 * @param warmup      预热次数
 * @param repeat      测量次数
 * @param topk        返回前 k 个结果
 * @param out_alg_ids      输出: 算法 ID 数组 [topk]
 * @param out_lat_us       输出: 延迟 (微秒) [topk]
 * @param out_tops         输出: 吞吐量 (TOPS) [topk]
 * @param out_workspace    输出: workspace 大小 [topk]
 * @param out_waves_count  输出: GPU 利用率 [topk]
 * @param out_algo_data    输出: 64 字节算法数据 [topk * 64]
 * @param out_valid        输出: 有效标志 [topk]
 * @param out_num_valid    输出: 有效结果数量
 * @param out_alg_count    输出: 启发式返回的算法总数
 * @param stream           CUDA 流
 *
 * @return 0 成功，-1 失败
 */
int cublaslt_search_single_m(
    const void* W_ptr,
    const void* A_ptr,
    void* R_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype,
    int warmup,
    int repeat,
    int topk,
    int* out_alg_ids,
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    float* out_waves_count,
    uint8_t* out_algo_data,
    uint8_t* out_valid,
    int* out_num_valid,
    int* out_alg_count,
    cudaStream_t stream
) {
  clear_error();
  
  try {
    if (!W_ptr || !A_ptr || !R_ptr) {
      set_error("Input/output pointers cannot be null");
      return -1;
    }
    if (topk <= 0) topk = 3;
    
    // ========================================================================
    // 维度检查
    // ========================================================================
    // 对于所有类型，cuBLASLt 对维度有一定对齐要求
    // FP4/FP8/INT8: 通常需要更严格的对齐 (16 或 32)
    // FP16/BF16:    8 的倍数
    // ========================================================================
    bool is_fp4 = (std::strcmp(dtype, "fp4e2m1") == 0 || std::strcmp(dtype, "fp4") == 0);
    bool is_8bit = (std::strcmp(dtype, "int8") == 0 || std::strcmp(dtype, "fp8e4m3") == 0 || 
                   std::strcmp(dtype, "fp8") == 0 || is_fp4);
    
    int align_req = is_8bit ? 16 : 8;  // 基本对齐要求
    
    // FP4 特殊要求：M 必须是 32 的倍数（与 cuSPARSELt 类似）
    if (is_fp4 && M % 32 != 0) {
        std::fprintf(stderr, "[cuBLASLt WARN] Skipping: FP4 requires M=%lld to be a multiple of 32\n",
                (long long)M);
        if (out_num_valid) *out_num_valid = 0;
        if (out_alg_count) *out_alg_count = 0;
        return 0;  // 返回成功但无结果
    }
    
    // FP4 特殊要求：K 必须是 32 的倍数
    if (is_fp4 && K % 32 != 0) {
        std::fprintf(stderr, "[cuBLASLt WARN] Skipping: FP4 requires K=%lld to be a multiple of 32\n",
                (long long)K);
        if (out_num_valid) *out_num_valid = 0;
        if (out_alg_count) *out_alg_count = 0;
        return 0;
    }

    // 获取数据类型配置（检查是否支持）
    bool dtype_supported = true;
    cudaDataType type_AB = to_cuda_input_dtype(dtype, &dtype_supported);
    if (!dtype_supported) {
        // dtype 不支持（如 FP4 在非 Blackwell 上），跳过
        if (out_num_valid) *out_num_valid = 0;
        if (out_alg_count) *out_alg_count = 0;
        set_error("Unsupported dtype for current GPU architecture");
        return 0;  // 返回成功但无结果（跳过）
    }
    
    // 关键修改：分开定义累加器类型 (C) 和输出类型 (D)
    // - type_C_acc: 累加器类型，必须是高精度 (INT32/FP32/BF16)
    // - type_D_out: 输出类型，低精度 (与输入相同)
    cudaDataType type_C_acc = to_cuda_accumulator_dtype(dtype);  // 累加器（高精度）
    cudaDataType type_D_out = to_cuda_output_dtype(dtype);       // 输出（低精度）
    cublasComputeType_t comp_type = get_compute_type(dtype);
    cudaDataType scale_type = get_scale_type(dtype);

    cublasLtHandle_t handle = get_cublaslt_handle();

    // 固定布局: T/N + Col/Col + Col
    cublasOperation_t opW = CUBLAS_OP_T;
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasLtOrder_t orderW = CUBLASLT_ORDER_COL;
    cublasLtOrder_t orderA = CUBLASLT_ORDER_COL;
    cublasLtOrder_t orderR = CUBLASLT_ORDER_COL;

    // W[K,N] 存储维度 (列主序)
    int64_t num_W_rows = K;
    int64_t num_W_cols = N;
    int64_t ldw = K;

    // A[K,M] 存储维度 (列主序)
    int64_t num_A_rows = K;
    int64_t num_A_cols = M;
    int64_t lda = K;

    // R[N,M] 存储维度 (列主序)
    // FP4 输出为 BF16，不需要打包
    int64_t num_R_rows = N;
    int64_t num_R_cols = M;
    int64_t ldr = num_R_rows;

    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmulDesc, comp_type, scale_type));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));

    // === FP4 Block Scale 配置 ===
    // FP4 需要 16-element 1D block scaling (CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
    // Scale 张量类型为 UE4M3 (无符号 E4M3)，每 16 个元素一个 scale
    void* scale_A_ptr = nullptr;
    void* scale_B_ptr = nullptr;
    int64_t scale_A_size = 0;
    int64_t scale_B_size = 0;

#if CUDART_VERSION >= 12050
    if (is_fp4) {
        // 设置 16-element 1D block scaling mode（仅 A 和 B）
        cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        
        // 分配 A/B scale 张量 (UE4M3 类型 = 1 字节，每 16 个元素一个 scale)
        // W (matA, 转置后 [N, K]): scale 大小 = N * ceil(K/16)
        // A (matB, [K, M]):        scale 大小 = M * ceil(K/16)
        int64_t K_blocks = (K + 15) / 16;
        scale_A_size = N * K_blocks;  // For W (作为 A 矩阵)
        scale_B_size = M * K_blocks;  // For A (作为 B 矩阵)
        
        CHECK_CUDA(cudaMalloc(&scale_A_ptr, scale_A_size * sizeof(uint8_t)));
        CHECK_CUDA(cudaMalloc(&scale_B_ptr, scale_B_size * sizeof(uint8_t)));
        
        // 初始化 A/B scale 为中性值
        // UE4M3 格式: 无符号 E4M3，值 1.0 的编码约为 0x38 (exp=7-bias=7=0, mant=0)
        CHECK_CUDA(cudaMemset(scale_A_ptr, 0x38, scale_A_size * sizeof(uint8_t)));
        CHECK_CUDA(cudaMemset(scale_B_ptr, 0x38, scale_B_size * sizeof(uint8_t)));
        
        // 设置 A/B scale 指针
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_A_ptr, sizeof(void*)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_B_ptr, sizeof(void*)));
    }
#endif

    // 创建矩阵布局描述符
    // 关键修改：分开创建 C (累加器) 和 D (输出) 的布局
    cublasLtMatrixLayout_t layoutW = nullptr, layoutA = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr, layoutD = nullptr;  // C 累加器, D 输出
    
    // W 和 A 输入布局
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutW, type_AB, num_W_rows, num_W_cols, ldw));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderW, sizeof(orderW)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutA, type_AB, num_A_rows, num_A_cols, lda));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
    
    // C 矩阵布局 - 累加器（高精度）
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutC, type_C_acc, num_R_rows, num_R_cols, ldr));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderR, sizeof(orderR)));
    
    // D 矩阵布局 - 输出（低精度，与输入相同）
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutD, type_D_out, num_R_rows, num_R_cols, ldr));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderR, sizeof(orderR)));

    // 创建算法偏好
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t max_workspace_size = 512 * 1024 * 1024;  // 512 MB
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_size, sizeof(max_workspace_size)));
    
    uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        &reduction_scheme_mask, sizeof(reduction_scheme_mask)));

    // 获取可用算法
    // 关键修改：使用分开的 layoutC 和 layoutD
    const int max_algo_count = 128;
    cublasLtMatmulHeuristicResult_t heuristicResult[max_algo_count];
    int returnedAlgoCount = 0;

    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutW, layoutA, layoutC, layoutD,
        preference, max_algo_count, heuristicResult, &returnedAlgoCount);

    if (out_alg_count) *out_alg_count = returnedAlgoCount;

    if (heur_status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
      // 清理并返回空结果
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layoutW);
      cublasLtMatrixLayoutDestroy(layoutA);
      cublasLtMatrixLayoutDestroy(layoutC);
      cublasLtMatrixLayoutDestroy(layoutD);
      cublasLtMatmulDescDestroy(matmulDesc);
      
      // 释放 FP4 scale 张量
      if (scale_A_ptr) cudaFree(scale_A_ptr);
      if (scale_B_ptr) cudaFree(scale_B_ptr);
      
      if (out_num_valid) *out_num_valid = 0;
      return 0;
    }

    // 分配共享 workspace
    void* workspace = nullptr;
    size_t current_workspace_size = max_workspace_size;
    CHECK_CUDA(cudaMalloc(&workspace, current_workspace_size));

    // alpha/beta 类型需要与 scale_type 匹配
    float alpha_f = 1.0f, beta_f = 0.0f;
    const void* alpha_ptr = &alpha_f;
    const void* beta_ptr = &beta_f;
    
    std::vector<AlgRecord> records;

    // 遍历所有算法
    for (int alg_idx = 0; alg_idx < returnedAlgoCount; ++alg_idx) {
      if (heuristicResult[alg_idx].state != CUBLAS_STATUS_SUCCESS) {
        continue;
      }

      const cublasLtMatmulAlgo_t* algo = &heuristicResult[alg_idx].algo;
      size_t ws_size = heuristicResult[alg_idx].workspaceSize;

      // 动态扩展 workspace
      if (ws_size > current_workspace_size) {
        void* new_workspace = nullptr;
        cudaError_t alloc_err = cudaMalloc(&new_workspace, ws_size);
        if (alloc_err == cudaSuccess) {
          cudaFree(workspace);
          workspace = new_workspace;
          current_workspace_size = ws_size;
        } else {
          continue;  // 分配失败，跳过此算法
        }
      }

      void* ws_ptr = (ws_size > 0) ? workspace : nullptr;
      bool success = true;

      // 预热
      // 关键修改：使用分开的 layoutC 和 layoutD
      for (int i = 0; i < warmup && success; ++i) {
        cublasStatus_t st = cublasLtMatmul(
            handle, matmulDesc, alpha_ptr,
            W_ptr, layoutW, A_ptr, layoutA, beta_ptr,
            R_ptr, layoutC, R_ptr, layoutD,
            algo, ws_ptr, ws_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) success = false;
      }
      if (success) CHECK_CUDA(cudaStreamSynchronize(stream));

      // 计时
      cudaEvent_t start, stop;
      float total_ms = 0.0f;
      
      if (success) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        for (int r = 0; r < repeat && success; ++r) {
          cublasStatus_t st = cublasLtMatmul(
              handle, matmulDesc, alpha_ptr,
              W_ptr, layoutW, A_ptr, layoutA, beta_ptr,
              R_ptr, layoutC, R_ptr, layoutD,
              algo, ws_ptr, ws_size, stream);
          if (st != CUBLAS_STATUS_SUCCESS) success = false;
        }
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
      }

      if (success) {
        AlgRecord rec;
        int algo_id = 0;
        cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID,
                                              &algo_id, sizeof(algo_id), nullptr);
        rec.alg_id = algo_id;
        rec.lat_us = (total_ms * 1000.0f) / static_cast<float>(repeat);
        double ops = 2.0 * M * N * K;
        rec.tops = static_cast<float>(ops / (rec.lat_us / 1e6) / 1e12);
        rec.workspace = static_cast<int64_t>(ws_size);
        rec.waves_count = heuristicResult[alg_idx].wavesCount;
        rec.valid = true;
        memcpy(rec.algo_data, algo, sizeof(cublasLtMatmulAlgo_t));
        records.push_back(rec);
      }
    }

    // 释放 workspace
    if (workspace) cudaFree(workspace);

    // 释放 FP4 scale 张量
    if (scale_A_ptr) cudaFree(scale_A_ptr);
    if (scale_B_ptr) cudaFree(scale_B_ptr);

    // 排序并填充输出
    std::sort(records.begin(), records.end(),
              [](const AlgRecord& a, const AlgRecord& b) {
                return a.lat_us < b.lat_us;
              });

    int fill = std::min(static_cast<int>(records.size()), topk);
    for (int i = 0; i < topk; ++i) {
      if (i < fill) {
        out_alg_ids[i] = records[i].alg_id;
        out_lat_us[i] = records[i].lat_us;
        out_tops[i] = records[i].tops;
        out_workspace[i] = records[i].workspace;
        out_waves_count[i] = records[i].waves_count;
        memcpy(out_algo_data + i * 64, records[i].algo_data, 64);
        out_valid[i] = 1;
      } else {
        out_alg_ids[i] = -1;
        out_lat_us[i] = 0.0f;
        out_tops[i] = 0.0f;
        out_workspace[i] = 0;
        out_waves_count[i] = 0.0f;
        memset(out_algo_data + i * 64, 0, 64);
        out_valid[i] = 0;
      }
    }
    
    if (out_num_valid) *out_num_valid = static_cast<int>(records.size());

    // 清理
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutD);
    cublasLtMatmulDescDestroy(matmulDesc);

    return 0;

  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("Unknown error in cublaslt_search_single_m");
    return -1;
  }
}

}  // extern "C"
