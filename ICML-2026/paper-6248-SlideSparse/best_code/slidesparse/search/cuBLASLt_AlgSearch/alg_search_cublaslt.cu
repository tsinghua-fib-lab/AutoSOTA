// cuBLASLt 算法离线搜索 (extern "C" 版本)
// 固定的 layout: 权重 W 在左，T/N + C/C GEMM，输出矩阵 order 固定为 Column 主序
// R[N,M]_col = W[N,K]^T_col * A[K,M]_col
// 支持的数据类型：int8, fp8e4m3
//
// 支持的数据类型组合:
// - FP8E4M3 输入 → BF16/FP32 输出 (CUBLAS_COMPUTE_32F)
// - INT8 输入 → INT32 输出 (CUBLAS_COMPUTE_32I，cuBLASLt INT8 不支持 BF16 输出)
//
// 导出方式: extern "C"，使用 ctypes 加载
// 参考: slidesparse/csrc/cublaslt_gemm/cublaslt_gemm.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cublas_v2.h>

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

static cudaDataType to_cuda_dtype(const char* dtype) {
  if (std::strcmp(dtype, "int8") == 0) return CUDA_R_8I;
  if (std::strcmp(dtype, "fp8e4m3") == 0) return CUDA_R_8F_E4M3;
  throw std::invalid_argument("Unsupported dtype. Supported: int8, fp8e4m3");
}

static cudaDataType cuda_out_dtype(const char* outdtype) {
  if (std::strcmp(outdtype, "bf16") == 0) return CUDA_R_16BF;
  if (std::strcmp(outdtype, "fp32") == 0) return CUDA_R_32F;
  if (std::strcmp(outdtype, "int32") == 0) return CUDA_R_32I;
  throw std::invalid_argument("Unsupported outdtype. Supported: bf16, fp32, int32");
}

static cublasComputeType_t compute_type_from_dtype(const char* dtype) {
  // FP8 使用 CUBLAS_COMPUTE_32F 支持 BF16/FP32 输出
  // INT8 使用 CUBLAS_COMPUTE_32I，只支持 INT32 输出（cuBLASLt 限制）
  if (std::strcmp(dtype, "int8") == 0) return CUBLAS_COMPUTE_32I;
  if (std::strcmp(dtype, "fp8e4m3") == 0) return CUBLAS_COMPUTE_32F;
  throw std::invalid_argument("Unsupported dtype");
}

static cudaDataType scale_type_from_dtype(const char* dtype) {
  // INT8 + COMPUTE_32I 需要 INT32 scale type
  // FP8 + COMPUTE_32F 需要 FP32 scale type
  if (std::strcmp(dtype, "int8") == 0) return CUDA_R_32I;
  return CUDA_R_32F;
}

static int dtype_elem_size(const char* dtype) {
  // int8 和 fp8e4m3 都是 1 字节
  return 1;
}

static int out_dtype_elem_size(const char* outdtype) {
  if (std::strcmp(outdtype, "bf16") == 0) return 2;
  if (std::strcmp(outdtype, "fp32") == 0) return 4;
  if (std::strcmp(outdtype, "int32") == 0) return 4;
  return 2;
}

// ============================================================================
// 搜索结果结构
// ============================================================================

// 编译期检查：确保 cublasLtMatmulAlgo_t 不超过我们预留的缓冲区大小
static_assert(sizeof(cublasLtMatmulAlgo_t) <= 64,
              "cublasLtMatmulAlgo_t size exceeds 64 bytes, increase algo_data buffer");

struct AlgRecord {
  int alg_id{-1};
  float lat_us{0.f};
  float tops{0.f};
  int64_t workspace{0};
  float waves_count{0.f};
  bool valid{false};
  uint8_t algo_data[64];  // 64 字节不透明算法数据
  
  AlgRecord() {
    memset(algo_data, 0, sizeof(algo_data));
  }
};

// ============================================================================
// 错误处理基础设施
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
 * 搜索单个 (N, K, M) 组合的最佳算法
 *
 * 固定布局: T/N + Col/Col + Col (权重 W 在左)
 * W[N,K]^T_col * A[K,M]_col = R[N,M]_col
 *
 * @param W_ptr       权重矩阵指针 [N, K]，已量化的数据
 * @param A_ptr       激活矩阵指针 [M, K]，已量化的数据
 * @param R_ptr       输出矩阵指针 [M, N] (Column Major [N,M])
 * @param N           权重行数
 * @param K           内维度
 * @param M           激活行数 / 输出列数
 * @param dtype       输入数据类型 "int8" 或 "fp8e4m3"
 * @param outdtype    输出数据类型 "bf16" 或 "fp32"
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
    const char* outdtype,
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

    cudaDataType type_AB = to_cuda_dtype(dtype);
    cudaDataType type_C = cuda_out_dtype(outdtype);
    cublasComputeType_t comp_type = compute_type_from_dtype(dtype);
    cudaDataType scale_type = scale_type_from_dtype(dtype);  // INT8 用 INT32, FP8 用 FP32

    cublasLtHandle_t handle = get_cublaslt_handle();

    // 固定布局: T/N + Col/Col + Col
    cublasOperation_t opW = CUBLAS_OP_T;
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasLtOrder_t orderW = CUBLASLT_ORDER_COL;
    cublasLtOrder_t orderA = CUBLASLT_ORDER_COL;
    cublasLtOrder_t orderR = CUBLASLT_ORDER_COL;

    // W[N,K] 存储维度
    int64_t num_W_rows = K;
    int64_t num_W_cols = N;
    int64_t ldw = K;

    // A[K,M] 存储维度
    int64_t num_A_rows = K;
    int64_t num_A_cols = M;
    int64_t lda = K;

    // C[N,M] 存储维度
    int64_t num_R_rows = N;
    int64_t num_R_cols = M;
    int64_t ldr = N;

    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmulDesc, comp_type, scale_type));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));

    // 创建矩阵布局描述符
    cublasLtMatrixLayout_t layoutW = nullptr, layoutA = nullptr, layoutR = nullptr;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutW, type_AB, num_W_rows, num_W_cols, ldw));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderW, sizeof(orderW)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutA, type_AB, num_A_rows, num_A_cols, lda));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutR, type_C, num_R_rows, num_R_cols, ldr));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutR, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderR, sizeof(orderR)));

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
    const int max_algo_count = 128;
    cublasLtMatmulHeuristicResult_t heuristicResult[max_algo_count];
    int returnedAlgoCount = 0;

    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutW, layoutA, layoutR, layoutR,
        preference, max_algo_count, heuristicResult, &returnedAlgoCount);

    if (out_alg_count) *out_alg_count = returnedAlgoCount;

    if (heur_status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
      // 清理并返回空结果
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layoutW);
      cublasLtMatrixLayoutDestroy(layoutA);
      cublasLtMatrixLayoutDestroy(layoutR);
      cublasLtMatmulDescDestroy(matmulDesc);
      
      if (out_num_valid) *out_num_valid = 0;
      return 0;
    }

    // 分配共享 workspace (初始 512MB, 可动态扩展)
    void* workspace = nullptr;
    size_t current_workspace_size = max_workspace_size;
    CHECK_CUDA(cudaMalloc(&workspace, current_workspace_size));

    // alpha/beta 类型需要与 scale_type 匹配
    // INT8 GEMM: scale_type = CUDA_R_32I, 需要 int32 alpha/beta
    // FP8 GEMM: scale_type = CUDA_R_32F, 需要 float alpha/beta
    float alpha_f = 1.0f, beta_f = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    const void* alpha_ptr = (scale_type == CUDA_R_32I) ? (const void*)&alpha_i : (const void*)&alpha_f;
    const void* beta_ptr = (scale_type == CUDA_R_32I) ? (const void*)&beta_i : (const void*)&beta_f;
    
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
      for (int i = 0; i < warmup && success; ++i) {
        cublasStatus_t st = cublasLtMatmul(
            handle, matmulDesc, alpha_ptr,
            W_ptr, layoutW, A_ptr, layoutA, beta_ptr,
            R_ptr, layoutR, R_ptr, layoutR,
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
              R_ptr, layoutR, R_ptr, layoutR,
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
    cublasLtMatrixLayoutDestroy(layoutR);
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
  return "fp8e4m3,int8";
}

/**
 * 获取对齐要求
 */
int cublaslt_alg_search_get_alignment(const char* dtype) {
  if (std::strcmp(dtype, "fp8e4m3") == 0) return 16;
  if (std::strcmp(dtype, "int8") == 0) return 4;
  return 1;
}

}  // extern "C"
