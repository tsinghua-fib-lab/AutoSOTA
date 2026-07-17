// cuBLASLt Layout 性能测试
// 提供 Python API，测试不同 layout 组合的 GEMM 性能
// 支持的数据类型：int8, fp8e4m3
//
// =====================================
// cuBLASLt API 约束 (基于官方文档):
// =====================================
//
// INT8 IMMA 内核要求 (使用常规数据排序):
//   - 所有矩阵指针必须 4 字节对齐 (16 字节对齐性能更佳)
//   - Leading dimensions 必须是 4 的倍数
//   - 只支持 TN 格式: A 必须转置, B 不转置
//   - m 和 k 必须是 4 的倍数
//   - computeType: CUBLAS_COMPUTE_32I
//
// FP8 内核要求:
//   - 所有矩阵维度必须满足 16 字节对齐
//   - TN 格式是首选 (在 Ada/Hopper/Blackwell GeForce 上)
//   - computeType 必须是 CUBLAS_COMPUTE_32F
//
// 数据类型支持 (本实现):
//   - INT8: Input=CUDA_R_8I, Output=bf16/fp32, Scale=CUDA_R_32F
//   - FP8:  Input=CUDA_R_8F_E4M3, Output=bf16/fp32, Scale=CUDA_R_32F
//
// 注意: 与 cuSPARSELt 不同, cuBLASLt 的 op 和 order 不是强制绑定的,
//       但某些组合可能没有优化的内核实现。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ===== 工具宏 =====
#define CHECK_CUDA_ERR(expr)                                                   \
  do {                                                                         \
    cudaError_t _status = (expr);                                              \
    if (_status != cudaSuccess) {                                              \
      std::ostringstream _oss;                                                 \
      _oss << "CUDA 调用失败: " << cudaGetErrorString(_status)                 \
           << " (code " << _status << ")";                                     \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS_ERR(expr)                                                 \
  do {                                                                         \
    cublasStatus_t _status = (expr);                                           \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream _oss;                                                 \
      _oss << "cuBLASLt 调用失败: " << cublasLtGetStatusString(_status)        \
           << " (code " << static_cast<int>(_status) << ")";                   \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// ===== dtype 相关 =====
static cudaDataType to_cuda_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUDA_R_8I;
  if (dtype == "fp8e4m3") return CUDA_R_8F_E4M3;
  throw std::invalid_argument("不支持的数据类型: " + dtype + "。支持: int8, fp8e4m3");
}

static cudaDataType cuda_out_dtype(const std::string &outdtype) {
  // 输出类型：支持 bf16 和 fp32
  if (outdtype == "bf16") return CUDA_R_16BF;
  if (outdtype == "fp32") return CUDA_R_32F;
  throw std::invalid_argument("不支持的输出数据类型: " + outdtype + "。支持: bf16, fp32");
}

static cublasComputeType_t compute_type_from_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUBLAS_COMPUTE_32I;
  if (dtype == "fp8e4m3") return CUBLAS_COMPUTE_32F;
  throw std::invalid_argument("不支持的数据类型: " + dtype);
}

static cudaDataType scale_type_from_dtype(const std::string &dtype) {
  // BF16 输出需要 CUDA_R_32F scale type
  // alpha/beta 使用 float 类型
  if (dtype == "int8" || dtype == "fp8e4m3") return CUDA_R_32F;
  throw std::invalid_argument("不支持的数据类型: " + dtype);
}

static int dtype_size(const std::string &dtype) {
  if (dtype == "int8" || dtype == "fp8e4m3") return 1;
  return 1;
}

static int out_dtype_size(const std::string &outdtype) {
  // 输出类型字节数：bf16=2, fp32=4
  if (outdtype == "bf16") return 2;  // bfloat16
  if (outdtype == "fp32") return 4;  // float32
  return 2;
}

// ===== 解析 layout 字符串 =====
static cublasOperation_t parse_op(const std::string &op) {
  if (op == "T") return CUBLAS_OP_T;
  if (op == "N") return CUBLAS_OP_N;
  throw std::invalid_argument("无效的操作: " + op + "。支持: T, N");
}

static cublasLtOrder_t parse_order(const std::string &order) {
  if (order == "Col") return CUBLASLT_ORDER_COL;
  if (order == "Row") return CUBLASLT_ORDER_ROW;
  throw std::invalid_argument("无效的顺序: " + order + "。支持: Col, Row");
}

// ===== 算法结果 =====
struct AlgResult {
  int alg_id;
  float lat_us;
  float tops;
  float waves_count;  // GPU 利用率指标
};

// ===== test_layout 主函数 =====
py::dict test_layout(
    int64_t N, int64_t K, int64_t M,
    const std::string &opW_str,     // "T" or "N"
    const std::string &opA_str,     // "T" or "N"
    const std::string &orderW_str,  // "Col" or "Row"
    const std::string &orderA_str,  // "Col" or "Row"
    const std::string &orderR_str,  // "Col" or "Row" (输出矩阵顺序)
    const std::string &dtype,
    const std::string &outdtype,    // "bf16" or "fp32" (输出数据类型)
    int warmup,
    int repeat
) {
  py::dict result;
  result["supported"] = false;
  result["algo_count"] = 0;
  result["top3"] = py::list();

  // 解析 layout 参数
  cublasOperation_t opW = parse_op(opW_str);
  cublasOperation_t opA = parse_op(opA_str);
  cublasLtOrder_t orderW = parse_order(orderW_str);
  cublasLtOrder_t orderA = parse_order(orderA_str);
  cublasLtOrder_t orderR = parse_order(orderR_str);

  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(outdtype);
  cublasComputeType_t comp_type = compute_type_from_dtype(dtype);
  cudaDataType scale_type = scale_type_from_dtype(dtype);

  // 计算矩阵维度
  // cuBLASLt GEMM: D = alpha * op(A) * op(B) + beta * C
  // 我们想要: R = W * A (保持 W 在左边)
  // 其中 W[N,K], A[K,M], R[N,M]
  bool isW_transposed = (opW == CUBLAS_OP_T);
  bool isA_transposed = (opA == CUBLAS_OP_T);
  bool isW_rowmajor = (orderW == CUBLASLT_ORDER_ROW);
  bool isA_rowmajor = (orderA == CUBLASLT_ORDER_ROW);
  bool isR_rowmajor = (orderR == CUBLASLT_ORDER_ROW);

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

  // 元素数量
  int64_t W_height = isW_rowmajor ? num_W_rows : num_W_cols;
  int64_t A_height = isA_rowmajor ? num_A_rows : num_A_cols;
  int64_t R_height = isR_rowmajor ? num_R_rows : num_R_cols;

  size_t W_elems = static_cast<size_t>(W_height) * ldw;
  size_t A_elems = static_cast<size_t>(A_height) * lda;
  size_t R_elems = static_cast<size_t>(R_height) * ldr;

  size_t W_size = W_elems * dtype_size(dtype);
  size_t A_size = A_elems * dtype_size(dtype);
  size_t R_size = R_elems * out_dtype_size(outdtype);

  // 分配设备内存
  void *dW = nullptr, *dA = nullptr, *dR = nullptr;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  CHECK_CUDA_ERR(cudaMalloc(&dW, W_size));
  CHECK_CUDA_ERR(cudaMalloc(&dA, A_size));
  CHECK_CUDA_ERR(cudaMalloc(&dR, R_size));

  // 随机初始化数据
  {
    std::vector<int8_t> hW(W_elems), hA(A_elems);
    for (size_t i = 0; i < W_elems; ++i) hW[i] = static_cast<int8_t>(rand() % 256 - 128);
    for (size_t i = 0; i < A_elems; ++i) hA[i] = static_cast<int8_t>(rand() % 256 - 128);
    CHECK_CUDA_ERR(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
  }
  CHECK_CUDA_ERR(cudaMemset(dR, 0, R_size));

  // 初始化 cuBLASLt
  cublasLtHandle_t handle;
  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 创建矩阵乘法描述符
  cublasLtMatmulDesc_t matmulDesc = nullptr;
  status = cublasLtMatmulDescCreate(&matmulDesc, comp_type, scale_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 设置转置操作
  status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 创建矩阵布局描述符
  cublasLtMatrixLayout_t layoutW = nullptr, layoutA = nullptr, layoutR = nullptr;

  // W 矩阵布局 (矩阵 A in cuBLASLt)
  status = cublasLtMatrixLayoutCreate(&layoutW, type_AB, num_W_rows, num_W_cols, ldw);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }
  status = cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderW, sizeof(orderW));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // A 矩阵布局 (矩阵 B in cuBLASLt)
  status = cublasLtMatrixLayoutCreate(&layoutA, type_AB, num_A_rows, num_A_cols, lda);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }
  status = cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // R 矩阵布局 (矩阵 C/D)
  status = cublasLtMatrixLayoutCreate(&layoutR, type_C, num_R_rows, num_R_cols, ldr);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }
  status = cublasLtMatrixLayoutSetAttribute(layoutR, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderR, sizeof(orderR));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutR);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 创建算法偏好
  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutR);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 设置 workspace 大小
  size_t workspace_size = 32 * 1024 * 1024;  // 32 MB
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutR);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // 获取可用算法
  const int max_algo_count = 128;
  cublasLtMatmulHeuristicResult_t heuristicResult[max_algo_count];
  int returnedAlgoCount = 0;

  status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      matmulDesc,
      layoutW,
      layoutA,
      layoutR,
      layoutR,
      preference,
      max_algo_count,
      heuristicResult,
      &returnedAlgoCount);

  if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutR);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR);
    return result;
  }

  // returnedAlgoCount 是启发式返回的算法数量（从1开始计数）
  result["alg_count"] = returnedAlgoCount;
  result["config_count"] = returnedAlgoCount;  // cuBLASLt 没有 split-k，所以 config_count = alg_count

  // 分配 workspace
  void *d_workspace = nullptr;
  if (workspace_size > 0) {
    CHECK_CUDA_ERR(cudaMalloc(&d_workspace, workspace_size));
  }

  // CUDA events for timing
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERR(cudaEventCreate(&stop_event));

  // 只记录最佳结果
  AlgResult best_result = {-1, 0.0f, 0.0f, 0.0f};
  int64_t best_workspace = 0;
  // alpha/beta 统一使用 float（因为 scale_type 已统一为 CUDA_R_32F 以支持 BF16 输出）
  float alpha = 1.0f, beta = 0.0f;
  const void *alpha_ptr = &alpha;
  const void *beta_ptr = &beta;

  // 遍历所有可用算法
  for (int alg_idx = 0; alg_idx < returnedAlgoCount; ++alg_idx) {
    if (heuristicResult[alg_idx].state != CUBLAS_STATUS_SUCCESS) {
      continue;
    }

    const cublasLtMatmulAlgo_t *algo = &heuristicResult[alg_idx].algo;
    size_t ws_size = heuristicResult[alg_idx].workspaceSize;
    float waves = heuristicResult[alg_idx].wavesCount;

    // 提取算法 ID（用于调试/显示）
    int algo_id = 0;
    cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID,
                                          &algo_id, sizeof(algo_id), nullptr);

    // Warmup
    bool warmup_success = true;
    for (int w = 0; w < warmup; ++w) {
      status = cublasLtMatmul(
          handle,
          matmulDesc,
          alpha_ptr,
          dW, layoutW,
          dA, layoutA,
          beta_ptr,
          dR, layoutR,
          dR, layoutR,
          algo,
          d_workspace,
          ws_size,
          stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        warmup_success = false;
        break;
      }
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

    if (!warmup_success) {
      continue;
    }

    // Benchmark
    CHECK_CUDA_ERR(cudaEventRecord(start_event, stream));
    for (int r = 0; r < repeat; ++r) {
      cublasLtMatmul(
          handle,
          matmulDesc,
          alpha_ptr,
          dW, layoutW,
          dA, layoutA,
          beta_ptr,
          dR, layoutR,
          dR, layoutR,
          algo,
          d_workspace,
          ws_size,
          stream);
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop_event, stream));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop_event));

    float total_ms = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&total_ms, start_event, stop_event));
    float avg_us = (total_ms * 1000.0f) / repeat;

    // 计算 TOPS
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tops = (flops / (avg_us * 1e-6)) / 1e12;

    // 更新最佳结果（按 tops 比较）
    if (best_result.alg_id < 0 || tops > best_result.tops) {
      best_result = {algo_id, avg_us, static_cast<float>(tops), waves};
      best_workspace = static_cast<int64_t>(ws_size);
    }
  }

  // 输出 best 结果
  result["supported"] = (best_result.alg_id >= 0);
  if (best_result.alg_id >= 0) {
    result["best_tops"] = best_result.tops;
    result["best_lat_us"] = best_result.lat_us;
    result["best_id"] = best_result.alg_id;
    result["best_ws"] = best_workspace;
    result["best_waves"] = best_result.waves_count;
  } else {
    result["best_tops"] = 0.0f;
    result["best_lat_us"] = 0.0f;
    result["best_id"] = -1;
    result["best_ws"] = 0;
    result["best_waves"] = 0.0f;
  }

  // 清理
  CHECK_CUDA_ERR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERR(cudaEventDestroy(stop_event));
  if (d_workspace) cudaFree(d_workspace);
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layoutW);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutR);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtDestroy(handle);
  cudaFree(dW);
  cudaFree(dA);
  cudaFree(dR);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_layout", &test_layout, "测试单个 layout 配置的 GEMM 性能 (cuBLASLt)",
        py::arg("N"), py::arg("K"), py::arg("M"),
        py::arg("opW"), py::arg("opA"),
        py::arg("orderW"), py::arg("orderA"), py::arg("orderR"),
        py::arg("dtype"),
        py::arg("outdtype") = "bf16",
        py::arg("warmup") = 10,
        py::arg("repeat") = 50);
}
