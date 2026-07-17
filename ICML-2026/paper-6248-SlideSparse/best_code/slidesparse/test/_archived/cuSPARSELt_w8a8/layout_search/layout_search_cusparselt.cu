// cuSPARSELt Layout 性能测试
// 提供 Python API，测试不同 layout 组合的 SpMM 性能
// 支持的数据类型：int8, fp8e4m3
//
// =====================================
// cuSPARSELt API 约束 (基于官方文档):
// =====================================
//
// 1. 稀疏矩阵 (Structured Matrix) 维度约束:
//    - rows, cols, ld 必须是以下的倍数:
//      32 if CUDA_R_8I, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_4F_E2M1
//      16 if CUDA_R_16F, CUDA_R_16BF
//      8  if CUDA_R_32F
//
// 2. 稠密矩阵 (Dense Matrix) 维度约束:
//    - rows, cols, ld 必须是以下的倍数:
//      16 if CUDA_R_8I, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_4F_E2M1
//      8  if CUDA_R_16F, CUDA_R_16BF
//      4  if CUDA_R_32F
//
// 3. op 和 order 的绑定约束 (针对 INT8/FP8):
//    - opA/opB = TN if orderA/orderB = Col/Col
//    - opA/opB = NT if orderA/orderB = Row/Row
//    - opA/opB = NN if orderA/orderB = Row/Col
//    - opA/opB = TT if orderA/orderB = Col/Row
//    (如果 B 是稀疏矩阵则相反)
//
// 4. 数据类型支持 (本实现):
//    - INT8: Input A/B=CUDA_R_8I, Output C/D=bf16/fp32
//    - FP8:  Input A/B=CUDA_R_8F_E4M3, Output=bf16/fp32
//
// 5. 内存对齐: alignment 必须是 16 的倍数

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
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

#define CHECK_CUSPARSE_ERR(expr)                                               \
  do {                                                                         \
    cusparseStatus_t _status = (expr);                                         \
    if (_status != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::ostringstream _oss;                                                 \
      _oss << "cuSPARSELt 调用失败: " << cusparseLtGetErrorString(_status)     \
           << " (code " << static_cast<int>(_status) << ")";                   \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// === 带超时的 planInit 包装函数 ===
// 在 cuSPARSELt 0.8.1 中，对不支持 Segment-K 的架构调用 split_k=-1 会导致
// planInit 无限期阻塞挂起。这个包装函数添加超时保护。
struct PlanInitResult {
  cusparseStatus_t status;
  bool timed_out;
};

static PlanInitResult planInitWithTimeout(
    cusparseLtHandle_t* handle,
    cusparseLtMatmulPlan_t* plan,
    cusparseLtMatmulDescriptor_t* matmul,
    cusparseLtMatmulAlgSelection_t* alg_sel,
    int timeout_seconds = 5) {
  
  // 使用 std::async 在另一个线程中运行 planInit
  auto future = std::async(std::launch::async, [&]() {
    return cusparseLtMatmulPlanInit(handle, plan, matmul, alg_sel);
  });
  
  // 等待指定超时时间
  auto wait_status = future.wait_for(std::chrono::seconds(timeout_seconds));
  
  if (wait_status == std::future_status::timeout) {
    // 超时！返回超时标志
    return PlanInitResult{CUSPARSE_STATUS_EXECUTION_FAILED, true};
  }
  
  // 正常完成
  return PlanInitResult{future.get(), false};
}

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

static cusparseComputeType compute_type_from_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUSPARSE_COMPUTE_32I;
  if (dtype == "fp8e4m3") return CUSPARSE_COMPUTE_32F;
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
static cusparseOperation_t parse_op(const std::string &op) {
  if (op == "T") return CUSPARSE_OPERATION_TRANSPOSE;
  if (op == "N") return CUSPARSE_OPERATION_NON_TRANSPOSE;
  throw std::invalid_argument("无效的操作: " + op + "。支持: T, N");
}

static cusparseOrder_t parse_order(const std::string &order) {
  if (order == "Col") return CUSPARSE_ORDER_COL;
  if (order == "Row") return CUSPARSE_ORDER_ROW;
  throw std::invalid_argument("无效的顺序: " + order + "。支持: Col, Row");
}

// ===== 算法结果 =====
struct AlgResult {
  int alg_id;
  int split_k;      // cuSPARSELt split-k 配置
  float lat_us;
  float tops;
  int64_t workspace;
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
    int repeat,
    bool test_segment_k  // 是否测试 Segment-K (split_k=-1)，由 Python 端根据架构决定
) {
  py::dict result;
  result["supported"] = false;
  result["max_alg_id"] = -1;
  result["top3"] = py::list();

  // 解析 layout 参数
  cusparseOperation_t opW = parse_op(opW_str);
  cusparseOperation_t opA = parse_op(opA_str);
  cusparseOrder_t orderW = parse_order(orderW_str);
  cusparseOrder_t orderA = parse_order(orderA_str);
  cusparseOrder_t orderR = parse_order(orderR_str);

  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(outdtype);
  cusparseComputeType comp_type = compute_type_from_dtype(dtype);
  unsigned alignment = 16;

  // 计算矩阵维度
  bool isW_transposed = (opW == CUSPARSE_OPERATION_TRANSPOSE);
  bool isA_transposed = (opA == CUSPARSE_OPERATION_TRANSPOSE);
  bool isW_rowmajor = (orderW == CUSPARSE_ORDER_ROW);
  bool isA_rowmajor = (orderA == CUSPARSE_ORDER_ROW);
  bool isR_rowmajor = (orderR == CUSPARSE_ORDER_ROW);

  // W[N,K]: 不转置时存储为[N,K]，转置时存储为[K,N]
  int64_t num_W_rows = isW_transposed ? K : N;
  int64_t num_W_cols = isW_transposed ? N : K;
  // A[M,K]: 转置时存储为[M,K]，不转置时存储为[K,M]
  int64_t num_A_rows = isA_transposed ? M : K;
  int64_t num_A_cols = isA_transposed ? K : M;
  // R[N,M]
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
  int *d_valid = nullptr;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  CHECK_CUDA_ERR(cudaMalloc(&dW, W_size));
  CHECK_CUDA_ERR(cudaMalloc(&dA, A_size));
  CHECK_CUDA_ERR(cudaMalloc(&dR, R_size));
  CHECK_CUDA_ERR(cudaMalloc(&d_valid, sizeof(int)));

  // 随机初始化数据
  {
    std::vector<int8_t> hW(W_elems), hA(A_elems);
    for (size_t i = 0; i < W_elems; ++i) hW[i] = static_cast<int8_t>(rand() % 256 - 128);
    for (size_t i = 0; i < A_elems; ++i) hA[i] = static_cast<int8_t>(rand() % 256 - 128);
    CHECK_CUDA_ERR(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
  }
  CHECK_CUDA_ERR(cudaMemset(dR, 0, R_size));

  // 初始化 cuSPARSELt
  cusparseLtHandle_t handle;
  cusparseStatus_t status = cusparseLtInit(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  cusparseLtMatDescriptor_t matW, matA, matR;
  cusparseLtMatmulDescriptor_t matmul;

  // 稀疏矩阵 W 描述符
  status = cusparseLtStructuredDescriptorInit(
      &handle, &matW, num_W_rows, num_W_cols, ldw, alignment,
      type_AB, orderW, CUSPARSELT_SPARSITY_50_PERCENT);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 稠密矩阵 A 描述符
  status = cusparseLtDenseDescriptorInit(
      &handle, &matA, num_A_rows, num_A_cols, lda, alignment,
      type_AB, orderA);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 输出矩阵 R 描述符
  status = cusparseLtDenseDescriptorInit(
      &handle, &matR, num_R_rows, num_R_cols, ldr, alignment,
      type_C, orderR);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 矩阵乘法描述符
  status = cusparseLtMatmulDescriptorInit(
      &handle, &matmul, opW, opA, &matW, &matA, &matR, &matR, comp_type);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 设置稀疏矩阵指针
  status = cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW));
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 剪枝
  status = cusparseLtSpMMAPrune(&handle, &matmul, dW, dW, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 检查剪枝
  status = cusparseLtSpMMAPruneCheck(&handle, &matmul, dW, d_valid, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  int is_valid = 0;
  CHECK_CUDA_ERR(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

  if (is_valid != 0) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 获取最大算法 ID
  int max_alg_id = -1;
  {
    cusparseLtMatmulAlgSelection_t alg_sel_tmp;
    status = cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel_tmp, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status == CUSPARSE_STATUS_SUCCESS) {
      cusparseLtMatmulAlgGetAttribute(
          &handle, &alg_sel_tmp, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
          &max_alg_id, sizeof(max_alg_id));
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    }
  }

  if (max_alg_id < 0) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // max_alg_id 是从 0 开始的最大算法 ID 且不允许取，所以 alg_count = max_alg_id
  result["alg_count"] = max_alg_id;

  // CUDA events for timing
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERR(cudaEventCreate(&stop_event));

  // 只记录最佳结果
  AlgResult best_result = {-1, 1, 0.0f, 0.0f, 0};
  int config_count = 0;  // 记录测试的配置总数
  float alpha = 1.0f, beta = 0.0f;

  // === 双层网格搜索：外层遍历 alg_id，内层自适应调整 split_k_val ===
  // split_k_val 候选列表：
  //   1: 不切分 (Baseline)
  //   2, 4, 8, 16, 32, 64: 传统 Split-K（自适应倍增，根据性能决定是否继续）
  //   -1: Segment-K (SM 9.0/10.x 架构特殊优化，由 Python 端控制是否测试)
  // 测试顺序：先测 k=1 得到 baseline，然后倍增 k，
  //   如果倍增失败则停止倍增，最后测试 k=-1
  
  // 注意：max_alg_id 是算法 ID 上界，有效范围为 [0, max_alg_id)，即 max_alg_id 不可取
  for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
    // 用于跟踪当前 alg_id 下的最优 split_k 的延时
    float best_lat_us_for_doubling = -1.0f;
    
    // === 阶段 1: 先测试 k=1 作为 baseline ===
    // === 阶段 2: 倍增测试 k=2,4,8,...，根据性能决定是否继续 ===
    // === 阶段 3: 最后测试 k=-1 (Segment-K)，仅当 test_segment_k=true ===
    
    // 构建 split_k 候选列表
    std::vector<int> split_k_candidates;
    split_k_candidates.push_back(1);  // 总是先测试不切分
    for (int sk = 2; sk <= K; sk *= 2) {
      split_k_candidates.push_back(sk);
    }
    if (test_segment_k) {
      split_k_candidates.push_back(-1);  // 仅当支持时才测试 Segment-K
    }

    bool stop_doubling = false;  // 用于控制倍增序列的终止
    
    for (int split_k_val : split_k_candidates) {
      // === 自适应倍增策略 ===
      // 如果已停止倍增且当前是倍增序列中的值 (>1)，则跳过（但 -1 除外）
      if (stop_doubling && split_k_val > 1) {
        continue;
      }

      cusparseLtMatmulAlgSelection_t alg_sel;
      status = cusparseLtMatmulAlgSelectionInit(
          &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
      if (status != CUSPARSE_STATUS_SUCCESS) {
        break;  // 算法选择初始化失败，跳出 split_k 循环
      }

      status = cusparseLtMatmulAlgSetAttribute(
          &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id));
      if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        continue;
      }

      // === 设置 Split-K 属性 ===
      // CUSPARSELT_MATMUL_SPLIT_K:
      //   1: 不切分 (默认)
      //   >1: 传统 Split-K
      //   -1: Segment-K (SM 9.0/10.x 特有优化，其他架构不支持)
      cusparseStatus_t split_k_status = cusparseLtMatmulAlgSetAttribute(
          &handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &split_k_val, sizeof(split_k_val));
      if (split_k_status != CUSPARSE_STATUS_SUCCESS) {
        // Split-K 设置失败
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        if (split_k_val > 1) {
          stop_doubling = true;  // 倍增失败，停止倍增
        }
        continue;
      }

      cusparseLtMatmulPlan_t plan;
      // 使用带超时的 planInit，防止在不支持的架构上无限期挂起
      auto plan_result = planInitWithTimeout(&handle, &plan, &matmul, &alg_sel, 5);
      if (plan_result.timed_out) {
        // planInit 超时（可能是 cuSPARSELt 0.8.1 在不支持 Segment-K 的架构上挂起）
        std::cerr << "[WARNING] planInit 超时 (alg_id=" << alg_id 
                  << ", split_k=" << split_k_val << ")" << std::endl;
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        continue;
      }
      if (plan_result.status != CUSPARSE_STATUS_SUCCESS) {
        // PlanInit 失败（算法+Split-K 组合不兼容）
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        if (split_k_val > 1) {
          stop_doubling = true;  // 倍增失败，停止倍增
        }
        continue;
      }

    // 压缩稀疏矩阵
    size_t compressed_size = 0, compressed_buffer_size = 0;
    CHECK_CUSPARSE_ERR(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buffer_size));

    void *dW_compressed = nullptr, *dW_compressedBuffer = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&dW_compressed, compressed_size));
    if (compressed_buffer_size > 0) {
      CHECK_CUDA_ERR(cudaMalloc(&dW_compressedBuffer, compressed_buffer_size));
    }

    status = cusparseLtSpMMACompress(&handle, &plan, dW, dW_compressed, dW_compressedBuffer, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cudaFree(dW_compressed);
      if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
      cusparseLtMatmulPlanDestroy(&plan);
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    // Workspace
    size_t workspace_size = 0;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    void *d_workspace = nullptr;
    if (workspace_size > 0) {
      CHECK_CUDA_ERR(cudaMalloc(&d_workspace, workspace_size));
    }

    // Warmup
    for (int w = 0; w < warmup; ++w) {
      status = cusparseLtMatmul(&handle, &plan, &alpha, dW_compressed, dA, &beta, dR, dR, d_workspace, &stream, 1);
      if (status != CUSPARSE_STATUS_SUCCESS) break;
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

    if (status != CUSPARSE_STATUS_SUCCESS) {
      if (d_workspace) cudaFree(d_workspace);
      cudaFree(dW_compressed);
      if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
      cusparseLtMatmulPlanDestroy(&plan);
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    // Benchmark
    CHECK_CUDA_ERR(cudaEventRecord(start_event, stream));
    for (int r = 0; r < repeat; ++r) {
      cusparseLtMatmul(&handle, &plan, &alpha, dW_compressed, dA, &beta, dR, dR, d_workspace, &stream, 1);
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop_event, stream));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop_event));

    float total_ms = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&total_ms, start_event, stop_event));
    float avg_us = (total_ms * 1000.0f) / repeat;

    // 计算 TOPS
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tops = (flops / (avg_us * 1e-6)) / 1e12;

    // 计数有效配置
    ++config_count;

    // 更新最佳结果（按 tops 比较）
    if (best_result.alg_id < 0 || tops > best_result.tops) {
      best_result = {alg_id, split_k_val, avg_us, static_cast<float>(tops), 
                     static_cast<int64_t>(workspace_size)};
    }

    // === 自适应倍增策略：根据性能决定是否继续倍增 ===
    // 对于倍增序列 (split_k_val >= 1)，更新 best 并决定是否继续
    if (split_k_val >= 1) {
      if (best_lat_us_for_doubling < 0 || avg_us < best_lat_us_for_doubling) {
        // 新延时更低，更新 best
        best_lat_us_for_doubling = avg_us;
      } else if (avg_us * 1.10f > best_lat_us_for_doubling && split_k_val > 1) {
        // 新延时 * 1.10 > 旧延时（10% 容限），停止倍增
        stop_doubling = true;
      }
    }
    // 注意：k=-1 (Segment-K) 不参与倍增策略，总是单独测试一次

    // 清理
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(dW_compressed);
    if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    }  // end split_k_val loop
  }  // end alg_id loop

  // 输出结果
  result["config_count"] = config_count;
  result["supported"] = (best_result.alg_id >= 0);
  if (best_result.alg_id >= 0) {
    result["best_tops"] = best_result.tops;
    result["best_lat_us"] = best_result.lat_us;
    result["best_id"] = best_result.alg_id;
    result["best_ws"] = best_result.workspace;
    result["best_split_k"] = best_result.split_k;
  } else {
    result["best_tops"] = 0.0f;
    result["best_lat_us"] = 0.0f;
    result["best_id"] = -1;
    result["best_ws"] = 0;
    result["best_split_k"] = 1;
  }

  // 清理
  CHECK_CUDA_ERR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERR(cudaEventDestroy(stop_event));
  cusparseLtMatDescriptorDestroy(&matW);
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matR);
  cusparseLtDestroy(&handle);
  cudaFree(dW);
  cudaFree(dA);
  cudaFree(dR);
  cudaFree(d_valid);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_layout", &test_layout, "测试单个 layout 配置的 SpMM 性能",
        py::arg("N"), py::arg("K"), py::arg("M"),
        py::arg("opW"), py::arg("opA"),
        py::arg("orderW"), py::arg("orderA"), py::arg("orderR"),
        py::arg("dtype"),
        py::arg("outdtype") = "bf16",
        py::arg("warmup") = 10,
        py::arg("repeat") = 50,
        py::arg("test_segment_k") = false);
}
