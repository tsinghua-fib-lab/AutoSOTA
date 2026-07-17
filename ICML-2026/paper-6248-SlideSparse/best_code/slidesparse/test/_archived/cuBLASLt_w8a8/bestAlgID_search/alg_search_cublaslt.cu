// cuBLASLt 算法离线搜索
// 固定的layout: 权重W在左，T/N + C/C GEMM，输出矩阵order固定为 Column 主序
// C[N,M]_col = W[N,K]^T_col * A[K,M]_col
// 支持的数据类型：int8, fp8e4m3
// 输出类型：bf16 (CUDA_R_16BF) 或 fp32 (CUDA_R_32F)

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
#define CHECK_CUDA_ERR(expr)                                                                        \
  do {                                                                                              \
    cudaError_t _status = (expr);                                                                  \
    if (_status != cudaSuccess) {                                                                  \
      std::ostringstream _oss;                                                                     \
      _oss << "CUDA 调用失败: " << cudaGetErrorString(_status) << " (code " << _status << ")";  \
      throw std::runtime_error(_oss.str());                                                        \
    }                                                                                              \
  } while (0)

#define CHECK_CUBLAS_ERR(expr)                                                                      \
  do {                                                                                              \
    cublasStatus_t _status = (expr);                                                               \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                                        \
      std::ostringstream _oss;                                                                     \
      _oss << "cuBLASLt 调用失败: " << cublasLtGetStatusString(_status)                           \
           << " (code " << static_cast<int>(_status) << ")";                                      \
      throw std::runtime_error(_oss.str());                                                        \
    }                                                                                              \
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

// ===== 简单量化：将 BF16/FP16 张量量化到 INT8 =====
static std::pair<torch::Tensor, double> quantize_int8(torch::Tensor x) {
  auto abs_max = x.abs().max().item<double>();
  double scale = abs_max > 0 ? 127.0 / abs_max : 1.0;
  auto q = (x * scale).round().clamp(-128, 127).to(torch::kChar);
  return {q, scale};
}

// ===== FP8 转换 =====
// FP8E4M3: 4位指数, 3位尾数, 动态范围小但精度高, 适合权重和激活
static torch::Tensor to_fp8_e4m3(torch::Tensor x) {
  if (!x.is_cuda()) throw std::runtime_error("FP8 转换需要 CUDA 张量");
  try {
    return x.to(torch::kFloat8_e4m3fn);
  } catch (...) {
    throw std::runtime_error("当前 PyTorch 版本不支持 FP8E4M3 类型，请升级到 PyTorch 2.1+");
  }
}

// ===== 搜索结果记录 =====
// 与 cuSPARSELt 对齐：存储算法配置
// 关键区别：cuBLASLt 需要完整的 64 字节不透明结构体（而非简单的 int alg_id）
struct AlgRecord {
  int alg_id{-1};             // 算法 ID（用于调试/显示）
  float lat_us{0.f};
  float tops{0.f};
  int64_t workspace{0};
  float waves_count{0.f};     // cuBLASLt 特有：GPU 利用率指标 (wavesCount)
  bool valid{false};
  float max_abs_err{0.f};
  
  // cuBLASLt 特有：64 字节不透明结构体（运行时直接加载使用）
  // 验证大小：static_assert(sizeof(cublasLtMatmulAlgo_t) == 64)
  uint8_t algo_data[64];
  
  AlgRecord() {
    memset(algo_data, 0, sizeof(algo_data));
  }
};

// ===== search_topk =====
// 固定布局: T/N + Col/Col + Col (权重W在左)
// W[N,K]^T_col * A[K,M]_col = C[N,M]_col
py::dict search_topk(torch::Tensor W_bf16, torch::Tensor A_bf16,
                     const std::vector<int64_t> &M_list,
                     const std::string &layout, const std::string &dtype,
                     const std::string &outdtype,
                     int warmup, int repeat, bool verify,
                     const std::vector<int64_t> &blacklist_ids,
                     int topk) {
  if (!W_bf16.is_cuda() || !A_bf16.is_cuda()) {
    throw std::invalid_argument("W 与 A 必须在 CUDA 上");
  }
  if (W_bf16.dim() != 2 || A_bf16.dim() != 2) {
    throw std::invalid_argument("输入张量必须是二维 [N,K] / [M,K]");
  }
  if (topk <= 0) topk = 3;

  auto W_fp = W_bf16.contiguous();
  auto A_fp = A_bf16.contiguous();
  int64_t N = W_fp.size(0);
  int64_t K = W_fp.size(1);
  
  // === 维度对齐检查 ===
  // cuBLASLt 要求：维度需满足对齐要求（INT8 需 4 对齐，FP8 需 16 对齐）
  auto check_alignment = [](int64_t val, int64_t align, const char* name) {
    if (val % align != 0) {
      std::ostringstream oss;
      oss << "维度不满足对齐要求: " << name << "=" << val << " 不是 " << align << " 的倍数";
      throw std::invalid_argument(oss.str());
    }
  };
  int64_t align_req = (dtype == "fp8e4m3") ? 16 : 4;
  check_alignment(N, align_req, "N");
  check_alignment(K, align_req, "K");
  for (auto m : M_list) {
    check_alignment(m, align_req, "M");
  }

  // 量化/转换
  torch::Tensor W_q;
  torch::Tensor A_q;
  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(outdtype);
  cublasComputeType_t comp_type = compute_type_from_dtype(dtype);
  cudaDataType scale_type = scale_type_from_dtype(dtype);

  if (dtype == "int8") {
    auto qW = quantize_int8(W_fp);
    auto qA = quantize_int8(A_fp);
    W_q = qW.first;
    A_q = qA.first;
  } else if (dtype == "fp8e4m3") {
    W_q = to_fp8_e4m3(W_fp);
    A_q = to_fp8_e4m3(A_fp);
  } else {
    throw std::invalid_argument("不支持的数据类型: " + dtype + "。支持: int8, fp8e4m3");
  }

  // === cuBLASLt 初始化 ===
  cublasLtHandle_t handle;
  CHECK_CUBLAS_ERR(cublasLtCreate(&handle));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // 固定布局: T/N + Col/Col + Col
  cublasOperation_t opW = CUBLAS_OP_T;
  cublasOperation_t opA = CUBLAS_OP_N;
  cublasLtOrder_t orderW = CUBLASLT_ORDER_COL;
  cublasLtOrder_t orderA = CUBLASLT_ORDER_COL;
  cublasLtOrder_t orderC = CUBLASLT_ORDER_COL;

  int64_t num_W_rows = K;  // 存储的行数
  int64_t num_W_cols = N;  // 存储的列数
  int64_t ldw = K;         // Col major: leading dim = 存储的行数

  // 为结果分配存储
  // 与 cuSPARSELt 对齐：存储 topk 的完整算法配置
  int64_t numM = static_cast<int64_t>(M_list.size());
  torch::Tensor topk_alg_id = torch::full({numM, topk}, -1, torch::dtype(torch::kInt32));
  torch::Tensor topk_lat = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_tops = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_workspace = torch::zeros({numM, topk}, torch::dtype(torch::kInt64));
  torch::Tensor topk_waves_count = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));  // cuBLASLt 特有
  torch::Tensor valid_mask = torch::zeros({numM, topk}, torch::dtype(torch::kUInt8));
  torch::Tensor num_valid = torch::zeros({numM}, torch::dtype(torch::kInt32));
  torch::Tensor verify_err = torch::zeros({numM}, torch::dtype(torch::kFloat32));
  std::vector<int> verify_failed_ids;
  
  // cuBLASLt 特有：存储 64 字节不透明结构体
  // 验证大小
  static_assert(sizeof(cublasLtMatmulAlgo_t) == 64, "cublasLtMatmulAlgo_t size mismatch");
  torch::Tensor topk_algo_data = torch::zeros({numM, topk, 64}, torch::dtype(torch::kUInt8));

  std::vector<AlgRecord> records;
  
  // === Workspace 配置 ===
  // 放宽限制：预分配 512MB workspace，确保不遗漏高性能算法
  size_t current_workspace_size = 512 * 1024 * 1024;  // 512 MB
  void *shared_workspace = nullptr;
  CHECK_CUDA_ERR(cudaMalloc(&shared_workspace, current_workspace_size));

  // 最大算法数（cuBLASLt 启发式搜索返回的最大数量）
  int alg_count = 0;       // 启发式搜索返回的最大算法数
  int config_count = 0;    // 实际测试的配置数（cuBLASLt 无 split-k，所以 config_count = 实测算法数）

  for (int64_t m_index = 0; m_index < numM; ++m_index) {
    int64_t M = M_list[m_index];
    
    // A 矩阵：PyTorch [M,K] Row Major = Col Major [K,M]
    int64_t num_A_rows = K;
    int64_t num_A_cols = M;
    int64_t lda = K;  // Col major: leading dim = 存储的行数

    // C 存储维度: [N,M] Col Major
    int64_t num_C_rows = N;
    int64_t num_C_cols = M;
    int64_t ldc = N;  // Col major: leading dim = N

    // B 切片（与 cuSPARSELt 命名对齐：这里取 A_q 的前 M 行）
    auto A_slice = A_q.narrow(0, 0, M).contiguous();

    // 输出 C 与 D（就地）
    // 关键：cuBLASLt 以 Column Major 存储 C 矩阵
    // 逻辑形状 [N, M]，Column Major 存储意味着 leading dim = N
    // PyTorch 默认 Row Major，为了兼容，我们创建 [M, N] 的 tensor
    // 这样 PyTorch 的 Row Major [M, N] 等价于 Column Major [N, M]
    torch::Tensor C_out;
    // 根据 outdtype 创建输出 tensor
    // Column Major [N, M] 等价于 Row Major [M, N] 的转置
    if (outdtype == "fp32") {
      C_out = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(W_q.device()));
    } else {
      C_out = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(W_q.device()));
    }

    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    CHECK_CUBLAS_ERR(cublasLtMatmulDescCreate(&matmulDesc, comp_type, scale_type));
    CHECK_CUBLAS_ERR(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
    CHECK_CUBLAS_ERR(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));

    // 创建矩阵布局描述符
    cublasLtMatrixLayout_t layoutW = nullptr, layoutA = nullptr, layoutC = nullptr;

    // W 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutW, type_AB, num_W_rows, num_W_cols, ldw));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderW, sizeof(orderW)));

    // A 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutA, type_AB, num_A_rows, num_A_cols, lda));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));

    // C 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutC, type_C, num_C_rows, num_C_cols, ldc));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderC, sizeof(orderC)));

    // 创建算法偏好
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLAS_ERR(cublasLtMatmulPreferenceCreate(&preference));
    
    // 设置最大 workspace 为 512MB
    CHECK_CUBLAS_ERR(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &current_workspace_size, sizeof(current_workspace_size)));
    
    // 允许所有 Reduction 方案（不进行位掩码限制）
    // CUBLASLT_REDUCTION_SCHEME_MASK 全 1 表示允许所有方案
    uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
    CHECK_CUBLAS_ERR(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_scheme_mask, sizeof(reduction_scheme_mask)));

    // 获取可用算法（启发式搜索）
    // 与 cuSPARSELt 不同：cuBLASLt 使用启发式搜索返回候选算法，cuSPARSELt 使用 ID 遍历
    const int max_algo_count = 128;  // 请求最多 128 个算法
    cublasLtMatmulHeuristicResult_t heuristicResult[max_algo_count];
    int returnedAlgoCount = 0;

    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutW,
        layoutA,
        layoutC,
        layoutC,
        preference,
        max_algo_count,
        heuristicResult,
        &returnedAlgoCount);

    if (heur_status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
      // 清理资源并跳过此 M
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layoutW);
      cublasLtMatrixLayoutDestroy(layoutA);
      cublasLtMatrixLayoutDestroy(layoutC);
      cublasLtMatmulDescDestroy(matmulDesc);
      continue;
    }

    if (returnedAlgoCount > alg_count) {
      alg_count = returnedAlgoCount;
    }

    // alpha/beta 统一使用 float（因为 scale_type 已统一为 CUDA_R_32F 以支持 BF16 输出）
    float alpha_f = 1.0f, beta_f = 0.0f;
    const void *alpha_ptr = &alpha_f;
    const void *beta_ptr = &beta_f;

    // 遍历所有返回的算法（与 cuSPARSELt 的 alg_id 遍历对齐）
    for (int alg_idx = 0; alg_idx < returnedAlgoCount; ++alg_idx) {
      if (heuristicResult[alg_idx].state != CUBLAS_STATUS_SUCCESS) {
        continue;
      }

      // 检查黑名单
      if (std::find(blacklist_ids.begin(), blacklist_ids.end(), alg_idx) != blacklist_ids.end()) {
        continue;
      }

      const cublasLtMatmulAlgo_t *algo = &heuristicResult[alg_idx].algo;
      size_t workspace_size = heuristicResult[alg_idx].workspaceSize;

      // === Workspace 回退机制 ===
      // 与 cuSPARSELt 对齐：如果当前共享 workspace 不够大，扩展它
      if (workspace_size > current_workspace_size) {
        if (shared_workspace != nullptr) {
          cudaFree(shared_workspace);
          shared_workspace = nullptr;
        }
        // 尝试分配更大的 workspace
        cudaError_t alloc_st = cudaMalloc(&shared_workspace, workspace_size);
        if (alloc_st != cudaSuccess) {
          // 分配失败，跳过此算法
          continue;
        }
        current_workspace_size = workspace_size;
      }

      // 使用共享 workspace（大小已保证足够）
      void *workspace = (workspace_size > 0) ? shared_workspace : nullptr;

      bool success = true;

      // 预热（显式传入 stream，保证一致性）
      if (warmup > 0) {
        for (int i = 0; i < warmup; ++i) {
          cublasStatus_t st = cublasLtMatmul(
              handle,
              matmulDesc,
              alpha_ptr,
              W_q.data_ptr(), layoutW,
              A_slice.data_ptr(), layoutA,
              beta_ptr,
              C_out.data_ptr(), layoutC,
              C_out.data_ptr(), layoutC,
              algo,
              workspace,
              workspace_size,
              stream);
          if (st != CUBLAS_STATUS_SUCCESS) {
            success = false;
            break;
          }
        }
        // 确保预热完成
        CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
      }

      // 正式计时：包住整个 repeat loop 一次，避免 event 开销污染小 M 测量
      cudaEvent_t start = nullptr, stop = nullptr;
      float total_ms = 0.0f;
      if (success) {
        CHECK_CUDA_ERR(cudaEventCreate(&start));
        CHECK_CUDA_ERR(cudaEventCreate(&stop));

        // 开始计时（只记录一次 start）
        CHECK_CUDA_ERR(cudaEventRecord(start, stream));
        
        for (int r = 0; r < repeat; ++r) {
          cublasStatus_t st = cublasLtMatmul(
              handle,
              matmulDesc,
              alpha_ptr,
              W_q.data_ptr(), layoutW,
              A_slice.data_ptr(), layoutA,
              beta_ptr,
              C_out.data_ptr(), layoutC,
              C_out.data_ptr(), layoutC,
              algo,
              workspace,
              workspace_size,
              stream);
          if (st != CUBLAS_STATUS_SUCCESS) {
            success = false;
            break;
          }
        }
        
        // 结束计时（只记录一次 stop，只同步一次）
        CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&total_ms, start, stop));
        CHECK_CUDA_ERR(cudaEventDestroy(start));
        CHECK_CUDA_ERR(cudaEventDestroy(stop));
      }

      if (success) {
        AlgRecord rec;
        // 提取算法 ID（用于调试/显示）
        int algo_id = 0;
        cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID,
                                              &algo_id, sizeof(algo_id), nullptr);
        rec.alg_id = algo_id;
        rec.lat_us = (total_ms * 1000.0f) / static_cast<float>(repeat);
        double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                     static_cast<double>(K);
        double tops = ops / (rec.lat_us / 1e6) / 1e12;
        rec.tops = static_cast<float>(tops);
        rec.workspace = static_cast<int64_t>(workspace_size);
        rec.waves_count = heuristicResult[alg_idx].wavesCount;  // GPU 利用率指标
        rec.valid = true;
        
        // 保存完整的 64 字节算法数据（cuBLASLt 特有）
        memcpy(rec.algo_data, algo, sizeof(cublasLtMatmulAlgo_t));

        // 校验（如需）
        if (verify) {
          // 使用量化后的数据做 FP32 参考计算
          // 这样才能和 cuBLASLt 的 INT8 GEMM 结果对比
          auto A_slice_v = A_q.narrow(0, 0, M);
          auto A_fp32 = A_slice_v.to(torch::kFloat32);
          auto W_fp32 = W_q.to(torch::kFloat32);
          auto ref = torch::matmul(W_fp32, A_fp32.transpose(0, 1));  // [N, M], Row Major
          
          // 将输出转为 FP32 比较
          // C_out 的创建已经考虑了布局：
          //   - Column Major 时创建为 [M, N]，转置后得到 [N, M]
          torch::Tensor out_fp32 = C_out.to(torch::kFloat32).t().contiguous();
          
          // 计算相对误差（相对于参考值的绝对值）
          auto ref_abs = ref.abs().clamp_min(1.0f);  // 避免除以0
          auto rel_diff = ((out_fp32 - ref) / ref_abs).abs();
          float max_rel_err = rel_diff.max().item<float>();
          rec.max_abs_err = max_rel_err;  // 存储的是相对误差
          
          // 相对误差容限：5%
          constexpr float tol = 0.05f;
          constexpr float critical_tol = 1.00f;  // 超过 100% 认为计算有严重问题
          
          if (max_rel_err > critical_tol || std::isnan(max_rel_err)) {
            // 严重错误：输出 warning，跳过当前 M 剩余算法
            std::cerr << "[WARNING] M=" << M << " alg_id=" << algo_id 
                      << " 相对误差=" << (max_rel_err * 100.0f) << "% > 100%，跳过剩余算法" << std::endl;
            rec.valid = false;
            verify_failed_ids.push_back(algo_id);
            break;  // 跳出 alg_idx 循环，进入下一个 M
          } else if (max_rel_err > tol) {
            // 超过容限但未达到严重级别：记录但不跳过
            std::cout << "[INFO] M=" << M << " alg_id=" << algo_id 
                      << " 相对误差=" << (max_rel_err * 100.0f) << "% > 5%" << std::endl;
            rec.valid = true;
            verify_failed_ids.push_back(algo_id);
          }
        }
        
        if (rec.valid) {
          records.push_back(rec);
          ++config_count;  // 统计实际测试的有效配置数
        }
      }

      // 注意：不再释放 workspace，因为使用共享 workspace
    }

    // 当前 M 取出记录并排序
    std::vector<AlgRecord> filtered;
    for (auto &r : records) {
      if (r.valid) filtered.push_back(r);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const AlgRecord &a, const AlgRecord &b) {
                return a.lat_us < b.lat_us;
              });

    int fill = std::min(static_cast<int>(filtered.size()), topk);
    for (int i = 0; i < fill; ++i) {
      topk_alg_id.index_put_({m_index, i}, filtered[i].alg_id);
      topk_lat.index_put_({m_index, i}, filtered[i].lat_us);
      topk_tops.index_put_({m_index, i}, filtered[i].tops);
      topk_workspace.index_put_({m_index, i}, filtered[i].workspace);
      topk_waves_count.index_put_({m_index, i}, filtered[i].waves_count);
      valid_mask.index_put_({m_index, i}, static_cast<uint8_t>(1));
      
      // 保存 64 字节算法数据（cuBLASLt 特有）
      auto algo_data_accessor = topk_algo_data.accessor<uint8_t, 3>();
      for (int b = 0; b < 64; ++b) {
        algo_data_accessor[m_index][i][b] = filtered[i].algo_data[b];
      }
      
      if (verify && i == 0) {
        verify_err.index_put_({m_index}, filtered[i].max_abs_err);
      }
    }
    num_valid.index_put_({m_index}, static_cast<int>(filtered.size()));

    // 清理当前 M 的资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    // 注意：cublasLtMatmulDesc_t 需要显式销毁
    records.clear();
  }

  // === 释放共享 workspace ===
  if (shared_workspace != nullptr) {
    cudaFree(shared_workspace);
    shared_workspace = nullptr;
  }

  // 清理
  cublasLtDestroy(handle);

  // 构造返回
  // 与 cuSPARSELt 对齐的输出格式
  py::dict out;
  out["M_list"] = torch::tensor(M_list, torch::dtype(torch::kInt32));
  out["NK"] = torch::tensor({static_cast<int32_t>(N), static_cast<int32_t>(K)},
                             torch::dtype(torch::kInt32));
  out["topk_alg_id"] = topk_alg_id;
  out["topk_lat_us"] = topk_lat;
  out["topk_tops"] = topk_tops;
  out["topk_workspace"] = topk_workspace;
  out["topk_waves_count"] = topk_waves_count;
  out["valid_mask"] = valid_mask;
  out["alg_count"] = alg_count;          // 启发式搜索返回的最大算法数
  out["config_count"] = config_count;    // 实际测试的有效配置数
  out["num_valid_algs_per_M"] = num_valid;                 // 每个 M 的有效算法数
  
  // cuBLASLt 特有：64 字节不透明结构体
  out["topk_algo_data"] = topk_algo_data;
  
  if (verify) {
    out["verify_max_abs_err"] = verify_err;
    out["verify_failed_algs"] = torch::tensor(verify_failed_ids,
                                              torch::dtype(torch::kInt32));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("search_topk", &search_topk, "枚举算法并返回 topk (cuBLASLt)", 
        py::arg("W_bf16"),
        py::arg("A_bf16"), 
        py::arg("M_list"), 
        py::arg("layout"),
        py::arg("dtype") = "int8",
        py::arg("outdtype") = "bf16",
        py::arg("warmup") = 25,
        py::arg("repeat") = 100, 
        py::arg("verify") = false,
        py::arg("blacklist_ids") = std::vector<int64_t>{},
        py::arg("topk") = 3);
}
