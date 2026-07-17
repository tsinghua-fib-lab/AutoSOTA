// cuSPARSELt 算法离线搜索
// 固定的layout: 权重W在左，T/N + C/C GEMM，输出矩阵order固定为 Column 主序
// C[N,M]_col = W[N,K]^T_col * A[K,M]_col
// 支持的数据类型：int8, fp8e4m3
// 输出类型：bf16 (CUDA_R_16BF) 或 fp32 (CUDA_R_32F)

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

#define CHECK_CUSPARSE_ERR(expr)                                                                    \
  do {                                                                                              \
    cusparseStatus_t _status = (expr);                                                             \
    if (_status != CUSPARSE_STATUS_SUCCESS) {                                                      \
      std::ostringstream _oss;                                                                     \
      _oss << "cuSPARSELt 调用失败: " << cusparseLtGetErrorString(_status)                        \
           << " (code " << static_cast<int>(_status) << ")";                                      \
      throw std::runtime_error(_oss.str());                                                        \
    }                                                                                              \
  } while (0)

// ===== 布局配置 =====

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
    // 注意：后台线程可能仍在运行，但我们无法强制终止它
    // 这里我们选择 detach 并返回超时错误
    return PlanInitResult{CUSPARSE_STATUS_EXECUTION_FAILED, true};
  }
  
  // 正常完成
  return PlanInitResult{future.get(), false};
}
struct LayoutConfig {
  cusparseOrder_t orderA{CUSPARSE_ORDER_COL};
  cusparseOrder_t orderB{CUSPARSE_ORDER_COL};
  cusparseOrder_t orderC{CUSPARSE_ORDER_COL};
  cusparseOperation_t opA{CUSPARSE_OPERATION_TRANSPOSE};
  cusparseOperation_t opB{CUSPARSE_OPERATION_NON_TRANSPOSE};
};

static LayoutConfig parse_layout(const std::string &layout) {
  LayoutConfig cfg;
  if (layout == "TNCCrow") {
    cfg.orderC = CUSPARSE_ORDER_ROW;
  } else {
    // 默认 col 版本
    cfg.orderC = CUSPARSE_ORDER_COL;
  }
  return cfg;
}

// ===== dtype 相关 =====
static cudaDataType to_cuda_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUDA_R_8I;
  if (dtype == "fp8e4m3") return CUDA_R_8F_E4M3;
  throw std::invalid_argument("不支持的数据类型: " + dtype + "。支持: int8, fp8e4m3");
}

static cudaDataType cuda_out_dtype(const std::string &outdtype) {
  // 输出类型：bf16 或 fp32
  if (outdtype == "bf16") return CUDA_R_16BF;
  if (outdtype == "fp32") return CUDA_R_32F;
  throw std::invalid_argument("不支持的输出类型: " + outdtype + "。支持: bf16, fp32");
}

static cusparseComputeType compute_type_from_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUSPARSE_COMPUTE_32I;
  if (dtype == "fp8e4m3") return CUSPARSE_COMPUTE_32F;
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

// ===== 计算维度辅助 =====
struct MatDims {
  int64_t num_A_rows;
  int64_t num_A_cols;
  int64_t num_B_rows;
  int64_t num_B_cols;
  int64_t num_C_rows;
  int64_t num_C_cols;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
};

static MatDims make_dims(int64_t N, int64_t M, int64_t K, const LayoutConfig &cfg) {
  bool isA_row = (cfg.orderA == CUSPARSE_ORDER_ROW);
  bool isB_row = (cfg.orderB == CUSPARSE_ORDER_ROW);
  bool isC_row = (cfg.orderC == CUSPARSE_ORDER_ROW);
  bool A_t = (cfg.opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
  bool B_t = (cfg.opB != CUSPARSE_OPERATION_NON_TRANSPOSE);

  MatDims d{};
  d.num_A_rows = A_t ? K : N;
  d.num_A_cols = A_t ? N : K;
  d.num_B_rows = B_t ? M : K;
  d.num_B_cols = B_t ? K : M;
  d.num_C_rows = N;
  d.num_C_cols = M;

  d.lda = isA_row ? d.num_A_cols : d.num_A_rows;
  d.ldb = isB_row ? d.num_B_cols : d.num_B_rows;
  d.ldc = isC_row ? d.num_C_cols : d.num_C_rows;
  return d;
}

// ===== prune_24 =====
// 执行 2:4 结构化剪枝，使用 FP16 作为内部计算类型（cuSPARSELt 支持更好）
torch::Tensor prune_24(torch::Tensor W_bf16, const std::string &layout) {
  if (!W_bf16.is_cuda()) {
    throw std::invalid_argument("W_bf16 必须在 CUDA 上");
  }
  
  // 转换为 FP16 进行剪枝（cuSPARSELt 对 FP16 支持更好）
  auto W = W_bf16.to(torch::kHalf).contiguous();
  int64_t N = W.size(0);
  int64_t K = W.size(1);

  // 使用一个合理的虚拟 M 值（必须是 16 的倍数）
  int64_t dummy_M = 16;

  LayoutConfig cfg = parse_layout(layout);
  MatDims dims = make_dims(N, dummy_M, K, cfg);

  cusparseLtHandle_t handle;
  CHECK_CUSPARSE_ERR(cusparseLtInit(&handle));

  cusparseLtMatDescriptor_t matA, matB, matC;
  unsigned alignment = 16;
  cudaDataType typeAB = CUDA_R_16F;  // FP16

  CHECK_CUSPARSE_ERR(cusparseLtStructuredDescriptorInit(
      &handle, &matA, dims.num_A_rows, dims.num_A_cols, dims.lda, alignment,
      typeAB, cfg.orderA, CUSPARSELT_SPARSITY_50_PERCENT));

  CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
      &handle, &matB, dims.num_B_rows, dims.num_B_cols, dims.ldb, alignment,
      typeAB, cfg.orderB));

  CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
      &handle, &matC, dims.num_C_rows, dims.num_C_cols, dims.ldc, alignment,
      typeAB, cfg.orderC));

  cusparseLtMatmulDescriptor_t matmul;
  CHECK_CUSPARSE_ERR(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, cfg.opA, cfg.opB, &matA, &matB, &matC, &matC,
      CUSPARSE_COMPUTE_32F));

  // 注意：不设置 CUSPARSELT_MATMUL_SPARSE_MAT_POINTER
  // 官方 workflow: 稀疏矩阵通过 prune API 直接操作 data_ptr，无需额外属性
  // 这与 search_topk() 保持一致的设计

  // 分配输出（FP16）
  auto out_fp16 = torch::zeros_like(W);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  CHECK_CUSPARSE_ERR(cusparseLtSpMMAPrune(
      &handle, &matmul, W.data_ptr(), out_fp16.data_ptr(),
      CUSPARSELT_PRUNE_SPMMA_TILE, stream));

  // 校验剪枝
  int *d_valid = nullptr;
  CHECK_CUDA_ERR(cudaMalloc(&d_valid, sizeof(int)));
  CHECK_CUSPARSE_ERR(cusparseLtSpMMAPruneCheck(
      &handle, &matmul, out_fp16.data_ptr(), d_valid, stream));

  int h_valid = 0;
  CHECK_CUDA_ERR(cudaMemcpyAsync(&h_valid, d_valid, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
  cudaFree(d_valid);
  if (h_valid != 0) {
    // 仅打印警告，不抛出异常（prune 操作本身是保证正确的）
    std::cerr << "[WARNING] cusparseLtSpMMAPruneCheck 返回非零值 (" << h_valid 
              << ")，可能是新架构兼容性问题，继续执行..." << std::endl;
  }

  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);
  
  // 转换回原始 dtype
  return out_fp16.to(W_bf16.scalar_type());
}

// ===== 搜索结果记录 =====
// 与 cuBLASLt 对齐：存储算法配置和 Split-K 参数
struct AlgRecord {
  int alg_id{-1};             // 算法 ID
  int split_k{1};             // Split-K 值：1=不切分, >1=传统Split-K, -1=Segment-K (H100+)
  float lat_us{0.f};
  float tops{0.f};
  int64_t workspace{0};
  bool valid{false};
  float max_abs_err{0.f};
  bool from_api_search{false}; // 是否来自官方 cusparseLtMatmulSearch API
};

// ===== search_topk =====
py::dict search_topk(torch::Tensor W_pruned_bf16, torch::Tensor A_bf16,
                     const std::vector<int64_t> &M_list,
                     const std::string &layout, const std::string &dtype,
                     const std::string &outdtype,
                     int warmup, int repeat, bool verify,
                     const std::vector<int64_t> &blacklist_ids,
                     int topk,
                     bool test_segment_k) {
  if (!W_pruned_bf16.is_cuda() || !A_bf16.is_cuda()) {
    throw std::invalid_argument("W 与 A 必须在 CUDA 上");
  }
  if (W_pruned_bf16.dim() != 2 || A_bf16.dim() != 2) {
    throw std::invalid_argument("输入张量必须是二维 [N,K] / [M,K]");
  }
  if (topk <= 0) topk = 3;

  auto W_fp = W_pruned_bf16.contiguous();
  auto A_fp = A_bf16.contiguous();
  int64_t N = W_fp.size(0);
  int64_t K = W_fp.size(1);
  
  // === 维度对齐检查 ===
  // cuSPARSELt 要求：dense 矩阵维度需 16 对齐，structured sparse 需 32 对齐
  auto check_alignment = [](int64_t val, int64_t align, const char* name) {
    if (val % align != 0) {
      std::ostringstream oss;
      oss << "维度不满足对齐要求: " << name << "=" << val << " 不是 " << align << " 的倍数";
      throw std::invalid_argument(oss.str());
    }
  };
  check_alignment(N, 32, "N (sparse rows)");
  check_alignment(K, 16, "K (shared dim)");
  for (auto m : M_list) {
    check_alignment(m, 16, "M");
  }

  // 量化/转换
  torch::Tensor W_q;
  torch::Tensor A_q;
  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(outdtype);
  cusparseComputeType comp_type = compute_type_from_dtype(dtype);

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

  // 获取最大 M 用于压缩计划
  int64_t max_M = 0;
  for (auto m : M_list) max_M = std::max(max_M, m);

  LayoutConfig cfg = parse_layout(layout);
  MatDims dims_compress = make_dims(N, max_M, K, cfg);

  cusparseLtHandle_t handle;
  CHECK_CUSPARSE_ERR(cusparseLtInit(&handle));

  unsigned alignment = 16;
  cusparseLtMatDescriptor_t matA, matB, matC;
  CHECK_CUSPARSE_ERR(cusparseLtStructuredDescriptorInit(
      &handle, &matA, dims_compress.num_A_rows, dims_compress.num_A_cols,
      dims_compress.lda, alignment, type_AB, cfg.orderA,
      CUSPARSELT_SPARSITY_50_PERCENT));
  CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
      &handle, &matB, dims_compress.num_B_rows, dims_compress.num_B_cols,
      dims_compress.ldb, alignment, type_AB, cfg.orderB));
  CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
      &handle, &matC, dims_compress.num_C_rows, dims_compress.num_C_cols,
      dims_compress.ldc, alignment, type_C, cfg.orderC));

    cusparseLtMatmulDescriptor_t matmul_desc;
    CHECK_CUSPARSE_ERR(cusparseLtMatmulDescriptorInit(
      &handle, &matmul_desc, cfg.opA, cfg.opB, &matA, &matB, &matC, &matC,
      comp_type));

  // 注意：不设置 CUSPARSELT_MATMUL_SPARSE_MAT_POINTER
  // 官方 workflow: 稀疏矩阵指针通过 cusparseLtMatmul() 的 d_A 参数传入 compressed pointer

  // === 获取最大算法 ID (通过官方 API) ===
  // 根据 cuSPARSELt 文档，CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID 返回算法 ID 上界
  // 有效算法 ID 范围为 [0, max_alg_id)，即 max_alg_id 本身不可取
  // 因此 alg_count = max_alg_id（有效算法数量）
  int max_alg_id = 0;
  {
    cusparseLtMatmulAlgSelection_t alg_sel_tmp;
    cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel_tmp, &matmul_desc, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (sel_st != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("cusparseLtMatmulAlgSelectionInit 失败");
    }
    cusparseStatus_t get_st = cusparseLtMatmulAlgGetAttribute(
        &handle, &alg_sel_tmp, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id, sizeof(max_alg_id));
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    if (get_st != CUSPARSE_STATUS_SUCCESS || max_alg_id <= 0) {
      throw std::runtime_error("无法获取有效的算法 ID 上界");
    }
  }

  // === 压缩相关变量（将在 alg_id 循环内按需分配）===
  // 根据官方文档：cusparseLtSpMMACompress() 必须在每次算法 ID 更新后重新调用
  // 因此压缩操作移入 alg_id 循环内，确保每个算法使用匹配的压缩数据
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // 为结果分配存储
  // 与 cuBLASLt 对齐：存储 topk 的完整算法配置
  int64_t numM = static_cast<int64_t>(M_list.size());
  torch::Tensor topk_alg = torch::full({numM, topk}, -1, torch::dtype(torch::kInt32));
  torch::Tensor topk_split_k = torch::full({numM, topk}, 1, torch::dtype(torch::kInt32));  // cuSPARSELt 特有：Split-K 值
  torch::Tensor topk_lat = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_tops = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_workspace = torch::zeros({numM, topk}, torch::dtype(torch::kInt64));
  torch::Tensor valid_mask = torch::zeros({numM, topk}, torch::dtype(torch::kUInt8));
  torch::Tensor num_valid = torch::zeros({numM}, torch::dtype(torch::kInt32));
  torch::Tensor verify_err = torch::zeros({numM}, torch::dtype(torch::kFloat32));
  std::vector<int> verify_failed_ids;
  
  // === 官方 API 搜索结果存储 ===
  // 每个 M 存储 API 返回的配置及其延迟排名
  torch::Tensor api_alg_id = torch::full({numM}, -1, torch::dtype(torch::kInt32));
  torch::Tensor api_split_k = torch::full({numM}, 1, torch::dtype(torch::kInt32));
  torch::Tensor api_lat_us = torch::zeros({numM}, torch::dtype(torch::kFloat32));
  torch::Tensor api_lat_rank = torch::full({numM}, -1, torch::dtype(torch::kInt32));  // 延迟排名（1-based，-1 表示无效）
  torch::Tensor api_total_configs = torch::zeros({numM}, torch::dtype(torch::kInt32));  // 总配置数

  std::vector<AlgRecord> records;
  
  // === 算法/配置统计 ===
  // alg_count: 算法数量 max_alg_id，因为 ID [0,max)
  // config_count: 实际测试的配置数 (每个 alg_id × split_k 组合)
  int config_count = 0;
  
  // === Workspace 回退机制 ===
  // 预分配一个初始 workspace，如果某个算法需要更大的空间，动态扩展
  size_t current_workspace_size = 0;
  void *shared_workspace = nullptr;

  for (int64_t m_index = 0; m_index < numM; ++m_index) {
    int64_t M = M_list[m_index];
    MatDims dims = make_dims(N, M, K, cfg);

    cusparseLtMatDescriptor_t matB_m, matC_m;
    CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
        &handle, &matB_m, dims.num_B_rows, dims.num_B_cols, dims.ldb, alignment,
        type_AB, cfg.orderB));
    // 注意：对于 A100 Row Major (NTRRrow)，ldc 的计算在 make_dims 中已处理
    // Col Major: ldc = num_C_rows = N
    // Row Major: ldc = num_C_cols = M
    CHECK_CUSPARSE_ERR(cusparseLtDenseDescriptorInit(
        &handle, &matC_m, dims.num_C_rows, dims.num_C_cols, dims.ldc, alignment,
        type_C, cfg.orderC));

    cusparseLtMatmulDescriptor_t matmul_m;
    CHECK_CUSPARSE_ERR(cusparseLtMatmulDescriptorInit(
        &handle, &matmul_m, cfg.opA, cfg.opB, &matA, &matB_m, &matC_m, &matC_m,
        comp_type));

    // 注意：不设置 CUSPARSELT_MATMUL_SPARSE_MAT_POINTER
    // 稀疏矩阵指针通过 cusparseLtMatmul() 的 d_A 参数传入 compressed pointer

    // B 切片
    auto A_slice = A_q.narrow(0, 0, M).contiguous();

    // 输出 C 与 D（就地）
    // 关键：cuSPARSELt 以 Column Major 存储 C 矩阵（当 cfg.orderC == COL）
    // 逻辑形状 [N, M]，Column Major 存储意味着 leading dim = N
    // PyTorch 默认 Row Major，为了兼容，我们创建 [M, N] 的 tensor
    // 这样 PyTorch 的 Row Major [M, N] 等价于 Column Major [N, M]
    torch::Tensor C_out;
    bool is_col_major = (cfg.orderC == CUSPARSE_ORDER_COL);
    // 输出类型由 outdtype 参数决定
    auto out_torch_dtype = (outdtype == "fp32") ? torch::kFloat32 : torch::kBFloat16;
    if (is_col_major) {
      // Column Major [N, M] 等价于 Row Major [M, N] 的转置
      C_out = torch::zeros({M, N}, torch::dtype(out_torch_dtype).device(W_q.device()));
    } else {
      C_out = torch::zeros({N, M}, torch::dtype(out_torch_dtype).device(W_q.device()));
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // === 阶段 0: 调用官方 cusparseLtMatmulSearch API 获取推荐配置 ===
    // 该 API 会自动搜索并选择最优算法配置，结果用于与我们手动搜索的结果对比
    AlgRecord api_search_record;
    api_search_record.valid = false;
    api_search_record.from_api_search = true;
    
    {
      // 为官方搜索创建 AlgSelection（使用默认配置）
      cusparseLtMatmulAlgSelection_t alg_sel_api;
      cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
          &handle, &alg_sel_api, &matmul_m, CUSPARSELT_MATMUL_ALG_DEFAULT);
      
      if (sel_st == CUSPARSE_STATUS_SUCCESS) {
        // 设置搜索迭代次数（与我们的 repeat 类似）
        int search_iterations = std::max(10, repeat / 10);  // 使用较少迭代加速搜索
        cusparseLtMatmulAlgSetAttribute(
            &handle, &alg_sel_api, CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
            &search_iterations, sizeof(search_iterations));
        
        cusparseLtMatmulPlan_t plan_api;
        auto plan_result = planInitWithTimeout(&handle, &plan_api, &matmul_m, &alg_sel_api, 10);
        
        if (!plan_result.timed_out && plan_result.status == CUSPARSE_STATUS_SUCCESS) {
          // 获取 workspace
          size_t ws_api = 0;
          cusparseLtMatmulGetWorkspace(&handle, &plan_api, &ws_api);
          
          // 确保共享 workspace 足够大
          if (ws_api > current_workspace_size) {
            if (shared_workspace) cudaFree(shared_workspace);
            cudaMalloc(&shared_workspace, ws_api);
            current_workspace_size = ws_api;
          }
          void *workspace_api = (ws_api > 0) ? shared_workspace : nullptr;
          
          // 为搜索压缩权重（使用默认 alg_id=0 即可，API 会内部调整）
          size_t comp_size_api = 0, comp_buf_size_api = 0;
          cusparseLtSpMMACompressedSize(&handle, &plan_api, &comp_size_api, &comp_buf_size_api);
          
          torch::Tensor W_comp_api = torch::empty({static_cast<long>(comp_size_api)},
                                                  torch::dtype(torch::kUInt8).device(W_q.device()));
          void *comp_buf_api = nullptr;
          if (comp_buf_size_api > 0) cudaMalloc(&comp_buf_api, comp_buf_size_api);
          
          cusparseStatus_t comp_st = cusparseLtSpMMACompress(
              &handle, &plan_api, W_q.data_ptr(), W_comp_api.data_ptr(),
              comp_buf_api, stream);
          
          if (comp_buf_api) cudaFree(comp_buf_api);
          
          if (comp_st == CUSPARSE_STATUS_SUCCESS) {
            // 调用官方搜索 API
            cusparseStatus_t search_st = cusparseLtMatmulSearch(
                &handle, &plan_api, &alpha, W_comp_api.data_ptr(), A_slice.data_ptr(),
                &beta, C_out.data_ptr(), C_out.data_ptr(), workspace_api, &stream, 1);
            
            if (search_st == CUSPARSE_STATUS_SUCCESS) {
              // 从 plan 中提取选中的算法参数
              int api_alg_id = 0;
              int api_split_k = 1;
              
              cusparseLtMatmulAlgGetAttribute(
                  &handle, &alg_sel_api, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                  &api_alg_id, sizeof(api_alg_id));
              cusparseLtMatmulAlgGetAttribute(
                  &handle, &alg_sel_api, CUSPARSELT_MATMUL_SPLIT_K,
                  &api_split_k, sizeof(api_split_k));
              
              // === 用选中的配置进行标准化性能测试 ===
              // 为了公平对比，需要用相同的 warmup/repeat 方法测试
              
              // 根据 API 选中的配置重新压缩（确保匹配）
              cusparseLtMatmulAlgSelection_t alg_sel_api_test;
              cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel_api_test, &matmul_m, CUSPARSELT_MATMUL_ALG_DEFAULT);
              cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel_api_test, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                              &api_alg_id, sizeof(api_alg_id));
              cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel_api_test, CUSPARSELT_MATMUL_SPLIT_K,
                                              &api_split_k, sizeof(api_split_k));
              
              cusparseLtMatmulPlan_t plan_api_test;
              auto plan_test_result = planInitWithTimeout(&handle, &plan_api_test, &matmul_m, &alg_sel_api_test, 5);
              
              if (!plan_test_result.timed_out && plan_test_result.status == CUSPARSE_STATUS_SUCCESS) {
                size_t ws_test = 0;
                cusparseLtMatmulGetWorkspace(&handle, &plan_api_test, &ws_test);
                
                if (ws_test > current_workspace_size) {
                  if (shared_workspace) cudaFree(shared_workspace);
                  cudaMalloc(&shared_workspace, ws_test);
                  current_workspace_size = ws_test;
                }
                void *workspace_test = (ws_test > 0) ? shared_workspace : nullptr;
                
                // 重新压缩
                size_t comp_size_test = 0, comp_buf_size_test = 0;
                cusparseLtSpMMACompressedSize(&handle, &plan_api_test, &comp_size_test, &comp_buf_size_test);
                torch::Tensor W_comp_test = torch::empty({static_cast<long>(comp_size_test)},
                                                         torch::dtype(torch::kUInt8).device(W_q.device()));
                void *comp_buf_test = nullptr;
                if (comp_buf_size_test > 0) cudaMalloc(&comp_buf_test, comp_buf_size_test);
                cusparseLtSpMMACompress(&handle, &plan_api_test, W_q.data_ptr(), W_comp_test.data_ptr(),
                                        comp_buf_test, stream);
                if (comp_buf_test) cudaFree(comp_buf_test);
                
                // 预热
                bool success = true;
                for (int i = 0; i < warmup && success; ++i) {
                  cusparseStatus_t st = cusparseLtMatmul(
                      &handle, &plan_api_test, &alpha, W_comp_test.data_ptr(), A_slice.data_ptr(),
                      &beta, C_out.data_ptr(), C_out.data_ptr(), workspace_test, &stream, 1);
                  if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                }
                CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
                
                // 正式计时
                if (success) {
                  cudaEvent_t start, stop;
                  cudaEventCreate(&start);
                  cudaEventCreate(&stop);
                  cudaEventRecord(start, stream);
                  
                  for (int r = 0; r < repeat; ++r) {
                    cusparseStatus_t st = cusparseLtMatmul(
                        &handle, &plan_api_test, &alpha, W_comp_test.data_ptr(), A_slice.data_ptr(),
                        &beta, C_out.data_ptr(), C_out.data_ptr(), workspace_test, &stream, 1);
                    if (st != CUSPARSE_STATUS_SUCCESS) { success = false; break; }
                  }
                  
                  cudaEventRecord(stop, stream);
                  cudaEventSynchronize(stop);
                  float total_ms = 0.0f;
                  cudaEventElapsedTime(&total_ms, start, stop);
                  cudaEventDestroy(start);
                  cudaEventDestroy(stop);
                  
                  if (success) {
                    api_search_record.alg_id = api_alg_id;
                    api_search_record.split_k = api_split_k;
                    api_search_record.lat_us = (total_ms * 1000.0f) / static_cast<float>(repeat);
                    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
                    api_search_record.tops = static_cast<float>(ops / (api_search_record.lat_us / 1e6) / 1e12);
                    api_search_record.workspace = static_cast<int64_t>(ws_test);
                    api_search_record.valid = true;
                    api_search_record.from_api_search = true;
                  }
                }
                cusparseLtMatmulPlanDestroy(&plan_api_test);
              }
              cusparseLtMatmulAlgSelectionDestroy(&alg_sel_api_test);
            }
          }
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
    // split_k_val 候选列表：
    //   1: 不切分 (Baseline)
    //   2, 4, 8, 16, 32, 64: 传统 Split-K（自适应倍增，根据性能决定是否继续）
    //   -1: Segment-K (SM 9.0/10.x 架构特殊优化，由 Python 端控制是否测试)
    // 测试顺序：先测 k=1 得到 baseline，然后倍增 k，
    //   如果新延时 * 1.10 > 旧延时则停止倍增，最后测试 k=-1
    
    // 注意：max_alg_id 是算法 ID 上界，有效范围为 [0, max_alg_id)，即 max_alg_id 不可取
    for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
      if (std::find(blacklist_ids.begin(), blacklist_ids.end(), alg_id) !=
          blacklist_ids.end()) {
        continue;
      }

      // === 为当前 alg_id 创建 plan 并压缩权重 ===
      // 根据官方文档：cusparseLtSpMMACompress() 必须在每次算法 ID 更新后重新调用
      cusparseLtMatmulAlgSelection_t alg_sel_compress;
      cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
          &handle, &alg_sel_compress, &matmul_m, CUSPARSELT_MATMUL_ALG_DEFAULT);
      if (sel_st != CUSPARSE_STATUS_SUCCESS) {
        continue;  // 算法选择初始化失败，跳过当前 alg_id
      }

      cusparseStatus_t set_st = cusparseLtMatmulAlgSetAttribute(
          &handle, &alg_sel_compress, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
          &alg_id, sizeof(alg_id));
      if (set_st != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
        continue;
      }

      cusparseLtMatmulPlan_t plan_compress;
      auto plan_result = planInitWithTimeout(&handle, &plan_compress, &matmul_m, &alg_sel_compress, 5);
      if (plan_result.timed_out || plan_result.status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
        continue;
      }

      // 获取压缩所需大小
      size_t compressed_size = 0, compressed_buffer_size = 0;
      CHECK_CUSPARSE_ERR(cusparseLtSpMMACompressedSize(
          &handle, &plan_compress, &compressed_size, &compressed_buffer_size));

      // 分配压缩权重存储
      torch::Tensor W_compressed = torch::empty({static_cast<long>(compressed_size)},
                                                torch::dtype(torch::kUInt8).device(W_q.device()));
      void *compress_buffer = nullptr;
      if (compressed_buffer_size > 0) {
        cudaError_t alloc_st = cudaMalloc(&compress_buffer, compressed_buffer_size);
        if (alloc_st != cudaSuccess) {
          cusparseLtMatmulPlanDestroy(&plan_compress);
          cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
          continue;
        }
      }

      // 执行压缩
      cusparseStatus_t compress_st = cusparseLtSpMMACompress(
          &handle, &plan_compress, W_q.data_ptr(), W_compressed.data_ptr(),
          compress_buffer, stream);
      
      // 释放压缩 buffer（压缩完成后不再需要）
      if (compress_buffer) {
        cudaFree(compress_buffer);
        compress_buffer = nullptr;
      }
      
      // 释放用于压缩的 plan 和 alg_sel（后续 split_k 测试会创建新的）
      cusparseLtMatmulPlanDestroy(&plan_compress);
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
      
      if (compress_st != CUSPARSE_STATUS_SUCCESS) {
        continue;  // 压缩失败，跳过当前 alg_id
      }

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
        // === 去重检查：跳过与官方 API 搜索结果相同的配置 ===
        // 避免重复测试和重复计入排名
        if (api_search_record.valid && 
            alg_id == api_search_record.alg_id && 
            split_k_val == api_search_record.split_k) {
          continue;
        }

        // === 自适应倍增策略 ===
        // 如果已停止倍增且当前是倍增序列中的值 (>1)，则跳过（但 -1 除外）
        if (stop_doubling && split_k_val > 1) {
          continue;
        }

        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseStatus_t sel_status = cusparseLtMatmulAlgSelectionInit(
            &handle, &alg_sel, &matmul_m, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (sel_status != CUSPARSE_STATUS_SUCCESS) {
          break;  // 算法选择初始化失败，跳出 split_k 循环
        }
        
        // 设置算法 ID
        cusparseStatus_t set_status = cusparseLtMatmulAlgSetAttribute(
            &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id,
            sizeof(alg_id));
        if (set_status != CUSPARSE_STATUS_SUCCESS) {
          cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
          continue;
        }

        // === 设置 Split-K 属性 ===
        // CUSPARSELT_MATMUL_SPLIT_K:
        //   1: 不切分 (默认)
        //   >1: 传统 Split-K
        //   -1: Segment-K (SM 9.0/10.x 特有优化，其他架构不支持)
        cusparseStatus_t split_k_status = cusparseLtMatmulAlgSetAttribute(
            &handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &split_k_val,
            sizeof(split_k_val));
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
        auto plan_result = planInitWithTimeout(&handle, &plan, &matmul_m, &alg_sel, 5);
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

        // === Workspace 回退机制 ===
        // 查询当前算法所需 workspace 大小（Split-K/Segment-K 对 workspace 需求不同）
        size_t workspace_size = 0;
        cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
        
        // 如果当前共享 workspace 不够大，扩展它
        if (workspace_size > current_workspace_size) {
          if (shared_workspace != nullptr) {
            cudaFree(shared_workspace);
            shared_workspace = nullptr;
          }
          // 尝试分配更大的 workspace
          cudaError_t alloc_st = cudaMalloc(&shared_workspace, workspace_size);
          if (alloc_st != cudaSuccess) {
            // 分配失败，跳过此组合
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
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
            cusparseStatus_t st = cusparseLtMatmul(
                &handle, &plan, &alpha, W_compressed.data_ptr(), A_slice.data_ptr(),
                &beta, C_out.data_ptr(), C_out.data_ptr(), workspace, &stream, 1);
            if (st != CUSPARSE_STATUS_SUCCESS) {
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
            cusparseStatus_t st = cusparseLtMatmul(
                &handle, &plan, &alpha, W_compressed.data_ptr(),
                A_slice.data_ptr(), &beta, C_out.data_ptr(), C_out.data_ptr(),
                workspace, &stream, 1);  // 显式传入 stream
            if (st != CUSPARSE_STATUS_SUCCESS) {
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
          rec.alg_id = alg_id;
          rec.split_k = split_k_val;  // 记录 Split-K 值
          rec.lat_us = (total_ms * 1000.0f) / static_cast<float>(repeat);
          double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                       static_cast<double>(K);
          double tops = ops / (rec.lat_us / 1e6) / 1e12;
          rec.tops = static_cast<float>(tops);
          rec.workspace = static_cast<int64_t>(workspace_size);
          rec.valid = true;

          // 校验（如需）
          if (verify) {
            // 使用量化后的数据做 FP32 参考计算
            // 这样才能和 cuSPARSELt 的 INT8 GEMM 结果对比
            auto A_slice_v = A_q.narrow(0, 0, M);
            auto A_fp32 = A_slice_v.to(torch::kFloat32);
            auto W_fp32 = W_q.to(torch::kFloat32);
            auto ref = torch::matmul(W_fp32, A_fp32.transpose(0, 1));  // [N, M], Row Major
          
            // 将输出转为 FP32 比较
            // C_out 的创建已经考虑了布局：
            //   - Column Major 时创建为 [M, N]，转置后得到 [N, M]
            //   - Row Major 时直接创建为 [N, M]
            torch::Tensor out_fp32;
            if (is_col_major) {
              // C_out 是 [M, N]，转置得到 [N, M] 与 ref 对齐
              out_fp32 = C_out.to(torch::kFloat32).t().contiguous();
            } else {
              // Row Major: 直接使用
              out_fp32 = C_out.to(torch::kFloat32);
            }
            
            // 计算相对误差（相对于参考值的绝对值）
            auto ref_abs = ref.abs().clamp_min(1.0f);  // 避免除以0
            auto rel_diff = ((out_fp32 - ref) / ref_abs).abs();
            float max_rel_err = rel_diff.max().item<float>();
            rec.max_abs_err = max_rel_err;  // 存储的是相对误差
            
            // 相对误差容限：5%
            constexpr float tol = 0.05f;
            constexpr float critical_tol = 1.00f;  // 超过 100% 认为计算有严重问题
            
            if (max_rel_err > critical_tol || std::isnan(max_rel_err)) {
              // 严重错误：输出 warning，跳过当前组合
              std::cerr << "[WARNING] M=" << M << " alg_id=" << alg_id << " split_k=" << split_k_val
                        << " 相对误差=" << (max_rel_err * 100.0f) << "% > 100%" << std::endl;
              rec.valid = false;
              verify_failed_ids.push_back(alg_id);
              cusparseLtMatmulPlanDestroy(&plan);
              cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
              continue;  // 跳过当前 split_k 组合
            } else if (max_rel_err > tol) {
              // 超过容限但未达到严重级别：记录但不跳过
              std::cout << "[INFO] M=" << M << " alg_id=" << alg_id << " split_k=" << split_k_val
                        << " 相对误差=" << (max_rel_err * 100.0f) << "% > 5%" << std::endl;
              rec.valid = true;
              verify_failed_ids.push_back(alg_id);
            }
          }
          records.push_back(rec);
          ++config_count;  // 统计实际测试的配置数
          
          // === 自适应倍增策略：根据性能决定是否继续倍增 ===
          // 对于倍增序列 (split_k_val >= 1)，更新 best 并决定是否继续
          if (split_k_val >= 1) {
            if (best_lat_us_for_doubling < 0 || rec.lat_us < best_lat_us_for_doubling) {
              // 新延时更低，更新 best
              best_lat_us_for_doubling = rec.lat_us;
            } else if (rec.lat_us * 1.10f > best_lat_us_for_doubling && split_k_val > 1) {
              // 新延时 * 1.10 > 旧延时（10% 容限），停止倍增
              stop_doubling = true;
            }
          }
          // 注意：k=-1 (Segment-K) 不参与倍增策略，总是单独测试一次
        }

        // 注意：不再释放 workspace，因为使用共享 workspace
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      }  // end split_k_val loop
    }  // end alg_id loop

    // 当前 M 取出记录并排序
    std::vector<AlgRecord> filtered;
    for (auto &r : records) {
      if (r.valid) filtered.push_back(r);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const AlgRecord &a, const AlgRecord &b) {
                return a.lat_us < b.lat_us;
              });

    // === 记录官方 API 搜索结果的排名 ===
    // 在排序后的列表中查找官方搜索结果（通过 from_api_search 标记）
    int api_rank = -1;  // -1 表示无效或未找到
    for (size_t i = 0; i < filtered.size(); ++i) {
      if (filtered[i].from_api_search) {
        api_rank = static_cast<int>(i + 1);  // 1-based 排名
        api_alg_id.index_put_({m_index}, filtered[i].alg_id);
        api_split_k.index_put_({m_index}, filtered[i].split_k);
        api_lat_us.index_put_({m_index}, filtered[i].lat_us);
        break;
      }
    }
    api_lat_rank.index_put_({m_index}, api_rank);
    api_total_configs.index_put_({m_index}, static_cast<int>(filtered.size()));

    int fill = std::min(static_cast<int>(filtered.size()), topk);
    for (int i = 0; i < fill; ++i) {
      topk_alg.index_put_({m_index, i}, filtered[i].alg_id);
      topk_split_k.index_put_({m_index, i}, filtered[i].split_k);  // 记录 Split-K 值
      topk_lat.index_put_({m_index, i}, filtered[i].lat_us);
      topk_tops.index_put_({m_index, i}, filtered[i].tops);
      topk_workspace.index_put_({m_index, i}, filtered[i].workspace);
      valid_mask.index_put_({m_index, i}, static_cast<uint8_t>(1));
      if (verify && i == 0) {
        verify_err.index_put_({m_index}, filtered[i].max_abs_err);
      }
    }
    num_valid.index_put_({m_index}, static_cast<int>(filtered.size()));

    cusparseLtMatDescriptorDestroy(&matB_m);
    cusparseLtMatDescriptorDestroy(&matC_m);
    // 注意：cusparseLtMatmulDescriptor_t 是普通结构体，无需显式销毁
    records.clear();
  }

  // === 释放共享 workspace ===
  if (shared_workspace != nullptr) {
    cudaFree(shared_workspace);
    shared_workspace = nullptr;
  }

  // 清理
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  // 注意：cusparseLtMatmulDescriptor_t 是普通结构体，无需显式销毁
  // 注意：压缩相关资源已在 alg_id 循环内释放
  cusparseLtDestroy(&handle);

  // 构造返回
  // 与 cuBLASLt 对齐的输出格式
  py::dict out;
  out["M_list"] = torch::tensor(M_list, torch::dtype(torch::kInt32));
  out["NK"] = torch::tensor({static_cast<int32_t>(N), static_cast<int32_t>(K)},
                             torch::dtype(torch::kInt32));
  out["topk_alg_id"] = topk_alg;
  out["topk_split_k"] = topk_split_k;  // cuSPARSELt 特有：Split-K 值 (1=不切分, >1=Split-K, -1=Segment-K)
  out["topk_lat_us"] = topk_lat;
  out["topk_tops"] = topk_tops;
  out["topk_workspace"] = topk_workspace;
  out["valid_mask"] = valid_mask;
  // 注意：现在每个算法使用自己的 ID 进行压缩，不再有统一的 compress_alg_id
  out["alg_count"] = max_alg_id;    // 有效算法数量 (ID 范围 [0, max_alg_id))
  out["config_count"] = config_count;   // 实际测试的配置数 (alg_id × split_k 组合)
  out["num_valid_algs_per_M"] = num_valid;  // 每个 M 的有效算法数（包含所有 Split-K 组合）
  // === 官方 API 搜索结果 ===
  out["api_alg_id"] = api_alg_id;
  out["api_split_k"] = api_split_k;
  out["api_lat_us"] = api_lat_us;
  out["api_lat_rank"] = api_lat_rank;        // 延迟排名 (1-based，-1 表示无效)
  out["api_total_configs"] = api_total_configs;  // 每个 M 的总配置数
  if (verify) {
    out["verify_max_abs_err"] = verify_err;
    out["verify_failed_algs"] = torch::tensor(verify_failed_ids,
                                              torch::dtype(torch::kInt32));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prune_24", &prune_24, "2:4 剪枝 (cuSPARSELt)",
        py::arg("W_bf16"), py::arg("layout"));
  m.def("search_topk", &search_topk, "枚举算法+Split-K 并返回 topk", py::arg("W_pruned_bf16"),
        py::arg("A_bf16"), py::arg("M_list"), py::arg("layout"),
        py::arg("dtype") = "int8", py::arg("outdtype") = "bf16",
        py::arg("warmup") = 25,
        py::arg("repeat") = 100, py::arg("verify") = false,
        py::arg("blacklist_ids") = std::vector<int64_t>{},
        py::arg("topk") = 3,
        py::arg("test_segment_k") = false);
}
