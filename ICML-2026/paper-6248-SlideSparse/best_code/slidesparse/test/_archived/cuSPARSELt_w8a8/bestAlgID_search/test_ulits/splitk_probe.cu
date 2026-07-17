// splitk_probe.cu - 探测 Split-K 行为的简单脚本
// 
// 编译运行:
//   python3 splitk_probe.py

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstdio>
#include <vector>

// 测试单个 Split-K 值，返回状态码，有超时机制
// 返回: 0=成功, >0=cuSPARSELt错误码, -999=超时
int test_splitk_value(cusparseLtHandle_t* handle, 
                      cusparseLtMatmulDescriptor_t* matmul,
                      int alg_id, int split_k, int timeout_ms) {
  cusparseLtMatmulAlgSelection_t sel;
  cusparseStatus_t st;
  
  st = cusparseLtMatmulAlgSelectionInit(handle, &sel, matmul,
                                        CUSPARSELT_MATMUL_ALG_DEFAULT);
  if (st != CUSPARSE_STATUS_SUCCESS) return (int)st;
  
  st = cusparseLtMatmulAlgSetAttribute(handle, &sel,
                                       CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                       &alg_id, sizeof(alg_id));
  if (st != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    return (int)st;
  }
  
  st = cusparseLtMatmulAlgSetAttribute(handle, &sel,
                                       CUSPARSELT_MATMUL_SPLIT_K,
                                       &split_k, sizeof(split_k));
  if (st != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    return (int)st;
  }
  
  // PlanInit - 这里可能会卡住
  cusparseLtMatmulPlan_t plan;
  st = cusparseLtMatmulPlanInit(handle, &plan, matmul, &sel);
  cusparseLtMatmulAlgSelectionDestroy(&sel);
  
  if (st == CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatmulPlanDestroy(&plan);
    return 0;
  }
  return (int)st;
}

// 主探测函数
std::vector<std::tuple<int, int, int>> probe_splitk(int N, int K, int M) {
  printf("========================================\n");
  printf("Split-K Probe (N=%d, K=%d, M=%d)\n", N, K, M);
  printf("========================================\n");
  fflush(stdout);
  
  std::vector<std::tuple<int, int, int>> results;  // (alg_id, split_k, status)
  
  // 获取 GPU 信息
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_major = prop.major;
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
  fflush(stdout);
  
  // 初始化
  cusparseLtHandle_t handle;
  if (cusparseLtInit(&handle) != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR: cusparseLtInit failed\n");
    return results;
  }
  
  // 创建描述符 (INT8, TN layout)
  unsigned alignment = 16;
  cusparseLtMatDescriptor_t matA, matB, matC;
  
  cusparseLtStructuredDescriptorInit(&handle, &matA, K, N, K, alignment,
                                     CUDA_R_8I, CUSPARSE_ORDER_COL,
                                     CUSPARSELT_SPARSITY_50_PERCENT);
  cusparseLtDenseDescriptorInit(&handle, &matB, K, M, K, alignment,
                                CUDA_R_8I, CUSPARSE_ORDER_COL);
  cusparseLtDenseDescriptorInit(&handle, &matC, N, M, N, alignment,
                                CUDA_R_32I, CUSPARSE_ORDER_COL);
  
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &matA, &matB, &matC, &matC,
                                 CUSPARSE_COMPUTE_32I);
  
  // === 1. 探测有效 alg_id ===
  printf("[1] Probing valid alg_ids...\n");
  fflush(stdout);
  
  std::vector<int> valid_alg_ids;
  for (int alg_id = 0; alg_id < 10; ++alg_id) {
    cusparseLtMatmulAlgSelection_t sel;
    cusparseStatus_t st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                                           CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) continue;
    
    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                         &alg_id, sizeof(alg_id));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }
    
    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    
    if (st == CUSPARSE_STATUS_SUCCESS) {
      valid_alg_ids.push_back(alg_id);
      cusparseLtMatmulPlanDestroy(&plan);
      printf("  alg_id=%d: OK\n", alg_id);
    } else {
      printf("  alg_id=%d: FAIL (code=%d)\n", alg_id, (int)st);
    }
    fflush(stdout);
  }
  
  if (valid_alg_ids.empty()) {
    printf("ERROR: No valid alg_id found!\n");
    cusparseLtDestroy(&handle);
    return results;
  }
  printf("\n");
  
  // === 2. 对每个有效 alg_id 测试 Split-K ===
  // 测试范围：1, 2, 4, 8, ..., 256, 512 (找到上限)
  // 注意：跳过 -1，因为可能会卡死
  printf("[2] Probing Split-K values (SKIP -1 to avoid hang)...\n");
  printf("    Testing: 1");
  for (int sk = 2; sk <= 512; sk *= 2) printf(", %d", sk);
  printf("\n\n");
  fflush(stdout);
  
  for (int alg_id : valid_alg_ids) {
    printf("  alg_id=%d:\n", alg_id);
    fflush(stdout);
    
    int last_valid_sk = 0;
    
    // 测试 split_k = 1, 2, 4, ..., 512
    for (int split_k = 1; split_k <= 512; split_k = (split_k == 1) ? 2 : split_k * 2) {
      int status = test_splitk_value(&handle, &matmul, alg_id, split_k, 1000);
      results.push_back({alg_id, split_k, status});
      
      if (status == 0) {
        printf("    split_k=%3d: OK\n", split_k);
        last_valid_sk = split_k;
      } else {
        printf("    split_k=%3d: FAIL (code=%d) -> STOP\n", split_k, status);
        break;  // Split-K 失败后停止（后面的更大值也会失败）
      }
      fflush(stdout);
    }
    
    printf("    => Max valid split_k: %d\n\n", last_valid_sk);
    fflush(stdout);
  }
  
  // === 3. 关于 -1 (Segment-K) 的说明 ===
  printf("[3] About split_k=-1 (Segment-K):\n");
  printf("    Segment-K is a Hopper (SM 9.0) specific optimization.\n");
  printf("    On non-Hopper GPUs, PlanInit may HANG indefinitely.\n");
  printf("    Current GPU SM: %d.x\n", sm_major);
  if (sm_major == 9) {
    printf("    => This is Hopper, -1 SHOULD work.\n");
  } else {
    printf("    => This is NOT Hopper, -1 will likely HANG. SKIP!\n");
  }
  printf("\n");
  fflush(stdout);
  
  // 清理
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);
  
  printf("========================================\n");
  printf("Done!\n");
  printf("========================================\n");
  fflush(stdout);
  
  return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_splitk", &probe_splitk, "Probe Split-K behavior",
        py::arg("N") = 64, py::arg("K") = 64, py::arg("M") = 32);
}
