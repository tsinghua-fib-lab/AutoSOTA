// debug_splitk.cu - 调试 cuSPARSELt Split-K 搜索卡住问题
// 编译: nvcc -O2 debug_splitk.cu -o debug_splitk -lcusparseLt -lcusparse -lcudart
// 运行: ./debug_splitk

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("[CUDA ERROR] %s at %s:%d\n", cudaGetErrorString(err), __FILE__,  \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUSPARSE(expr)                                                   \
  do {                                                                         \
    cusparseStatus_t st = (expr);                                              \
    if (st != CUSPARSE_STATUS_SUCCESS) {                                       \
      printf("[cuSPARSELt ERROR] code=%d at %s:%d\n", (int)st, __FILE__,        \
             __LINE__);                                                        \
      return st;                                                               \
    }                                                                          \
  } while (0)

// 测试参数（满足对齐要求）
constexpr int N = 64;   // 稀疏矩阵行数 (需 32 对齐)
constexpr int K = 64;   // 共享维度 (需 16 对齐)
constexpr int M = 32;   // 批次大小 (需 16 对齐)

int main() {
  printf("========================================\n");
  printf("cuSPARSELt Split-K Debug Tool\n");
  printf("========================================\n");
  printf("Test dims: N=%d, K=%d, M=%d\n\n", N, K, M);

  // 初始化 CUDA
  CHECK_CUDA(cudaSetDevice(0));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

  // === Step 1: 初始化 cuSPARSELt handle ===
  printf("[Step 1] Init cuSPARSELt handle...\n");
  fflush(stdout);
  cusparseLtHandle_t handle;
  cusparseStatus_t st = cusparseLtInit(&handle);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("  FAILED: cusparseLtInit returned %d\n", (int)st);
    return 1;
  }
  printf("  OK\n\n");

  // === Step 2: 创建矩阵描述符 ===
  printf("[Step 2] Create matrix descriptors...\n");
  fflush(stdout);
  
  unsigned alignment = 16;
  cudaDataType type_AB = CUDA_R_16F;  // FP16
  cudaDataType type_C = CUDA_R_16F;
  cusparseComputeType comp_type = CUSPARSE_COMPUTE_16F;
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  cusparseOperation_t opA = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // 根据 TN 布局计算维度
  int num_A_rows = K;  // opA=T -> A 的存储是 [K, N]
  int num_A_cols = N;
  int num_B_rows = K;  // opB=N -> B 的存储是 [K, M]
  int num_B_cols = M;
  int lda = num_A_rows;  // col major
  int ldb = num_B_rows;
  int ldc = N;

  cusparseLtMatDescriptor_t matA, matB, matC;
  
  st = cusparseLtStructuredDescriptorInit(&handle, &matA, num_A_rows, num_A_cols,
                                          lda, alignment, type_AB, order,
                                          CUSPARSELT_SPARSITY_50_PERCENT);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("  FAILED: matA init returned %d\n", (int)st);
    return 1;
  }
  
  st = cusparseLtDenseDescriptorInit(&handle, &matB, num_B_rows, num_B_cols,
                                     ldb, alignment, type_AB, order);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("  FAILED: matB init returned %d\n", (int)st);
    return 1;
  }
  
  st = cusparseLtDenseDescriptorInit(&handle, &matC, N, M, ldc, alignment,
                                     type_C, order);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("  FAILED: matC init returned %d\n", (int)st);
    return 1;
  }
  printf("  OK\n\n");

  // === Step 3: 创建 matmul 描述符 ===
  printf("[Step 3] Create matmul descriptor...\n");
  fflush(stdout);
  
  cusparseLtMatmulDescriptor_t matmul;
  st = cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB,
                                      &matA, &matB, &matC, &matC, comp_type);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("  FAILED: matmul descriptor init returned %d\n", (int)st);
    return 1;
  }
  printf("  OK\n\n");

  // === Step 4: 探测最大有效 alg_id ===
  printf("[Step 4] Probe max valid alg_id...\n");
  fflush(stdout);
  
  int max_alg_id = -1;
  int consecutive_failures = 0;
  constexpr int kMaxConsecutiveFailures = 5;
  
  for (int probe = 0; probe < 100; ++probe) {  // 加上限，防止无限循环
    printf("  Probing alg_id=%d... ", probe);
    fflush(stdout);
    
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("AlgSelectionInit FAILED (%d)\n", (int)st);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) break;
      continue;
    }
    
    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                         &probe, sizeof(probe));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttribute(ALG_CONFIG_ID) FAILED (%d)\n", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) break;
      continue;
    }
    
    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    
    if (st == CUSPARSE_STATUS_SUCCESS) {
      printf("OK\n");
      max_alg_id = probe;
      consecutive_failures = 0;
      cusparseLtMatmulPlanDestroy(&plan);
    } else {
      printf("PlanInit FAILED (%d)\n", (int)st);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) break;
    }
  }
  
  printf("\n  Max valid alg_id = %d\n\n", max_alg_id);
  
  if (max_alg_id < 0) {
    printf("ERROR: No valid alg_id found!\n");
    return 1;
  }

  // === Step 5: 测试 Split-K 设置 ===
  printf("[Step 5] Test Split-K attribute setting...\n");
  printf("  Using alg_id=%d as base\n\n", max_alg_id);
  fflush(stdout);
  
  std::vector<int> split_k_vals = {1, 2, 4, 8, 16, 32, 64, -1};
  
  for (int split_k : split_k_vals) {
    printf("  Testing split_k=%d... ", split_k);
    fflush(stdout);
    
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("AlgSelectionInit FAILED (%d)\n", (int)st);
      continue;
    }
    
    // 设置 alg_id
    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                         &max_alg_id, sizeof(max_alg_id));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttribute(ALG_CONFIG_ID) FAILED (%d)\n", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }
    
    // === 关键：测试 SPLIT_K 属性设置 ===
    printf("setting SPLIT_K... ");
    fflush(stdout);
    
    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_SPLIT_K,
                                         &split_k, sizeof(split_k));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttribute(SPLIT_K) FAILED (%d)\n", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }
    printf("OK... ");
    
    // === 测试 PlanInit ===
    printf("PlanInit... ");
    fflush(stdout);
    
    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    
    if (st == CUSPARSE_STATUS_SUCCESS) {
      printf("OK\n");
      cusparseLtMatmulPlanDestroy(&plan);
    } else {
      printf("FAILED (%d)\n", (int)st);
    }
  }

  // === Step 6: 检查枚举属性 ===
  printf("\n[Step 6] Check SPLIT_K_MODE attribute (if supported)...\n");
  fflush(stdout);
  
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      // 尝试获取 split_k 属性
      int current_split_k = 0;
      size_t ret_size = 0;
      st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &current_split_k, sizeof(current_split_k),
                                           &ret_size);
      if (st == CUSPARSE_STATUS_SUCCESS) {
        printf("  Default SPLIT_K value: %d\n", current_split_k);
      } else {
        printf("  GetAttribute(SPLIT_K) FAILED (%d) - attribute may not exist\n", (int)st);
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }

  // === 清理 ===
  printf("\n[Step 7] Cleanup...\n");
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);
  printf("  Done\n");

  printf("\n========================================\n");
  printf("Debug completed successfully!\n");
  printf("========================================\n");

  return 0;
}
