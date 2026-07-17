// debug_verbose.cu - 带详细输出的 cuSPARSELt 调试版本
// 直接在关键位置添加 printf 来定位卡住的位置
//
// 编译方法（使用 PyTorch 扩展）:
// python3 -c "
// import torch
// from torch.utils.cpp_extension import load
// prop = torch.cuda.get_device_properties(0)
// sm = f'sm_{prop.major}{prop.minor}'
// ext = load('debug_verbose', ['debug_verbose.cu'], 
//            extra_cuda_cflags=['-O0', '-g', f'-arch={sm}'],
//            extra_ldflags=['-lcusparseLt'], verbose=True)
// "
//
// 运行: python3 -c "import debug_verbose; debug_verbose.test_search()"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <cstdio>
#include <vector>

#define CHECK_CUDA_ERR(expr)                                                   \
  do {                                                                         \
    cudaError_t _st = (expr);                                                  \
    if (_st != cudaSuccess) {                                                  \
      printf("[CUDA ERR] %s at line %d\n", cudaGetErrorString(_st), __LINE__); \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// 使用最小尺寸进行测试
void test_search() {
  printf("========================================\n");
  printf("Verbose cuSPARSELt Debug\n");
  printf("========================================\n");
  fflush(stdout);

  constexpr int N = 64;
  constexpr int K = 64;
  constexpr int M = 32;

  printf("[1] Init cuSPARSELt handle...\n");
  fflush(stdout);
  cusparseLtHandle_t handle;
  cusparseStatus_t st = cusparseLtInit(&handle);
  printf("    cusparseLtInit: %d\n", (int)st);
  fflush(stdout);
  if (st != CUSPARSE_STATUS_SUCCESS) return;

  printf("[2] Creating matrix descriptors...\n");
  fflush(stdout);

  unsigned alignment = 16;
  cudaDataType type_AB = CUDA_R_8I;  // INT8
  cudaDataType type_C = CUDA_R_32I;
  cusparseComputeType comp_type = CUSPARSE_COMPUTE_32I;

  cusparseLtMatDescriptor_t matA, matB, matC;

  // TN layout: A[K,N], B[K,M], C[N,M]
  st = cusparseLtStructuredDescriptorInit(&handle, &matA, K, N, K, alignment,
                                          type_AB, CUSPARSE_ORDER_COL,
                                          CUSPARSELT_SPARSITY_50_PERCENT);
  printf("    matA (sparse): %d\n", (int)st);
  fflush(stdout);

  st = cusparseLtDenseDescriptorInit(&handle, &matB, K, M, K, alignment,
                                     type_AB, CUSPARSE_ORDER_COL);
  printf("    matB (dense): %d\n", (int)st);
  fflush(stdout);

  st = cusparseLtDenseDescriptorInit(&handle, &matC, N, M, N, alignment, type_C,
                                     CUSPARSE_ORDER_COL);
  printf("    matC (output): %d\n", (int)st);
  fflush(stdout);

  printf("[3] Creating matmul descriptor...\n");
  fflush(stdout);

  cusparseLtMatmulDescriptor_t matmul;
  st = cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                      CUSPARSE_OPERATION_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &matA, &matB, &matC, &matC, comp_type);
  printf("    matmul: %d\n", (int)st);
  fflush(stdout);

  printf("[4] Probing max alg_id...\n");
  fflush(stdout);

  int max_alg_id = -1;
  int consecutive_failures = 0;
  constexpr int kMaxConsecutiveFailures = 5;
  constexpr int kMaxProbe = 20;  // 限制最大探测数

  for (int probe = 0; probe < kMaxProbe; ++probe) {
    printf("    probe=%d: ", probe);
    fflush(stdout);

    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("AlgSelInit=%d ", (int)st);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) {
        printf("-> STOP (consecutive failures)\n");
        break;
      }
      printf("\n");
      continue;
    }

    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                         &probe, sizeof(probe));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttr(ID)=%d ", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) {
        printf("-> STOP\n");
        break;
      }
      printf("\n");
      continue;
    }

    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);

    if (st == CUSPARSE_STATUS_SUCCESS) {
      printf("PlanInit=OK -> VALID\n");
      max_alg_id = probe;
      consecutive_failures = 0;
      cusparseLtMatmulPlanDestroy(&plan);
    } else {
      printf("PlanInit=%d ", (int)st);
      ++consecutive_failures;
      if (consecutive_failures >= kMaxConsecutiveFailures) {
        printf("-> STOP\n");
        break;
      }
      printf("\n");
    }
    fflush(stdout);
  }

  printf("\n    max_alg_id = %d\n\n", max_alg_id);
  fflush(stdout);

  if (max_alg_id < 0) {
    printf("ERROR: No valid alg_id found!\n");
    cusparseLtDestroy(&handle);
    return;
  }

  printf("[5] Testing Split-K values (alg_id=%d)...\n", max_alg_id);
  fflush(stdout);

  std::vector<int> split_k_vals = {1, 2, 4, 8, 16, 32, 64, -1};

  for (int split_k : split_k_vals) {
    printf("    split_k=%d: ", split_k);
    fflush(stdout);

    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("AlgSelInit=%d\n", (int)st);
      continue;
    }

    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                         &max_alg_id, sizeof(max_alg_id));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttr(ID)=%d\n", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }

    printf("SetAttr(SPLIT_K)...");
    fflush(stdout);

    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel, CUSPARSELT_MATMUL_SPLIT_K,
                                         &split_k, sizeof(split_k));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("=%d\n", (int)st);
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }
    printf("OK ");

    printf("PlanInit...");
    fflush(stdout);

    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);

    if (st == CUSPARSE_STATUS_SUCCESS) {
      printf("OK\n");
      cusparseLtMatmulPlanDestroy(&plan);
    } else {
      printf("=%d\n", (int)st);
    }
    fflush(stdout);
  }

  printf("\n[6] Cleanup...\n");
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);

  printf("\n========================================\n");
  printf("Debug completed!\n");
  printf("========================================\n");
  fflush(stdout);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_search", &test_search, "Verbose debug test");
}
