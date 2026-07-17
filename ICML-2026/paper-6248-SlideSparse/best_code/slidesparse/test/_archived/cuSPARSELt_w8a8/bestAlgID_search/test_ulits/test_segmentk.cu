// test_segmentk.cu - 专门测试 split_k=-1 (Segment-K)
// 每一步都有 printf，用子进程超时来定位卡住位置

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstdio>

const char* st2s(cusparseStatus_t st) {
  switch (st) {
    case CUSPARSE_STATUS_SUCCESS: return "SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUSPARSE_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "UNKNOWN";
  }
}

// 测试指定步骤，返回能走到第几步
// step: 0=init_only, 1=alg_sel, 2=set_alg_id, 3=set_splitk, 4=plan_init
int test_step(int max_step, int N, int K, int M, int alg_id) {
  printf(">>> Testing up to step %d (N=%d, K=%d, M=%d, alg_id=%d)\n", max_step, N, K, M, alg_id);
  fflush(stdout);
  
  cusparseStatus_t st;
  
  // Step 0: Init handle
  printf("  [Step 0] cusparseLtInit... ");
  fflush(stdout);
  cusparseLtHandle_t handle;
  st = cusparseLtInit(&handle);
  printf("%s\n", st2s(st));
  fflush(stdout);
  if (st != CUSPARSE_STATUS_SUCCESS) return 0;
  if (max_step == 0) { cusparseLtDestroy(&handle); return 0; }
  
  // 创建矩阵描述符
  printf("  [Step 0.5] Creating matrix descriptors... ");
  fflush(stdout);
  unsigned alignment = 16;
  cusparseLtMatDescriptor_t matA, matB, matC;
  st = cusparseLtStructuredDescriptorInit(&handle, &matA, K, N, K, alignment,
                                          CUDA_R_8I, CUSPARSE_ORDER_COL,
                                          CUSPARSELT_SPARSITY_50_PERCENT);
  if (st != CUSPARSE_STATUS_SUCCESS) { printf("matA FAILED: %s\n", st2s(st)); cusparseLtDestroy(&handle); return 0; }
  
  st = cusparseLtDenseDescriptorInit(&handle, &matB, K, M, K, alignment,
                                     CUDA_R_8I, CUSPARSE_ORDER_COL);
  if (st != CUSPARSE_STATUS_SUCCESS) { printf("matB FAILED: %s\n", st2s(st)); cusparseLtDestroy(&handle); return 0; }
  
  st = cusparseLtDenseDescriptorInit(&handle, &matC, N, M, N, alignment,
                                     CUDA_R_32I, CUSPARSE_ORDER_COL);
  if (st != CUSPARSE_STATUS_SUCCESS) { printf("matC FAILED: %s\n", st2s(st)); cusparseLtDestroy(&handle); return 0; }
  
  cusparseLtMatmulDescriptor_t matmul;
  st = cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                      CUSPARSE_OPERATION_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &matA, &matB, &matC, &matC,
                                      CUSPARSE_COMPUTE_32I);
  if (st != CUSPARSE_STATUS_SUCCESS) { printf("matmul FAILED: %s\n", st2s(st)); cusparseLtDestroy(&handle); return 0; }
  printf("OK\n");
  fflush(stdout);
  
  // Step 1: AlgSelectionInit
  printf("  [Step 1] cusparseLtMatmulAlgSelectionInit... ");
  fflush(stdout);
  cusparseLtMatmulAlgSelection_t sel;
  st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
  printf("%s\n", st2s(st));
  fflush(stdout);
  if (st != CUSPARSE_STATUS_SUCCESS) { cusparseLtDestroy(&handle); return 1; }
  if (max_step == 1) { cusparseLtMatmulAlgSelectionDestroy(&sel); cusparseLtDestroy(&handle); return 1; }
  
  // Step 2: SetAttribute ALG_CONFIG_ID
  printf("  [Step 2] SetAttribute(ALG_CONFIG_ID=%d)... ", alg_id);
  fflush(stdout);
  st = cusparseLtMatmulAlgSetAttribute(&handle, &sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                       &alg_id, sizeof(alg_id));
  printf("%s\n", st2s(st));
  fflush(stdout);
  if (st != CUSPARSE_STATUS_SUCCESS) { cusparseLtMatmulAlgSelectionDestroy(&sel); cusparseLtDestroy(&handle); return 2; }
  if (max_step == 2) { cusparseLtMatmulAlgSelectionDestroy(&sel); cusparseLtDestroy(&handle); return 2; }
  
  // Step 3: SetAttribute SPLIT_K=-1
  printf("  [Step 3] SetAttribute(SPLIT_K=-1)... ");
  fflush(stdout);
  int split_k = -1;
  st = cusparseLtMatmulAlgSetAttribute(&handle, &sel, CUSPARSELT_MATMUL_SPLIT_K,
                                       &split_k, sizeof(split_k));
  printf("%s\n", st2s(st));
  fflush(stdout);
  if (st != CUSPARSE_STATUS_SUCCESS) { cusparseLtMatmulAlgSelectionDestroy(&sel); cusparseLtDestroy(&handle); return 3; }
  if (max_step == 3) { cusparseLtMatmulAlgSelectionDestroy(&sel); cusparseLtDestroy(&handle); return 3; }
  
  // Step 4: PlanInit
  printf("  [Step 4] cusparseLtMatmulPlanInit... ");
  fflush(stdout);
  cusparseLtMatmulPlan_t plan;
  st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
  printf("%s\n", st2s(st));
  fflush(stdout);
  
  if (st == CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatmulPlanDestroy(&plan);
  }
  cusparseLtMatmulAlgSelectionDestroy(&sel);
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);
  
  printf(">>> Completed all steps!\n");
  fflush(stdout);
  return 4;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_step", &test_step, "Test Segment-K step by step",
        py::arg("max_step"), py::arg("N") = 64, py::arg("K") = 64, 
        py::arg("M") = 32, py::arg("alg_id") = 0);
}
