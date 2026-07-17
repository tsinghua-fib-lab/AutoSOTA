// splitk_deep_probe.cu - 深度探测 Split-K 行为
// 
// 测试内容：
// 1. CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID 查询
// 2. Split-K 各种值 (1, 2, 4, ..., K)
// 3. CUSPARSELT_MATMUL_SPLIT_K_MODE 枚举
// 4. CUSPARSELT_MATMUL_SPLIT_K_BUFFERS 值
// 5. -1 (Segment-K) 的 SetAttribute 是否真的成功
//
// 运行: python3 splitk_deep_probe.py

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstdio>
#include <vector>

// 辅助：cusparseStatus_t 转字符串
const char* status_to_str(cusparseStatus_t st) {
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

void deep_probe(int N, int K, int M) {
  printf("================================================================\n");
  printf("cuSPARSELt Split-K Deep Probe\n");
  printf("================================================================\n");
  printf("Matrix dims: N=%d, K=%d, M=%d\n\n", N, K, M);
  fflush(stdout);
  
  // GPU 信息
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("[0] GPU Info:\n");
  printf("    Name: %s\n", prop.name);
  printf("    SM: %d.%d\n", prop.major, prop.minor);
  printf("    Expected Split-K behavior (per docs):\n");
  if (prop.major < 9) {
    printf("      - pre-SM9: Split-K range [1, K], 1=disabled\n");
    printf("      - SPLIT_K_BUFFERS: [0, SplitK-1]\n");
  } else {
    printf("      - SM9/10/12: -1 (segment-K) or 1 (disabled)\n");
    printf("      - SPLIT_K_BUFFERS: only 0\n");
  }
  printf("\n");
  fflush(stdout);
  
  // 初始化
  cusparseLtHandle_t handle;
  cusparseStatus_t st = cusparseLtInit(&handle);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR: cusparseLtInit failed: %s\n", status_to_str(st));
    return;
  }
  
  // 创建描述符
  unsigned alignment = 16;
  cusparseLtMatDescriptor_t matA, matB, matC;
  
  st = cusparseLtStructuredDescriptorInit(&handle, &matA, K, N, K, alignment,
                                          CUDA_R_8I, CUSPARSE_ORDER_COL,
                                          CUSPARSELT_SPARSITY_50_PERCENT);
  st = cusparseLtDenseDescriptorInit(&handle, &matB, K, M, K, alignment,
                                     CUDA_R_8I, CUSPARSE_ORDER_COL);
  st = cusparseLtDenseDescriptorInit(&handle, &matC, N, M, N, alignment,
                                     CUDA_R_32I, CUSPARSE_ORDER_COL);
  
  cusparseLtMatmulDescriptor_t matmul;
  st = cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                      CUSPARSE_OPERATION_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &matA, &matB, &matC, &matC,
                                      CUSPARSE_COMPUTE_32I);
  
  // === Test 1: 查询 MAX_ALG_ID ===
  printf("[1] Query CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:\n");
  fflush(stdout);
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      int max_id = -1;
      st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                           &max_id, sizeof(max_id));
      if (st == CUSPARSE_STATUS_SUCCESS) {
        printf("    MAX_ALG_ID = %d\n", max_id);
      } else {
        printf("    GetAttribute failed: %s\n", status_to_str(st));
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }
  printf("\n");
  fflush(stdout);
  
  // === Test 2: 查询默认 SPLIT_K 值 ===
  printf("[2] Query default SPLIT_K value:\n");
  fflush(stdout);
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      int split_k = -999;
      st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &split_k, sizeof(split_k));
      if (st == CUSPARSE_STATUS_SUCCESS) {
        printf("    Default SPLIT_K = %d\n", split_k);
      } else {
        printf("    GetAttribute(SPLIT_K) failed: %s\n", status_to_str(st));
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }
  printf("\n");
  fflush(stdout);
  
  // === Test 3: 查询 SPLIT_K_MODE ===
  printf("[3] Query default SPLIT_K_MODE:\n");
  fflush(stdout);
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      int mode = -999;
      st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                           &mode, sizeof(mode));
      if (st == CUSPARSE_STATUS_SUCCESS) {
        printf("    Default SPLIT_K_MODE = %d\n", mode);
      } else {
        printf("    GetAttribute(SPLIT_K_MODE) failed: %s\n", status_to_str(st));
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }
  printf("\n");
  fflush(stdout);
  
  // === Test 4: 查询 SPLIT_K_BUFFERS ===
  printf("[4] Query default SPLIT_K_BUFFERS:\n");
  fflush(stdout);
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      int buffers = -999;
      st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                           &buffers, sizeof(buffers));
      if (st == CUSPARSE_STATUS_SUCCESS) {
        printf("    Default SPLIT_K_BUFFERS = %d\n", buffers);
      } else {
        printf("    GetAttribute(SPLIT_K_BUFFERS) failed: %s\n", status_to_str(st));
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }
  printf("\n");
  fflush(stdout);
  
  // === Test 5: 测试 SetAttribute(SPLIT_K=-1) 然后 GetAttribute 验证 ===
  printf("[5] Test SetAttribute(SPLIT_K=-1) then GetAttribute:\n");
  fflush(stdout);
  {
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      int split_k = -1;
      st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &split_k, sizeof(split_k));
      printf("    SetAttribute(SPLIT_K=-1): %s\n", status_to_str(st));
      
      if (st == CUSPARSE_STATUS_SUCCESS) {
        // 回读验证
        int readback = -999;
        st = cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                             CUSPARSELT_MATMUL_SPLIT_K,
                                             &readback, sizeof(readback));
        if (st == CUSPARSE_STATUS_SUCCESS) {
          printf("    GetAttribute(SPLIT_K) after set: %d\n", readback);
        } else {
          printf("    GetAttribute failed: %s\n", status_to_str(st));
        }
      }
      cusparseLtMatmulAlgSelectionDestroy(&sel);
    }
  }
  printf("\n");
  fflush(stdout);
  
  // === Test 6: 测试各种 Split-K 值的 SetAttribute + PlanInit ===
  printf("[6] Test Split-K values (SetAttribute + PlanInit):\n");
  printf("    Testing: 1, 2, 4, 8, 16, 32, 64, 128, K=%d, K*2=%d\n\n", K, K*2);
  fflush(stdout);
  
  std::vector<int> test_vals = {1, 2, 4, 8, 16, 32, 64, 128, K, K*2};
  
  for (int split_k : test_vals) {
    printf("    split_k=%d: ", split_k);
    fflush(stdout);
    
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("AlgSelInit=%s\n", status_to_str(st));
      continue;
    }
    
    st = cusparseLtMatmulAlgSetAttribute(&handle, &sel,
                                         CUSPARSELT_MATMUL_SPLIT_K,
                                         &split_k, sizeof(split_k));
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("SetAttr=%s\n", status_to_str(st));
      cusparseLtMatmulAlgSelectionDestroy(&sel);
      continue;
    }
    printf("SetAttr=OK ");
    fflush(stdout);
    
    cusparseLtMatmulPlan_t plan;
    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
    cusparseLtMatmulAlgSelectionDestroy(&sel);
    
    if (st == CUSPARSE_STATUS_SUCCESS) {
      printf("PlanInit=OK\n");
      cusparseLtMatmulPlanDestroy(&plan);
    } else {
      printf("PlanInit=%s\n", status_to_str(st));
    }
    fflush(stdout);
  }
  printf("\n");
  
  // === Test 7: 测试 -1 的 PlanInit（带超时警告）===
  printf("[7] About SPLIT_K=-1 (Segment-K):\n");
  printf("    SM %d.%d detected.\n", prop.major, prop.minor);
  if (prop.major == 9) {
    printf("    This is Hopper, -1 should work.\n");
  } else if (prop.major == 10 || prop.major == 12) {
    printf("    This is SM %d.x (Blackwell). Per docs, -1 should work.\n", prop.major);
    printf("    BUT: Previous test showed PlanInit hangs!\n");
    printf("    This suggests a cuSPARSELt BUG or driver issue.\n");
  } else {
    printf("    This is pre-SM9, -1 is NOT supported.\n");
  }
  printf("\n");
  printf("    SKIPPING -1 PlanInit test to avoid hang.\n");
  printf("\n");
  fflush(stdout);
  
  // === Test 8: 使用 cusparseLtMatmulSearch 让库自动选择 ===
  printf("[8] Test cusparseLtMatmulSearch (auto-tuning):\n");
  fflush(stdout);
  {
    // 分配数据
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, K * N);
    cudaMalloc(&d_B, K * M);
    cudaMalloc(&d_C, N * M * sizeof(int32_t));
    cudaMemset(d_A, 0, K * N);
    cudaMemset(d_B, 0, K * M);
    cudaMemset(d_C, 0, N * M * sizeof(int32_t));
    
    cusparseLtMatmulAlgSelection_t sel;
    st = cusparseLtMatmulAlgSelectionInit(&handle, &sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("    AlgSelectionInit failed: %s\n", status_to_str(st));
      cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    } else {
      cusparseLtMatmulPlan_t plan;
      st = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &sel);
      if (st != CUSPARSE_STATUS_SUCCESS) {
        printf("    PlanInit failed: %s\n", status_to_str(st));
        cusparseLtMatmulAlgSelectionDestroy(&sel);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
      } else {
        // 压缩
        size_t comp_size = 0, comp_buf_size = 0;
        cusparseLtSpMMACompressedSize(&handle, &plan, &comp_size, &comp_buf_size);
        void *d_A_comp, *d_comp_buf = nullptr;
        cudaMalloc(&d_A_comp, comp_size);
        if (comp_buf_size > 0) cudaMalloc(&d_comp_buf, comp_buf_size);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cusparseLtSpMMACompress(&handle, &plan, d_A, d_A_comp, d_comp_buf, stream);
        cudaStreamSynchronize(stream);
        
        // Workspace
        size_t ws_size = 0;
        cusparseLtMatmulGetWorkspace(&handle, &plan, &ws_size);
        void *d_ws = nullptr;
        if (ws_size > 0) cudaMalloc(&d_ws, ws_size);
        
        printf("    Calling cusparseLtMatmulSearch...\n");
        fflush(stdout);
        
        float alpha = 1.0f, beta = 0.0f;
        st = cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                    d_A_comp, d_B, &beta, d_C, d_C,
                                    d_ws, &stream, 1);
        
        if (st == CUSPARSE_STATUS_SUCCESS) {
          printf("    Search completed!\n");
          
          // 读取 Search 选择的参数
          int found_alg_id = -999;
          int found_split_k = -999;
          
          cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                          CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                          &found_alg_id, sizeof(found_alg_id));
          cusparseLtMatmulAlgGetAttribute(&handle, &sel,
                                          CUSPARSELT_MATMUL_SPLIT_K,
                                          &found_split_k, sizeof(found_split_k));
          printf("    Search result: alg_id=%d, split_k=%d\n", found_alg_id, found_split_k);
        } else {
          printf("    Search failed: %s\n", status_to_str(st));
        }
        
        cudaStreamDestroy(stream);
        if (d_ws) cudaFree(d_ws);
        if (d_comp_buf) cudaFree(d_comp_buf);
        cudaFree(d_A_comp);
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&sel);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
      }
    }
  }
  printf("\n");
  fflush(stdout);
  
  // 清理
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtDestroy(&handle);
  
  printf("================================================================\n");
  printf("Done!\n");
  printf("================================================================\n");
  fflush(stdout);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deep_probe", &deep_probe, "Deep probe Split-K behavior",
        py::arg("N") = 64, py::arg("K") = 64, py::arg("M") = 32);
}
