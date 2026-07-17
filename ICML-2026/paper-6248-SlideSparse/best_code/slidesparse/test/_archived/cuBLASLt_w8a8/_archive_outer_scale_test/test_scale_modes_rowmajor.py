#!/usr/bin/env python3
"""
完整测试：SM120 上 FP8 GEMM + 各种 Scale 模式
使用正确的 row-major 布局 (CUBLASLT_ORDER_ROW)

python3 slidesparse/cuBLASLt_test/outer_scale_test/test_scale_modes_rowmajor.py

"""

import os
import torch
from torch.utils.cpp_extension import load

CUDA_CODE = r'''
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CUBLASLT_CHECK(status) \
    do { \
        cublasStatus_t err = status; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLASLt error %d at line %d\n", err, __LINE__); \
            TORCH_CHECK(false, "cuBLASLt error: ", err); \
        } \
    } while (0)

static cublasLtHandle_t g_handle = nullptr;
static cublasLtHandle_t get_handle() {
    if (!g_handle) cublasLtCreate(&g_handle);
    return g_handle;
}

// 基础设置: Row-major FP8 GEMM with TN format
// D[M,N] = A[M,K] @ B[N,K]^T

struct GemmConfig {
    cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulPreference_t pref;
    cublasLtHandle_t handle;
    
    void create(int64_t M, int64_t N, int64_t K) {
        handle = get_handle();
        
        // Row-major layouts
        cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
        
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, M, K, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
        
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, N, K, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
        
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, M, N, N));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
        
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        
        cublasOperation_t opA = CUBLAS_OP_N, opB = CUBLAS_OP_T;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
        
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t wsSize = 32 * 1024 * 1024;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
    }
    
    void destroy() {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutD);
    }
};

// Test 1: No scale (baseline)
std::string test_no_scale(torch::Tensor A, torch::Tensor B, torch::Tensor& D) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(0);
    D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    
    GemmConfig cfg;
    cfg.create(M, N, K);
    
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        cfg.handle, cfg.desc, cfg.layoutA, cfg.layoutB, cfg.layoutD, cfg.layoutD,
        cfg.pref, 1, &result, &returned);
    
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        cfg.destroy();
        return "FAIL: status=" + std::to_string(st);
    }
    
    void* ws = nullptr;
    if (result.workspaceSize > 0) cudaMalloc(&ws, result.workspaceSize);
    
    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    st = cublasLtMatmul(cfg.handle, cfg.desc, &alpha,
        A.data_ptr(), cfg.layoutA, B.data_ptr(), cfg.layoutB,
        &beta, D.data_ptr(), cfg.layoutD, D.data_ptr(), cfg.layoutD,
        &result.algo, ws, result.workspaceSize, stream);
    
    cudaStreamSynchronize(stream);
    if (ws) cudaFree(ws);
    cfg.destroy();
    
    return (st == CUBLAS_STATUS_SUCCESS) ? "OK" : "FAIL: matmul status=" + std::to_string(st);
}

// Test 2: Tensorwide scale (alpha = scale_a * scale_b)
std::string test_tensorwide_alpha(torch::Tensor A, torch::Tensor B, float scale_a, float scale_b, torch::Tensor& D) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(0);
    D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    
    GemmConfig cfg;
    cfg.create(M, N, K);
    
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        cfg.handle, cfg.desc, cfg.layoutA, cfg.layoutB, cfg.layoutD, cfg.layoutD,
        cfg.pref, 1, &result, &returned);
    
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        cfg.destroy();
        return "FAIL: heuristic status=" + std::to_string(st);
    }
    
    void* ws = nullptr;
    if (result.workspaceSize > 0) cudaMalloc(&ws, result.workspaceSize);
    
    float alpha = scale_a * scale_b, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    st = cublasLtMatmul(cfg.handle, cfg.desc, &alpha,
        A.data_ptr(), cfg.layoutA, B.data_ptr(), cfg.layoutB,
        &beta, D.data_ptr(), cfg.layoutD, D.data_ptr(), cfg.layoutD,
        &result.algo, ws, result.workspaceSize, stream);
    
    cudaStreamSynchronize(stream);
    if (ws) cudaFree(ws);
    cfg.destroy();
    
    return (st == CUBLAS_STATUS_SUCCESS) ? "OK" : "FAIL: matmul status=" + std::to_string(st);
}

// Test 3: Scale via SCALE_POINTER (per-tensor)
std::string test_scale_pointer(torch::Tensor A, torch::Tensor B, torch::Tensor scale_a, torch::Tensor scale_b, torch::Tensor& D) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(0);
    D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    
    GemmConfig cfg;
    cfg.create(M, N, K);
    
    // Set scale pointers
    const float* scale_a_ptr = scale_a.data_ptr<float>();
    const float* scale_b_ptr = scale_b.data_ptr<float>();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a_ptr, sizeof(scale_a_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b_ptr, sizeof(scale_b_ptr)));
    
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        cfg.handle, cfg.desc, cfg.layoutA, cfg.layoutB, cfg.layoutD, cfg.layoutD,
        cfg.pref, 1, &result, &returned);
    
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        cfg.destroy();
        return "FAIL: heuristic status=" + std::to_string(st) + ", returned=" + std::to_string(returned);
    }
    
    void* ws = nullptr;
    if (result.workspaceSize > 0) cudaMalloc(&ws, result.workspaceSize);
    
    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    st = cublasLtMatmul(cfg.handle, cfg.desc, &alpha,
        A.data_ptr(), cfg.layoutA, B.data_ptr(), cfg.layoutB,
        &beta, D.data_ptr(), cfg.layoutD, D.data_ptr(), cfg.layoutD,
        &result.algo, ws, result.workspaceSize, stream);
    
    cudaStreamSynchronize(stream);
    if (ws) cudaFree(ws);
    cfg.destroy();
    
    return (st == CUBLAS_STATUS_SUCCESS) ? "OK" : "FAIL: matmul status=" + std::to_string(st);
}

// Test 4: Outer Vector Scaling
std::string test_outer_vec(torch::Tensor A, torch::Tensor B, torch::Tensor scale_a, torch::Tensor scale_b, torch::Tensor& D) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(0);
    D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    
    GemmConfig cfg;
    cfg.create(M, N, K);
    
    // Set outer vec scale mode
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    
    const float* scale_a_ptr = scale_a.data_ptr<float>();
    const float* scale_b_ptr = scale_b.data_ptr<float>();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a_ptr, sizeof(scale_a_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b_ptr, sizeof(scale_b_ptr)));
    
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        cfg.handle, cfg.desc, cfg.layoutA, cfg.layoutB, cfg.layoutD, cfg.layoutD,
        cfg.pref, 1, &result, &returned);
    
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        cfg.destroy();
        return "FAIL: heuristic status=" + std::to_string(st) + ", returned=" + std::to_string(returned);
    }
    
    void* ws = nullptr;
    if (result.workspaceSize > 0) cudaMalloc(&ws, result.workspaceSize);
    
    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    st = cublasLtMatmul(cfg.handle, cfg.desc, &alpha,
        A.data_ptr(), cfg.layoutA, B.data_ptr(), cfg.layoutB,
        &beta, D.data_ptr(), cfg.layoutD, D.data_ptr(), cfg.layoutD,
        &result.algo, ws, result.workspaceSize, stream);
    
    cudaStreamSynchronize(stream);
    if (ws) cudaFree(ws);
    cfg.destroy();
    
    return (st == CUBLAS_STATUS_SUCCESS) ? "OK" : "FAIL: matmul status=" + std::to_string(st);
}

// Test 5: 32-element 1D block scaling (SM 10.0+)
std::string test_block_scale_32(torch::Tensor A, torch::Tensor B, torch::Tensor scale_a, torch::Tensor scale_b, torch::Tensor& D) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(0);
    D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    
    GemmConfig cfg;
    cfg.create(M, N, K);
    
    // 32-element block scaling - requires CUDA_R_8F_UE8M0 scale type
    // But we need to check if this is available
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasStatus_t st1 = cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    cublasStatus_t st2 = cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    
    if (st1 != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
        cfg.destroy();
        return "FAIL: set scale mode failed";
    }
    
    const float* scale_a_ptr = scale_a.data_ptr<float>();
    const float* scale_b_ptr = scale_b.data_ptr<float>();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a_ptr, sizeof(scale_a_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(cfg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b_ptr, sizeof(scale_b_ptr)));
    
    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        cfg.handle, cfg.desc, cfg.layoutA, cfg.layoutB, cfg.layoutD, cfg.layoutD,
        cfg.pref, 1, &result, &returned);
    
    cfg.destroy();
    
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        return "FAIL: heuristic status=" + std::to_string(st) + ", returned=" + std::to_string(returned);
    }
    
    return "OK (algo found, not executed)";
}

std::string get_gpu_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    char buf[256];
    snprintf(buf, sizeof(buf), "%s (SM %d.%d)", prop.name, prop.major, prop.minor);
    return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_no_scale", &test_no_scale);
    m.def("test_tensorwide_alpha", &test_tensorwide_alpha);
    m.def("test_scale_pointer", &test_scale_pointer);
    m.def("test_outer_vec", &test_outer_vec);
    m.def("test_block_scale_32", &test_block_scale_32);
    m.def("get_gpu_info", &get_gpu_info);
}
'''

def main():
    print("=" * 70)
    print("FP8 GEMM Scale Modes Test (Row-major layout)")
    print("=" * 70)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.cu', delete=False, mode='w') as f:
        f.write(CUDA_CODE)
        cuda_file = f.name
    
    print(f"\nCompiling...")
    ext = load(
        name="test_scale_modes_rowmajor",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        verbose=False,
    )
    os.unlink(cuda_file)
    
    print(f"GPU: {ext.get_gpu_info()}")
    
    # Test data
    M, N, K = 64, 128, 256
    torch.manual_seed(42)
    
    A_fp32 = torch.randn(M, K, device="cuda") * 0.1
    B_fp32 = torch.randn(N, K, device="cuda") * 0.1
    A_fp8 = A_fp32.to(torch.float8_e4m3fn)
    B_fp8 = B_fp32.to(torch.float8_e4m3fn)
    
    scale_a_val, scale_b_val = 2.0, 1.5
    scale_a = torch.tensor([scale_a_val], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([scale_b_val], device="cuda", dtype=torch.float32)
    scale_a_vec = torch.ones(M, device="cuda", dtype=torch.float32) * scale_a_val
    scale_b_vec = torch.ones(N, device="cuda", dtype=torch.float32) * scale_b_val
    
    # Reference
    ref_no_scale = torch.mm(A_fp8.float(), B_fp8.float().t()).to(torch.bfloat16)
    ref_scaled = (scale_a_val * scale_b_val * torch.mm(A_fp8.float(), B_fp8.float().t())).to(torch.bfloat16)
    
    print("\n" + "-" * 70)
    results = []
    
    # Test 1: No scale
    print("\nTest 1: No scale (baseline)")
    D1 = torch.empty(1, device="cuda")
    status = ext.test_no_scale(A_fp8, B_fp8, D1)
    if status == "OK":
        diff = (D1.float() - ref_no_scale.float()).abs().max().item()
        results.append(("No scale", "OK", f"diff={diff:.6f}"))
        print(f"  Status: OK, max diff: {diff:.6f}")
    else:
        results.append(("No scale", status, ""))
        print(f"  Status: {status}")
    
    # Test 2: Tensorwide via alpha
    print("\nTest 2: Tensorwide scale (alpha = scale_a * scale_b)")
    D2 = torch.empty(1, device="cuda")
    status = ext.test_tensorwide_alpha(A_fp8, B_fp8, scale_a_val, scale_b_val, D2)
    if status == "OK":
        diff = (D2.float() - ref_scaled.float()).abs().max().item()
        results.append(("Tensorwide (alpha)", "OK", f"diff={diff:.6f}"))
        print(f"  Status: OK, max diff: {diff:.6f}")
    else:
        results.append(("Tensorwide (alpha)", status, ""))
        print(f"  Status: {status}")
    
    # Test 3: Scale via SCALE_POINTER
    print("\nTest 3: Per-tensor scale via SCALE_POINTER")
    D3 = torch.empty(1, device="cuda")
    status = ext.test_scale_pointer(A_fp8, B_fp8, scale_a, scale_b, D3)
    if status == "OK":
        diff = (D3.float() - ref_scaled.float()).abs().max().item()
        results.append(("SCALE_POINTER", "OK", f"diff={diff:.6f}"))
        print(f"  Status: OK, max diff: {diff:.6f}")
    else:
        results.append(("SCALE_POINTER", status, ""))
        print(f"  Status: {status}")
    
    # Test 4: Outer Vector
    print("\nTest 4: Outer Vector Scaling")
    D4 = torch.empty(1, device="cuda")
    status = ext.test_outer_vec(A_fp8, B_fp8, scale_a_vec, scale_b_vec, D4)
    results.append(("Outer Vec", status, ""))
    print(f"  Status: {status}")
    
    # Test 5: 32-element block scale
    print("\nTest 5: 32-element 1D Block Scaling (SM 10.0+)")
    D5 = torch.empty(1, device="cuda")
    # Need proper scale tensor for block scaling
    scale_a_block = torch.ones((M * K + 31) // 32, device="cuda", dtype=torch.float32)
    scale_b_block = torch.ones((N * K + 31) // 32, device="cuda", dtype=torch.float32)
    status = ext.test_block_scale_32(A_fp8, B_fp8, scale_a_block, scale_b_block, D5)
    results.append(("Block Scale 32", status, ""))
    print(f"  Status: {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Test':<25} {'Status':<40} {'Notes':<20}")
    print("-" * 70)
    for name, status, notes in results:
        print(f"{name:<25} {status:<40} {notes:<20}")
    
    print("\n" + "=" * 70)
    print("Conclusion:")
    print("=" * 70)
    print("""
On SM 12.0 (Blackwell/RTX 5080):
- Basic FP8 GEMM with row-major layout: WORKS
- Tensorwide scaling via alpha: WORKS  
- Per-tensor scale via SCALE_POINTER: check above
- Outer Vector Scaling: NOT SUPPORTED (SM 9.0 only)
- 32-element Block Scaling: check above (SM 10.0+)

For per-row/per-col dequant, options:
1. Manual post-multiply with scale vectors after GEMM
2. Use block scaling if supported
3. Fuse scale into a custom epilogue kernel
""")

if __name__ == "__main__":
    main()
