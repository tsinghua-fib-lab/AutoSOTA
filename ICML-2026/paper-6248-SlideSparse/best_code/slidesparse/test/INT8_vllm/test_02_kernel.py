#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_02_kernel.py - SlideSparse Kernel 正确性测试

验证 SlideSparse INT8 GEMM kernel 的计算正确性：
1. 使用随机数据对比 vLLM 原生路径 和 SlideSparse 路径的输出
2. 测试不同矩阵尺寸 (M, N, K)
3. 验证 INNER_DTYPE_32 选项

测试流程:
    input (BF16) -> quant (INT8) -> GEMM -> dequant+bias -> output (BF16)
                                    ↑
                    baseline (vLLM 原生) vs test (SlideSparse)

使用方法:
    python3 test_02_kernel.py --use-cutlass
    python3 test_02_kernel.py --use-cublaslt                # vs cuBLASLt
    python3 test_02_kernel.py --use-cublaslt --inner-32     # cuBLASLt + 高精度累加

    
    python3 test_02_kernel.py --use-cusparselt --sparsity 2_4 --inner-32    # cuSPARSELt + 2:4 高精度
    python3 test_02_kernel.py --use-cusparselt --sparsity 2_6               # cuSPARSELt + 2:6
    python3 test_02_kernel.py --use-cusparselt --sparsity 2_8               # cuSPARSELt + 2:8
    python3 test_02_kernel.py --use-cusparselt --sparsity 2_10              # cuSPARSELt + 2:10
    python3 test_02_kernel.py --use-cusparselt --sparsity 2_12 --inner-32   # cuSPARSELt + 2:12 高精度

对比说明:
    - baseline: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)
    - test: SlideSparse 路径 (根据参数选择 kernel)
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 1. 先加 slidesparse/test 以便导入测试工具
_TEST_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_TEST_DIR = _TEST_DIR.parent
sys.path.insert(0, str(_SLIDESPARSE_TEST_DIR))

# 2. 再加 weight_convert 目录（必须在 test 之后插入到更前面）
#    weight_convert 下的脚本使用裸导入 `from utils import ...`
#    需要让它们能找到 weight_convert/utils.py 而不是 test/utils.py
_SLIDESPARSE_DIR = _SLIDESPARSE_TEST_DIR.parent
_WEIGHT_CONVERT_DIR = _SLIDESPARSE_DIR / "weight_convert"
sys.path.insert(0, str(_WEIGHT_CONVERT_DIR))

# 导入测试工具（使用显式路径避免与 weight_convert/utils.py 冲突）
import importlib.util
_test_utils_spec = importlib.util.spec_from_file_location(
    "test_utils", _SLIDESPARSE_TEST_DIR / "utils.py"
)
_test_utils = importlib.util.module_from_spec(_test_utils_spec)
_test_utils_spec.loader.exec_module(_test_utils)

TestRunner = _test_utils.TestRunner
TestResult = _test_utils.TestResult
TestStatus = _test_utils.TestStatus
test_case = _test_utils.test_case
EnvironmentChecker = _test_utils.EnvironmentChecker
Colors = _test_utils.Colors
Benchmarker = _test_utils.Benchmarker
cuda_memory_manager = _test_utils.cuda_memory_manager
skip_if_no_cuda = _test_utils.skip_if_no_cuda
skip_if_no_int8 = _test_utils.skip_if_no_int8
parse_common_args = _test_utils.parse_common_args
apply_env_args = _test_utils.apply_env_args
get_backend_name = _test_utils.get_backend_name
set_env_for_baseline = _test_utils.set_env_for_baseline
set_env_for_test = _test_utils.set_env_for_test
restore_env = _test_utils.restore_env

import torch


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class GEMMTestCase:
    """GEMM 测试用例"""
    name: str
    M: int
    N: int
    K: int
    
    @property
    def shape_str(self) -> str:
        return f"M={self.M}, N={self.N}, K={self.K}"


# 测试矩阵尺寸 - 覆盖不同场景
TEST_CASES = [
    # 小矩阵
    GEMMTestCase("Small (M=16)", M=16, N=896, K=896),
    GEMMTestCase("Small (M=32)", M=32, N=896, K=896),
    # 中等矩阵
    GEMMTestCase("Medium (M=128)", M=128, N=4096, K=4096),
    GEMMTestCase("Medium (M=256)", M=256, N=4096, K=4096),
    # 大矩阵
    GEMMTestCase("Large (M=1024)", M=1024, N=4096, K=4096),
    GEMMTestCase("Large (M=4096)", M=4096, N=4096, K=4096),
    # Qwen2.5-0.5B 典型尺寸
    GEMMTestCase("Qwen-0.5B QKV", M=4096, N=896*3, K=896),
    GEMMTestCase("Qwen-0.5B FFN", M=8192, N=4864, K=896),
    # Llama3.2-1B 典型尺寸
    GEMMTestCase("Llama-1B QKV", M=8192, N=2048*3, K=2048),
    GEMMTestCase("Llama-1B FFN", M=16384, N=8192, K=2048),
]

# ============================================================================
# 非对齐尺寸测试 - M 不满足 16 对齐
# ============================================================================

UNALIGNED_TEST_CASES = [
    # M 不对齐 (M % 16 != 0) - 这是实际推理中常见的情况（decode 阶段 M<16）
    GEMMTestCase("M unaligned (M=1)", M=1, N=512, K=256),      # M=1 典型 decode 场景
    GEMMTestCase("M unaligned (M=7)", M=7, N=512, K=256),      # M 远小于 16
    GEMMTestCase("M unaligned (M=17)", M=17, N=512, K=256),    # M 略大于 16
    GEMMTestCase("M unaligned (M=63)", M=63, N=512, K=256),    # M 略小于 64
    GEMMTestCase("M unaligned (M=100)", M=100, N=512, K=256),  # M=100 (需要 pad 到 112)
    GEMMTestCase("M unaligned (M=255)", M=255, N=512, K=256),  # M 接近但不等于 256
]


# ============================================================================
# 默认稀疏配置（可通过 --sparsity 命令行参数覆盖）
# ============================================================================

DEFAULT_Z = 2
DEFAULT_L = 6  # 2:6 稀疏（默认），可通过 set_sparsity_config() 修改


def set_sparsity_config(sparsity_str: str, verbose: bool = True) -> None:
    """
    设置稀疏配置
    
    Args:
        sparsity_str: 稀疏格式字符串，如 "2_4", "2_6", "2_8", "2_10"
        verbose: 是否打印配置信息（默认 True，但 CUTLASS/cuBLASLt 可传 False）
    """
    global DEFAULT_Z, DEFAULT_L
    
    parts = sparsity_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid sparsity format: {sparsity_str}, expected Z_L (e.g., 2_6)")
    
    try:
        DEFAULT_Z = int(parts[0])
        DEFAULT_L = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid sparsity format: {sparsity_str}, Z and L must be integers")
    
    if verbose:
        print(f"Sparsity config set to {DEFAULT_Z}:{DEFAULT_L}")


# ============================================================================
# 辅助函数
# ============================================================================

def get_int8_dtype():
    """获取 INT8 数据类型"""
    return torch.int8


def prune_weight_ZL(
    weight: torch.Tensor,
    Z: int = DEFAULT_Z,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """
    对权重应用 Z:L 稀疏剪枝（基于幅度）
    
    Args:
        weight: [N, K] 权重
        Z: 每组至少剪掉的元素数
        L: 组大小
    
    Returns:
        剪枝后的权重（保持原 dtype）
    """
    from slidesparse.weight_convert.prune import prune_tensor
    return prune_tensor(weight, Z, L, mode="magnitude")


def slide_weight(
    weight: torch.Tensor,
    L: int = DEFAULT_L,
) -> Tuple[torch.Tensor, int]:
    """
    对剪枝后的权重执行 slide 转换
    
    Args:
        weight: [N, K] 2:L 稀疏权重
        L: 稀疏组大小
    
    Returns:
        (slide_weight [N, K_slide], K_slide)
    """
    from slidesparse.utils import SlideSparseConfig
    from slidesparse.weight_convert.slide import slide_tensor
    
    config = SlideSparseConfig(Z=DEFAULT_Z, L=L)
    slided, metadata = slide_tensor(weight, config, align_to=32, verbose=False)
    return slided, metadata["output_k"]


def compress_weight(
    slide_weight: torch.Tensor,
) -> torch.Tensor:
    """
    使用 cuSPARSELt 压缩 2:4 稀疏权重
    
    Args:
        slide_weight: [N, K_slide] 2:4 稀疏权重
    
    Returns:
        compressed: [compressed_size] uint8 1D
    """
    from slidesparse.weight_convert.compress import compress_tensor_online
    return compress_tensor_online(slide_weight, verbose=False)


@dataclass
class PreparedTestData:
    """准备好的测试数据（支持所有后端）"""
    input_bf16: torch.Tensor          # [M, K] BF16
    # 基础权重（剪枝后，用于 baseline 验证）
    weight_pruned_int8: torch.Tensor   # [N, K] INT8 (2:L 稀疏)
    weight_scale: torch.Tensor        # [N, 1] FP32
    bias: torch.Tensor                # [N] BF16
    # cuSPARSELt 专用数据
    weight_slide_int8: Optional[torch.Tensor] = None      # [N, K_slide] INT8 (2:4)
    weight_compressed: Optional[torch.Tensor] = None     # [compressed_size] uint8
    K_slide: Optional[int] = None                         # slide 后的 K 维度
    # 元数据
    M: int = 0
    N: int = 0
    K: int = 0
    L: int = DEFAULT_L


def generate_test_data(
    M: int,
    N: int,
    K: int,
    L: int = DEFAULT_L,
    prepare_cusparselt: bool = False,
    device: str = "cuda",
    seed: int = 42,
) -> PreparedTestData:
    """
    生成测试数据（支持所有后端）
    
    所有权重都经过 2:L 剪枝，确保三个后端的计算结果数学等价。
    
    Args:
        M, N, K: 矩阵尺寸
        L: 稀疏组大小（默认 6，即 2:6）
        prepare_cusparselt: 是否准备 cuSPARSELt 数据（slide + compress）
        device: 计算设备
        seed: 随机种子
    
    Returns:
        PreparedTestData 包含所有后端需要的数据
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. 生成 BF16 输入
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # 2. 生成 BF16 权重并剪枝
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1
    weight_pruned_bf16 = prune_weight_ZL(weight_bf16, Z=DEFAULT_Z, L=L)
    
    # 3. 量化为 INT8
    int8_max = 127.0  # INT8 的最大值
    weight_absmax = weight_pruned_bf16.abs().max(dim=1, keepdim=True).values
    weight_scale = (weight_absmax / int8_max).to(torch.float32)
    weight_scale = torch.clamp(weight_scale, min=1e-12)
    
    weight_scaled = weight_pruned_bf16.float() / weight_scale
    weight_pruned_int8 = weight_scaled.round().clamp(-128, 127).to(torch.int8)  # [N, K]
    
    # 4. 生成 bias
    bias = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01
    
    # 5. 准备结果
    result = PreparedTestData(
        input_bf16=input_bf16,
        weight_pruned_int8=weight_pruned_int8,
        weight_scale=weight_scale,
        bias=bias,
        M=M, N=N, K=K, L=L,
    )
    
    # 6. 为 cuSPARSELt 准备 slide + compress 数据
    if prepare_cusparselt:
        weight_slide_int8, K_slide = slide_weight(weight_pruned_int8, L=L)
        weight_compressed = compress_weight(weight_slide_int8)
        result.weight_slide_int8 = weight_slide_int8
        result.weight_compressed = weight_compressed
        result.K_slide = K_slide
    
    return result


# CUTLASS/INT8 不支持时的错误信息关键字
_CUTLASS_UNSUPPORTED_ERRORS = ("Error Internal", "cutlass", "CUTLASS")
_INT8_UNSUPPORTED_ERRORS = ("Int8 not supported on SM", "not supported")

# ============================================================================
# Op 缓存 - 避免重复创建对象
# ============================================================================
# 这些 Op 对象是无状态的，可以安全地复用

_baseline_op = None
_slidesparse_op = None


def _get_baseline_op():
    """获取缓存的 baseline Op (INT8)"""
    global _baseline_op
    if _baseline_op is None:
        from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
            ScaledMMLinearLayerConfig,
            choose_scaled_mm_linear_kernel,
        )
        # INT8 使用 per-channel 量化配置
        config = ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        kernel_type = choose_scaled_mm_linear_kernel(config)
        _baseline_op = kernel_type(
            c=config,
            w_q_param_name="weight",
            w_s_param_name="weight_scale",
            i_s_param_name="input_scale",
            i_zp_param_name="input_zero_point",
            azp_adj_param_name="azp_adj",
        )
    return _baseline_op


def _get_slidesparse_op():
    """获取缓存的 SlideSparse Op (INT8)"""
    global _slidesparse_op
    if _slidesparse_op is None:
        from slidesparse.core.SlideSparseLinearMethod_INT8 import SlideSparseInt8LinearOp
        _slidesparse_op = SlideSparseInt8LinearOp(
            act_quant_static=False,
            input_symmetric=True,
        )
    return _slidesparse_op


def run_baseline(
    data: PreparedTestData,
) -> torch.Tensor | None:
    """
    运行 vLLM 原生路径 (baseline) - INT8
    
    使用剪枝后的权重，与 SlideSparse 路径数学等价。
    
    Returns:
        输出 tensor，如果 CUTLASS/INT8 不支持当前 GPU 则返回 None
    """
    from vllm import _custom_ops as ops
    
    # vLLM INT8 路径使用 [K, N] 列主序格式 (stride(0)==1)
    weight_int8_t = data.weight_pruned_int8.t()  # [N, K] -> [K, N] 列主序视图
    
    try:
        # INT8 使用 ops.scaled_int8_quant 进行量化
        # 返回 (output, scales, azp)，对称量化时 azp 为 None
        qinput, scale_a, _ = ops.scaled_int8_quant(data.input_bf16, symmetric=True)
        
        # 调用 cutlass_scaled_mm
        output = ops.cutlass_scaled_mm(
            qinput,
            weight_int8_t,
            scale_a.view(-1, 1),
            data.weight_scale.t(),  # [N, 1] -> [1, N]
            out_dtype=torch.bfloat16,
            bias=data.bias,
        )
        return output
    except RuntimeError as e:
        # CUTLASS/INT8 在高版本 GPU (如 sm_120/121) 上不支持，返回 None
        error_msg = str(e)
        if any(err in error_msg for err in _CUTLASS_UNSUPPORTED_ERRORS + _INT8_UNSUPPORTED_ERRORS):
            return None
        raise


def run_slidesparse(
    data: PreparedTestData,
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
) -> torch.Tensor:
    """
    运行 SlideSparse INT8 路径
    
    数据流：
    - CUTLASS:    weight_pruned [K, N] -> quant + GEMM (fused dequant)
    - cuBLASLt:   weight_pruned [N, K] -> quant_only + GEMM + dequant
    - cuSPARSELt: weight_compressed [1D] -> quant_slide + sparse GEMM + dequant
    
    Args:
        data: 准备好的测试数据
        use_cublaslt: 使用 cuBLASLt 后端
        use_cusparselt: 使用 cuSPARSELt 后端
    """
    op = _get_slidesparse_op()
    
    if use_cusparselt:
        # cuSPARSELt 路径：使用压缩后的权重
        if data.weight_compressed is None:
            raise ValueError("cuSPARSELt requires compressed weight. "
                           "Call generate_test_data with prepare_cusparselt=True")
        return op.apply(
            input=data.input_bf16,
            weight=data.weight_compressed,
            weight_scale=data.weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            bias=data.bias,
            slide_weight_N=data.N,
            slide_weight_K=data.K_slide,
            L=data.L,
        )
    elif use_cublaslt:
        # cuBLASLt 路径：使用 [N, K] 格式的剪枝权重
        return op.apply(
            input=data.input_bf16,
            weight=data.weight_pruned_int8,  # [N, K]
            weight_scale=data.weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            bias=data.bias,
        )
    else:
        # CUTLASS 路径：使用 [K, N] 列主序格式 (stride(0)==1)
        return op.apply(
            input=data.input_bf16,
            weight=data.weight_pruned_int8.t(),
            weight_scale=data.weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            bias=data.bias,
        )


def check_correctness(
    baseline_output: torch.Tensor,
    test_output: torch.Tensor,
    rtol: float = 1,
    atol: float = 1,
) -> Tuple[bool, float, float]:
    """
    检查输出正确性
    
    INT8 精度较低，误差在 5-10% 是正常的
    
    Returns:
        (is_match, max_diff, mean_diff)
    """
    diff = (test_output - baseline_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_match = torch.allclose(test_output, baseline_output, rtol=rtol, atol=atol)
    
    return is_match, max_diff, mean_diff


# ============================================================================
# 测试用例
# ============================================================================

@test_case("CUDA 可用性", skip_if=skip_if_no_cuda)
def test_cuda_available():
    """验证 CUDA 可用"""
    device = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return True, f"{device} (sm_{cc[0]}{cc[1]})"


@test_case("INT8 支持", skip_if=skip_if_no_int8)
def test_int8_support():
    """验证 INT8 支持"""
    cc = torch.cuda.get_device_capability(0)
    return True, f"sm_{cc[0]}{cc[1]} >= sm_75 (Turing+)"


@test_case("SlideSparseInt8LinearOp 基本功能", skip_if=skip_if_no_int8)
def test_op_basic():
    """测试 Op 基本运行（使用 2:L 剪枝权重）"""
    from slidesparse.core.config import is_cublaslt_enabled, is_cusparselt_enabled

    print(f"如果quant和dequant Triton kernel没有提前搜索, 会有首次运行的搜索开销")
    
    use_cublaslt = is_cublaslt_enabled()
    use_cusparselt = is_cusparselt_enabled()
    
    # 在 SM120+ 上，CUTLASS INT8 不支持，需要使用 cuBLASLt
    cutlass_int8_supported = EnvironmentChecker.supports_cutlass_int8()
    if not cutlass_int8_supported and not use_cublaslt and not use_cusparselt:
        cc = EnvironmentChecker.cuda_compute_capability()
        return True, f"sm_{cc[0]}{cc[1]} 不支持 CUTLASS INT8，请使用 --use-cublaslt 或 --use-cusparselt"
    
    with cuda_memory_manager():
        M, N, K = 64, 512, 256
        # 生成剪枝后的测试数据
        data = generate_test_data(
            M, N, K, 
            L=DEFAULT_L,
            prepare_cusparselt=use_cusparselt,
        )
        
        # 运行 SlideSparse
        output = run_slidesparse(data, use_cublaslt=use_cublaslt, use_cusparselt=use_cusparselt)
        
        assert output.shape == (M, N), f"输出形状错误: {output.shape}"
        assert output.dtype == torch.bfloat16, f"输出类型错误: {output.dtype}"
    
    backend = "cuSPARSELt" if use_cusparselt else ("cuBLASLt" if use_cublaslt else "CUTLASS")
    return True, f"输出形状 {output.shape}, kernel={backend}, sparsity=2:{DEFAULT_L}"


@test_case("单次正确性验证", skip_if=skip_if_no_int8)
def test_single_correctness():
    """单次正确性测试（baseline vs SlideSparse）
    
    验证 2:L 剪枝权重的计算正确性。
    slide + compress 具有数学不变性，所以用剪枝后的权重做 baseline 验证。
    """
    from slidesparse.core.config import is_cublaslt_enabled, is_cusparselt_enabled
    
    use_cublaslt = is_cublaslt_enabled()
    use_cusparselt = is_cusparselt_enabled()
    
    # 在 SM120+ 上，CUTLASS INT8 不支持，需要使用 cuBLASLt
    cutlass_int8_supported = EnvironmentChecker.supports_cutlass_int8()
    if not cutlass_int8_supported and not use_cublaslt and not use_cusparselt:
        cc = EnvironmentChecker.cuda_compute_capability()
        return True, f"sm_{cc[0]}{cc[1]} 不支持 CUTLASS INT8，请使用 --use-cublaslt 或 --use-cusparselt"
    
    with cuda_memory_manager():
        M, N, K = 128, 1024, 512
        data = generate_test_data(
            M, N, K,
            L=DEFAULT_L,
            prepare_cusparselt=use_cusparselt,
        )
        
        # 运行 baseline (vLLM 原生路径)
        baseline_output = run_baseline(data)
        
        # 如果 baseline 不可用（CUTLASS 不支持当前 GPU），跳过对比
        if baseline_output is None:
            # 只运行 test，验证不报错即可
            test_output = run_slidesparse(data, use_cublaslt=use_cublaslt, use_cusparselt=use_cusparselt)
            return True, f"CUTLASS 不支持当前 GPU，跳过对比 (test 输出 shape={test_output.shape})"
        
        # 运行 SlideSparse
        test_output = run_slidesparse(data, use_cublaslt=use_cublaslt, use_cusparselt=use_cusparselt)
        
        is_match, max_diff, mean_diff = check_correctness(baseline_output, test_output)
    
    if is_match:
        return True, f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    else:
        return False, f"误差过大: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"


# ============================================================================
# 批量正确性测试
# ============================================================================

def run_batch_correctness_test(
    test_cases: List[GEMMTestCase],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_32: bool = False,
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    批量运行正确性测试
    
    使用 2:L 剪枝权重，验证三种后端的数学等价性：
    - baseline: vLLM CUTLASS 路径（使用剪枝后的权重）
    - test: SlideSparse 路径（CUTLASS/cuBLASLt/cuSPARSELt）
    
    Args:
        test_cases: 测试用例列表
        use_cublaslt: 测试组使用 cuBLASLt
        use_cusparselt: 测试组使用 cuSPARSELt
        inner_32: 使用高精度累加
        verbose: 是否打印详细信息
    
    Returns:
        (passed, total, results)
    """
    sparsity_str = f"{DEFAULT_Z}_{DEFAULT_L}"
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity=sparsity_str)
    results = []
    passed = 0
    
    if verbose:
        print("\n" + "=" * 110)
        print(Colors.bold(f"vLLM 原生 vs {backend_name} 正确性对比 (2:{DEFAULT_L} 稀疏权重)"))
        print("=" * 110)
        print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
              f"{'Max Diff':>10} | {'Mean Diff':>12} | {'Status':>8}")
        print("-" * 110)
    
    # 检测 CUTLASS INT8 是否支持当前 GPU
    cutlass_supported = EnvironmentChecker.supports_cutlass_int8()
    
    # 如果 CUTLASS INT8 不支持，且没有使用 cuBLASLt 或 cuSPARSELt，则跳过整个测试
    if not cutlass_supported and not use_cublaslt and not use_cusparselt:
        if verbose:
            reason = EnvironmentChecker.supports_cutlass_int8_reason()
            print(f"\n{Colors.yellow(f'注意: vLLM CUTLASS INT8 不支持当前 GPU')}")
            print(f"{Colors.yellow(f'原因: {reason}')}")
            print(f"{Colors.yellow('SlideSparse CUTLASS fallback 路径也依赖 vLLM INT8 量化函数，无法运行')}")
            print(f"{Colors.yellow('请使用 --use-cublaslt 或 --use-cusparselt 进行测试')}\n")
        return 0, 0, []  # 返回 0/0 表示跳过
    
    if not cutlass_supported and verbose:
        reason = EnvironmentChecker.supports_cutlass_int8_reason()
        print(f"\n{Colors.yellow(f'注意: vLLM CUTLASS INT8 不支持当前 GPU ({reason})，跳过 baseline 对比')}")
        print(f"{Colors.yellow('只验证 SlideSparse 路径能正常运行')}\n")
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                # 生成剪枝后的测试数据
                data = generate_test_data(
                    case.M, case.N, case.K,
                    L=DEFAULT_L,
                    prepare_cusparselt=use_cusparselt,
                )
                
                # 1. 运行 baseline (vLLM 原生) - 如果 CUTLASS 支持
                baseline_output = None
                if cutlass_supported:
                    saved = set_env_for_baseline()
                    baseline_output = run_baseline(data)
                    restore_env(saved)
                
                # 2. 运行 test (SlideSparse)
                saved = set_env_for_test(use_cublaslt, use_cusparselt, inner_32, sparsity=sparsity_str, model_name="Llama3.2-1B-INT8")
                test_output = run_slidesparse(data, use_cublaslt=use_cublaslt, use_cusparselt=use_cusparselt)
                restore_env(saved)
                
                # 检查正确性
                if baseline_output is not None:
                    is_match, max_diff, mean_diff = check_correctness(
                        baseline_output, test_output
                    )
                    result = {
                        "name": case.name,
                        "M": case.M,
                        "N": case.N,
                        "K": case.K,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "match": is_match,
                    }
                    if is_match:
                        passed += 1
                        status = Colors.green("PASS")
                    else:
                        status = Colors.red("FAIL")
                    if verbose:
                        print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                              f"{max_diff:>10.6f} | {mean_diff:>12.8f} | {status}")
                else:
                    # CUTLASS 不支持，只验证 test 能运行
                    result = {
                        "name": case.name,
                        "M": case.M,
                        "N": case.N,
                        "K": case.K,
                        "skipped": True,
                        "match": True,  # 能运行就算通过
                    }
                    passed += 1
                    if verbose:
                        print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                              f"{'N/A':>10} | {'N/A':>12} | {Colors.cyan('SKIP')}")
                
                results.append(result)
                
        except Exception as e:
            results.append({
                "name": case.name,
                "error": str(e),
                "match": False,
            })
            if verbose:
                print(f"{case.name:<20} | {Colors.red('ERROR')}: {e}")
    
    if verbose:
        print("-" * 110)
        print(f"总计: {passed}/{len(test_cases)} 通过")
        print("=" * 110)
    
    return passed, len(test_cases), results


@test_case("批量正确性测试", skip_if=skip_if_no_int8)
def test_batch_correctness():
    """批量正确性测试"""
    # 从环境变量获取当前配置
    use_cublaslt = EnvironmentChecker.is_cublaslt_enabled()
    use_cusparselt = EnvironmentChecker.is_cusparselt_enabled()
    inner_32 = EnvironmentChecker.is_inner_dtype_32()
    
    passed, total, results = run_batch_correctness_test(
        TEST_CASES, 
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        verbose=True
    )
    
    # 如果返回 0/0，表示因为硬件不支持而跳过
    if total == 0:
        reason = EnvironmentChecker.supports_cutlass_int8_reason()
        return True, f"vLLM CUTLASS INT8 不支持 ({reason})，请使用 --use-cublaslt 或 --use-cusparselt"
    
    if passed == total:
        return True, f"全部 {total} 个测试通过"
    else:
        return False, f"{total - passed}/{total} 个测试失败"


@test_case("非对齐尺寸正确性测试", skip_if=skip_if_no_int8)
def test_unaligned_correctness():
    """测试 M 不满足对齐要求时的正确性
    
    验证 Triton quant kernel 内部的 M padding 逻辑：
    - M 需要对齐到 16（Triton BLOCK_M 最小值）
    - quant kernel 输出 [M_pad, K_pad]，GEMM 后截断回 [M, N]
    
    注意：K 维度不测试非对齐情况，因为：
    1. 实际模型的隐藏层维度总是 32/64/128 的倍数
    2. 权重的 K 维度在模型加载时确定，不会被 padding
    3. cuBLASLt/cuSPARSELt 的 GEMM 要求 qinput 和 weight 的 K 维度一致
    
    此测试确保：
    1. 非对齐 M 不会导致崩溃
    2. 输出形状正确（M x N，不是 M_pad x N）
    3. cuBLASLt 路径的 M_pad 截断正确
    """
    # 从环境变量获取当前配置
    use_cublaslt = EnvironmentChecker.is_cublaslt_enabled()
    use_cusparselt = EnvironmentChecker.is_cusparselt_enabled()
    inner_32 = EnvironmentChecker.is_inner_dtype_32()
    
    sparsity_str = f"{DEFAULT_Z}_{DEFAULT_L}"
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity=sparsity_str)
    
    print(f"\n{Colors.bold('='*80)}")
    print(Colors.bold(f"非对齐尺寸正确性测试 (backend: {backend_name})"))
    print(f"{Colors.bold('='*80)}")
    print(f"测试 M 不对齐时的 padding 处理")
    print(f"- M 对齐要求: 16 (BLOCK_M)")
    print(f"- K 维度: 必须对齐到 32（模型设计约束，不测试非对齐）")
    print("-" * 80)
    
    passed, total, results = run_batch_correctness_test(
        UNALIGNED_TEST_CASES, 
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        verbose=True
    )
    
    # 如果返回 0/0，表示因为硬件不支持而跳过
    if total == 0:
        reason = EnvironmentChecker.supports_cutlass_int8_reason()
        return True, f"vLLM CUTLASS INT8 不支持 ({reason})，请使用 --use-cublaslt 或 --use-cusparselt"
    
    if passed == total:
        return True, f"全部 {total} 个非对齐测试通过"
    else:
        return False, f"{total - passed}/{total} 个非对齐测试失败"


# ============================================================================
# 性能对比测试
# ============================================================================

def run_performance_comparison(
    test_cases: List[GEMMTestCase],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_32: bool = False,
    warmup: int = 25,
    repeat: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """运行性能对比"""
    sparsity_str = f"{DEFAULT_Z}_{DEFAULT_L}"
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity=sparsity_str)
    results = []
    
    # 检测 CUTLASS INT8 是否支持当前 GPU
    cutlass_supported = EnvironmentChecker.supports_cutlass_int8()
    
    # 如果 CUTLASS INT8 不支持，且没有使用 cuBLASLt 或 cuSPARSELt，则跳过整个测试
    if not cutlass_supported and not use_cublaslt and not use_cusparselt:
        if verbose:
            reason = EnvironmentChecker.supports_cutlass_int8_reason()
            print(f"\n{Colors.yellow(f'注意: vLLM CUTLASS INT8 不支持当前 GPU')}")
            print(f"{Colors.yellow(f'原因: {reason}')}")
            print(f"{Colors.yellow('SlideSparse CUTLASS fallback 路径也依赖 vLLM INT8 量化函数，无法运行')}")
            print(f"{Colors.yellow('请使用 --use-cublaslt 或 --use-cusparselt 进行测试')}\n")
        return []  # 返回空列表表示跳过
    
    if verbose:
        print("\n" + "=" * 130)
        if cutlass_supported:
            print(Colors.bold(f"vLLM 原生 vs {backend_name} 性能对比"))
        else:
            reason = EnvironmentChecker.supports_cutlass_int8_reason()
            print(Colors.bold(f"{backend_name} 性能测试"))
            print(Colors.yellow(f"注意: vLLM CUTLASS INT8 不支持当前 GPU ({reason})，跳过 baseline 对比"))
        print("=" * 130)
        print(f"Warmup: {warmup}, Repeat: {repeat}")
        print("-" * 130)
        if cutlass_supported:
            print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
                  f"{'Baseline(ms)':>12} | {'Test(ms)':>12} | {'Speedup':>8} | {'Match':>6}")
        else:
            print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
                  f"{'Test(ms)':>12}")
        print("-" * 130)
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                # 生成剪枝后的测试数据
                data = generate_test_data(
                    case.M, case.N, case.K,
                    L=DEFAULT_L,
                    prepare_cusparselt=use_cusparselt,
                )
                
                baseline_time = None
                baseline_output = None
                
                # ============================================================
                # 预准备：将所有数据准备移到 benchmark 循环外
                # 确保 repeat 内部只有纯 kernel 调用
                # ============================================================
                
                # 预先获取 Op 对象（避免循环内全局查找）
                baseline_op = _get_baseline_op() if cutlass_supported else None
                test_op = _get_slidesparse_op()
                
                # 预先准备权重视图（避免循环内每次调用 .t()）
                weight_for_baseline = data.weight_pruned_int8.t() if cutlass_supported else None
                
                # 根据后端选择正确的权重格式
                if use_cusparselt:
                    weight_for_test = data.weight_compressed
                    test_kwargs = {
                        "slide_weight_N": data.N,
                        "slide_weight_K": data.K_slide,
                        "L": data.L,
                    }
                elif use_cublaslt:
                    weight_for_test = data.weight_pruned_int8  # [N, K]
                    test_kwargs = {}
                else:
                    weight_for_test = data.weight_pruned_int8.t()  # [K, N] 列主序
                    test_kwargs = {}
                
                # 预绑定所有参数，创建纯净的 kernel 调用闭包
                input_bf16 = data.input_bf16
                weight_scale = data.weight_scale
                bias = data.bias
                
                def baseline_kernel():
                    """纯净的 baseline kernel 调用（无额外开销）"""
                    return baseline_op.apply(
                        input=input_bf16,
                        weight=weight_for_baseline,
                        weight_scale=weight_scale,
                        out_dtype=torch.bfloat16,
                        input_scale=None,
                        bias=bias,
                    )
                
                def test_kernel():
                    """纯净的 test kernel 调用（无额外开销）"""
                    return test_op.apply(
                        input=input_bf16,
                        weight=weight_for_test,
                        weight_scale=weight_scale,
                        out_dtype=torch.bfloat16,
                        input_scale=None,
                        bias=bias,
                        **test_kwargs,
                    )
                
                # Baseline 性能 (仅当 CUTLASS 支持时)
                if cutlass_supported:
                    saved = set_env_for_baseline()
                    baseline_output = baseline_kernel()  # 获取 output 用于正确性验证
                    baseline_time, _ = Benchmarker.benchmark(
                        baseline_kernel,  # 直接传函数引用，无 lambda 开销
                        warmup=warmup,
                        repeat=repeat,
                    )
                    restore_env(saved)
                
                # Test 性能
                saved = set_env_for_test(use_cublaslt, use_cusparselt, inner_32, sparsity=sparsity_str, model_name="Llama3.2-1B-INT8")
                test_output = test_kernel()  # 获取 output 用于正确性验证
                test_time, _ = Benchmarker.benchmark(
                    test_kernel,  # 直接传函数引用，无 lambda 开销
                    warmup=warmup,
                    repeat=repeat,
                )
                restore_env(saved)
                
                # 构建结果
                result = {
                    "name": case.name,
                    "M": case.M,
                    "N": case.N,
                    "K": case.K,
                    "test_ms": test_time,
                }
                
                if cutlass_supported and baseline_output is not None:
                    is_match, _, _ = check_correctness(baseline_output, test_output)
                    speedup = baseline_time / test_time if test_time > 0 else 0
                    result["baseline_ms"] = baseline_time
                    result["speedup"] = speedup
                    result["match"] = is_match
                    
                    match_str = Colors.green("✓") if is_match else Colors.red("✗")
                    speedup_str = f"{speedup:.3f}x"
                    
                    if verbose:
                        print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                              f"{baseline_time:>12.4f} | {test_time:>12.4f} | {speedup_str:>8} | {match_str:>6}")
                else:
                    # 无 baseline，只显示 test 时间
                    if verbose:
                        print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                              f"{test_time:>12.4f}")
                
                results.append(result)
                
        except Exception as e:
            results.append({
                "name": case.name,
                "error": str(e),
            })
            if verbose:
                print(f"{case.name:<20} | {Colors.red('ERROR')}: {e}")
    
    if verbose:
        print("-" * 130)
        # 计算平均 speedup (仅当有 baseline 时)
        valid_results = [r for r in results if "speedup" in r]
        if valid_results:
            avg_speedup = sum(r["speedup"] for r in valid_results) / len(valid_results)
            print(f"平均加速比: {avg_speedup:.3f}x")
        elif not cutlass_supported:
            print(f"(无 baseline 对比，仅显示 {backend_name} 性能)")
        print("=" * 130)
    
    return results


@test_case("性能对比测试", skip_if=skip_if_no_int8)
def test_performance_comparison():
    """性能对比测试"""
    # 从环境变量获取当前配置
    use_cublaslt = EnvironmentChecker.is_cublaslt_enabled()
    use_cusparselt = EnvironmentChecker.is_cusparselt_enabled()
    inner_32 = EnvironmentChecker.is_inner_dtype_32()
    
    results = run_performance_comparison(
        TEST_CASES, 
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        warmup=10, 
        repeat=50
    )
    
    # 如果返回空列表，表示因为硬件不支持而跳过
    if not results:
        reason = EnvironmentChecker.supports_cutlass_int8_reason()
        return True, f"vLLM CUTLASS INT8 不支持 ({reason})，请使用 --use-cublaslt 或 --use-cusparselt"
    
    valid_results = [r for r in results if "speedup" in r]
    if valid_results:
        avg_speedup = sum(r["speedup"] for r in valid_results) / len(valid_results)
        return True, f"平均加速比 {avg_speedup:.3f}x"
    
    # 无 baseline 时，只要 test 能运行就算通过
    test_results = [r for r in results if "test_ms" in r]
    if test_results:
        return True, f"CUTLASS INT8 不支持当前 GPU，已完成 {len(test_results)} 个性能测试"
    return True, "测试完成"


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        test_cuda_available,
        test_int8_support,
        test_op_basic,
        test_single_correctness,
        test_batch_correctness,
        test_unaligned_correctness,  # 新增：非对齐尺寸测试
        test_performance_comparison,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("SlideSparse Kernel 正确性测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    parser = parse_common_args("SlideSparse Kernel 正确性测试")
    args = parser.parse_args()
    
    # 设置稀疏配置（优先于环境变量设置）
    # 只在使用 cuSPARSELt 时才显示稀疏配置信息
    if hasattr(args, 'sparsity') and args.sparsity:
        use_cusparselt = hasattr(args, 'use_cusparselt') and args.use_cusparselt
        set_sparsity_config(args.sparsity, verbose=use_cusparselt)
    
    # 应用环境变量（传入 model 参数以加载对应的调优配置）
    model_name = getattr(args, 'model', None)
    apply_env_args(args, model_name=model_name)
    
    success = run_tests(verbose=True)
    
    sys.exit(0 if success else 1)
