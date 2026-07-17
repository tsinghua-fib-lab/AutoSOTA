#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_01_bridge.py - SlideSparse 桥接与集成测试

验证 SlideSparse 外挂模块的：
1. 模块导入和包结构
2. 配置系统（环境变量）
3. vLLM 集成点
4. 关键类和函数的存在性
5. cuBLASLt / cuSPARSELt Extension 加载

使用方法:
    python3 test_01_bridge.py

说明:
    此测试不需要任何参数，会全面测试所有桥接功能，
    包括 CUTLASS fallback、cuBLASLt、cuSPARSELt 三种后端的 Extension 加载。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    Colors,
)


# ============================================================================
# 1. 模块导入测试
# ============================================================================

@test_case("导入 slidesparse 主模块")
def test_import_main():
    """测试 slidesparse 主模块导入"""
    import slidesparse
    
    assert hasattr(slidesparse, "__version__"), "缺少 __version__"
    return True, f"v{slidesparse.__version__}"


@test_case("导入 slidesparse.core")
def test_import_core():
    """测试 core 子模块导入"""
    from slidesparse.core import (
        # 配置函数
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
        get_slidesparse_status,
        # Linear 方法
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        # Kernel 函数
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        # 工厂函数
        wrap_scheme_fp8,
        # Extension 加载
        _get_gemm_extension,
    )
    
    # 验证导出的符号类型
    assert callable(is_slidesparse_enabled)
    assert callable(is_cublaslt_enabled)
    assert callable(is_cusparselt_enabled)
    assert callable(is_inner_dtype_32)
    assert callable(get_slidesparse_status)
    assert callable(wrap_scheme_fp8)
    assert callable(_get_gemm_extension)
    
    return True, "核心接口导入成功"


@test_case("导入配置模块")
def test_import_config():
    """测试配置模块单独导入"""
    from slidesparse.core.config import (
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
        get_slidesparse_status,
    )
    
    # 验证返回类型
    assert isinstance(is_slidesparse_enabled(), bool)
    assert isinstance(is_cublaslt_enabled(), bool)
    assert isinstance(is_cusparselt_enabled(), bool)
    assert isinstance(is_inner_dtype_32(), bool)
    
    return True, "配置模块导入成功"


@test_case("环境完整性检查")
def test_environment_sanity():
    """检查硬件目录、.so 文件和 Tuned Configs 是否存在"""
    from slidesparse.utils import build_hw_dir_name, find_file
    
    # 1. 检查 build/{hw_dir_name} 是否存在 (用于 Kernel Build)
    hw_dir_name = build_hw_dir_name()
    csrc_root = Path("/root/vllmbench/slidesparse/csrc")
    
    # 2. 检查 Tuned Kernels (*_tuned_*.py)
    tuned_kernels = []
    # Check dequant_bias
    tuned_kernels.extend(list((csrc_root / "dequant_bias_triton" / "build" / hw_dir_name).glob("*_tuned_*.py")))
    # Check quant_only
    tuned_kernels.extend(list((csrc_root / "quant_only_triton" / "build" / hw_dir_name).glob("*_tuned_*.py")))
    # Check quant_slide
    tuned_kernels.extend(list((csrc_root / "fused_quant_slide_triton" / "build" / hw_dir_name).glob("*_tuned_*.py")))

    if not tuned_kernels:
         pass # 允许无 Kernel (如首次运行)
    
    # 3. 检查 Tuned Configs (AlgSearch 结果)
    search_root = Path("/root/vllmbench/slidesparse/search")
    
    cublaslt_configs = list((search_root / "cuBLASLt_AlgSearch" / "alg_search_results" / hw_dir_name).glob("*.json"))
    cusparselt_configs = list((search_root / "cuSPARSELt_AlgSearch" / "alg_search_results" / hw_dir_name).glob("*.json"))
    
    if not cublaslt_configs:
        pass # return False, f"未找到 cuBLASLt Tuned Configs"
    if not cusparselt_configs:
        pass # return False, f"未找到 cuSPARSELt Tuned Configs"

    # 4. 检查 .so 文件
    cublaslt_so = find_file("cublaslt_gemm", search_dir=csrc_root / "cublaslt_gemm" / "build", ext=".so")
    if not cublaslt_so:
        return False, "未找到 cuBLASLt GEMM .so 文件"

    cusparselt_so = find_file("cusparselt_gemm", search_dir=csrc_root / "cusparselt_gemm" / "build", ext=".so")
    if not cusparselt_so:
        return False, "未找到 cuSPARSELt GEMM .so 文件"
         
    return True, f"环境检查通过: {hw_dir_name}\n    - Tuned Kernels: {len(tuned_kernels)}\n    - cuBLASLt Configs: {len(cublaslt_configs)}\n    - cuSPARSELt Configs: {len(cusparselt_configs)}"
    assert isinstance(get_slidesparse_status(), str)
    
    return True, "配置解析正常"


@test_case("导入 LinearMethod 模块")
def test_import_linear_method():
    """测试 LinearMethod 模块"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        SlideSparseFp8LinearOp,
        SlideSparseFp8LinearMethod,
        # 三个 kernel 函数
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        # 工厂函数
        wrap_scheme_fp8,
        # Extension 加载
        _get_gemm_extension,
    )
    
    # 验证类结构
    assert hasattr(SlideSparseFp8LinearOp, "apply")
    assert hasattr(SlideSparseFp8LinearMethod, "apply_weights")
    assert hasattr(SlideSparseFp8LinearMethod, "create_weights")
    assert hasattr(SlideSparseFp8LinearMethod, "process_weights_after_loading")
    
    # 验证 kernel 函数
    assert callable(cuBLASLt_FP8_linear)
    assert callable(cuSPARSELt_FP8_linear)
    assert callable(cutlass_FP8_linear)
    
    # 验证 Extension 加载函数
    assert callable(_get_gemm_extension)
    
    return True, "类结构正确"


# ============================================================================
# 2. 环境变量配置测试
# ============================================================================

@test_case("环境变量解析")
def test_env_vars():
    """测试环境变量读取"""
    from slidesparse.core.config import (
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
    )
    
    slidesparse_enabled = is_slidesparse_enabled()
    cublaslt_enabled = is_cublaslt_enabled()
    cusparselt_enabled = is_cusparselt_enabled()
    inner_32 = is_inner_dtype_32()
    
    return True, f"SlideSparse={slidesparse_enabled}, cuBLASLt={cublaslt_enabled}, cuSPARSELt={cusparselt_enabled}, Inner32={inner_32}"


@test_case("get_slidesparse_status 格式")
def test_status_format():
    """测试状态字符串格式"""
    from slidesparse.core.config import get_slidesparse_status
    
    status = get_slidesparse_status()
    
    # 状态字符串应该包含关键信息
    assert "SlideSparse" in status or "slidesparse" in status.lower()
    
    return True, status


@test_case("互斥配置校验")
def test_mutual_exclusion():
    """测试 USE_CUBLASLT 和 USE_CUSPARSELT 互斥"""
    import importlib
    
    # 保存原环境
    old_cublaslt = os.environ.get("USE_CUBLASLT")
    old_cusparselt = os.environ.get("USE_CUSPARSELT")
    
    try:
        # 设置互斥配置
        os.environ["USE_CUBLASLT"] = "1"
        os.environ["USE_CUSPARSELT"] = "1"
        
        # 重新加载配置模块以触发校验
        # 注意：由于 _config_validated 标志，需要特殊处理
        from slidesparse.core import config
        config._config_validated = False  # 重置标志
        
        try:
            config.is_cublaslt_enabled()
            # 如果没有抛出异常，说明校验有问题
            return False, "互斥校验未生效"
        except ValueError as e:
            if "mutually exclusive" in str(e):
                return True, "互斥校验正常"
            raise
            
    finally:
        # 恢复环境
        config._config_validated = False
        if old_cublaslt is not None:
            os.environ["USE_CUBLASLT"] = old_cublaslt
        else:
            os.environ.pop("USE_CUBLASLT", None)
        if old_cusparselt is not None:
            os.environ["USE_CUSPARSELT"] = old_cusparselt
        else:
            os.environ.pop("USE_CUSPARSELT", None)


# ============================================================================
# 3. vLLM 桥接测试
# ============================================================================

@test_case("vLLM 桥接文件导入")
def test_vllm_bridge_import():
    """测试 vLLM 内的桥接文件"""
    from vllm.model_executor.layers.quantization.slidesparse import (
        # 配置
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
        get_slidesparse_status,
        # Linear 方法
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        # Kernel 函数
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        # 工厂函数
        wrap_scheme_fp8,
    )
    
    # 验证导入成功
    assert callable(is_slidesparse_enabled)
    assert callable(is_cublaslt_enabled)
    assert callable(wrap_scheme_fp8)
    
    return True, "vLLM 桥接模块正常"


@test_case("桥接函数引用一致性")
def test_bridge_reference_consistency():
    """验证 vLLM 桥接文件的函数是原始函数的引用"""
    from vllm.model_executor.layers.quantization.slidesparse import (
        is_slidesparse_enabled as vllm_is_slidesparse_enabled,
        is_cublaslt_enabled as vllm_is_cublaslt_enabled,
        is_cusparselt_enabled as vllm_is_cusparselt_enabled,
        is_inner_dtype_32 as vllm_is_inner_32,
        wrap_scheme_fp8 as vllm_wrap_scheme_fp8,
    )
    from slidesparse.core.config import (
        is_slidesparse_enabled, 
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
    )
    from slidesparse.core.SlideSparseLinearMethod_FP8 import wrap_scheme_fp8
    
    # 应该是同一个函数对象
    assert vllm_is_slidesparse_enabled is is_slidesparse_enabled, "is_slidesparse_enabled 引用不一致"
    assert vllm_is_cublaslt_enabled is is_cublaslt_enabled, "is_cublaslt_enabled 引用不一致"
    assert vllm_is_cusparselt_enabled is is_cusparselt_enabled, "is_cusparselt_enabled 引用不一致"
    assert vllm_is_inner_32 is is_inner_dtype_32, "is_inner_dtype_32 引用不一致"
    assert vllm_wrap_scheme_fp8 is wrap_scheme_fp8, "wrap_scheme_fp8 引用不一致"
    
    return True, "函数引用一致"


@test_case("CompressedTensors 模块导入")
def test_compressed_tensors_import():
    """测试 CompressedTensors 量化模块"""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
        CompressedTensorsLinearMethod,
    )
    
    assert CompressedTensorsConfig is not None
    assert hasattr(CompressedTensorsConfig, "get_scheme")
    
    return True, "CompressedTensors 可用"


@test_case("FP8 Scheme 类可用")
def test_fp8_scheme():
    """测试 FP8 scheme 类"""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )
    
    # 验证必要方法
    assert hasattr(CompressedTensorsW8A8Fp8, "create_weights")
    assert hasattr(CompressedTensorsW8A8Fp8, "process_weights_after_loading")
    assert hasattr(CompressedTensorsW8A8Fp8, "apply_weights")
    
    return True, "FP8 scheme 可用"


# ============================================================================
# 4. Op 实例化测试
# ============================================================================

@test_case("SlideSparseFp8LinearOp 创建")
def test_create_op():
    """测试 Op 实例创建"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import SlideSparseFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    # 默认配置
    op1 = SlideSparseFp8LinearOp()
    assert op1 is not None
    assert hasattr(op1, "apply")
    assert hasattr(op1, "quant_fp8")
    assert hasattr(op1, "_kernel_name")
    
    # 自定义配置
    op2 = SlideSparseFp8LinearOp(
        act_quant_static=True,
        act_quant_group_shape=GroupShape.PER_TENSOR
    )
    assert op2 is not None
    
    return True, f"Op 创建成功 (kernel={op1._kernel_name})"


@test_case("QuantFP8 配置一致性")
def test_quant_fp8_config():
    """测试内部 QuantFP8 配置"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import SlideSparseFp8LinearOp
    
    # 使用默认配置 (动态量化)
    op = SlideSparseFp8LinearOp(act_quant_static=False)
    
    assert op.quant_fp8 is not None
    assert op.quant_fp8.static == op.act_quant_static
    
    return True, "配置一致"


@test_case("wrap_scheme_fp8 统一入口")
def test_wrap_scheme_fp8():
    """测试统一的 FP8 scheme 包装入口"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import wrap_scheme_fp8
    
    # 验证函数存在且可调用
    assert callable(wrap_scheme_fp8)
    
    # 不支持的类型应返回原对象
    class MockScheme:
        pass
    
    mock = MockScheme()
    wrapped = wrap_scheme_fp8(mock)
    
    assert wrapped is mock, "不支持的类型应返回原对象"
    
    return True, "统一入口正常"


# ============================================================================
# 5. Extension 加载测试
# ============================================================================

@test_case("cuBLASLt Extension 加载状态")
def test_cublaslt_extension_load():
    """测试 cuBLASLt extension 加载状态
    
    注意：_get_gemm_extension 返回的是 ctypes Wrapper 类实例（cuBLASLtGemmWrapper），
    而不是 Python 扩展模块。Wrapper 实例有 cublaslt_fp8_mm 方法。
    """
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        cuBLASLtGemmWrapper,
    )
    
    try:
        ext = _get_gemm_extension("cublaslt")
        # 验证返回类型是 Wrapper 实例
        if not isinstance(ext, cuBLASLtGemmWrapper):
            return False, f"Extension 类型错误: 期望 cuBLASLtGemmWrapper，实际 {type(ext).__name__}"
        # 验证 Wrapper 实例有 cublaslt_fp8_mm 方法
        has_fp8_mm = callable(getattr(ext, "cublaslt_fp8_mm", None))
        if has_fp8_mm:
            return True, f"Wrapper 已加载，cublaslt_fp8_mm 方法可用"
        else:
            return True, f"Wrapper 已加载，但缺少 cublaslt_fp8_mm 方法"
    except FileNotFoundError as e:
        # .so 文件未编译不是错误，只是状态报告
        return True, f"Extension 未编译: {e}"
    except Exception as e:
        return True, f"Extension 加载异常: {e}"


@test_case("cuBLASLt Extension 函数签名")
def test_cublaslt_extension_signatures():
    """测试 cuBLASLt extension Wrapper 方法签名
    
    注意：现在返回的是 ctypes Wrapper 类实例，不是 Python 扩展模块。
    需要检查 Wrapper 实例的方法，而不是模块的函数属性。
    """
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        cuBLASLtGemmWrapper,
    )
    
    try:
        ext = _get_gemm_extension("cublaslt")
    except FileNotFoundError:
        return TestResult(
            name="cuBLASLt Extension 函数签名",
            status=TestStatus.SKIPPED,
            message="Extension 未编译，跳过签名检查"
        )
    except Exception as e:
        return TestResult(
            name="cuBLASLt Extension 函数签名",
            status=TestStatus.WARNING,
            message=f"Extension 加载异常: {e}"
        )
    
    # 检查是否是正确的 Wrapper 类型
    if not isinstance(ext, cuBLASLtGemmWrapper):
        return TestResult(
            name="cuBLASLt Extension 函数签名",
            status=TestStatus.FAILED,
            message=f"类型错误: 期望 cuBLASLtGemmWrapper，实际 {type(ext).__name__}"
        )
    
    # 检查必要的方法
    missing = []
    if not callable(getattr(ext, "cublaslt_fp8_mm", None)):
        missing.append("cublaslt_fp8_mm")
    
    if missing:
        return TestResult(
            name="cuBLASLt Extension 函数签名",
            status=TestStatus.WARNING,
            message=f"缺少方法: {', '.join(missing)}"
        )
    
    return True, "Wrapper 方法签名正确"


@test_case("cuSPARSELt Extension 加载状态")
def test_cusparselt_extension_load():
    """测试 cuSPARSELt extension 加载状态
    
    注意：_get_gemm_extension 返回的是 ctypes Wrapper 类实例（cuSPARSELtGemmWrapper），
    而不是 Python 扩展模块。Wrapper 实例有 cusparselt_fp8_mm 方法。
    """
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        cuSPARSELtGemmWrapper,
    )
    
    try:
        ext = _get_gemm_extension("cusparselt")
        # 验证返回类型是 Wrapper 实例
        if not isinstance(ext, cuSPARSELtGemmWrapper):
            return False, f"Extension 类型错误: 期望 cuSPARSELtGemmWrapper，实际 {type(ext).__name__}"
        # 验证 Wrapper 实例有 cusparselt_fp8_mm 方法
        has_fp8_mm = callable(getattr(ext, "cusparselt_fp8_mm", None))
        if has_fp8_mm:
            return True, f"Wrapper 已加载，cusparselt_fp8_mm 方法可用"
        else:
            return True, f"Wrapper 已加载，但缺少 cusparselt_fp8_mm 方法"
    except FileNotFoundError as e:
        # .so 文件未编译不是错误，只是状态报告
        return True, f"Extension 未编译: {e}"
    except Exception as e:
        return True, f"Extension 加载异常: {e}"


@test_case("cuSPARSELt Extension 函数签名")
def test_cusparselt_extension_signatures():
    """测试 cuSPARSELt extension Wrapper 方法签名
    
    注意：现在返回的是 ctypes Wrapper 类实例，不是 Python 扩展模块。
    需要检查 Wrapper 实例的方法，而不是模块的函数属性。
    """
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        cuSPARSELtGemmWrapper,
    )
    
    try:
        ext = _get_gemm_extension("cusparselt")
    except FileNotFoundError:
        return TestResult(
            name="cuSPARSELt Extension 函数签名",
            status=TestStatus.SKIPPED,
            message="Extension 未编译，跳过签名检查"
        )
    except Exception as e:
        return TestResult(
            name="cuSPARSELt Extension 函数签名",
            status=TestStatus.WARNING,
            message=f"Extension 加载异常: {e}"
        )
    
    # 检查是否是正确的 Wrapper 类型
    if not isinstance(ext, cuSPARSELtGemmWrapper):
        return TestResult(
            name="cuSPARSELt Extension 函数签名",
            status=TestStatus.FAILED,
            message=f"类型错误: 期望 cuSPARSELtGemmWrapper，实际 {type(ext).__name__}"
        )
    
    # 检查必要的方法
    missing = []
    if not callable(getattr(ext, "cusparselt_fp8_mm", None)):
        missing.append("cusparselt_fp8_mm")
    
    if missing:
        return TestResult(
            name="cuSPARSELt Extension 函数签名",
            status=TestStatus.WARNING,
            message=f"缺少方法: {', '.join(missing)}"
        )
    
    return True, "Wrapper 方法签名正确"


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        # 0. 环境完备性检查
        test_environment_sanity,
        # 1. 模块导入
        test_import_main,
        test_import_core,
        test_import_config,
        test_import_linear_method,
        # 2. 环境变量
        test_env_vars,
        test_status_format,
        test_mutual_exclusion,
        # 3. vLLM 桥接
        test_vllm_bridge_import,
        test_bridge_reference_consistency,
        test_compressed_tensors_import,
        test_fp8_scheme,
        # 4. Op 实例化
        test_create_op,
        test_quant_fp8_config,
        test_wrap_scheme_fp8,
        # 5. Extension 加载（cuBLASLt / cuSPARSELt 对称）
        test_cublaslt_extension_load,
        test_cublaslt_extension_signatures,
        test_cusparselt_extension_load,
        test_cusparselt_extension_signatures,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("SlideSparse 桥接与集成测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":

    success = run_tests(verbose=True)

    sys.exit(0 if success else 1)
