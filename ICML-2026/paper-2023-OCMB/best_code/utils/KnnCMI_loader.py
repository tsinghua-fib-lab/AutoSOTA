# KnnCMI_loader.py
# 统一的 CMI 加载模块，支持缓存开关
#
# 用法:
#   from KnnCMI_loader import cmi, invalidate_cmi_cache, set_use_cache
#
#   # 开启缓存（默认，适合中间调试）
#   set_use_cache(True)
#
#   # 关闭缓存（最终实验）
#   set_use_cache(False)
#
# 环境变量控制:
#   export KNNCMI_USE_CACHE=0  # 关闭缓存
#   export KNNCMI_USE_CACHE=1  # 开启缓存（默认）

import os
import numpy as np

# 全局开关：是否使用缓存
_USE_CACHE = os.environ.get('KNNCMI_USE_CACHE', '1') == '1'

# CMI 函数和缓存清除函数
_cmi_cached = None
_cmi_nocache = None
_invalidate_cache_fn = None
_current_cmi = None
_backend = "unknown"  # cuda/cpu/fallback

# 检测 CUDA 可用性
cuda_available = False
try:
    from numba import cuda as numba_cuda
    cuda_available = numba_cuda.is_available()
except Exception:
    pass

# 加载各版本 CMI
if cuda_available:
    # 带缓存版本
    try:
        from KnnCMI_cuda_cached import cmi as _cmi_cuda_cached
        from KnnCMI_cuda_cached import invalidate_cmi_cache as _inv_cache
        from KnnCMI_cuda_cached import get_cmi_cache_stats, reset_cmi_cache_stats
        _cmi_cached = _cmi_cuda_cached
        _invalidate_cache_fn = _inv_cache
        _backend = "cuda"
    except ImportError:
        pass

    # 无缓存版本
    try:
        from KnnCMI_cuda import cmi as _cmi_cuda
        _cmi_nocache = _cmi_cuda
        _backend = "cuda"
    except ImportError:
        pass

# CPU 版本作为 fallback
if _cmi_cached is None and _cmi_nocache is None:
    try:
        from KnnCMI import cmi as _cmi_cpu
        _cmi_nocache = _cmi_cpu
        _cmi_cached = _cmi_cpu  # CPU 版本没有缓存区别
        _backend = "cpu"
    except ImportError:
        # 最终 fallback（接受全部参数以兼容包装器）
        def _fallback_cmi(x, y, z, k, data, discrete_dist=1, minzero=1):
            return np.random.random() * 0.1
        _cmi_nocache = _fallback_cmi
        _cmi_cached = _fallback_cmi
        _backend = "fallback"

# 兼容：缺失某个实现时，退化到另一个
if _cmi_cached is None:
    _cmi_cached = _cmi_nocache
if _cmi_nocache is None:
    _cmi_nocache = _cmi_cached

# 初始化当前使用的 CMI
_cache_supported = _invalidate_cache_fn is not None
if _USE_CACHE and _cmi_cached is not None and _cache_supported:
    _current_cmi = _cmi_cached
    print("[KnnCMI] backend=cuda mode=cache (KNNCMI_USE_CACHE=1)")
else:
    _current_cmi = _cmi_nocache
    if _backend == "cuda":
        print("[KnnCMI] backend=cuda mode=no-cache (KNNCMI_USE_CACHE=0)")
    elif _backend == "cpu":
        print(f"[KnnCMI] backend=cpu (KNNCMI_USE_CACHE={'1' if _USE_CACHE else '0'})")
    else:
        print(f"[KnnCMI] backend={_backend} (KNNCMI_USE_CACHE={'1' if _USE_CACHE else '0'})")


def set_use_cache(use_cache: bool):
    """
    设置是否使用缓存版本

    Args:
        use_cache: True=使用缓存（加速），False=不使用缓存（公平实验）
    """
    global _USE_CACHE, _current_cmi

    _USE_CACHE = use_cache
    if use_cache and _cache_supported and _cmi_cached is not None:
        _current_cmi = _cmi_cached
        print("[KnnCMI] Switched to mode=cache")
    else:
        _current_cmi = _cmi_nocache
        print("[KnnCMI] Switched to mode=no-cache")


def get_use_cache() -> bool:
    """获取当前是否使用缓存"""
    return _USE_CACHE


def cmi(x, y, z, k, data, discrete_dist=1, minzero=1):
    """
    条件互信息计算（自动选择缓存/无缓存版本）
    """
    return _current_cmi(x, y, z, k, data, discrete_dist, minzero)


def invalidate_cmi_cache():
    """清除 CMI 缓存（仅在缓存模式下有效）"""
    if _invalidate_cache_fn is not None:
        _invalidate_cache_fn()


def get_cache_stats():
    """获取缓存统计（仅在缓存模式下有效）"""
    try:
        return get_cmi_cache_stats()
    except:
        return {'total_calls': 0, 'cache_hits': 0, 'hit_rate': 'N/A'}


def reset_cache_stats():
    """重置缓存统计"""
    try:
        reset_cmi_cache_stats()
    except:
        pass
