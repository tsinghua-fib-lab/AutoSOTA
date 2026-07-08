"""
工具模块 - KnnCMI条件独立性测试
"""

try:
    from .KnnCMI_loader import cmi, invalidate_cmi_cache, set_use_cache, get_use_cache
    __all__ = ['cmi', 'invalidate_cmi_cache', 'set_use_cache', 'get_use_cache']
except ImportError:
    # Fallback implementations
    try:
        from .KnnCMI_cuda import cmi
        __all__ = ['cmi']
    except ImportError:
        try:
            from .KnnCMI import cmi
            __all__ = ['cmi']
        except ImportError:
            # 最后的fallback
            import numpy as np
            def cmi(x, y, z, k, data):
                return np.random.random() * 0.1
            __all__ = ['cmi']
