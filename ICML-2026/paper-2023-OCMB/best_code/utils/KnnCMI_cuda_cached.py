# KnnCMI_cuda_cached.py - GPU Accelerated Version with Data Caching
# 优化：缓存 GPU 数据，避免重复传输

import numpy as np
import pandas as pd
from scipy.special import digamma
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError
import math
import weakref

# ==============================================================================
# 全局缓存管理器
# ==============================================================================

class CMICache:
    """CMI 数据缓存管理器 - 避免重复的 CPU→GPU 数据传输"""

    def __init__(self):
        self._cache_key = None      # 用于验证缓存有效性的 key
        self._data_gpu = None       # GPU 上的数据
        self._n = None              # 样本数
        self._p = None              # 变量数
        self._call_count = 0        # 调用计数（用于调试）
        self._hit_count = 0         # 缓存命中计数

    def _cleanup_gpu_memory(self):
        """清理 GPU 内存"""
        try:
            # 释放旧的 GPU 数据
            if self._data_gpu is not None:
                del self._data_gpu
                self._data_gpu = None
            # 强制释放 CUDA 内存池
            cuda.current_context().memory_manager.deallocations.clear()
        except Exception:
            pass

    def get_data_gpu(self, data: pd.DataFrame):
        """获取 GPU 数据，如果缓存有效则复用"""
        # 生成缓存 key：基于 DataFrame 的 id 和形状
        # 注意：我们用 id(data) + shape 作为简单的缓存 key
        # 如果数据内容可能变化但 id 不变，需要更复杂的 key
        cache_key = (id(data), data.shape)

        self._call_count += 1

        if self._cache_key == cache_key and self._data_gpu is not None:
            # 缓存命中
            self._hit_count += 1
            return self._data_gpu, self._n, self._p

        # 缓存未命中，需要传输数据
        n, p = data.shape
        # 兼容 DataFrame 和 ndarray
        if hasattr(data, 'to_numpy'):
            data_np = data.to_numpy(dtype=np.float64)
        else:
            data_np = np.asarray(data, dtype=np.float64)

        # 清理旧缓存以释放 GPU 内存
        self._cleanup_gpu_memory()

        try:
            data_gpu = cuda.to_device(data_np)
        except CudaAPIError as e:
            # GPU 内存分配失败，尝试清理后重试
            self._cleanup_gpu_memory()
            import gc
            gc.collect()
            try:
                data_gpu = cuda.to_device(data_np)
            except CudaAPIError:
                # 如果仍然失败，抛出更友好的错误
                raise RuntimeError(
                    f"GPU 内存分配失败 (数据大小: {n}x{p})。"
                    f"尝试: 1) 减少数据量 2) 重启GPU 3) 使用CPU版本"
                ) from e

        # 更新缓存
        self._cache_key = cache_key
        self._data_gpu = data_gpu
        self._n = n
        self._p = p

        return data_gpu, n, p

    def invalidate(self):
        """手动清除缓存（在数据变化时调用）"""
        self._cleanup_gpu_memory()
        self._cache_key = None
        self._n = None
        self._p = None

    def get_stats(self):
        """获取缓存统计信息"""
        hit_rate = self._hit_count / max(1, self._call_count) * 100
        return {
            'total_calls': self._call_count,
            'cache_hits': self._hit_count,
            'hit_rate': f'{hit_rate:.1f}%'
        }

    def reset_stats(self):
        """重置统计计数"""
        self._call_count = 0
        self._hit_count = 0


# 全局缓存实例
_cmi_cache = CMICache()


def invalidate_cmi_cache():
    """清除 CMI 缓存（在数据变化时调用）"""
    _cmi_cache.invalidate()


def get_cmi_cache_stats():
    """获取 CMI 缓存统计"""
    return _cmi_cache.get_stats()


def reset_cmi_cache_stats():
    """重置缓存统计"""
    _cmi_cache.reset_stats()


# ==============================================================================
# 1. GPU DEVICE FUNCTIONS
# ==============================================================================

@cuda.jit(device=True)
def l_infinity_dist_gpu(point_i_data, point_j_data, variables):
    """在GPU上计算两个点在指定变量子空间上的L-无穷范数距离。"""
    max_dist = 0.0
    for var_idx in variables:
        dist = abs(point_i_data[var_idx] - point_j_data[var_idx])
        if dist > max_dist:
            max_dist = dist
    return max_dist


@cuda.jit(device=True)
def find_knn_distance_gpu(i, n, data, k, variables):
    """对于给定的点 i, 在GPU上找到其第 k 个最近邻的距离 (rho)。"""
    # 维持一个大小为 k+1 的小数组来存储最近的k个邻居的距离
    k_distances = cuda.local.array(shape=11, dtype=np.float64)  # 假定k不超过10

    for idx in range(k + 1):
        k_distances[idx] = np.inf

    point_i_data = data[i]

    # 遍历所有其他点 j，找到与点 i 的距离
    for j in range(n):
        dist = l_infinity_dist_gpu(point_i_data, data[j], variables)

        # 插入排序：将新距离插入到 k_distances 数组中，并保持有序
        if dist < k_distances[k]:
            k_distances[k] = dist
            for l in range(k, 0, -1):
                if k_distances[l] < k_distances[l - 1]:
                    temp = k_distances[l]
                    k_distances[l] = k_distances[l - 1]
                    k_distances[l - 1] = temp
                else:
                    break

    # 找到严格在 rho 距离内的点的数量 (k_tilde)
    rho = k_distances[k]
    k_tilde = 0
    for j in range(n):
        dist = l_infinity_dist_gpu(point_i_data, data[j], variables)
        if dist <= rho:
            k_tilde += 1

    return k_tilde - 1, rho


@cuda.jit(device=True)
def count_neighbors_gpu(i, n, data, rho, variables):
    """在GPU上计算点 i 在半径 rho 内的邻居数量。"""
    count = 0
    point_i_data = data[i]
    for j in range(n):
        if i == j:
            continue
        dist = l_infinity_dist_gpu(point_i_data, data[j], variables)
        if dist <= rho:
            count += 1
    return count


# ==============================================================================
# 2. GPU KERNEL
# ==============================================================================

@cuda.jit
def cmi_kernel(data, x, y, z, k, out_values):
    """CUDA Kernel to compute counts for CMI calculation for each point in parallel."""
    i = cuda.grid(1)
    n = data.shape[0]

    if i >= n:
        return

    xyz_vars = x + y + z
    xz_vars = x + z
    yz_vars = y + z
    z_vars = z

    k_tilde, rho = find_knn_distance_gpu(i, n, data, k, xyz_vars)

    nz = count_neighbors_gpu(i, n, data, rho, z_vars) + 1
    nxz = count_neighbors_gpu(i, n, data, rho, xz_vars) + 1
    nyz = count_neighbors_gpu(i, n, data, rho, yz_vars) + 1

    out_values[i, 0] = k_tilde
    out_values[i, 1] = nz
    out_values[i, 2] = nxz
    out_values[i, 3] = nyz


@cuda.jit
def mi_kernel(data, x, y, k, out_values):
    """CUDA Kernel for Mutual Information (when z is empty)."""
    i = cuda.grid(1)
    n = data.shape[0]

    if i >= n:
        return

    xy_vars = x + y
    x_vars = x
    y_vars = y

    k_tilde, rho = find_knn_distance_gpu(i, n, data, k, xy_vars)

    nx = count_neighbors_gpu(i, n, data, rho, x_vars) + 1
    ny = count_neighbors_gpu(i, n, data, rho, y_vars) + 1

    out_values[i, 0] = k_tilde
    out_values[i, 1] = nx
    out_values[i, 2] = ny


# ==============================================================================
# 3. HOST FUNCTION (带缓存)
# ==============================================================================

def cmi(x, y, z, k, data, discrete_dist=1, minzero=1):
    """
    GPU-accelerated CMI calculation with caching, I(x,y|z).

    优化：
    - 缓存 GPU 数据，避免重复的 CPU→GPU 传输
    - 同一个 IAMB 过程中的多次 CMI 调用只需传输一次数据
    """
    # --- 使用缓存获取 GPU 数据 ---
    data_gpu, n, p = _cmi_cache.get_data_gpu(data)

    # --- 列名转换为索引 ---
    vrbls = [x, y, z]
    for i, lst in enumerate(vrbls):
        if all(isinstance(elem, str) for elem in lst) and len(lst) > 0:
            vrbls[i] = list(data.columns.get_indexer(lst))
    x, y, z = vrbls

    # --- Kernel launch configuration ---
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # --- Execute on GPU ---
    if len(z) > 0:
        out_values_gpu = cuda.device_array((n, 4), dtype=np.float64)
        cmi_kernel[blocks_per_grid, threads_per_block](data_gpu, tuple(x), tuple(y), tuple(z), k, out_values_gpu)
        out_values = out_values_gpu.copy_to_host()

        with np.errstate(divide='ignore', invalid='ignore'):
            psis = digamma(out_values[:, 0]) - digamma(out_values[:, 2]) - digamma(out_values[:, 3]) + digamma(out_values[:, 1])

    else:  # MI calculation
        out_values_gpu = cuda.device_array((n, 3), dtype=np.float64)
        mi_kernel[blocks_per_grid, threads_per_block](data_gpu, tuple(x), tuple(y), k, out_values_gpu)
        out_values = out_values_gpu.copy_to_host()

        with np.errstate(divide='ignore', invalid='ignore'):
            psis = digamma(out_values[:, 0]) + digamma(n) - digamma(out_values[:, 1]) - digamma(out_values[:, 2])

    total_cmi = np.nan_to_num(psis, nan=0.0, posinf=0.0, neginf=0.0).mean()

    if minzero == 1:
        return max(total_cmi, 0)
    else:
        return total_cmi
