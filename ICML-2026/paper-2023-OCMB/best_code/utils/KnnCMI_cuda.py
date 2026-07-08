# KnnCMI.py - GPU Accelerated Version using Numba CUDA

import numpy as np
import pandas as pd
from scipy.special import digamma
from numba import cuda
import math

MAX_K = 10  # 受 device local array 限制（见 find_knn_distance_gpu）


# ==============================================================================
# 1. GPU DEVICE FUNCTIONS
# 这些函数将被编译成在GPU设备上运行的代码，只能被其他GPU函数调用。
# 它们是构建我们主内核的"积木"。
# ==============================================================================

@cuda.jit(device=True)
def l_infinity_dist_gpu(point_i_data, point_j_data, variables):
    """
    在GPU上计算两个点在指定变量子空间上的L-无穷范数距离。
    """
    max_dist = 0.0
    for var_idx in variables:
        # 假设数据已经是数值型
        dist = abs(point_i_data[var_idx] - point_j_data[var_idx])
        if dist > max_dist:
            max_dist = dist
    return max_dist


@cuda.jit(device=True)
def find_knn_distance_gpu(i, n, data, k, variables):
    """
    对于给定的点 i, 在GPU上找到其第 k 个最近邻的距离 (rho)。
    这是一个简化的KNN实现，适用于在设备函数内部运行。
    """
    # 维持一个大小为 k+1 的小数组来存储最近的k个邻居的距离
    # +1 是因为包含了点自身
    k_distances = cuda.local.array(shape=11, dtype=np.float64)  # k <= 10

    for idx in range(k + 1):
        k_distances[idx] = np.inf

    point_i_data = data[i]

    # 遍历所有其他点 j，找到与点 i 的距离
    for j in range(n):
        dist = l_infinity_dist_gpu(point_i_data, data[j], variables)

        # 插入排序：将新距离插入到 k_distances 数组中，并保持有序
        if dist < k_distances[k]:
            k_distances[k] = dist
            # 从后向前冒泡，将新距离放到正确的位置
            for l in range(k, 0, -1):
                if k_distances[l] < k_distances[l - 1]:
                    # 交换
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

    return k_tilde - 1, rho  # -1 去掉点自身


@cuda.jit(device=True)
def count_neighbors_gpu(i, n, data, rho, variables):
    """
    在GPU上计算点 i 在半径 rho 内的邻居数量。
    """
    count = 0
    point_i_data = data[i]
    for j in range(n):
        if i == j:  # 不计算自身
            continue
        dist = l_infinity_dist_gpu(point_i_data, data[j], variables)
        if dist <= rho:
            count += 1
    return count


# ==============================================================================
# 2. GPU KERNEL
# 这是将在GPU上并行执行的主函数。
# 我们将启动 n 个线程，每个线程处理一个数据点 i。
# ==============================================================================

@cuda.jit
def cmi_kernel(data, x, y, z, k, out_values):
    """
    CUDA Kernel to compute counts for CMI calculation for each point in parallel.
    out_values is an array of shape (n, 4) to store [k_tilde, nz, nxz, nyz]
    """
    # 获取当前GPU线程的全局唯一ID
    i = cuda.grid(1)
    n = data.shape[0]

    # 确保线程ID在数据范围内
    if i >= n:
        return

    # 定义变量子空间
    xyz_vars = x + y + z
    xz_vars = x + z
    yz_vars = y + z
    z_vars = z

    # 1. 找到点 i 的第 k 个邻居的距离 rho
    k_tilde, rho = find_knn_distance_gpu(i, n, data, k, xyz_vars)

    # 2. 在不同的子空间中计算邻居数量
    nz = count_neighbors_gpu(i, n, data, rho, z_vars) + 1  # +1 包含点自身
    nxz = count_neighbors_gpu(i, n, data, rho, xz_vars) + 1
    nyz = count_neighbors_gpu(i, n, data, rho, yz_vars) + 1

    # 3. 将计算出的四个计数值写回输出数组
    # 我们不在GPU上做digamma，因为它在numba.cuda中不可用。
    # 我们把计数结果传回CPU，在CPU上完成最后计算。
    out_values[i, 0] = k_tilde
    out_values[i, 1] = nz
    out_values[i, 2] = nxz
    out_values[i, 3] = nyz


@cuda.jit
def mi_kernel(data, x, y, k, out_values):
    """
    CUDA Kernel for Mutual Information (when z is empty).
    out_values is an array of shape (n, 3) to store [k_tilde, nx, ny]
    """
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
# 3. HOST FUNCTION
# 这是我们在主程序中调用的CPU函数。它负责管理数据传输和启动GPU内核。
# ==============================================================================

def cmi(x, y, z, k, data, discrete_dist=1, minzero=1):
    """
    GPU-accelerated CMI calculation, I(x,y|z).
    """
    if k is None:
        raise ValueError("k must be a positive integer")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > MAX_K:
        raise ValueError(f"GPU KnnCMI only supports k <= {MAX_K} (got k={k}). "
                         f"Please lower k or use the CPU KnnCMI implementation.")

    # CUDA 不可用时，自动回退到CPU版本，避免脚本直接崩溃
    try:
        cuda_available = cuda.is_available()
    except Exception:
        cuda_available = False
    if not cuda_available:
        try:
            from KnnCMI import cmi as cmi_cpu
        except Exception as err:
            raise RuntimeError("CUDA is not available and CPU fallback (KnnCMI.py) is not importable.") from err
        return cmi_cpu(x, y, z, k, data, discrete_dist=discrete_dist, minzero=minzero)

    # --- Host-side data preparation ---
    n, p = data.shape

    # 将列名转换为索引
    vrbls = [x, y, z]
    for i, lst in enumerate(vrbls):
        if all(isinstance(elem, str) for elem in lst) and len(lst) > 0:
            if not hasattr(data, "columns"):
                raise ValueError("When x/y/z are column names, `data` must be a pandas DataFrame.")
            vrbls[i] = list(data.columns.get_indexer(lst))
    x, y, z = vrbls

    # 将数据转换为Numpy数组，并确保是float32/64以便GPU处理
    if hasattr(data, "to_numpy"):
        data_np = data.to_numpy(dtype=np.float64)
    else:
        data_np = np.asarray(data, dtype=np.float64)

    # 将数据从CPU内存拷贝到GPU显存
    try:
        data_gpu = cuda.to_device(data_np)
    except Exception as err:
        # driver init / memory allocation 等问题时回退CPU，保证可用性
        try:
            from KnnCMI import cmi as cmi_cpu
        except Exception as cpu_err:
            raise RuntimeError("Failed to move data to GPU and CPU fallback is not importable.") from cpu_err
        return cmi_cpu(x, y, z, k, data, discrete_dist=discrete_dist, minzero=minzero)

    # --- Kernel launch configuration ---
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # --- Execute on GPU ---
    if len(z) > 0:
        # 在GPU上创建一个空的输出数组来接收结果
        out_values_gpu = cuda.device_array((n, 4), dtype=np.float64)

        # 启动 CMI 内核！
        cmi_kernel[blocks_per_grid, threads_per_block](data_gpu, tuple(x), tuple(y), tuple(z), k, out_values_gpu)

        # 将结果从GPU显存拷回CPU内存
        out_values = out_values_gpu.copy_to_host()

        # --- Final calculation on CPU ---
        # 使用从GPU获取的计数值，在CPU上完成最后的digamma计算
        # np.errstate 忽略除以0的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            # k_tilde, nz, nxz, nyz
            psis = digamma(out_values[:, 0]) - digamma(out_values[:, 2]) - digamma(out_values[:, 3]) + digamma(
                out_values[:, 1])

    else:  # MI calculation
        out_values_gpu = cuda.device_array((n, 3), dtype=np.float64)
        mi_kernel[blocks_per_grid, threads_per_block](data_gpu, tuple(x), tuple(y), k, out_values_gpu)
        out_values = out_values_gpu.copy_to_host()

        with np.errstate(divide='ignore', invalid='ignore'):
            # k_tilde, nx, ny
            psis = digamma(out_values[:, 0]) + digamma(n) - digamma(out_values[:, 1]) - digamma(out_values[:, 2])

    # 替换nan为0，并求均值
    total_cmi = np.nan_to_num(psis, nan=0.0, posinf=0.0, neginf=0.0).mean()

    if minzero == 1:
        return max(total_cmi, 0)
    else:
        return total_cmi
