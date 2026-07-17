#!/usr/bin/env python3
"""BitNet structured-sparse checkpoint compression

将 2:4 结构化稀疏的 .pt  state_dict 转换为预编码和压缩后的 .pt 文件：
仅将特定的 int8 二维权重替换为压缩描述符，其余条目完全保留

在这里，压缩时的M维度被固定为4096，但是这个可以和推理时的M不同，无关
但是，压缩时的GEMM算法固定为id=6，因此推理时也必须指定此id才能正确解码压缩权重

分别执行：
python convert_checkpoint_compress.py --input ./checkpoints/model_state_8I_pruned_2_8_magnitude_expand.pt
python convert_checkpoint_compress.py --input ./checkpoints/model_state_8I_pruned_2_8_magnitude_base.pt

"""

from __future__ import annotations

# 优先载入自定义的cuSPARSELt动态库，避免与torch版本冲突
import argparse

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import logging


# GPU 侧仍写入独立文件，CPU 侧直接打印到终端
# 日志文件放在 bitnet_sparse 文件夹内，与压缩库保持一致
GPU_LOG_PATH = Path(__file__).resolve().parent / "bitnet_sparse" / "bitnet_compress.log"
if GPU_LOG_PATH.exists():
    GPU_LOG_PATH.unlink()  # 确保每次运行都会从空日志开始，避免旧记录混入

# 通过环境变量把日志路径暴露给 C++ 压缩库，使 GPU 侧 debug 输出写入独立文件
os.environ["BITNET_DEBUG_LOG_PATH"] = str(GPU_LOG_PATH)

# --- 日志记录器设置 -----------------------------------------------------------------
# 我们只保留文件输出，不再把详细调试信息写到终端，终端仅打印简洁的进度信息
logger = logging.getLogger("bitnet_compress")
logger.setLevel(logging.INFO)
logger.handlers.clear()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(stream_handler)
logger.propagate = False  # 阻止日志向根 logger 传播，避免被其他默认 handler 重复打印

import torch


@dataclass(eq=False)
class BitNetCompressedTensor:
    """用于封装单块压缩权重及其元数据的轻量结构。

    之所以不直接返回压缩后的 torch.Tensor，是为了让加载端能够读取额外的维度、
    算法、稀疏性等信息，快速恢复执行计划。"""

    data: torch.Tensor  # 压缩后的权重数据，存储为一维的 int8 张量
    m: int              # 原始 M 维度 (批大小)，在压缩时可能使用，在推理时可变
    n: int              # 权重矩阵的 N 维度 (输出特征数)
    k: int              # 权重矩阵的 K 维度 (输入特征数)
    algorithm: str = "cusparseLt_int8_2to4"  # 使用的压缩算法标识
    extra: Dict[str, object] = field(default_factory=dict)  # 存储额外信息，如原始张量名

    def __post_init__(self) -> None:
        """在对象初始化后进行数据校验和规整。"""
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if self.data.dtype != torch.int8:
            raise TypeError("Compressed tensor requires torch.int8 storage")
        # 确保数据在 CPU 上，并且是内存连续的，以便后续处理
        if self.data.device.type != "cpu":
            self.data = self.data.cpu()
        self.data = self.data.contiguous()
        # 校验维度信息
        for name, value in {"m": self.m, "n": self.n, "k": self.k}.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")

    @property
    def size_bytes(self) -> int:
        """返回压缩后数据的字节大小。"""
        return self.data.numel()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "checkpoints" / "model_state_int8_expand.pt"
# 默认输入指向同级目录的结构化稀疏检查点，运行时也可以传入自定义路径覆盖


def _format_size(bytes_count: int) -> str:
    """字节数格式化"""
    gb = bytes_count / (1024 ** 3)
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    mb = bytes_count / (1024 ** 2)
    if mb >= 0.1:
        return f"{mb:.2f} MB"
    kb = bytes_count / 1024
    if kb >= 0.1:
        return f"{kb:.2f} KB"
    return f"{bytes_count} B"


if __name__ == "__main__":
    # ----------------------------- 命令行参数解析 -----------------------------
    # 仅支持一个可选位置参数：输入模型路径。若未提供，则使用默认的结构化稀疏 checkpoint。
    parser = argparse.ArgumentParser(description="BitNet structured sparse compression")
    parser.add_argument(
        "--input",
        nargs="?",
        default=DEFAULT_INPUT,
        type=Path,
        help=(
            "Path to the structured sparse .pt file "
            "(default: ./checkpoints/model_state_int8_expand.pt)"
        ),
    )
    args = parser.parse_args()

    # ----------------------------- 输入 / 输出路径推导 ------------------------
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    # 输出路径基于输入路径自动生成，添加后缀 "_compressed"
    output_path = input_path.with_name(input_path.stem + "_compressed.pt")
    
    # 压缩库位于 bitnet_kernels_sparse 文件夹内
    library_path = Path(__file__).resolve().parent / "bitnet_kernels_sparse" / "libbitnet_sparse.so"
    if not library_path.exists():
        raise FileNotFoundError(f"Compression library not found: {library_path}")
    
    m_dim = 4096  # 为了压缩所设计的伪维度，在M>=512影响不大，不需要与运行时保持一致

    # ----------------------------- 日志入口 & 终端提示 -------------------------
    # 写入日志的同时，终端仅展示关键路径及大小。
    source_size = input_path.stat().st_size
    logger.info("Compression started")
    logger.info("Source checkpoint: %s (%s)", input_path, _format_size(source_size))
    logger.info("Destination checkpoint: %s", output_path)
    # 终端仅保留简洁的进度提示，方便人工观察
    print("[BitNet] Compression started")
    print(f"  Source      : {input_path} ({_format_size(source_size)})")
    print(f"  Destination : {output_path}")

    # ----------------------------- 动态库加载逻辑 -----------------------------
    preferred_paths = []  # 搜索 cuSPARSELt 动态库的路径优先级列表
    env_path = os.environ.get("CUSPARSELT_PATH")
    if env_path:
        preferred_paths.append(env_path)
    preferred_paths.extend(
        [
            # aarch64 (ARM64) 路径
            "/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            # x86_64 路径
            "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            # 通用 CUDA 路径
            "/usr/local/cuda/lib64/libcusparseLt.so.0",
        ]
    )
    found = ctypes.util.find_library("cusparseLt")
    if found:
        preferred_paths.append(found)

    cusparselt_loaded = False
    for path in dict.fromkeys(preferred_paths):  # dict.fromkeys() 用于保持顺序同时去除重复路径
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            getattr(lib, "cusparseLtMatmulAlgSelectionDestroy")
            logger.info("[cuSPARSELt] loaded %s", Path(path).resolve())
            cusparselt_loaded = True
            break
        except (OSError, AttributeError):
            continue
    if not cusparselt_loaded:
        raise OSError(
            "Unable to locate a compatible libcusparseLt. Set CUSPARSELT_PATH or install"
            " CUDA 12.9+ with libcusparseLt.so.0 available."
        )

    # ----------------------------- CUDA 初始化与函数签名 ---------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for compression")
    torch.cuda.init()  # 显式初始化 CUDA runtime，防止后续首次调用延迟
    stream = torch.cuda.current_stream()  # 使用默认流执行压缩操作

    compress_lib = ctypes.CDLL(str(library_path))
    # 以下显式声明函数签名是为了让 ctypes 进行参数类型检查，避免运行时崩溃
    compress_lib.bitlinear_get_compress_sizes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    compress_lib.bitlinear_get_compress_sizes.restype = None
    compress_lib.bitlinear_compress_weight.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    compress_lib.bitlinear_compress_weight.restype = None

    # ----------------------------- 读取原始 state_dict ------------------------
    raw_state = torch.load(str(input_path), map_location="cpu")  # 支持直接加载 state_dict 或包装结构
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    elif isinstance(raw_state, dict):
        state_dict = raw_state
    else:
        raise TypeError("Checkpoint is not a state_dict or wrapped state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict is not a mapping")

    enhanced_state: Dict[str, object] = {}  # 输出 state_dict，保持原始结构但替换关键权重
    compressible_suffix = ("wqkv.weight", "wo.weight", "w13.weight", "w2.weight")
    compressed_count = 0

    for name, value in state_dict.items():
        # ------------------------------------------------------------------
        # 循环一：遍历原始 state_dict
        #   1) 筛选出需要压缩的矩阵：二维 / int8 / 目标关键字后缀
        #   2) 在 GPU 上调用 C++ 压缩库完成压缩
        #   3) 回写 BitNetCompressedTensor 描述符
        # 非目标权重将原封不动写回，保证模型配置、标量向量完整保留。
        # ------------------------------------------------------------------
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.dtype == torch.int8 and any(
            name.endswith(suffix) for suffix in compressible_suffix
        ):
            # Step 1: 准备输入矩阵，确保是连续内存；记录维度信息用于 metadata
            tensor = value.contiguous()
            n, k = tensor.shape

            # Step 2: 将矩阵拷贝至 GPU，准备调用压缩函数
            weight_gpu = tensor.to(device="cuda", dtype=torch.int8, non_blocking=True)

            # Step 3: 询问压缩库需要的输出 buffer 大小 & 临时工作区大小
            compressed = ctypes.c_size_t()
            temp = ctypes.c_size_t()
            compress_lib.bitlinear_get_compress_sizes(
                ctypes.c_int(m_dim),
                ctypes.c_int(n),
                ctypes.c_int(k),
                ctypes.byref(compressed),
                ctypes.byref(temp),
            )
            compressed_size = int(compressed.value)
            temp_size = int(temp.value)
            if compressed_size <= 0:
                raise RuntimeError(f"Invalid compressed size: {compressed_size}")

            # Step 4: 根据返回的大小分配输出 buffer / 临时 buffer（如果需要）
            compressed_gpu = torch.empty(compressed_size, dtype=torch.int8, device="cuda")
            temp_gpu = (
                torch.empty(temp_size, dtype=torch.int8, device="cuda")
                if temp_size > 0 else None
            )
            temp_ptr = (
                ctypes.c_void_p(temp_gpu.data_ptr()) if temp_gpu is not None else ctypes.c_void_p(0)
            )

            # Step 5: 调用压缩函数，将结果写入 compressed_gpu
            compress_lib.bitlinear_compress_weight(
                ctypes.c_void_p(weight_gpu.data_ptr()),
                ctypes.c_void_p(compressed_gpu.data_ptr()),
                temp_ptr,
                ctypes.c_int(m_dim),
                ctypes.c_int(n),
                ctypes.c_int(k),
                ctypes.c_void_p(stream.cuda_stream),
            )
            torch.cuda.synchronize()  # 等待 GPU 压缩完成，确保随后从 CPU 读取的数据完整有效

            # Step 6: 拷贝结果回 CPU，并清理中间 GPU 资源
            compressed_tensor = compressed_gpu.cpu()  # 将结果拷贝回 CPU，方便写入 state_dict
            del weight_gpu, compressed_gpu
            if temp_gpu is not None:
                del temp_gpu

            # Step 7: 构造描述符，写回到增强后的 state_dict 中
            descriptor = BitNetCompressedTensor(
                data=compressed_tensor,
                m=m_dim,
                n=n,
                k=k,
                extra={
                    "source": name,
                    "compressed_size": compressed_tensor.numel(),
                    "sparsity": "2:4",
                },
            )
            enhanced_state[name] = descriptor
            compressed_count += 1
            logger.info(
                "[compress] %s: shape=(%d, %d) -> compressed %d bytes",
                name,
                n,
                k,
                descriptor.size_bytes,
            )
        else:
            enhanced_state[name] = value  # 非目标权重保持原状，确保配置/向量完整保留

    if compressed_count == 0:
        raise RuntimeError(
            "No eligible int8 weight matrices were found. Check that the input checkpoint"
            " has already been quantized and contains wqkv/w13/w2/wo weights."
        )

    torch.save(enhanced_state, str(output_path))  # 输出增强型 state_dict，供后续加载使用
    output_size = output_path.stat().st_size if output_path.exists() else 0
    logger.info(
        "[compress] Saved enhanced checkpoint to %s (compressed %d tensors).",
        output_path,
        compressed_count,
    )
    logger.info(
        "[compress] Preserved %d entries without modification.",
        len(state_dict) - compressed_count,
    )
    logger.info("Destination checkpoint size: %s", _format_size(output_size))

    # 收尾时展示输出文件大小和完成状态
    print(f"  Output size : {_format_size(output_size)}")
    print("[BitNet] Compression finished")
