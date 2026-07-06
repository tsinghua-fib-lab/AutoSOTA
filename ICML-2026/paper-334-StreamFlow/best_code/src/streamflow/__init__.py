"""
StreamFlow: PeRFlow优化的流式扩散管道

基于StreamDiffusion，但专门为PeRFlow优化，解决了：
1. 调度器step方法被覆盖的问题
2. VAE解码后处理导致偏暗的问题
3. 时间步映射不匹配的问题
"""

from .pipeline import StreamFlow
from .image_utils import postprocess_image

__version__ = "0.1.0"
__all__ = ["StreamFlow", "postprocess_image"]