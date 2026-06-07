# Patched for eval - lazy imports
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

# Core modules that don't need mmengine
from .sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
try:
    from .vq_sam2 import VQ_SAM2Model
except ImportError:
    VQ_SAM2Model = None

# Optional heavy modules
try:
    from .qwen25vl import QWEN25VL_VQSAM2Model
except ImportError:
    QWEN25VL_VQSAM2Model = None

try:
    from .qwen3vl import QWEN3VL_VQSAM2Model
except ImportError:
    QWEN3VL_VQSAM2Model = None

try:
    from .perceptionlm import PerceptionLM_TokenMask
except ImportError:
    PerceptionLM_TokenMask = None

try:
    from .processing_perception_lm import PerceptionLMProcessor
except ImportError:
    PerceptionLMProcessor = None
