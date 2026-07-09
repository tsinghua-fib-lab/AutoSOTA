from peft.utils import PeftType, register_peft_method

from ._version import __version__
from .config import PLATEConfig
from .layer import PLATELinear
from .model import PLATETuner, get_plate_model

__all__ = ["PLATEConfig", "PLATETuner", "PLATELinear", "get_plate_model", "__version__"]

# Dynamically add PLATE to PeftType enum if not already present
if not hasattr(PeftType, "PLATE"):
    PeftType.PLATE = "PLATE"
    PeftType._member_names_.append("PLATE")
    PeftType._member_map_["PLATE"] = PeftType.PLATE
    PeftType._value2member_map_["PLATE"] = PeftType.PLATE

# Register PLATE as a PEFT method (skip if already registered)
# The prefix "plate_" is used by PEFT to filter state_dict keys when saving/loading
# This is essential for DeepSpeed ZeRO-3 and FSDP compatibility
try:
    register_peft_method(
        name="plate",
        config_cls=PLATEConfig,
        model_cls=PLATETuner,
        prefix="plate_",  
    )
except KeyError:
    pass

