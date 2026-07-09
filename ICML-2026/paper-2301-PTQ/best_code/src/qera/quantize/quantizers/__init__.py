from .block_fp import block_fp_quantizer
from .integer import int_quantizer
from .minifloat import minifloat_ieee_quantizer
from .bypass import bypass_quantizer
from .mxint import mxint_quantizer
from .normalfloat import normalfloat_quantizer

def get_quantizer(name: str):
    match name:
        case "bypass":
            return bypass_quantizer
        case "block_fp":
            return block_fp_quantizer
        case "mxint":
            return mxint_quantizer
        case "normalfloat":
            return normalfloat_quantizer
        case "integer":
            return int_quantizer
        case "minifloat":
            return minifloat_ieee_quantizer
        case _:
            raise ValueError(f"quantizer {name} not supported")
