from __future__ import annotations

# Import strategies for side-effect registration.
from . import full as _full  # noqa: F401
from . import linear_probe as _linear_probe  # noqa: F401

# Keep PEFT optional so non-PEFT strategies remain usable
# in environments without the full PEFT dependency stack.
try:
    from . import peft_lora as _peft_lora  # noqa: F401
except Exception:
    _peft_lora = None
