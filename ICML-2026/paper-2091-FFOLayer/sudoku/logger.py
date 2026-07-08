from typing import Optional

_writer = None
_tag = "BLO"
_step = 0

def set_writer(writer, tag: Optional[str] = None):
    global _writer, _tag
    _writer = writer
    if tag is not None:
        _tag = tag

def log_scalar(name: str, value, step: Optional[int] = None):
    global _writer, _step, _tag
    if _writer is None:
        return
    if hasattr(value, "item"):
        value = value.item()
    if step is None:
        _step += 1
        step = _step
    _writer.add_scalar(f"{_tag}/{name}", float(value), step)

def log_text(name: str, text: str, step: Optional[int] = None):
    global _writer, _step, _tag
    if _writer is None:
        return
    if step is None:
        _step += 1
        step = _step
    _writer.add_text(f"{_tag}/{name}", str(text), step)
