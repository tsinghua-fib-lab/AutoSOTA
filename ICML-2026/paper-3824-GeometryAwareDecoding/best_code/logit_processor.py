"""Stub for TopH_LogitsProcessor from the Top-H repo.
Only needed when alpha is passed; the stub raises if actually called.
"""
from transformers import LogitsProcessor

class TopH_LogitsProcessor(LogitsProcessor):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TopH_LogitsProcessor is not installed. "
            "This stub exists only so that the huggingface.py importer does not crash "
            "when alpha is NOT passed. If you need Top-H decoding, install the Top-H repo."
        )
    
    def __call__(self, input_ids, scores):
        raise NotImplementedError("TopH_LogitsProcessor stub")
