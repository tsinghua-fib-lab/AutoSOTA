import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[0].joinpath("src").as_posix())


from qera.ptq_pipeline import pipeline_fp16_bf16_fp32
from qera.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    pipeline_fp16_bf16_fp32()
