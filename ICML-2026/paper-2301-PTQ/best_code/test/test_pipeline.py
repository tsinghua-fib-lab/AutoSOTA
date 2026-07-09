import sys
from pathlib import Path

src_path = Path(__file__).parents[1].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"

sys.path.append(src_path.as_posix())


from qera.pipeline import pipeline_qera
from qera.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    pipeline_qera()
