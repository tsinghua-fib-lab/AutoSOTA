import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "scripts/score.py"), run_name="__main__")
