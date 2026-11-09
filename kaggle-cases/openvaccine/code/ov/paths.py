import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
BPPS = os.path.join(DATA, "bpps")
KOV = os.path.join(ROOT, "KaggleOpenVaccine")
ETERNAFOLD = os.path.join(ROOT, "tools", "EternaFold")
RUNS = os.path.join(ROOT, "runs")

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# output column order of the Nullrecurrent model (scored columns first, matches LOSS_WGTS)
MODEL_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]
SCORED = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
ERRORS = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_pH10", "deg_error_Mg_50C", "deg_error_50C"]
