from problems.ou import get_problem as ou_problem
from problems.double_well import get_problem as double_well_problem
from problems.cir import get_problem as cir_problem
from problems.muller_brown import get_problem as muller_brown_problem
from problems.cell_diffusion import get_problem as cell_diffusion_problem

PROBLEMS = {
    "ou": ou_problem,
    "double_well": double_well_problem,
    "cir": cir_problem,
    "muller_brown": muller_brown_problem,
    "cell_diffusion": cell_diffusion_problem,
}