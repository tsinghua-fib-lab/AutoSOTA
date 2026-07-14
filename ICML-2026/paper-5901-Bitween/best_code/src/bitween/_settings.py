from enum import Enum

FILE_PATH = "benchmarks/bitween/dig/bresenham.dig.traces.csv"  # NOTE: NO NEED MILP
# FILE_PATH = "benchmarks/bitween/dig/cohencu.dig.dyn.traces.csv"  # NOTE: NO NEED MILP
# FILE_PATH = "benchmarks/bitween/dig/cohendiv.dig.dyn.traces.csv"  # NOTE: NO NEED MILP
# FILE_PATH = "benchmarks/bitween/dig/dijkstra.dig.dyn.traces.csv"  # NOTE: NO NEED MILP
# FILE_PATH = "benchmarks/bitween/dig/egcd.dig.dyn.traces.csv"  # NOTE: NO NEED MILP / ForwardSelection performs well
# FILE_PATH = "benchmarks/bitween/dig/egcd2.dig.dyn.traces.csv"  # NOTE: NO NEED MILP
# FILE_PATH = "benchmarks/bitween/dig/egcd3.dig.dyn.traces.csv"  # NOTE: NO NEED MILP / COMPARE this with egcd2
# FILE_PATH = "benchmarks/bitween/dig/fermat1.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/fermat2.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/freire1_int.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/freire1.dig.dyn.traces.csv" # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/freire2.dig.dyn.traces.csv" # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/geo1.dig.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/geo2.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/geo3.dig.traces.csv" # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/hard.dig.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/knuth.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/lcm1.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/lcm2.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/mannadiv.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/prod4br.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/prodbin.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/ps1.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/ps2.dig.dyn.traces.csv"  # NOTE: Check on DIG
# FILE_PATH = "benchmarks/bitween/dig/ps3.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/ps4.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/ps5.dig.dyn.traces.csv"  # NOTE: Check on BITWEEN
# FILE_PATH = "benchmarks/bitween/dig/ps5.dig.dyn.traces1.csv"  # NOTE: Check on BITWEEN
# FILE_PATH = "benchmarks/bitween/dig/ps5.dig.traces.csv"  # NOTE: Check on BITWEEN
# FILE_PATH = "benchmarks/bitween/dig/ps6.dig.traces.csv"  # NOTE: Check on BITWEEN
# FILE_PATH = "benchmarks/bitween/dig/sqrt1.dig.traces.csv"

# FILE_PATH = "benchmarks/bitween/dig/poly3.dig.dyn.traces1.csv"
# FILE_PATH = "benchmarks/bitween/dig/poly3.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/poly4.dig.dyn.traces.csv"
# FILE_PATH = "benchmarks/bitween/dig/poly5.dig.dyn.traces.csv"
# FILE_PATH = "trace.csv"

LOGGER_LEVEL = 3

DEGREE = 2
EPSILON = 0.001
PRECISION = 3  # Precision for the floating-point numbers

TYPE = "INT"  # INT or REAL


class InitialMethod(Enum):
    MULTIPLE_REGRESSION = 0
    SIMPLE_REGRESSION = 1
    FORWARD_SELECTION = 2
    EAGER_MILP = 3  # don't use this, it is for the ablation study
    PYSR = 4  # PySR is a symbolic regression library in Python
    GPLEARN = 5  # gplearn is a symbolic regression library in Python
    KAN = 6  # KAN: Kolmogorov–Arnold Networks


INITIAL_METHOD: InitialMethod = InitialMethod.MULTIPLE_REGRESSION

# NOTE: Forward Selection Method
SELECTOR_INITIAL_RATE = 0.8
SELECTOR_DECAY_RATE = 0.3
SELECTOR_PARALLEL = True
SELECTOR_MAX_FEATURES = 10 if DEGREE > 2 else 7


# NOTE: Multiple Regression Method
CROSS_VALIDATION = 3
REGRESSION_SCORE = "r2"  # neg_mean_squared_error or r2

# NOTE: Regression Refinement Method
REGRESSION_REFINEMENT = True  # if True, then use the regression refinement algorithm

# NOTE: Regression Sample Size Limit (for Regression-based Methods)
REGRESSION_SAMPLE_RATE = 10  # if the number of data points is larger than this value, then we use the sample rate
REGRESSION_SAMPLE_THRESHOLD = 50  # if the number of data points is larger than this value, then we use this number of data points
REGRESSION_USE_SAMPLE_RATE = True  # if True, then we use the sample rate

# NOTE: Equation inference parameters for Regression-based Methods
COEFF_THRESHOLD = 0.01  # remove coefficients that are smaller than this value
USE_CUTOFF = True  # if True, then we use the cutoffs
COEFF_CUTOFF = 30  # remove equalities that are larger than this value
INTERCEPT_CUTOFF = 50  # remove equalities that are larger than this value


class MILPSolver(Enum):
    PULP = 0
    GUROBI = 1
    GLPK = 2

    def __str__(self):
        return self.name


# NOTE: MILP Method parameters
MILP = True  # if True, then we use the MILP solver after the regression-based methods
MILP_SOLVER: MILPSolver = MILPSolver.GUROBI  # PULP or GUROBI or GLPK
OBJECTIVE_THRESHOLD = 1e-9
MILP_BOUND = 20
MILP_TIME_LIMIT = 1  # in seconds
PARALLEL_MILP = True  # if True, then we use the parallel MILP solver
MILP_WARNINGS = False  # if True, then we show the MILP warnings


# NOTE: Eager MILP
# if ALWAYS, then use MILP from all data points for each pivot
# if AUTO, then we use MILP from all data points for each pivot if the number of data points is less than MILP_SAMPLE_THRESHOLD
class FullMILP(Enum):
    NEVER = 0
    ALWAYS = 1
    AUTO = 2


FULL_MILP = FullMILP.AUTO
FULL_MILP_THRESHOLD = 9  # if the number of data points is less than this value, then we use MILP from all data points for each pivot

# NOTE: Construct a model from the Regression models based on frequency, and then refine the model using MILP
MILP_FREQ_REFINE = False

# NOTE: MILP Sample Size Limit
MILP_SAMPLE_RATE = 1.5
MILP_SAMPLE_THRESHOLD = 30  # if the number of data points is larger than this value, then we use this number of data points
MILP_USE_SAMPLE_RATE = True  # if True, then we use the sample rate for MILP

# NOTE: Sympy Linear Solver
# Solve system of linear equations with  variables, which means both under- and overdetermined systems are supported.
# https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve_linear_system
LINEAR_SOLVER = False
USE_LINSOLVE = LINEAR_SOLVER and TYPE == "INT"

# NOTE: Reductions
UGLY_FACTOR = 20  # remove equalities that have lots of terms and "large" coefficients

# NOTE: Z3
SOLVER_TIMEOUT = 10  # in seconds
SLOW_SIMPLIFY = False  # simplify results, e.g., removing weaker invariants
CONSISTENCY_CHECK = False  # if True, then we use the consistency check method

# NOTE: Reporting
PROPERTY_TABLE_WIDTH = 75

# NOTE: Fuzzing
FUZZ_TIMEOUT = 2  # in seconds

# NOTE: Verification
# if True, then we use the is_close function for floating-point numbers
VERIFICATION_IS_CLOSE_FOR_FLOAT = True

# This is for accidentially running the main function with very high degree;
# running with high degrees may freeze the system
LIMIT_DEGREE = 10  # if the degree is larger than this value, then we use this value

if DEGREE > LIMIT_DEGREE:
    DEGREE = LIMIT_DEGREE

if __name__ == "__main__":
    from bitween.main import bitween

    bitween()
