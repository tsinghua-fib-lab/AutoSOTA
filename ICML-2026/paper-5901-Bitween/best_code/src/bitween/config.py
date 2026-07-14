import configparser
import os
from enum import StrEnum


class Method(StrEnum):
    """Method for Analysis."""

    MULTIPLE_REGRESSION = "multiple_regression"
    SIMPLE_REGRESSION = "simple_regression"
    FORWARD_SELECTION = "forward_selection"
    EAGER_MILP = "eager_milp"
    PYSR = "pysr"
    GPLEARN = "gplearn"
    KAN = "kan"
    RAG_SR = "rag_sr"
    PARFAM = "parfam"
    MetaSymNet = "metasymnet"


class Correctness(StrEnum):
    """Post Correctness Check."""

    VERIFICATION = "verification"
    FUZZING = "fuzzing"
    NONE = "none"


class MILPSolver(StrEnum):
    """MILP Solver."""

    PULP = "pulp"
    GUROBI = "gurobi"
    GLPK = "glpk"


class FullMILP(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


class RegressionScore(StrEnum):
    NEG_MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    R2 = "r2"


class InvariantType(StrEnum):
    INT = "int"
    REAL = "real"
    MIXED = "mixed"


class Config:
    _instance = None
    _working_dir = os.getcwd()
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _config_file = os.path.join(_current_dir, "../..", "bitween.ini")
    _alternate_config_file = os.path.join(_working_dir, "bitween.ini")

    def __new__(cls, config_file=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = configparser.ConfigParser(
                inline_comment_prefixes=("#", ";")
            )
            if cls._alternate_config_file and os.path.exists(
                cls._alternate_config_file
            ):
                cls._instance._config_file = cls._alternate_config_file
            else:
                cls._instance._config_file = config_file or cls._config_file
            if not os.path.exists(cls._instance._config_file):
                raise FileNotFoundError(
                    f"Config file '{cls._instance._config_file}' not found."
                )
            cls._instance._config.read(cls._instance._config_file)
        return cls._instance

    @property
    def project_dir(self):
        """Get the project directory."""
        return os.path.join(self._current_dir, "../..")

    # NOTE: General settings
    @property
    def file_path(self):
        """Get the file path."""
        return self._config.get("general", "file_path", fallback="")

    @file_path.setter
    def file_path(self, value):
        self._config.set("general", "file_path", value)

    @property
    def degree(self):
        """Get the degree."""
        return self._config.getint("general", "degree", fallback=2)

    @degree.setter
    def degree(self, value):
        self._config.set("general", "degree", str(value))

    @property
    def epsilon(self):
        """Get the epsilon value. Tolerance for squared error."""
        return self._config.getfloat("general", "epsilon", fallback=0.001)

    @epsilon.setter
    def epsilon(self, value):
        self._config.set("general", "epsilon", str(value))

    @property
    def delta(self):
        """Get the delta value."""
        return self._config.getfloat("general", "delta", fallback=0.2)

    @delta.setter
    def delta(self, value):
        self._config.set("general", "delta", str(value))

    @property
    def precision(self):
        """Get the precision value. Precision for the floating-point numbers"""
        return self._config.getint("general", "precision", fallback=3)

    @precision.setter
    def precision(self, value):
        self._config.set("general", "precision", str(value))

    @property
    def method(self):
        """Get the initial method."""
        value = self._config.get("general", "method", fallback="MULTIPLE_REGRESSION")
        return Method[value]

    @method.setter
    def method(self, value):
        self._config.set("general", "method", value.name)

    @property
    def correctness(self):
        """Check or Verify correctness."""
        value = self._config.get("general", "correctness", fallback="VERIFICATION")
        return Correctness[value]

    @correctness.setter
    def correctness(self, value):
        self._config.set("general", "correctness", value.name)

    @property
    def n(self):
        """Get the number of data points."""
        return self._config.getint("general", "n", fallback=20)

    @n.setter
    def n(self, value):
        self._config.set("general", "n", str(value))

    @property
    def logger_level(self):
        """Get the logger level."""
        return self._config.getint("general", "logger_level", fallback=3)

    @logger_level.setter
    def logger_level(self, value):
        self._config.set("general", "logger_level", str(value))

    @property
    def invariant_type(self):
        """Get the invariant type (INT, REAL, or MIXED)."""
        value = self._config.get("general", "invariant_type", fallback="MIXED")
        return InvariantType[value]

    @invariant_type.setter
    def invariant_type(self, value):
        self._config.set("general", "invariant_type", value.name)

    # NOTE: Forward Selection Method
    @property
    def selector_initial_rate(self):
        """Get the selector initial rate for forward selection."""
        return self._config.getfloat(
            "forward_selection", "selector_initial_rate", fallback=0.8
        )

    @selector_initial_rate.setter
    def selector_initial_rate(self, value):
        self._config.set("forward_selection", "selector_initial_rate", str(value))

    @property
    def selector_decay_rate(self):
        """Get the selector decay rate for forward selection"""
        return self._config.getfloat(
            "forward_selection", "selector_decay_rate", fallback=0.3
        )

    @selector_decay_rate.setter
    def selector_decay_rate(self, value):
        self._config.set("forward_selection", "selector_decay_rate", str(value))

    @property
    def selector_parallel(self):
        """Get whether the selector is parallel for forward selection."""
        return self._config.getboolean(
            "forward_selection", "selector_parallel", fallback=True
        )

    @selector_parallel.setter
    def selector_parallel(self, value):
        self._config.set("forward_selection", "selector_parallel", str(value))

    @property
    def selector_max_features(self):
        """Get the maximum number of selector features for forward selection."""
        features = self._config.getint(
            "forward_selection",
            "selector_max_features",
            fallback=10 if self.degree > 2 else 6,
        )
        return 10 if features > 10 else features

    @selector_max_features.setter
    def selector_max_features(self, value):
        self._config.set("forward_selection", "selector_max_features", str(value))

    # NOTE: Multiple Regression Method
    @property
    def cross_validation(self):
        """Get the number of cross validations for multiple regression."""
        return self._config.getint(
            "multiple_regression", "cross_validation", fallback=3
        )

    @cross_validation.setter
    def cross_validation(self, value):
        self._config.set("multiple_regression", "cross_validation", str(value))

    @property
    def regression_score(self):
        """Get the regression score for multiple regression."""
        value = self._config.get(
            "multiple_regression", "regression_score", fallback="r2"
        )
        return RegressionScore(value)

    @regression_score.setter
    def regression_score(self, value):
        self._config.set("multiple_regression", "regression_score", value.value)

    # NOTE: Regression Refinement Method
    @property
    def regression_refinement(self):
        """Get whether regression refinement is enabled."""
        return self._config.getboolean(
            "regression_refinement", "regression_refinement", fallback=True
        )

    @regression_refinement.setter
    def regression_refinement(self, value):
        self._config.set("regression_refinement", "regression_refinement", str(value))

    # NOTE: Regression Sample Size Limit (for Regression-based Methods)
    @property
    def use_sample_rate_regression(self):
        """Get whether to use sample rate for regression."""
        return self._config.getboolean(
            "regression_sample", "use_sample_rate", fallback=True
        )

    @use_sample_rate_regression.setter
    def use_sample_rate_regression(self, value):
        self._config.set("regression_sample", "use_sample_rate", str(value))

    @property
    def sample_rate_regression(self):
        """Get the sample rate for regression."""
        return self._config.getint("regression_sample", "sample_rate", fallback=10)

    @sample_rate_regression.setter
    def sample_rate_regression(self, value):
        self._config.set("regression_sample", "sample_rate", str(value))

    @property
    def sample_threshold_regression(self):
        """Get the sample threshold for regression."""
        return self._config.getint("regression_sample", "sample_threshold", fallback=50)

    @sample_threshold_regression.setter
    def sample_threshold_regression(self, value):
        self._config.set("regression_sample", "sample_threshold", str(value))

    # NOTE: Equation inference parameters for Regression-based Methods
    @property
    def coeff_threshold(self):
        """Get the coefficient threshold for regression."""
        return self._config.getfloat(
            "equation_inference", "coeff_threshold", fallback=0.01
        )

    @coeff_threshold.setter
    def coeff_threshold(self, value):
        self._config.set("equation_inference", "coeff_threshold", str(value))

    @property
    def use_cutoff(self):
        """Get whether to use cutoff for regression."""
        return self._config.getboolean(
            "equation_inference", "use_cutoff", fallback=True
        )

    @use_cutoff.setter
    def use_cutoff(self, value):
        self._config.set("equation_inference", "use_cutoff", str(value))

    @property
    def coeff_cutoff(self):
        """Get the coefficient cutoff value for regression."""
        return self._config.getint("equation_inference", "coeff_cutoff", fallback=30)

    @coeff_cutoff.setter
    def coeff_cutoff(self, value):
        self._config.set("equation_inference", "coeff_cutoff", str(value))

    @property
    def intercept_cutoff(self):
        """Get the intercept cutoff value for regression."""
        return self._config.getint(
            "equation_inference", "intercept_cutoff", fallback=50
        )

    @intercept_cutoff.setter
    def intercept_cutoff(self, value):
        self._config.set("equation_inference", "intercept_cutoff", str(value))

    # NOTE: MILP Method parameters
    @property
    def milp_enabled(self):
        """Get whether MILP is enabled."""
        return self._config.getboolean("milp", "enabled", fallback=False)

    @milp_enabled.setter
    def milp_enabled(self, value):
        self._config.set("milp", "enabled", str(value))

    @property
    def milp_solver(self):
        """Get the MILP solver."""
        value = self._config.get("milp", "solver", fallback="GUROBI")
        return MILPSolver[value]

    @milp_solver.setter
    def milp_solver(self, value):
        self._config.set("milp", "solver", value.name)

    @property
    def objective_threshold(self):
        """Get the objective threshold."""
        return self._config.getfloat("milp", "objective_threshold", fallback=1e-9)

    @objective_threshold.setter
    def objective_threshold(self, value):
        self._config.set("milp", "objective_threshold", str(value))

    @property
    def bound(self):
        """Get the MILP bound."""
        return self._config.getint("milp", "bound", fallback=20)

    @bound.setter
    def bound(self, value):
        self._config.set("milp", "bound", str(value))

    @property
    def milp_timeout(self):
        """Get the MILP timeout."""
        return self._config.getint("milp", "timeout", fallback=1)

    @milp_timeout.setter
    def milp_timeout(self, value):
        self._config.set("milp", "timeout", str(value))

    @property
    def milp_parallel(self):
        """Get whether MILP is parallel."""
        return self._config.getboolean("milp", "parallel", fallback=True)

    @milp_parallel.setter
    def milp_parallel(self, value):
        self._config.set("milp", "parallel", str(value))

    @property
    def milp_warnings(self):
        """Get whether MILP warnings are enabled."""
        return self._config.getboolean("milp", "warnings", fallback=False)

    @milp_warnings.setter
    def milp_warnings(self, value):
        self._config.set("milp", "warnings", str(value))

    # NOTE: MILP Sample Size Limit
    @property
    def sample_rate_milp(self):
        """Get the MILP sample rate."""
        return self._config.getfloat("milp_sample", "sample_rate", fallback=1.5)

    @sample_rate_milp.setter
    def sample_rate_milp(self, value):
        self._config.set("milp_sample", "sample_rate", str(value))

    @property
    def sample_threshold_milp(self):
        """Get the MILP sample threshold."""
        return self._config.getint("milp_sample", "sample_threshold", fallback=30)

    @sample_threshold_milp.setter
    def sample_threshold_milp(self, value):
        self._config.set("milp_sample", "sample_threshold", str(value))

    @property
    def use_sample_rate_milp(self):
        """Get whether to use MILP sample rate."""
        return self._config.getboolean("milp_sample", "use_sample_rate", fallback=True)

    @use_sample_rate_milp.setter
    def use_sample_rate_milp(self, value):
        self._config.set("milp_sample", "use_sample_rate", str(value))

    # NOTE: Eager MILP
    @property
    def full_milp_strategy(self):
        """
        Get the full MILP strategy. If ALWAYS, then use MILP from all data points
        for each pivot, if AUTO, then we use MILP from all data points for each pivot
        if the number of data points is less than SAMPLE_THRESHOLD
        """
        value = self._config.get("full_milp", "strategy", fallback="AUTO")
        return FullMILP[value]

    @full_milp_strategy.setter
    def full_milp_strategy(self, value):
        self._config.set("full_milp", "strategy", value.name)

    @property
    def full_milp_threshold(self):
        """Get the full MILP threshold."""
        return self._config.getint("full_milp", "threshold", fallback=9)

    @full_milp_threshold.setter
    def full_milp_threshold(self, value):
        self._config.set("full_milp", "threshold", str(value))

    # NOTE: Construct a model from the Regression models based on frequency,
    @property
    def milp_freq_refine_enabled(self):
        """Get whether MILP frequency refinement model is enabled."""
        return self._config.getboolean("milp_freq_refine", "enabled", fallback=False)

    @milp_freq_refine_enabled.setter
    def milp_freq_refine_enabled(self, value):
        self._config.set("milp_freq_refine", "enabled", str(value))

    # NOTE: Sympy Linear Solver
    @property
    def linear_solver_enabled(self):
        """Get whether the linear solver is enabled."""
        return self._config.getboolean("linear_solver", "enabled", fallback=False)

    @linear_solver_enabled.setter
    def linear_solver_enabled(self, value):
        self._config.set("linear_solver", "enabled", str(value))

    # NOTE: Reductions
    @property
    def ugly_factor(self):
        """Get the ugly factor."""
        return self._config.getint("reductions", "ugly_factor", fallback=20)

    @ugly_factor.setter
    def ugly_factor(self, value):
        self._config.set("reductions", "ugly_factor", str(value))

    # NOTE: Z3 SMT Solver
    @property
    def z3_timeout(self):
        """Get the Z3 SMT solver timeout."""
        return self._config.getint("z3", "timeout", fallback=10)

    @z3_timeout.setter
    def z3_timeout(self, value):
        self._config.set("z3", "timeout", str(value))

    @property
    def slow_simplify(self):
        """Get whether slow simplify is enabled."""
        return self._config.getboolean("z3", "slow_simplify", fallback=False)

    @slow_simplify.setter
    def slow_simplify(self, value):
        self._config.set("z3", "slow_simplify", str(value))

    @property
    def consistency_check(self):
        """Get whether consistency check is enabled."""
        return self._config.getboolean("z3", "consistency_check", fallback=False)

    @consistency_check.setter
    def consistency_check(self, value):
        self._config.set("z3", "consistency_check", str(value))

    # NOTE: Reporting
    @property
    def property_table_width(self):
        """Get the property table width."""
        return self._config.getint("reporting", "property_table_width", fallback=70)

    @property_table_width.setter
    def property_table_width(self, value):
        self._config.set("reporting", "property_table_width", str(value))

    # NOTE: Fuzzing
    @property
    def fuzz_timeout(self):
        """Get the fuzzing timeout."""
        return self._config.getint("fuzzing", "timeout", fallback=2)

    @fuzz_timeout.setter
    def fuzz_timeout(self, value):
        self._config.set("fuzzing", "timeout", str(value))

    # NOTE: Verification
    @property
    def is_close_for_float(self):
        """Get whether is_close_for_float is enabled."""
        return self._config.getboolean(
            "verification", "is_close_for_float", fallback=True
        )

    @is_close_for_float.setter
    def is_close_for_float(self, value):
        self._config.set("verification", "is_close_for_float", str(value))

    @property
    def civl_path(self):
        """Get the path to the CIVL verifier."""
        return self._config.get("verification", "civl_path", fallback="")

    @civl_path.setter
    def civl_path(self, value):
        self._config.set("verification", "civl_path", value)

    # NOTE: Limit Degree
    @property
    def limit_degree(self):
        """Get the limit degree."""
        return self._config.getint("limits", "limit_degree", fallback=10)

    @limit_degree.setter
    def limit_degree(self, value):
        self._config.set("limits", "limit_degree", str(value))

    @property
    def pysr_timeout(self):
        """Get the PySR timeout."""
        return self._config.getfloat("pysr", "timeout", fallback=600)

    @pysr_timeout.setter
    def pysr_timeout(self, value):
        self._config.set("pysr", "timeout", str(value))

    def print_all(self):
        for section in self._config.sections():
            print(f"[{section}]")
            for key in self._config[section]:
                print(f"{key} = {self._config[section][key]}")
            print()


# Example usage:
if __name__ == "__main__":
    try:
        config = Config()
        print()
        # Print all configuration settings
        config.print_all()

        print(f"Logger Level: {config.logger_level}")

        config.logger_level = 5
        print(f"Updated Logger Level: {Config().logger_level}")

        config.method = Method.GPLEARN
        print(f"Method: {config.method}")
        assert config.method == Method.GPLEARN

        config.invariant_type = InvariantType.INT
        print(f"Invariant Type: {config.invariant_type}")
        assert config.invariant_type == InvariantType.INT

    except FileNotFoundError as e:
        print(e)
