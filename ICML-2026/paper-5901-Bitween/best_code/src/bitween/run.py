import argparse
from pathlib import Path

from bitween.checker import fuzz_and_check
from bitween.config import Config, Correctness, Method, MILPSolver, RegressionScore
from bitween.decorator import decorate_assertions
from bitween.fuzzer import fuzz_and_trace
from bitween.main import bitween
from bitween.verifier import fuzz_and_verify

config = Config()


def run():
    parser = argparse.ArgumentParser(
        prog="bitween",
        description="Run analysis based on provided configurations.",
        usage="%(prog)s -f <file_path> -n <func_name> [options]",
        epilog="For more information, visit the documentation.",
    )

    parser.add_argument(
        "-f", "--file_path", required=True, help="Path to the input file."
    )
    parser.add_argument(
        "-m", "--func_name", required=True, help="Name of the function to analyze."
    )
    parser.add_argument(
        "-o", "--out_path", help="Path to the output file (default: file_path)."
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=int,
        help="Degree of the polynomial (default: 2).",
        default=2,
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of samples to collect (default: 20).",
        default=20,
    )
    parser.add_argument(
        "--correctness",
        choices=["verification", "fuzzing"],
        help="Check or verify correctness of inferred invariants.",
        default="verification",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Tolerance for squared error (default: 0.001).",
        default=0.001,
    )
    parser.add_argument(
        "--delta", type=float, help="Delta value (default: 0.2).", default=0.2
    )
    parser.add_argument(
        "--precision",
        type=int,
        help="Precision for floating-point numbers (default: 2).",
        default=2,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[m.name for m in Method],
        help="Initial method for analysis (default: MULTIPLE_REGRESSION).",
        default="MULTIPLE_REGRESSION",
    )
    parser.add_argument(
        "--cross_validation",
        type=int,
        help="Number of cross validations (default: 3).",
        default=3,
    )
    parser.add_argument(
        "--regression_score",
        type=str,
        choices=[s.name for s in RegressionScore],
        help="Regression score method (default: r2).",
        default="r2",
    )
    parser.add_argument(
        "--coeff_threshold",
        type=float,
        help="Coefficient threshold (default: 0.01).",
        default=0.01,
    )
    parser.add_argument(
        "--use_cutoff",
        type=bool,
        help="Whether to use cutoff (default: True).",
        default=True,
    )
    parser.add_argument(
        "--coeff_cutoff",
        type=int,
        help="Coefficient cutoff value (default: 30).",
        default=30,
    )
    parser.add_argument(
        "--intercept_cutoff",
        type=int,
        help="Intercept cutoff value (default: 50).",
        default=50,
    )
    parser.add_argument(
        "--ugly_factor",
        type=int,
        help="Ugly factor for reductions (default: 20).",
        default=20,
    )
    parser.add_argument(
        "--milp",
        type=str,
        choices=[solver.name for solver in MILPSolver],
        help="MILP solver to use (default: None).",
        default=None,
    )

    config = Config()

    args = parser.parse_args(
        # Test arguments
        # args=[
        #     "-f",
        #     config.project_dir + "/benchmarks/rsr/sigmoid.c",
        #     "--func_name",
        #     # "sine",
        #     "sigmoid",
        #     "--method",
        #     "MULTIPLE_REGRESSION",
        #     "-o",
        #     config.project_dir + "/benchmarks/rsr/.out.c",
        #     "-d",
        #     "2",
        #     "-n",
        #     "150",
        #     "--correctness",
        #     "verification",
        # ]
    )

    target_file = Path(args.file_path)
    if not target_file.exists():
        parser.exit(1, f"The file '{args.file_path}' does not exist.\n")

    # Override configurations from command-line arguments if provided
    if args.file_path:
        config.file_path = args.file_path
    if args.func_name:
        config.func_name = args.func_name
    out_path = args.out_path if args.out_path else args.file_path
    if args.degree:
        config.degree = args.degree
    if args.epsilon:
        config.epsilon = args.epsilon
    if args.delta:
        config.delta = args.delta
    if args.precision:
        config.precision = args.precision
    if args.correctness:
        config.correctness = Correctness[args.correctness.upper()]
        print(f"Correctness: {config.correctness}")
    if args.n:
        config.n = args.n
    if args.method:
        config.method = Method[args.method.upper()]
    if args.cross_validation:
        config.cross_validation = args.cross_validation
    if args.regression_score:
        config.regression_score = RegressionScore[args.regression_score.upper()]
    if args.coeff_threshold:
        config.coeff_threshold = args.coeff_threshold
    if args.use_cutoff is not None:
        config.use_cutoff = args.use_cutoff
    if args.coeff_cutoff:
        config.coeff_cutoff = args.coeff_cutoff
    if args.intercept_cutoff:
        config.intercept_cutoff = args.intercept_cutoff
    if args.ugly_factor:
        config.ugly_factor = args.ugly_factor

    # Enable MILP if specified
    if args.milp:
        config.milp_enabled = True
        config.milp_solver = MILPSolver[args.milp.upper()]

    # Print the current configuration
    config.print_all()

    # Implement your analysis logic here
    print(f"Running analysis on {config.file_path} with function {config.func_name}...")

    # Load the vtrace, vassume, and vdistr data
    trace_file = fuzz_and_trace(config.file_path, config.func_name, config.n)

    equations, _, _ = bitween(trace_file)

    if config.correctness == Correctness.VERIFICATION:
        fuzz_and_verify(config.file_path, config.func_name, equations)
    elif config.correctness == Correctness.FUZZING:
        fuzz_and_check(config.file_path, config.func_name, equations, config.n * 10)
    elif config.correctness == Correctness.NONE:
        pass
    else:
        raise ValueError(f"Invalid correctness option: {config.correctness}")

    decorate_assertions(config.file_path, config.func_name, out_path, equations)

    return


if __name__ == "__main__":
    run()
