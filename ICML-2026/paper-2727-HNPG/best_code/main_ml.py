import argparse
import os
import sys
from pathlib import Path

HYPHOP_ROOT = Path(__file__).resolve().parent / "ml" / "hyphop"


def _setup_hyphop():
    os.chdir(HYPHOP_ROOT)
    if str(HYPHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(HYPHOP_ROOT))


def main():
    ap = argparse.ArgumentParser(description="ML benchmarks: MNIST or MIL")
    ap.add_argument("--task", required=True, choices=["mnist", "mil"])
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="run full benchmark table (run_mnist_table / run_mil_table)",
    )
    args, rest = ap.parse_known_args()

    _setup_hyphop()

    if args.benchmark:
        if args.task == "mnist":
            from run_mnist_table import main as run
        else:
            from run_mil_table import main as run
        run()
        return

    script = "test_mnist.py" if args.task == "mnist" else "test_mil.py"
    sys.argv = [script] + rest

    if args.task == "mnist":
        from test_mnist import main as run
    else:
        from test_mil import main as run

    run()


if __name__ == "__main__":
    main()
