"""Batch experiment runner."""
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import decoupledmarket.constant as constant
from decoupledmarket.main_parallel import overall_test_parallel, setup_simple_logger
from decoupledmarket.performance_monitor import reset_monitor


AGENT_COUNTS = [14]
EXPERIMENT_DAYS = 2

def run_experiments():
    base_dir = constant.Save_Path
    data_db = osp.join(base_dir, "data.db")

    print("=" * 60)
    print("Batch experiment: different agent counts")
    print("  Agent counts:", AGENT_COUNTS)
    print("  Simulation days:", EXPERIMENT_DAYS)
    print("  Output directory:", base_dir)
    print("=" * 60)

    setup_simple_logger()
    original_days = constant.No_Days
    original_agent = constant.Num_agent

    try:
        for num_agent in AGENT_COUNTS:
            print("\n>>> Running Num_agent = {} ...".format(num_agent))
            constant.Num_agent = num_agent
            constant.No_Days = EXPERIMENT_DAYS
            reset_monitor()
            overall_test_parallel(
                executor_type="thread",
                max_workers=int(os.getenv("EXPERIMENT_MAX_WORKERS", "4")),
                batch_size=20,
                enable_monitoring=False,
            )
            dest = osp.join(base_dir, "{}.db".format(num_agent))
            if osp.isfile(data_db):
                shutil.copy(data_db, dest)
                print("  Copied data.db -> {}.db".format(num_agent))
            else:
                print("  warning: data.db not found; skipping copy")
    finally:
        constant.No_Days = original_days
        constant.Num_agent = original_agent

    print("Batch experiments completed. Use scripts/analyze_num_agents.py for analysis.")


if __name__ == "__main__":
    run_experiments()
