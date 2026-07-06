from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.ablation.config_sketch_solve import CONFIG
from bench.ablation.run_utils import run_ablation


if __name__ == "__main__":
    run_ablation(CONFIG, "bench.ablation.run_sketch_solve")
