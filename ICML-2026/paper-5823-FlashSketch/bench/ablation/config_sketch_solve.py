from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.ablation.config_base import AblationConfig
from bench.ablation.datasets import ablation_datasets
from bench.ablation.methods import ablation_methods
from bench.e2e.tasks.sketch_and_solve_ls import SketchAndSolveConfig


CONFIG = AblationConfig(
    datasets=ablation_datasets(),
    methods=ablation_methods(),
    tasks=[SketchAndSolveConfig()],
)
