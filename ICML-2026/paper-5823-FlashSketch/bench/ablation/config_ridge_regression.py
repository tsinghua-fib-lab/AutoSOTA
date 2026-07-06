from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.ablation.config_base import AblationConfig
from bench.ablation.datasets import ablation_datasets
from bench.ablation.methods import ablation_methods
from bench.e2e.tasks.ridge_regression import RidgeRegressionConfig


CONFIG = AblationConfig(
    datasets=ablation_datasets(),
    methods=ablation_methods(),
    tasks=[RidgeRegressionConfig()],
)
