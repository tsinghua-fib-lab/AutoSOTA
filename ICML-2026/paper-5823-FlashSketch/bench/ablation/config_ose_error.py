from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.ablation.config_base import AblationConfig
from bench.ablation.datasets import ablation_datasets
from bench.ablation.methods import ablation_methods
from bench.e2e.tasks.ose_error import OseErrorConfig


CONFIG = AblationConfig(
    datasets=ablation_datasets(),
    methods=ablation_methods(),
    tasks=[OseErrorConfig()],
)
