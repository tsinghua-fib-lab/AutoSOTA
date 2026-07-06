from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.camera_ready.config_base import CameraReadyConfig
from bench.camera_ready.datasets import camera_ready_datasets
from bench.camera_ready.methods import camera_ready_methods
from bench.e2e.tasks.gram_approx import GramApproxConfig


CONFIG = CameraReadyConfig(
    datasets=camera_ready_datasets(),
    methods=camera_ready_methods(),
    tasks=[GramApproxConfig()],
)
