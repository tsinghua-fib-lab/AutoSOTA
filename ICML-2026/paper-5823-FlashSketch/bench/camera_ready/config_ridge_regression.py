from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.camera_ready.config_base import CameraReadyConfig
from bench.camera_ready.datasets import camera_ready_datasets_ridge
from bench.camera_ready.methods import camera_ready_methods
from bench.e2e.tasks.ridge_regression import RidgeRegressionConfig


CONFIG = CameraReadyConfig(
    datasets=camera_ready_datasets_ridge(),
    methods=camera_ready_methods(),
    tasks=[RidgeRegressionConfig()],
)
