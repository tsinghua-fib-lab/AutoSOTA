from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from bench.camera_ready.config_ridge_regression import CONFIG
from bench.camera_ready.run_utils import run_camera_ready


if __name__ == "__main__":
    run_camera_ready(CONFIG, "bench.camera_ready.run_ridge_regression")
