from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import subprocess

from logging_utils import get_logger


_LOGGER = get_logger("kernels.build")


def main():
    """Entry point for building CUDA extensions via Makefile."""
    cmd = ["make", "kernels.build"]
    _LOGGER.info("Running %s", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
