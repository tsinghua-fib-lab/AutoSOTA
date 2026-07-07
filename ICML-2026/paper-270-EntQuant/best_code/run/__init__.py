import logging

logger = logging.getLogger(__name__)


def setup_env(
    read_bashrc: bool = True,
    pythonpath: bool = True,
    dotenv: bool = True,
    cwd: bool = True,
    project_root_env_var: bool = True,
    set_ninja_path: bool = True,
):
    """
    Utility function to make the setup of the project root robust and allows you to run the entrypoints from anywhere.
    This is a simple wrapper around the `rootutils.setup_root` function from https://github.com/ashleve/rootutils.
    To make this work properly, an empty file `.project-root` must be present in the root of the project.
    By Default, we also use a `.env` file to set environment variables.

    This function also parses the environment variables from the bashrc file to make sure that the environment
    variables are set.

    Notes: Execute this before Huggingface packages are imported to make sure that `HF_HOME` is already defined as
        an environment variable.
    """

    if set_ninja_path:
        try:
            import os

            import ninja

            os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
            logger.debug(f"Added ninja.BIN_DIR to PATH: {ninja.BIN_DIR}")
        except Exception as e:
            logger.warning(
                f"Failed to add ninja.BIN_DIR to PATH: {e}\n"
                "ninja is required for compiling the nvCOMP and qfloat8 backend."
            )

    def _can_reach_hf(timeout=1):
        """Check if huggingface.co is reachable."""
        import socket

        try:
            socket.setdefaulttimeout(timeout)
            socket.create_connection(("huggingface.co", 443), timeout=timeout)
            return True
        except (socket.error, OSError):
            return False

    if not _can_reach_hf():
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"

    if read_bashrc:
        try:
            import os
            import subprocess

            # Parse bashrc and extract environment variables
            output = subprocess.check_output(["bash", "-c", "source ~/.bashrc && env"], text=True)
            for line in output.splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key] = value
        except Exception as e:
            logger.warning(f"Failed to parse environment variables from bashrc: {e}")

    import rootutils

    rootutils.setup_root(
        __file__,
        indicator=".project-root",
        pythonpath=pythonpath,
        dotenv=dotenv,
        cwd=cwd,
        project_root_env_var=project_root_env_var,  # or set PROJECT_ROOT in .env
    )
