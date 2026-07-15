import os
import subprocess


def install_compressor():
    """
    Install APPFL supported compressors into .compressor directory.
    """
    current_path = os.path.dirname(__file__)
    script_path = os.path.join(current_path, "install.sh")
    subprocess.run(["bash", script_path])
