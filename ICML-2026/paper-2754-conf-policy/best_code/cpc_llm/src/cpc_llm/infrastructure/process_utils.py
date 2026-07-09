import logging
import subprocess

from omegaconf import DictConfig
from typing import List, Tuple

logger = logging.getLogger(__name__)


def get_output_or_exit(command: str) -> str:
    """
    Runs the bash command and streams the output
    """
    logger.info(f"Running command: {command}")
    p = subprocess.Popen(
        args=command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in p.stdout:
        logger.info(f"Stdout: {line}")
    p.wait()
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise ValueError(
            f"Command '{command}' has failed.\nStdout: {stdout}\nStderr: {stderr}"
        )
    return stdout


def create_slurm_submission(
    job_name: str,
    py_commands: str,
    slurm_script_path: str,
    wait: bool = True,
) -> str:
    wait_str = ""
    if wait:
        wait_str = "--wait"
    return f'sbatch {wait_str} --parsable --job-name={job_name} {slurm_script_path} "{py_commands}"'


def create_lsf_submission(
    job_name: str,
    py_commands: str,
    lsf_script_path: str,
    str_to_replace: str = "command",
    wait: bool = True,
) -> str:
    wait_str = ""
    if wait:
        wait_str = "-K"
    return f'sed "s|{str_to_replace}|{py_commands}|g" < {lsf_script_path} | bsub -J {job_name} {wait_str}'


def create_job_submission(
    cfg: DictConfig,
    job_name: str,
    py_commands: str,
    lsf_script_path: str,
    slurm_script_path: str,
    wait: bool = True,
) -> str:
    if cfg.job_submission_system == "slurm":
        return create_slurm_submission(
            job_name, py_commands, slurm_script_path, wait=wait
        )
    elif cfg.job_submission_system == "lsf":
        return create_lsf_submission(job_name, py_commands, lsf_script_path, wait=wait)
    else:
        raise ValueError(
            f"Unrecognized job submission system: {cfg.job_submission_system}"
        )


def run_all_commands_and_wait_until_all_completed(
    commands: List[str], ignore_failures: bool = False
) -> Tuple[List[str], List[str], List[int]]:
    """
    Runs all the commands in bash shells in parallel processes and
    waits until all child processes have completed. Returns the outputs from all commands.
    """
    all_processes = []
    for cmd in commands:
        p = subprocess.Popen(
            args=cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"Running command: {cmd}")
        all_processes.append(p)
    all_outputs = []
    all_stderrs = []
    returncodes = []
    for p in all_processes:
        stdout_str = ""
        for line in p.stdout:
            logger.info(f"Stdout: {line}")
            stdout_str += f"{line}\n"
        all_outputs.append(stdout_str)

    for p, cmd in zip(all_processes, commands):
        p.wait()
        # If the process failed, return early!
        _, stderr = p.communicate()
        if p.returncode != 0 and not ignore_failures:
            raise ValueError(f"Command '{cmd}' has failed.\nStderr: {stderr}")
        all_stderrs.append(stderr)
        returncodes.append(p.returncode)
    return all_outputs, all_stderrs, returncodes
