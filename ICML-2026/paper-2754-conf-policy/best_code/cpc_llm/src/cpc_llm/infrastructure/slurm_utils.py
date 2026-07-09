import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid


from typing import IO, Callable, List, Optional, Tuple

# Optional hook called after each direct subprocess completes.
# Used by modal_runner to commit the outputs volume mid-pipeline
# so logs survive hard kills (OOM, timeout).
_post_subprocess_hook: Callable[[], None] | None = None


def set_post_subprocess_hook(hook: Callable[[], None] | None) -> None:
    """Register a callback invoked after each direct subprocess completes."""
    global _post_subprocess_hook
    _post_subprocess_hook = hook


@contextlib.contextmanager
def rm_slurm_env():
    """Context manager for temporarily removing slurm environment variables that
    may be mistakenly inherited by new slurm jobs launched from the current program.
    """
    old_environ = {
        x: os.environ.pop(x, None)
        for x in [
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPU_BIND",
            "SLURM_CPU_BIND_TYPE",
            "SLURM_CPU_BIND_LIST",
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_LOCALID",
            "SLURM_NODEID",
            "SLURM_NTASKS",
            "SLURM_PROCID",
        ]
    }
    try:
        yield
    finally:
        for k, val in old_environ.items():
            if val is not None:
                os.environ[k] = val


def submit_sbatch_job(
    sbatch_cmd: str, dump_dir: str, blocking: bool = True
) -> Tuple[subprocess.Popen, str]:
    """Submit sbatch job, get slurm job ID, and decide whether to wait or not.
    Returns both the process and the slurm job ID
    """
    sbatch_output = os.path.join(dump_dir, f"sbatch_output_{uuid.uuid1()}")
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(sbatch_cmd)
        f.close()
        p = subprocess.Popen(
            args=[f"sbatch --wait --parsable {f.name} > {sbatch_output}"],
            shell=True,
            executable="/bin/bash",
        )
        # now wait for the sbatch_output file to appear
        logging.info("Waiting for job submission...")
        while not os.path.exists(sbatch_output) or os.path.getsize(sbatch_output) == 0:
            time.sleep(1)
        # get job ID
        slurm_job_id = open(sbatch_output).read().strip()
        logging.info(f"Submitted slurm job {slurm_job_id}")
        os.remove(sbatch_output)
        # copy submission file to current directory with job ID in filename
        new_submission_fp = os.path.join(dump_dir, f"{slurm_job_id}_submission.sbatch")
        shutil.copy(f.name, new_submission_fp)
        logging.info(
            f"Slurm submission script for job {slurm_job_id} written to {new_submission_fp}."
        )
        if blocking:
            logging.info(f"Waiting for job {slurm_job_id}...")
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Slurm job {slurm_job_id} failed.")
            logging.info(f"Slurm job {slurm_job_id} succeeded")
        return p, slurm_job_id


def _get_gpu_check_script(num_gpus_required: int) -> str:
    """Returns a bash script snippet that verifies GPU availability before running the job."""
    if num_gpus_required <= 0:
        return ""
    return f'''
# GPU availability check
echo "Checking GPU availability..."
NUM_GPUS_VISIBLE=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS_VISIBLE" -lt "{num_gpus_required}" ]; then
    echo "ERROR: Required {num_gpus_required} GPU(s) but only $NUM_GPUS_VISIBLE available/visible."
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    exit 1
fi
echo "GPU check passed: $NUM_GPUS_VISIBLE GPU(s) available."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

'''


@rm_slurm_env()
def submit_cmd_to_slurm(
    py_cmd: str,
    dump_dir: str,
    blocking: bool = False,
    setup_str: str = (
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        '[ -f "$HOME/.env.slurm" ] && source "$HOME/.env.slurm"'
    ),
    path_to_repo: Optional[str] = None,
    nodes_to_exclude_str: str = 'ai4dd-b200-st-comp-48xl-51,ai4dd-b200-st-comp-48xl-69,ai4dd-b200-st-comp-48xl-68,ai4dd-b200-st-comp-48xl-82,ai4dd-b200-st-comp-48xl-7', #None,  ## Example: 'b200-st-b200-2-4,b200-st-b200-2-1'
    **slurm_kwargs,
) -> Tuple[subprocess.Popen, str]:
    if path_to_repo is None:
        path_to_repo = "~/conformal-policy-control" #"~/cdt-agents"

    # Add GPU check if GPUs are requested
    num_gpus_required = slurm_kwargs.get("gpus_per_node", 0)
    gpu_check_script = _get_gpu_check_script(num_gpus_required)

    sbatch_str = f"source ~/.bashrc\ncd {path_to_repo}\nsource .venv/bin/activate\n{gpu_check_script}{py_cmd}"
    # add #SBATCH commands on top
    if "output" not in slurm_kwargs:
        slurm_kwargs["output"] = f"{dump_dir}/%x_%j.logs"
    if "error" not in slurm_kwargs:
        slurm_kwargs["error"] = f"{dump_dir}/%x_%j.logs"
    sbatch_prefix = ["#!/usr/bin/env bash"]
    sbatch_prefix.extend(
        [f"#SBATCH --{k.replace('_', '-')}={v}" for k, v in slurm_kwargs.items()]
    )
    # Add node exclusion as a #SBATCH directive.
    # Slurm parses each #SBATCH line by splitting on whitespace, so the exclude
    # list must be comma-separated with no spaces (e.g. "node1,node2"), otherwise
    # anything after the first space is treated as a separate invalid directive.
    if nodes_to_exclude_str is not None and nodes_to_exclude_str.strip():
        exclude_str = ",".join(n.strip() for n in nodes_to_exclude_str.split(","))
        sbatch_prefix.append(f"#SBATCH --exclude={exclude_str}")
    sbatch_str = "\n".join(sbatch_prefix) + "\n\n" + setup_str + "\n\n" + sbatch_str
    return submit_sbatch_job(sbatch_str, dump_dir, blocking=blocking)


def wait_for_slurm_jobs_to_complete(jobs: List[Tuple[subprocess.Popen, str]]):
    """Given a list of tuples of (process, slurm job ID), wait for all to complete."""
    processes = ", ".join([f"{p.pid}" for (p, _) in jobs])
    job_ids = ", ".join([f"{job_id}" for (_, job_id) in jobs])
    logging.info(
        f"Waiting for processes {processes} (slurm jobs {job_ids}) to complete..."
    )
    for p, job_id in jobs:
        rc = p.wait()
        stdout, stderr = p.communicate()
        logging.info(f"slurm job {job_id} stdout: {stdout}")
        logging.info(f"slurm job {job_id} stderr: {stderr}")
        if rc != 0:
            raise RuntimeError(f"Process {p.pid} (slurm job {job_id}) failed!")
        logging.info(f"Slurm job {job_id} succeeded!")


def _tee_stream(source: IO[bytes], *destinations: IO[bytes]) -> None:
    """Copy lines from source to all destinations until EOF."""
    for line in iter(source.readline, b""):
        for dest in destinations:
            dest.write(line)
            dest.flush()


def submit_cmd_direct(
    py_cmd: str,
    dump_dir: str,
    blocking: bool = True,
    **kwargs,
) -> Tuple[subprocess.Popen, str]:
    """Run a Python command directly as a subprocess (no SLURM).

    Drop-in alternative to submit_cmd_to_slurm for environments without
    a job scheduler (e.g., Modal, local development).

    Subprocess output is tee'd to both a log file and the parent's stderr,
    making it visible in ``modal app logs`` while still preserving a
    persistent log on disk.

    Extra kwargs (slurm_args, path_to_repo, etc.) are ignored.
    """
    os.makedirs(dump_dir, exist_ok=True)
    log_path = os.path.join(dump_dir, f"direct_{uuid.uuid1()}.log")
    logging.info(f"Running directly: {py_cmd}")
    logging.info(f"Log path: {log_path}")

    log_file = open(log_path, "wb")  # noqa: SIM115
    p = subprocess.Popen(
        py_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Tee subprocess output to both the log file and parent stderr
    tee_thread = threading.Thread(
        target=_tee_stream,
        args=(p.stdout, log_file, sys.stderr.buffer),
        daemon=True,
    )
    tee_thread.start()

    job_id = f"direct-{p.pid}"
    # Attach resources so wait_for_direct_jobs_to_complete can clean up
    p._direct_log_path = log_path
    p._tee_thread = tee_thread
    p._log_file = log_file
    if blocking:
        logging.info(f"Waiting for process {p.pid}...")
        p.wait()
        tee_thread.join()
        p.stdout.close()
        log_file.close()
        if p.returncode != 0:
            if _post_subprocess_hook:
                _post_subprocess_hook()
            _log_direct_failure(p.returncode, log_path, py_cmd)
            raise RuntimeError(f"Direct execution failed: {py_cmd}")
        logging.info(f"Process {p.pid} succeeded")
        if _post_subprocess_hook:
            _post_subprocess_hook()

    return p, job_id


def _log_direct_failure(rc: int, log_path: str, cmd: str = ""):
    """Log the tail of a failed direct subprocess for debugging."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
            tail = "".join(lines[-30:])
            logging.error(f"Command failed (rc={rc}). Output tail:\n{tail}")
    except OSError:
        logging.error(f"Command failed (rc={rc}). See {log_path}")


def wait_for_direct_jobs_to_complete(jobs: List[Tuple[subprocess.Popen, str]]):
    """Wait for direct subprocess jobs to complete."""
    for p, job_id in jobs:
        rc = p.wait()
        # Clean up tee thread and log file handle from submit_cmd_direct
        tee_thread = getattr(p, "_tee_thread", None)
        log_file_obj = getattr(p, "_log_file", None)
        if tee_thread:
            tee_thread.join()
            p.stdout.close()
        if log_file_obj:
            log_file_obj.close()
        if rc != 0:
            if _post_subprocess_hook:
                _post_subprocess_hook()
            log_path = getattr(p, "_direct_log_path", None)
            if log_path:
                _log_direct_failure(rc, log_path)
            raise RuntimeError(f"Process {job_id} failed with return code {rc}")
        logging.info(f"Process {job_id} succeeded")
        if _post_subprocess_hook:
            _post_subprocess_hook()
