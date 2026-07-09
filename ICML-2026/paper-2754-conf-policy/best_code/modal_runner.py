"""
Modal script for running CPC-LLM experiments on GPU.

Usage:
    # Environment test (verifies GPU, imports, configs)
    modal run modal_runner.py --test

    # Smoke test (tiny model, 1 round, ~5 min)
    modal run modal_runner.py --smoke

    # Smoke test with cached outputs (skips completed stages on re-run)
    modal run modal_runner.py --smoke --cache

    # Check progress (tail latest subprocess log from outputs volume)
    modal run modal_runner.py --check-progress

    # Custom config
    modal run modal_runner.py --config-name cpc_llm

    # With cached outputs (skips completed stages on re-run)
    modal run modal_runner.py --config-name cpc_llm --cache

    # With overrides
    modal run modal_runner.py --config-name cpc_llm \
        --overrides "num_marge_rounds=1"

Headless / deploy mode (survives laptop sleep or terminal close):
    # One-time setup — redeploy whenever modal_runner.py changes
    modal deploy modal_runner.py

    # Trigger a run that continues even if the local client disconnects
    modal run modal_runner.py --deploy
    modal run modal_runner.py --deploy --smoke --cache
    modal run modal_runner.py --deploy --config-name cpc_llm

    # Check status of a headless job (prints call ID after --deploy)
    modal run modal_runner.py --status <call_id>

    # Cancel a running headless job
    modal run modal_runner.py --cancel <call_id>

Sweep mode (parallel runs across seeds/alpha/etc.):
    # Define a sweep config (see sweep_configs/ for examples)
    # Requires: modal deploy modal_runner.py  (run once first)
    modal run modal_runner.py --sweep sweep_configs/test_sweep.yaml

    # Check status of all jobs from a sweep
    modal run modal_runner.py --sweep-status .sweep_runs/<record>.json
"""

from pathlib import Path

import logging

import modal

app = modal.App("cpc-llm")

script_dir = Path(__file__).parent.absolute()

LOCAL_DIR_IGNORE = [
    "**/__pycache__/**",
    "**/.git/**",
    "**/.pytest_cache/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/*.log",
    "**/outputs/**",
    "**/wandb/**",
    "**/.wandb/**",
    "**/.ruff_cache/**",
    "**/.venv/**",
    "**/notebooks/**",
]

# Persistent volume for HuggingFace model cache
hf_cache_volume = modal.Volume.from_name("cpc-llm-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/vol/hf-cache"

# Persistent volume for pipeline outputs (training checkpoints, generated data)
outputs_volume = modal.Volume.from_name("cpc-llm-outputs", create_if_missing=True)
OUTPUTS_PATH = "/vol/outputs"

# Base dependencies (cached across code changes)
# Use uv for fast dependency resolution (pip backtracking on s3fs/aiobotocore is brutal)
_base_deps = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "'torch>=2.2,<2.9' "
        "transformers>=4.41 "
        "datasets>=2.20 "
        "accelerate>=0.31 "
        "peft>=0.11 "
        "trl>=0.9 "
        "botorch>=0.11 "
        "pytorch-holo>=0.0.1 "
        "numpy>=1.26 "
        "pandas>=2.2 "
        "scipy>=1.13 "
        "scikit-learn>=1.5 "
        "pynndescent>=0.5 "
        "hydra-core>=1.3 "
        "omegaconf>=2.3 "
        "tqdm>=4.66 "
        "s3fs>=2024.5 "
        "boto3>=1.34 "
        "backoff>=2.2 "
        "wandb>=0.17 "
    )
)

# Image with local code (add_local_dir must be last for cache efficiency)
image = _base_deps.add_local_dir(
    str(script_dir),
    remote_path="/app/cpc",
    ignore=LOCAL_DIR_IGNORE,
)


def _setup_env():
    """Common environment setup for remote functions."""
    import os
    import sys

    app_path = "/app/cpc"
    sys.path.insert(0, app_path)
    sys.path.insert(0, f"{app_path}/cpc_llm/src")
    os.chdir(app_path)

    # Point HuggingFace cache at the persistent volume
    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_PATH}/hub"

    # Reduce CUDA memory fragmentation from repeated model load/unload cycles
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Ensure subprocesses can find cpc_llm
    python_path = os.environ.get("PYTHONPATH", "")
    new_paths = f"{app_path}:{app_path}/cpc_llm/src"
    if new_paths not in python_path:
        os.environ["PYTHONPATH"] = (
            f"{new_paths}:{python_path}" if python_path else new_paths
        )

    return app_path


def _persist_wandb_logs(
    app_path: str, output_dir: str, logger: "logging.Logger"
) -> None:
    """Copy wandb offline logs from container-local /app/cpc/outputs/ to the persistent volume."""
    import shutil
    from pathlib import Path

    wandb_src = Path(app_path) / "outputs"
    wandb_dst = Path(output_dir) / "wandb_logs"
    if not wandb_src.exists():
        logger.info("No wandb logs found to persist")
        return
    wandb_runs = list(wandb_src.rglob("wandb/offline-run-*")) + list(
        wandb_src.rglob("wandb/run-*")
    )
    if not wandb_runs:
        logger.info("No wandb run directories found")
        return
    wandb_dst.mkdir(parents=True, exist_ok=True)
    for run_dir in wandb_runs:
        dst = wandb_dst / run_dir.name
        if dst.exists():
            continue
        logger.info(f"Copying wandb logs: {run_dir} -> {dst}")
        shutil.copytree(run_dir, dst)
    logger.info(f"Persisted {len(wandb_runs)} wandb run(s) to {wandb_dst}")


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=57600,  # 16 hours
    secrets=[modal.Secret.from_name("wandb")],
    volumes={HF_CACHE_PATH: hf_cache_volume, OUTPUTS_PATH: outputs_volume},
)
def run_experiment_remote(
    config_name: str = "cpc_llm",
    overrides: list | None = None,
    cache: bool = False,
):
    """Run CPC-LLM pipeline with the specified Hydra config.

    Args:
        config_name: Hydra config to use.
        overrides: List of Hydra overrides.
        cache: If True, reuse pipeline outputs (checkpoints, generated data)
            from previous runs, skipping completed stages. If False (default),
            clear the outputs volume first. Either way, outputs are persisted
            to the volume for future --cache runs.
    """
    import hashlib
    import logging
    import shutil

    app_path = _setup_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CPC-LLM with config: {config_name}")

    import torch

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    output_dir = OUTPUTS_PATH  # fallback if exception occurs before config resolution
    try:
        import hydra
        from omegaconf import OmegaConf

        override_list = overrides or []
        # Force direct execution (no SLURM) and local storage.
        # Use placeholder output dir — we'll override after hashing the config.
        modal_overrides = [
            "job_submission_system=direct",
            "parent_output_dir=null",
            "local_output_dir=null",
            f"path_to_repo={app_path}",
        ]
        override_list.extend(modal_overrides)

        config_path = f"{app_path}/cpc_llm/config"
        with hydra.initialize_config_dir(config_dir=config_path, version_base="1.1"):
            cfg = hydra.compose(config_name=config_name, overrides=override_list)

            # Hash the resolved config (excluding infrastructure fields that
            # vary per-run) to get a stable, content-addressed cache key.
            cfg_for_hash = OmegaConf.to_container(cfg, resolve=True)
            for key in ("local_output_dir", "parent_output_dir", "path_to_repo"):
                cfg_for_hash.pop(key, None)
            config_hash = hashlib.sha256(
                OmegaConf.to_yaml(cfg_for_hash).encode()
            ).hexdigest()[:12]

            output_dir = f"{OUTPUTS_PATH}/{config_name}_{config_hash}"
            output_path = Path(output_dir)
            logger.info(f"Output dir: {output_dir}")

            if not cache:
                if output_path.exists():
                    for child in output_path.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                    outputs_volume.commit()
                    logger.info("Cleared outputs (--no-cache)")
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Reusing cached outputs at {output_dir}")

            # Now set the real output dir in the config
            cfg.local_output_dir = output_dir
            logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

            from cpc_llm.infrastructure.slurm_utils import set_post_subprocess_hook
            from cpc_llm.main import run_pipeline

            set_post_subprocess_hook(outputs_volume.commit)
            run_pipeline(cfg, on_round_complete=outputs_volume.commit)

        # Copy wandb logs to the persistent volume before container exits
        _persist_wandb_logs(app_path, output_dir, logger)

        # Persist any newly downloaded models and pipeline outputs to volumes
        hf_cache_volume.commit()
        outputs_volume.commit()

        logger.info("CPC-LLM pipeline completed!")
        return "Success"

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        _persist_wandb_logs(app_path, output_dir, logger)
        outputs_volume.commit()  # persist partial work so --cache can resume
        raise


@app.function(
    image=image,
    gpu="A100",
    memory=16384,
    timeout=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def test_environment():
    """Verify GPU, imports, configs, and HF cache."""
    _setup_env()

    import torch

    results = []
    results.append(f"Python: {__import__('sys').version}")
    results.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        results.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    results.append(f"PyTorch: {torch.__version__}")

    import transformers

    results.append(f"Transformers: {transformers.__version__}")

    from pathlib import Path

    config_path = Path("cpc_llm/config")
    if config_path.exists():
        config_files = sorted(f.name for f in config_path.glob("*.yaml"))
        results.append(f"Configs: {config_files}")

    import cpc_llm  # noqa: F401

    results.append("cpc_llm: imported OK")

    hf_cache = Path(HF_CACHE_PATH)
    cached_models = list(hf_cache.glob("hub/models--*"))
    results.append(f"HF cache: {len(cached_models)} models cached")

    for line in results:
        print(line)

    return "Environment test passed!"


@app.function(
    image=image,
    timeout=60,
    volumes={OUTPUTS_PATH: outputs_volume},
)
def check_progress_remote() -> str:
    """Print the tail of the most recent subprocess log on the outputs volume."""
    from datetime import datetime

    outputs_volume.reload()

    outputs = Path(OUTPUTS_PATH)
    log_files = sorted(outputs.rglob("direct_*.log"), key=lambda p: p.stat().st_mtime)

    if not log_files:
        print("No subprocess logs found on the outputs volume.")
        return "No logs found"

    latest = log_files[-1]
    st = latest.stat()
    print(f"Latest log: {latest}")
    print(f"Size: {st.st_size} bytes")
    print(f"Modified: {datetime.fromtimestamp(st.st_mtime).isoformat()}")
    print("--- Last 50 lines ---")

    lines = latest.read_text().splitlines()
    for line in lines[-50:]:
        print(line)

    return str(latest)


def _build_sweep_jobs(
    sweep_path: str,
) -> tuple[str, bool, list[str], list[list[str]]]:
    """Parse sweep config and return job override lists.

    The sweep config is a plain YAML file (not Hydra) with:
        base_config: <hydra config name>
        cache: <bool>
        overrides: [<fixed hydra overrides>]
        parameters:
            <hydra.override.key>: [val1, val2, ...]

    Returns the cartesian product of all parameter values as override lists.
    Workaround: if ``initial_seed`` is swept, ``last_seed`` is auto-set to
    match so each container runs exactly one seed.

    Args:
        sweep_path: Path to sweep YAML file.

    Returns:
        Tuple of (base_config, cache, fixed_overrides, jobs) where each job
        is a list of Hydra override strings.
    """
    import itertools

    import yaml

    sweep_file = Path(sweep_path)
    if not sweep_file.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")

    with open(sweep_file) as f:
        sweep_cfg = yaml.safe_load(f)

    if not isinstance(sweep_cfg, dict):
        raise ValueError(f"Sweep config is empty or invalid: {sweep_path}")

    if "base_config" not in sweep_cfg:
        raise ValueError("Sweep config must specify 'base_config'")

    base_config: str = sweep_cfg["base_config"]
    cache: bool = sweep_cfg.get("cache", False)
    fixed_overrides: list[str] = sweep_cfg.get("overrides", [])
    parameters: dict[str, list] = sweep_cfg.get("parameters", {})

    for key, values in parameters.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Parameter '{key}' must be a list, got {type(values).__name__}"
            )

    if not parameters:
        return base_config, cache, fixed_overrides, [list(fixed_overrides)]

    param_names = list(parameters.keys())
    param_values = [parameters[k] for k in param_names]

    jobs: list[list[str]] = []
    for combo in itertools.product(*param_values):
        job_overrides = list(fixed_overrides)
        for name, value in zip(param_names, combo):
            job_overrides.append(f"{name}={value}")

        # Seed workaround: keep last_seed in sync so the main loop runs once.
        # Skipped if last_seed is already set in fixed_overrides or parameters.
        if "initial_seed" in parameters and "last_seed" not in parameters:
            has_last_seed = any(o.startswith("last_seed=") for o in fixed_overrides)
            if not has_last_seed:
                seed_val = combo[param_names.index("initial_seed")]
                job_overrides.append(f"last_seed={seed_val}")

        jobs.append(job_overrides)

    return base_config, cache, fixed_overrides, jobs


def _run_sweep(sweep_path: str) -> None:
    """Parse sweep config, spawn all jobs, print call ID table.

    Always headless — requires prior ``modal deploy modal_runner.py``.

    Args:
        sweep_path: Path to sweep YAML file.
    """
    import json
    from datetime import datetime

    base_config, cache, fixed_overrides, jobs = _build_sweep_jobs(sweep_path)

    fn = modal.Function.from_name("cpc-llm", "run_experiment_remote")

    print(f"Sweep: {sweep_path}")
    print(f"Base config: {base_config}")
    print(f"Cache: {cache}")
    if fixed_overrides:
        print(f"Fixed overrides: {fixed_overrides}")
    print(f"Total jobs: {len(jobs)}")
    print()

    call_ids: list[tuple[int, list[str], str]] = []
    for i, job_overrides in enumerate(jobs):
        call = fn.spawn(base_config, job_overrides, cache=cache)
        call_id = call.object_id
        call_ids.append((i, job_overrides, call_id))
        override_str = ", ".join(job_overrides)
        print(f"  [{i}] {call_id}  {override_str}")

    print()
    print("=" * 72)
    print(f"Spawned {len(call_ids)} jobs")
    print("=" * 72)

    # Save record for --sweep-status
    sweep_record = {
        "sweep_path": sweep_path,
        "timestamp": datetime.now().isoformat(),
        "base_config": base_config,
        "cache": cache,
        "jobs": [
            {"index": i, "call_id": cid, "overrides": ovr} for i, ovr, cid in call_ids
        ],
    }
    record_dir = Path(".sweep_runs")
    record_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(sweep_path).stem
    record_path = record_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    record_path.write_text(json.dumps(sweep_record, indent=2))

    print()
    print(f"Sweep record: {record_path}")
    print()
    print("Check all statuses:")
    print(f"  modal run modal_runner.py --sweep-status {record_path}")
    print()
    print("Check individual job:")
    print("  modal run modal_runner.py --status <call_id>")
    print()
    print("Dashboard: https://modal.com/apps/samuelstanton/cpc-llm")


def _show_sweep_status(record_path: str) -> None:
    """Show status of all jobs from a sweep run.

    Args:
        record_path: Path to a ``.sweep_runs/*.json`` record file.
    """
    import json

    record = json.loads(Path(record_path).read_text())

    print(f"Sweep: {record['sweep_path']}")
    print(f"Started: {record['timestamp']}")
    print(f"Jobs: {len(record['jobs'])}")
    print()
    print(f"{'Job':>4}  {'Status':<12}  {'Call ID':<32}  Overrides")
    print("-" * 80)

    for job in record["jobs"]:
        call_id = job["call_id"]
        try:
            call = modal.FunctionCall.from_id(call_id)
            graph = call.get_call_graph()
            status = graph[0].status.name
        except Exception as exc:
            status = f"UNKNOWN ({type(exc).__name__})"
        override_str = ", ".join(job["overrides"])
        print(f"{job['index']:>4}  {status:<12}  {call_id:<32}  {override_str}")

    print()


def _show_status(call_id: str) -> None:
    """Show status and result of a spawned job. Pure local operation."""
    from modal.call_graph import InputStatus

    call = modal.FunctionCall.from_id(call_id)
    graph = call.get_call_graph()
    root = graph[0]
    status = root.status

    print(f"Call ID:  {call_id}")
    print(f"Status:  {status.name}")

    if status == InputStatus.SUCCESS:
        try:
            result = call.get(timeout=0)
            print(f"Result:  {result}")
        except TimeoutError:
            print("Result:  (retry in a moment)")
    elif status in (InputStatus.FAILURE, InputStatus.INIT_FAILURE):
        try:
            call.get(timeout=0)
        except TimeoutError:
            print("Error:   (details not yet cached — retry in a moment)")
        except Exception as exc:
            print(f"Error:   {exc}")
    elif status == InputStatus.TERMINATED:
        print("(Job was cancelled)")
    elif status == InputStatus.TIMEOUT:
        print("(Job timed out)")
    elif status == InputStatus.PENDING:
        print("(Job is queued or running)")
        print()
        print("Tail logs (if the job has started):")
        print("  modal run modal_runner.py --check-progress")
    else:
        print("(Unrecognised status — check Modal dashboard)")


def _cancel_job(call_id: str) -> None:
    """Cancel a running spawned job. Pure local operation."""
    call = modal.FunctionCall.from_id(call_id)
    call.cancel()
    print(f"Cancellation requested for call: {call_id}")
    print("The job may take a few seconds to terminate.")


@app.local_entrypoint()
def main_entrypoint(
    config_name: str = "cpc_llm",
    test: bool = False,
    smoke: bool = False,
    cache: bool = False,
    check_progress: bool = False,
    deploy: bool = False,
    sweep: str = "",
    sweep_status: str = "",
    status: str = "",
    cancel: str = "",
    overrides: str = None,
):
    """
    Local entrypoint for running CPC-LLM on Modal.

    Usage:
        # Environment test
        modal run modal_runner.py --test

        # Smoke test (tiny model, ~5 min)
        modal run modal_runner.py --smoke

        # Smoke test with cached outputs (skips completed stages)
        modal run modal_runner.py --smoke --cache

        # Check progress (tail latest subprocess log)
        modal run modal_runner.py --check-progress

        # Custom config
        modal run modal_runner.py --config-name cpc_llm

        # With cached outputs (skips completed stages)
        modal run modal_runner.py --config-name cpc_llm --cache

        # With overrides
        modal run modal_runner.py --config-name cpc_llm \
            --overrides "num_marge_rounds=1"

        # Headless run (survives laptop sleep / terminal close)
        # Requires: modal deploy modal_runner.py  (run once first)
        modal run modal_runner.py --deploy
        modal run modal_runner.py --deploy --smoke --cache

        # Check status of a headless job (call ID printed after --deploy)
        modal run modal_runner.py --status <call_id>

        # Cancel a running headless job
        modal run modal_runner.py --cancel <call_id>

        # Sweep mode (parallel runs across seeds/alpha/etc.)
        # Requires: modal deploy modal_runner.py  (run once first)
        modal run modal_runner.py --sweep sweep_configs/test_sweep.yaml

        # Check status of all jobs from a sweep
        modal run modal_runner.py --sweep-status .sweep_runs/<record>.json
    """
    if sweep:
        _run_sweep(sweep)
    elif sweep_status:
        _show_sweep_status(sweep_status)
    elif test:
        print("Running environment test...")
        result = test_environment.remote()
        print(f"Result: {result}")
    elif check_progress:
        print("Checking progress...")
        result = check_progress_remote.remote()
        print(f"Result: {result}")
    elif status:
        _show_status(status)
    elif cancel:
        _cancel_job(cancel)
    else:
        override_list = []
        if overrides:
            override_list = [o.strip() for o in overrides.split(",")]
        if smoke:
            config_name = "smoke"
            print("Running smoke test (pythia-14m, 1 round)...")
        else:
            print(f"Running CPC-LLM with config: {config_name}")
        if cache:
            print("Using cached outputs volume")
        if deploy:
            fn = modal.Function.from_name("cpc-llm", "run_experiment_remote")
            call = fn.spawn(config_name, override_list, cache=cache)
            call_id = call.object_id
            print(f"Job spawned. Call ID: {call_id}")
            print()
            print("Check status:")
            print(f"  modal run modal_runner.py --status {call_id}")
            print()
            print("Tail logs (most recent subprocess):")
            print("  modal run modal_runner.py --check-progress")
            print()
            print("Cancel:")
            print(f"  modal run modal_runner.py --cancel {call_id}")
            print()
            print("Dashboard: https://modal.com/apps/samuelstanton/cpc-llm")
        else:
            result = run_experiment_remote.remote(
                config_name, override_list, cache=cache
            )
            print(f"Result: {result}")
