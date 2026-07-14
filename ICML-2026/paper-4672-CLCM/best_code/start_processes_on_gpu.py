#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import yaml
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch WandB sweep runs on specified GPUs, managing GPU allocation automatically."
    )
    parser.add_argument('gpu_ids', type=int, nargs='+', help='List of GPU IDs to use')
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Existing WandB sweep ID (e.g., entity/project/sweep_id). If not provided, creates a new sweep.')
    parser.add_argument('--config', type=str, default='configs/sweep_all_modalities.yaml',
                        help='Path to sweep config YAML (used when creating a new sweep)')
    parser.add_argument('--project', type=str, default='all_modalities_sweep',
                        help='WandB project name (used when creating a new sweep)')
    parser.add_argument('--entity', type=str, default='equilibrium-fisher-control',
                        help='WandB entity name (used when creating a new sweep)')
    parser.add_argument('--min_free_vram', type=int, default=2000,
                        help='Minimum free VRAM (in MiB) required on a GPU to launch an agent (default: 2000)')
    parser.add_argument('--poll_interval', type=int, default=10,
                        help='Seconds to wait between checking for free GPUs (default: 10)')
    return parser.parse_args()


def create_sweep(config_path: str, project: str, entity: str) -> str:
    """Create a new wandb sweep from YAML config and return the full sweep ID."""
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )   
    full_sweep_id = f"{entity}/{project}/{sweep_id}"
    print(f"Created sweep: {full_sweep_id}")
    return full_sweep_id


def get_free_memory(gpu_id):
    """Returns the free memory (in MiB) for the given GPU ID."""
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.free --format=csv,noheader,nounits"
    try:
        output = subprocess.check_output(command, shell=True)
        free_memory = int(output.decode().strip())
        return free_memory
    except Exception as e:
        print(f"Error getting free memory for GPU {gpu_id}: {e}")
        return 0


def get_sweep_status(sweep_id: str) -> dict:
    """Query WandB API to get sweep status and remaining runs."""
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_id)
        state = sweep.state
        # Count runs by state
        runs = list(sweep.runs)
        finished = sum(1 for r in runs if r.state == 'finished')
        running = sum(1 for r in runs if r.state == 'running')
        pending = sum(1 for r in runs if r.state == 'pending')
        failed = sum(1 for r in runs if r.state in ('failed', 'crashed'))

        # For grid search, expected_run_count tells us total runs
        expected = getattr(sweep, 'expected_run_count', None)

        return {
            'state': state,
            'finished': finished,
            'running': running,
            'pending': pending,
            'failed': failed,
            'expected': expected,
            'total_completed': finished + failed,
        }
    except Exception as e:
        print(f"Error querying sweep status: {e}")
        return None


def cancel_sweep(sweep_id):
    """Cancel the sweep via WandB CLI."""
    cancel_cmd = f"wandb sweep --cancel {sweep_id}"
    try:
        subprocess.call(cancel_cmd, shell=True)
        print(f"Triggered sweep cancellation: {cancel_cmd}")
    except Exception as e:
        print(f"Failed to cancel sweep: {e}")


def find_available_gpu(gpu_ids: list, min_free_vram: int, gpu_processes: dict) -> int | None:
    """Find a GPU with enough free VRAM and no active process."""
    for gpu_id in gpu_ids:
        # Check if there's already a process running on this GPU
        if gpu_id in gpu_processes:
            proc = gpu_processes[gpu_id]
            poll_result = proc.poll()
            if poll_result is None:  # Still running
                continue
            else:
                # Process finished, remove from tracking
                print(f"[DEBUG] GPU {gpu_id} process finished with exit code {poll_result}", flush=True)
                del gpu_processes[gpu_id]

        # Check VRAM
        free_mem = get_free_memory(gpu_id)
        if free_mem >= min_free_vram:
            return gpu_id
        else:
            print(f"[DEBUG] GPU {gpu_id} has {free_mem} MiB free (need {min_free_vram})", flush=True)

    return None


def main():
    args = parse_arguments()

    # Track which process is running on which GPU
    gpu_processes: dict[int, subprocess.Popen] = {}
    total_launched = 0

    # Create sweep if no sweep_id provided
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        print(f"Creating new sweep from {args.config}...")
        sweep_id = create_sweep(args.config, args.project, args.entity)

    print(f"\nSweep ID: {sweep_id}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Min free VRAM: {args.min_free_vram} MiB")
    print(f"Poll interval: {args.poll_interval}s\n")

    try:
        while True:
            # Clean up finished processes
            finished_gpus = []
            for gpu_id, proc in gpu_processes.items():
                if proc.poll() is not None:
                    finished_gpus.append(gpu_id)
            for gpu_id in finished_gpus:
                del gpu_processes[gpu_id]

            # Check sweep status
            status = get_sweep_status(sweep_id)
            if status:
                print(f"[Sweep status] state={status['state']}, "
                      f"finished={status['finished']}, running={status['running']}, "
                      f"pending={status['pending']}, failed={status['failed']}, "
                      f"expected={status['expected']}", flush=True)

                # Check if sweep is done
                if status['state'] in ('FINISHED', 'CANCELED'):
                    print(f"\nSweep {status['state'].lower()}!")
                    break

                # Only consider stopping if we've actually launched something
                # and there's nothing left to do
                if total_launched > 0 and len(gpu_processes) == 0:
                    # All our processes finished - check if sweep is truly done
                    if status['state'] == 'FINISHED':
                        print("\nSweep finished!")
                        break
                    # If expected is set and we've completed all, we're done
                    if status['expected'] and status['total_completed'] >= status['expected']:
                        print(f"\nAll {status['expected']} runs completed!")
                        break

            # Try to launch on available GPUs
            available_gpu = find_available_gpu(args.gpu_ids, args.min_free_vram, gpu_processes)

            if available_gpu is not None:
                # Launch a single run on this GPU using --count 1
                command = f"CUDA_VISIBLE_DEVICES={available_gpu} WANDB_START_METHOD=thread wandb agent --count 1 {sweep_id}"
                print(f"Launching agent on GPU {available_gpu}: {command}", flush=True)
                proc = subprocess.Popen(command, shell=True)
                gpu_processes[available_gpu] = proc
                total_launched += 1
                print(f"[DEBUG] Launched on GPU {available_gpu}, total_launched={total_launched}, gpu_processes keys: {list(gpu_processes.keys())}", flush=True)
                time.sleep(2)  # Brief pause before checking for more GPUs
            else:
                # No GPU available, wait and retry
                active_gpus = list(gpu_processes.keys())
                # Debug: show which processes are still running vs finished
                for gid, p in gpu_processes.items():
                    poll_res = p.poll()
                    print(f"[DEBUG] gpu_processes[{gid}]: poll()={poll_res}", flush=True)
                print(f"Waiting for GPU... Active processes on GPUs: {active_gpus}", flush=True)
                time.sleep(args.poll_interval)

        # Wait for any remaining processes
        if gpu_processes:
            print(f"\nWaiting for {len(gpu_processes)} remaining process(es) to finish...")
            for gpu_id, proc in gpu_processes.items():
                proc.wait()
                print(f"GPU {gpu_id} process finished.")

        print(f"\nAll done! Total agents launched: {total_launched}")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating launched agents and cancelling sweep...")
        for proc in gpu_processes.values():
            proc.terminate()
        cancel_sweep(sweep_id)
        sys.exit(0)


if __name__ == '__main__':
    main()
