import argparse
import subprocess
import math
import os
import logging
import time

from task_queue import TaskQueue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Path to the script to run (e.g., a.py or b.py)")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--use_queue", action="store_true")
    parser.add_argument("--extra_args", type=str, default="", help="Additional arguments to pass to the script that is going to be gpu parallelized")
    parser.add_argument("--gpus_per_process", type=int, default=1)
    return parser.parse_args()


def init_main_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[GPU Parallel] %(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    return logger


def init_worker_logger():
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Process {get_worker_rank()}] %(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    return logger


def get_worker_rank():
    return int(dict(os.environ).get('GPU_PARALLEL_PROCESS_ID', '0'))


def wait_for_done_or_error(*processes: subprocess.Popen) -> bool:
    # Keep track of whether we've handled a failure
    error_occurred = False

    # Polling loop to check for any process completion
    while True:
        for i, proc in enumerate(processes):
            retcode = proc.poll()
            if retcode is not None:
                # Process has finished
                if retcode != 0:
                    logging.error(f"Process {i} exited with error code {retcode}. Terminating others.")
                    error_occurred = True
                    # Terminate all other processes
                    for j, p in enumerate(processes):
                        if p != proc and p.poll() is None:
                            logging.info(f"Terminating process {j} because of the above error")
                            p.terminate()
                    # Optionally wait for them to terminate
                    for j, p in enumerate(processes):
                        if p != proc:
                            p.wait()
                else:
                    logging.info(f"Process {i} completed successfully.")
                break  # Exit the loop after first process finishes
        else:
            time.sleep(1)  # Avoid busy waiting
            continue
        break  # Exit the outer loop if inner loop was broken

    # Wait for the remaining (non-terminated) processes to finish
    for proc in processes:
        if proc.poll() is None:
            proc.wait()
    return error_occurred


def main():
    logger = init_main_logger()
    args = parse_args()

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set.")
    logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

    gpus = visible_devices.split(",")
    n_gpus = len(gpus)
    assert n_gpus % args.gpus_per_process == 0, "total CUDA_VISIBLE_DEVICES must be divisible by gpus_per_process"
    logger.info(f"Number of GPUs: {n_gpus}")

    total = args.end - args.start
    logger.info(f"Processing samples [{args.start}; {args.end})")
    logger.info(f"Total samples: {total}")
    queue = None
    if args.use_queue:
        queue = TaskQueue(range(args.start, args.end), hostname="localhost", start=True)
    else:
        logger.info(f"Chunk size (per process): {math.ceil(total / n_gpus)}")

    processes = []
    for i in range(n_gpus // args.gpus_per_process):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[i * args.gpus_per_process: (i + 1) * args.gpus_per_process])
        env["GPU_PARALLEL_PROCESS_ID"] = str(i)
        if args.use_queue:
            cmd = f"python3 {args.script} --queue {queue.endpoint} {args.extra_args}"
        else:
            chunk_size = math.ceil(total / n_gpus)
            chunk_start = args.start + i * chunk_size
            chunk_end = min(args.start + (i + 1) * chunk_size, args.end)
            if chunk_start >= args.end:
                break
            cmd = f"python3 {args.script} --start {chunk_start} --end {chunk_end} {args.extra_args}"
        logger.info(f"Launching on GPU {env['CUDA_VISIBLE_DEVICES']}: {cmd}")
        processes.append(subprocess.Popen(cmd, shell=True, env=env))

    error_occurred = wait_for_done_or_error(*processes)
    if queue is not None:
        queue.shutdown()
    if error_occurred:
        raise RuntimeError("Exited due error in one of the processes")
    logger.info("Done")


if __name__ == "__main__":
    main()
