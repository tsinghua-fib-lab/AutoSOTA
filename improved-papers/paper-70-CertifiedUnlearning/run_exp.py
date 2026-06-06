import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import os
import queue
import logging


def run_experiment(gpu_id, config_path, folder):
    try:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # my_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        cmd = f"python experiment.py --config {os.path.join(folder, config_path)}"
        status = subprocess.run(
            cmd.split(),
            stdout=open(f"run_logs/{folder}/{config_path}.log", "w"),
            stderr=subprocess.STDOUT,
            env=my_env,
        ).returncode
        # print(f"Running {cmd} on {gpu_id}")
        # time.sleep(random.randint(1, 10))
        # print(f"lol {gpu_id}")
        logging.info(f"Command '{cmd}' completed with status {status}")
        return gpu_id
    except Exception as e:
        logging.error(f"Error in {config_path}: {str(e)}")
        return gpu_id



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-e", type=str, default="auto_configs/Simplenet/PABI")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "-j",
        "--jobs_per_gpu",
        type=int,
        default=1,
        help="Number of concurrent jobs per GPU",
    )
    args = parser.parse_args()
    os.makedirs(f"run_logs/{args.e}", exist_ok=True)

    config_queue = queue.Queue()
    [config_queue.put((cfg, args.e)) for cfg in os.listdir(args.e)]
    
    gpu_pool = queue.Queue()
    for gpu_id in range(args.offset, args.offset + args.n):
        for _ in range(args.jobs_per_gpu):
            gpu_pool.put(gpu_id)

    total_concurrent = args.n * args.jobs_per_gpu
    logging.info(f"Running up to {total_concurrent} concurrent jobs on {args.n} GPUs")

    with ProcessPoolExecutor(max_workers=total_concurrent) as executor:
        futures = {}

        for _ in range(min(total_concurrent, config_queue.qsize())):
            cfg_path, cfg_dir = config_queue.get()
            gpu = gpu_pool.get()
            future = executor.submit(run_experiment, gpu, cfg_path, cfg_dir)
            futures[future] = (gpu, cfg_path)

        while futures:
            for future in as_completed(futures):
                gpu, cfg_path = futures.pop(future)
                try:
                    returned_gpu = future.result()
                    gpu_pool.put(returned_gpu)
                    logging.info(f"Job {cfg_path} returned GPU {returned_gpu}")
                except Exception as e:
                    logging.error(f"Job failed: {str(e)}")
                    gpu_pool.put(gpu)

                if not config_queue.empty():
                    new_cfg, new_dir = config_queue.get()
                    new_gpu = gpu_pool.get()
                    new_future = executor.submit(run_experiment, new_gpu, new_cfg, new_dir)
                    futures[new_future] = (new_gpu, new_cfg)

    logging.info("All experiments completed successfully")