import csv
import os
import sys
import time
import traceback

from pipeline import Pipeline
from search.configs import Arguments


# Table-configured hyperparameters:
# dataset -> (eta, doob_gamma, doob_tau, doob_t_star_idx, particles)
DATASET_CONFIGS = {
    # Medium-Expert
    "halfcheetah-medium-expert-v2": (0.7, 0.25, 0.5, 8, 4),
    "hopper-medium-expert-v2": (0.8, 0.25, 0.7, 8, 4),
    "walker2d-medium-expert-v2": (0.4, 0.5, 0.4, 10, 8),
    # Medium
    "halfcheetah-medium-v2": (0.2, 0.25, 0.5, 10, 4),
    "hopper-medium-v2": (0.6, 0.75, 0.4, 8, 8),
    "walker2d-medium-v2": (0.3, 0.5, 0.3, 10, 4),
    # Medium-Replay
    "halfcheetah-medium-replay-v2": (0.4, 0.5, 0.3, 10, 8),
    "hopper-medium-replay-v2": (0.2, 0.5, 0.5, 6, 4),
    "walker2d-medium-replay-v2": (0.2, 0.25, 0.7, 8, 8),
}

datasets = list(DATASET_CONFIGS.keys())

# How many jobs to run in parallel per GPU (override with JOBS_PER_GPU env var)
MAX_JOBS_PER_GPU = int(os.environ.get("JOBS_PER_GPU", "8"))


def experiment(dataset: str = "halfcheetah-medium-expert-v2", device: str = "cuda:0", seed: int = 0):
    eta, doob_gamma, doob_tau, doob_t_star_idx, particles = DATASET_CONFIGS[dataset]

    args = Arguments()
    args.dataset = dataset
    args.device = device
    args.seed = seed
    args.per_sample_batch_size = particles

    args.eta = eta
    args.doob_M = 32
    args.doob_gamma = doob_gamma
    args.doob_tau = doob_tau
    args.doob_t_star_idx = doob_t_star_idx
    args.doob_antithetic_sampling = True

    output_file = f"results/{args.dataset}/doob.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "dataset",
                    "seed",
                    "particles",
                    "eta",
                    "doob_M",
                    "doob_gamma",
                    "doob_tau",
                    "doob_t_star_idx",
                    "mean",
                    "std",
                ]
            )

    t0 = time.time()
    print(
        f"[START] ds={dataset} seed={seed} "
        f"particles={args.per_sample_batch_size} "
        f"eta={args.eta} M={args.doob_M} gamma={args.doob_gamma} "
        f"tau={args.doob_tau} t_star={args.doob_t_star_idx}",
        flush=True,
    )
    try:
        pipeline = Pipeline(args)
        mean, std = pipeline.eval()
    except Exception as e:
        print(f"[FAIL] ds={dataset} seed={seed}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

    dt = time.time() - t0
    print(f"[DONE ] ds={dataset} seed={seed} time={dt/60:.1f}min", flush=True)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                dataset,
                seed,
                args.per_sample_batch_size,
                args.eta,
                args.doob_M,
                args.doob_gamma,
                args.doob_tau,
                args.doob_t_star_idx,
                mean,
                std,
            ]
        )


def _worker_init(gpu_id):
    import torch

    torch.cuda.set_device(gpu_id)
    global _WORKER_DEVICE, _WORKER_GPU_ID
    _WORKER_DEVICE = f"cuda:{gpu_id}"
    _WORKER_GPU_ID = gpu_id
    print(
        f"[worker start] pid={os.getpid()} gpu_id={gpu_id} "
        f"cuda_count={torch.cuda.device_count()} name={torch.cuda.get_device_name(gpu_id)}",
        flush=True,
    )


def _run_job(job):
    ds, seed = job
    print(f"[worker {_WORKER_GPU_ID}] running ds={ds} seed={seed}", flush=True)
    experiment(dataset=ds, device=_WORKER_DEVICE, seed=seed)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    import torch

    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0

    jobs_per_gpu = max(1, MAX_JOBS_PER_GPU)
    print(f"[main] gpus={n_gpus} jobs_per_gpu={jobs_per_gpu}", flush=True)

    # build all jobs (dataset x seed)
    all_jobs = []
    for ds in datasets:
        for seed in range(5):
            all_jobs.append((ds, seed))

    # split jobs across GPUs
    buckets = [[] for _ in range(n_gpus)]
    for k, job in enumerate(all_jobs):
        buckets[k % n_gpus].append(job)

    ctx = multiprocessing.get_context("spawn")
    pools = []
    for gpu_id in range(n_gpus):
        pool = ctx.Pool(processes=jobs_per_gpu, initializer=_worker_init, initargs=(gpu_id,))
        for job in buckets[gpu_id]:
            pool.apply_async(_run_job, (job,))
        pools.append(pool)

    # wait for all pools
    for pool in pools:
        pool.close()
    for pool in pools:
        pool.join()
