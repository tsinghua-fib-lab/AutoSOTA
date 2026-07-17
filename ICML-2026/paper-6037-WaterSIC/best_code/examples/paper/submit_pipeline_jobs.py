from __future__ import annotations

from functools import partial

from scheduler import run_tasks
from scripts.run_pipeline_job import run_pipeline_job


def main():
    models = ["3-8B"]

    # Layers to quantize
    layer_begin = 0
    layer_end = 32
    weights = "wq,wk,wv,wo,w1,w3,w2"  # w2 last so Hessian sees quantized w1,w3

    # Calibration setup
    seqlen = 2048

    # Sweeps
    gptq_rates = [2, 3, 4]
    gptq_groups = [-1, 64, 128]

    zsic_rates = [2.0, 3.0, 4.0]

    tasks = []
    for model in models:
        # GPTQ sweep
        for R in gptq_rates:
            for g in gptq_groups:
                f = partial(
                    run_pipeline_job,
                    model,
                    "gptq",
                    float(R),
                    layer_begin=layer_begin,
                    layer_end=layer_end,
                    weights=weights,
                    seqlen=seqlen,

                    groupsize=g,
                    actorder=False,
                    # Single-process dist init (ONLY if ckpt_dir has one shard)
                    init_dist=True,
                )
                tasks.append((f, tuple()))

        # ZSIC sweep
        for R in zsic_rates:
            f = partial(
                run_pipeline_job,
                model,
                "zsic",
                float(R),
                layer_begin=layer_begin,
                layer_end=layer_end,
                weights=weights,
                seqlen=seqlen,
                init_dist=True,
            )
            tasks.append((f, tuple()))

    run_tasks(tasks, gpu_list=[0, 1, 2, 3])


if __name__ == "__main__":
    main()
