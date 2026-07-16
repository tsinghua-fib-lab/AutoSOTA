import multiprocessing
import os
import signal
import sys

from tqdm import tqdm

from watermark_benchmark.utils import (
    get_input_file,
    get_output_file,
    load_config,
    setup_randomness,
)
from watermark_benchmark.utils.classes import Generation

from ..metrics.code import CodeRating
from ..metrics.llm_compare import LLMCompareRating
from ..metrics.llm_rating import LLMRating
from ..metrics.MAUVE import MAUVERating
from ..metrics.ppl import PPLRating
from ..metrics.repetition import RepetitionRating


def writer_process(queue, config, w_count):
    from watermark_benchmark.utils import get_output_file, setup_randomness

    setup_randomness(config)
    outfilepath = get_output_file(config)

    for _ in tqdm(range(w_count), total=w_count, desc="Rating"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        with open(outfilepath, "a") as outfile:
            outfile.write("\n".join(str(gen) for gen in task) + "\n")


def rating_process(config, generations, writer_queue, device, baselines):
    if config.quality_metric == "llm":
        metric = LLMRating(config, writer_queue, device)
    elif config.quality_metric == "llm_cot":
        metric = LLMRating(config, writer_queue, device, cot=True)
    elif config.quality_metric == "mauve":
        metric = MAUVERating(config, writer_queue, device)
    elif config.quality_metric == "ppl":
        metric = PPLRating(config, writer_queue, device)
    elif config.quality_metric == "repetition":
        metric = RepetitionRating(config, writer_queue, device)
    elif config.quality_metric == "llm_compare":
        metric = LLMCompareRating(config, writer_queue, device)
    elif config.quality_metric == "code":
        metric = CodeRating(config, writer_queue, device)
    else:
        raise ValueError(
            "Unknown quality metric: {}. Valid metrics are [llm, mauve, ppl, repetition]".format(
                config.quality_metric
            )
        )

    metric.rate(generations, baselines)


def run(config_file, generations=None):
    # load config
    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    setup_randomness(config)

    # load generations
    generations = (
        Generation.from_file(get_input_file(config))
        if generations is None
        else generations
    )
    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)
    existing = {
        (
            (
                str(g.watermark.to_dict(True, True))
                if g.watermark is not None
                else g.temp
            ),
            g.attack,
        )
        for g in Generation.from_file(outfilepath)
        if g.rating != -1
    }

    settings = [
        v
        for v in set(
            (
                (
                    str(g.watermark.to_dict(True, True))
                    if g.watermark is not None
                    else g.temp
                ),
                g.attack,
            )
            for g in generations
        )
        if v not in existing
    ]

    if not len(settings):
        return

    baselines = {
        (float(g.temp), g.id): g.response
        for g in generations
        if g.watermark is None and g.attack is None
    }

    devices = []
    for device in config.get_devices():
        devices.extend([device])
    ct = 1 + (len(settings) // len(devices))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()

    writer = multiprocessing.Process(
        target=writer_process,
        args=(writer_queue, config, len(config.get_devices())),
    )
    writer.start()

    for idx, device in enumerate(devices):
        local_settings = set(settings[idx * ct : (idx + 1) * ct])
        local_tasks = [
            g
            for g in generations
            if (
                (
                    str(g.watermark.to_dict(True, True))
                    if g.watermark is not None
                    else g.temp
                ),
                g.attack,
            )
            in local_settings
        ]
        if len(devices) > 1:
            processes.append(
                multiprocessing.Process(
                    target=rating_process,
                    args=(config, local_tasks, writer_queue, device, baselines),
                )
            )
            processes[-1].start()
        else:
            rating_process(config, local_tasks, writer_queue, device, baselines)

    # Setup signal handler
    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in processes:
            p.terminate()
        writer.terminate()
        exit()

    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in processes:
        p.terminate()

    return Generation.from_file(get_output_file(config))


def main():
    multiprocessing.set_start_method("spawn")
    run(sys.argv[1])


def rate(config_file, generations):
    """
    Standalone perturb procedure.

    Args:
        config_file (str or ConfigSpec): Config file or path to config file
        generations (list of Generation or None): List of generations to perturb.

    If config does not contain a results directory, it will be created.
    This procedure sets the appropriate input and output files for the generation procedure.

    Return:
        generations (list): A list of generations.
        config (ConfigSpec): The updated configuration object.
    """
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn")

    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    config.input_file = config.results + "/perturbed{}.tsv".format(
        "_val" if config.validation else ""
    )
    config.output_file = config.results + "/rated{}.tsv".format(
        "_val" if config.validation else ""
    )

    if generations is None:
        generations = Generation.from_file(get_input_file(config))

    return run(config, generations), config
