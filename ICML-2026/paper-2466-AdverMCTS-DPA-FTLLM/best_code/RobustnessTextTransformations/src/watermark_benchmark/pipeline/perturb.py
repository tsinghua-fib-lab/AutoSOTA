import math
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import replace

from tqdm import tqdm

from watermark_benchmark.attacks.helm_attacks import setup
from watermark_benchmark.utils import (
    get_input_file,
    get_output_file,
    load_config,
    setup_randomness,
)
from watermark_benchmark.utils.classes import Generation


def init_attacks(
    config,
    dispatch_queue=None,
    results_queue=None,
    synonym_cache=None,
    names_only=False,
):
    """
    Initializes a dictionary of attack objects based on the given configuration.

    Args:
        config: A configuration object containing attack parameters.
        dispatch_queue: A queue for dispatching generation tasks to worker processes.
        results_queue: A queue for receiving generation results from worker processes.
        synonym_cache: A cache for storing synonym generation results.
        names_only: A boolean indicating whether to only include attack names in the dictionary.

    Returns:
        A dictionary of attack objects, keyed by attack name.
    """
    from watermark_benchmark.attacks.helm_attacks import init_helm_attacks
    from watermark_benchmark.attacks.paraphrase_attack import ParaphraseAttack
    from watermark_benchmark.attacks.swap_attack import SwapAttack
    from watermark_benchmark.attacks.synonym_attack import SynonymAttack

    attack_list = {}

    if config.paraphrase:
        for name, params in ParaphraseAttack.get_param_list(reduced=False):
            attack_list[name] = (
                ParaphraseAttack(
                    *params, queue=dispatch_queue, resp_queue=results_queue
                )
                if not names_only
                else True
            )
        return attack_list

    # Helm attacks
    attack_list.update(init_helm_attacks(names_only=names_only))

    for name, params in SwapAttack.get_param_list():
        attack_list[name] = SwapAttack(*params) if not names_only else True
    for name, params in ParaphraseAttack.get_param_list(
        reduced=not config.paraphrase
    ):
        attack_list[name] = (
            ParaphraseAttack(
                *params, queue=dispatch_queue, resp_queue=results_queue
            )
            if not names_only
            else True
        )
    for name, params in SynonymAttack.get_param_list():
        attack_list[name] = (
            SynonymAttack(
                *params,
                generation_queue=dispatch_queue,
                resp_queue=results_queue,
                cache=synonym_cache
            )
            if not names_only
            else True
        )

    return attack_list


def perturb_process(
    task_queue, writer_queue, results_queue, dispatch, config, synonym_cache
):
    """
    This function is responsible for perturbing the input text using various attacks.

    Args:
        task_queue (Queue): A queue of tasks to be processed.
        writer_queue (Queue): A queue to write the results to.
        results_queue (Queue): A queue to store the results of the attacks.
        dispatch (function): A function to dispatch the results to.
        config (dict): A dictionary containing the configuration for the attacks.
        synonym_cache (dict): A dictionary containing the synonym cache.

    Returns:
        None
    """
    from watermark_benchmark.utils import setup_randomness

    # Setup attacks
    setup_randomness(config)
    setup(config)
    attack_list = init_attacks(config, dispatch, results_queue, synonym_cache)

    while True:
        task = task_queue.get(block=True)
        if task is None:
            task_queue.put(None)
            return

        attack_name, generation = task
        if attack_name is None:
            writer_queue.put(generation)
        else:
            setup_randomness(config)
            attack = attack_list[attack_name]
            generation = replace(
                generation,
                response=attack.warp(generation.response, generation.prompt),
                attack=attack_name,
            )
            writer_queue.put(generation)


def writer_process(queue, config, w_count):
    """
    This function writes the perturbed data to the output file.

    Args:
        queue: multiprocessing.Queue object
        config: dict containing configuration parameters
        w_count: int, number of perturbations to write to the output file
    """
    from watermark_benchmark.utils import get_output_file

    outfilepath = get_output_file(config)

    for _ in tqdm(range(w_count), total=w_count, desc="Perturb process"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        with open(outfilepath, "a") as outfile:
            outfile.write(str(task) + "\n")


def run(config_file, generations=None):
    """
    Runs the perturbation process on the given generations and configuration file.

    Args:
        config_file (str or dict): Path to the configuration file or the configuration dictionary.
        generations (list of Generation, optional): List of generations to perturb. If not provided, it will be loaded from the input file specified in the configuration.

    Returns:
        None
    """
    import torch

    from watermark_benchmark.utils.apis import (
        dipper_server,
        openai_process,
        translate_process,
    )

    # Load config
    if type(config_file) == str:
        config = load_config(config_file)
    else:
        config = config_file

    setup_randomness(config)
    setup(config)

    # Load generations
    if not generations:
        generations = Generation.from_file(get_input_file(config))

    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)

    existing = {
        str(
            (
                g.watermark.to_dict(True, True)
                if g.watermark is not None
                else g.temp,
                g.id,
                g.attack,
            )
        )
        for g in Generation.from_file(outfilepath)
    }

    # Count attacks
    attack_list = init_attacks(config, names_only=True)

    global_manager = multiprocessing.Manager()
    processes = []
    tasks = []
    task_count = 0

    # for generation in tqdm(generations, total=len(generations), desc="Preparing tasks"):
    for generation in generations:
        wid, gid = (
            generation.watermark.to_dict(True, True)
            if generation.watermark is not None
            else generation.temp
        ), generation.id
        if str((wid, gid, None)) not in existing:
            tasks.append((None, generation))
            task_count += 1

        # Only perturb 100 generations
        if generation.id % 100 > 33 or generation.watermark is None:
            continue

        for attack in attack_list:
            if str((wid, gid, attack)) not in existing:
                tasks.append((attack, generation))
                task_count += 1

    if not task_count:
        return

    # Setup dipper & google translate processes
    dipper_queue, translate_queue = (
        global_manager.Queue(),
        global_manager.Queue(),
    )
    if config.paraphrase:
        devices = config.get_devices()
        for i in range(config.dipper_processes):
            processes.append(
                multiprocessing.Process(
                    target=dipper_server,
                    args=(
                        dipper_queue,
                        [devices[i % len(devices)]] if len(devices) else [],
                    ),
                )
            )
            processes[-1].start()

    for d in config.get_devices():
        process_count = (
            int(
                math.floor(
                    torch.cuda.get_device_properties(d).total_memory
                    / 2000000000
                )
            )
            if d != "cpu"
            else 1
        )
        for _ in range(process_count):
            processes.append(
                multiprocessing.Process(
                    target=translate_process,
                    args=(translate_queue, ["en", "fr", "ru"], d),
                )
            )
            processes[-1].start()
            time.sleep(5)

    # Setup synonym cache
    synonym_cache = global_manager.dict()

    # Setup openai
    openai_queue = global_manager.Queue()
    openai_cache = global_manager.dict()
    if config.paraphrase:
        for i in range(config.openai_processes):
            processes.append(
                multiprocessing.Process(
                    target=openai_process,
                    args=(openai_queue, config.openai_key, openai_cache),
                )
            )
            processes[-1].start()

    # Setup dispatch process
    dispatch_queues = {
        "dipper": dipper_queue,
        "translate": translate_queue,
        "openai": openai_queue,
    }

    # Setup perturbation processes
    task_queue = global_manager.Queue()
    writer_queue = global_manager.Queue()
    results_queues = []
    for _ in range(config.threads):
        results_queues.append(global_manager.Queue())
        processes.append(
            multiprocessing.Process(
                target=perturb_process,
                args=(
                    task_queue,
                    writer_queue,
                    results_queues[-1],
                    dispatch_queues,
                    config,
                    synonym_cache,
                ),
            )
        )
        processes[-1].start()

    for t in tasks:
        task_queue.put(t)

    # Setup writer
    writer = multiprocessing.Process(
        target=writer_process, args=(writer_queue, config, task_count)
    )
    writer.start()

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


def perturb(config_file, generations):
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
    config.input_file = config.results + "/generations{}.tsv".format(
        "_val" if config.validation else ""
    )
    config.output_file = config.results + "/perturbed{}.tsv".format(
        "_val" if config.validation else ""
    )

    if generations is None:
        generations = Generation.from_file(get_input_file(config))

    return run(config, generations), config
