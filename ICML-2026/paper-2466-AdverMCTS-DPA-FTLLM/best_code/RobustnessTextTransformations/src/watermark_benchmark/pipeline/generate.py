import multiprocessing
import os
import random
import signal
import sys
from dataclasses import replace

from tqdm import tqdm

from watermark_benchmark.utils import (
    get_output_file,
    get_server_args,
    load_config,
    setup_randomness,
)
from watermark_benchmark.utils.generation_prompts import (
    raw_prompts as default_prompts,
)


def writer_process(queue, config, w_count, use_tqdm=True):
    """
    This function is a process that writes the generated outputs to a file.

    Args:
        queue (multiprocessing.Queue): A queue containing the generated outputs.
        config (Config): The configuration object.
        w_count (int): The number of watermark generations to write to the file.
    """

    outfilepath = get_output_file(config)

    for _ in tqdm(
        range(w_count), total=w_count, desc="Generations", disable=not use_tqdm
    ):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        with open(outfilepath, "a", encoding="utf-8") as outfile:
            outfile.write("\n".join(str(gen) for gen in task) + "\n")


def gen_process(
    config,
    tasks,
    writer_queue,
    device,
    prompts,
    custom_builder=None,
    use_tqdm=False,
):
    """
    This function is a process that generates watermarked text.

    Args:
        config (Config): The configuration object.
        tasks (list): A list of tuples containing the watermark, keys, and temperature.
        writer_queue (multiprocessing.Queue): A queue to store the generated outputs.
        device (int): The device to use for generating the watermarked text.
        prompts (list): A list of prompts to use for generating the watermarked text.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Imports
    import torch

    torch.set_num_threads(1)

    from watermark_benchmark.servers import get_model
    from watermark_benchmark.utils.bit_tokenizer import Binarization
    from watermark_benchmark.watermark import get_watermark

    setup_randomness(config)

    # Setup server
    server = get_model(
        config.engine, config, max_model_len=2048, **get_server_args(config)
    )
    tokenizer = server.tokenizer()
    binarizer = Binarization(
        tokenizer,
        server.devices,
        use_huffman_coding=config.huffman_coding is not None,
        huffman_coding_path=config.huffman_coding,
    )

    def run_instance(watermark, keys, temp):
        # Setup watermark
        setup_randomness(config)
        watermark_engine = (
            get_watermark(
                watermark,
                tokenizer,
                binarizer,
                server.devices,
                keys,
                builder=custom_builder,
            )
            if watermark is not None
            else None
        )

        # Install and run
        server.install(watermark_engine)
        outputs = server.run(
            prompts, config, temp, keys, watermark, use_tqdm=use_tqdm
        )

        writer_queue.put(outputs)

        # Reset server
        server.install(None)
        torch.cuda.empty_cache()

    for t in tasks:
        run_instance(*t)


def run(
    config_file,
    watermarks=None,
    custom_builder=None,
    raw_prompts=default_prompts,
    use_tqdm=False,
):
    """
    This function runs the watermark generation process.

    Args:
        config_file (str): The path to the configuration file.
        watermarks (list): A list of watermarks to use for generating the watermarked text.
    """
    from watermark_benchmark.utils.classes import Generation, WatermarkSpec
    from watermark_benchmark.utils.standardize import standardize

    # Load config
    if isinstance(config_file, str):
        config = load_config(config_file)
    else:
        config = config_file
    setup_randomness(config)

    # Setup watermarks
    if not watermarks:
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(line.strip()), tokenizer=config.model
                )
                for line in infile.read().split("\n")
                if len(line)
            ]

    # Generate tasks
    prompts = [standardize(config.model, s, p) for p, s in raw_prompts]

    unique_temps, tasks = set(), []
    for watermark in watermarks:
        # Randomly sample key if needed
        if watermark.randomize:
            keys = [random.randint(0, 1000000) for _ in prompts]
        else:
            keys = [watermark.secret_key for _ in prompts]

        # Add task
        tasks.append((watermark, keys))
        unique_temps.add(watermark.temp)

    # Load previous generations
    outfilepath = get_output_file(config)
    if not os.path.isfile(outfilepath):
        Generation.to_file(outfilepath)

    all_tasks = [(watermark, keys, watermark.temp) for watermark, keys in tasks]
    if config.baseline:
        all_tasks.extend([(None, None, temp) for temp in unique_temps])

    existing = {
        str(
            g.watermark.to_dict(True, True)
            if g.watermark is not None
            else g.temp
        )
        for g in Generation.from_file(outfilepath)
    }
    filtered_tasks = []
    for w, k, t in all_tasks:
        if w is not None and str(w.to_dict(True, True)) not in existing:
            filtered_tasks.append((w, k, t))
        elif w is None and str(t) not in existing:
            filtered_tasks.append((w, k, t))

    if not len(filtered_tasks):
        return

    # Setup processes
    ct = 1 + (len(filtered_tasks) // len(config.get_devices()))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()
    random.shuffle(filtered_tasks)

    if len(config.get_devices()) > 1:
        for idx, device in enumerate(config.get_devices()):
            local = filtered_tasks[idx * ct : (idx + 1) * ct]
            processes.append(
                multiprocessing.Process(
                    target=gen_process,
                    args=(
                        config,
                        local,
                        writer_queue,
                        device,
                        prompts,
                        custom_builder,
                        use_tqdm,
                    ),
                )
            )
            processes[-1].start()

        writer = multiprocessing.Process(
            target=writer_process,
            args=(writer_queue, config, len(filtered_tasks)),
        )
        writer.start()
    else:
        writer = multiprocessing.Process(
            target=writer_process,
            args=(writer_queue, config, len(filtered_tasks), False),
        )
        writer.start()
        device = config.get_devices()[0]
        gen_process(
            config,
            filtered_tasks,
            writer_queue,
            device,
            prompts,
            custom_builder,
            True,
        )

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
    run(sys.argv[1])


def generate(config_file, watermarks, custom_builder=None):
    """
    Standalone generation procedure.

    Args:
        config_file (str or ConfigSpec): Config file or path to config file
        watermarks (list): A list of watermark specs to use for generating the watermarked text.
        custom_builder (function): A custom builder function to use for generating the watermarks.
        Set to none if not using custom watermarks.

    If config does not contain a results directory, it will be created.
    This procedure sets the appropriate input and output files for the generation procedure.

    Return:
        generations (list): A list of generations.
        config (ConfigSpec): The updated configuration object.
    """
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        print("Cannot set spawn multiprocessing method.")

    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    config.input_file = None
    config.output_file = config.results + "/generations{}.tsv".format(
        "_val" if config.validation else ""
    )

    return run(config, watermarks, custom_builder), config
