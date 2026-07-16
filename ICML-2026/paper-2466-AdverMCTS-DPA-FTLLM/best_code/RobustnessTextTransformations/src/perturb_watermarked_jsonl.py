import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from watermark_benchmark.attacks.helm_attacks_new import setup

# =========================
# User settings: fill these
# =========================
INPUT_FILE = "input/train.jsonl"
OUTPUT_FILE = "output/train_perturbed.jsonl"
MISSPELLINGS_PATH = "RobustnessTextTransformations/run/static_data/misspellings.json"
SEED = 0

# Whether to run full paraphrase mode
# False -> helm + swap + reduced paraphrase + synonym
# True  -> full paraphrase attack list only
PARAPHRASE = False

THREADS = 8
DEVICES = []          # e.g. [] for CPU only, or [0], or [0,1]
DIPPER_PROCESSES = 0
OPENAI_PROCESSES = 0
OPENAI_KEY = "YOUR_OPENAI_KEY_HERE"

# =========================
# Environment
# =========================
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "submodules"))


class LocalConfig:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_file = OUTPUT_FILE
        self.seed = SEED
        self.paraphrase = PARAPHRASE
        self.threads = THREADS
        self.devices = DEVICES
        self.dipper_processes = DIPPER_PROCESSES
        self.openai_processes = OPENAI_PROCESSES
        self.openai_key = OPENAI_KEY

    def get_devices(self):
        return self.devices if self.devices is not None else []

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def setup_randomness(config):
    seed = getattr(config, "seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_sample_id"] = idx
            data.append(item)
    return data


def normalize_text(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    return text.strip().replace("\r\n", "\n")


def make_dedup_key(sample, attack):
    return str(
        (
            sample.get("_sample_id"),
            normalize_text(sample.get("watermarked")),
            attack,
        )
    )


def build_existing_keys(outfilepath):
    existing = set()
    if not os.path.exists(outfilepath):
        return existing

    with open(outfilepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                existing.add(make_dedup_key(item, item.get("attack")))
            except Exception:
                continue

    return existing


def init_attacks(
    config,
    dispatch_queue=None,
    results_queue=None,
    synonym_cache=None,
    names_only=False,
):
    from watermark_benchmark.attacks.helm_attacks_new import init_helm_attacks
    from watermark_benchmark.attacks.paraphrase_attack_new import ParaphraseAttack
    from watermark_benchmark.attacks.swap_attack_new import SwapAttack
    from watermark_benchmark.attacks.synonym_attack_new import SynonymAttack

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
                cache=synonym_cache,
            )
            if not names_only
            else True
        )

    return attack_list


def perturb_process(
    task_queue, writer_queue, results_queue, dispatch, config, synonym_cache
):
    setup_randomness(config)
    setup(misspellings_path=MISSPELLINGS_PATH)  # helm_attacks.py already has hard-coded misspellings path
    # attack_list = init_attacks(config, dispatch, results_queue, synonym_cache)
    attack_list = init_attacks(config, None, None, synonym_cache)

    while True:
        task = task_queue.get(block=True)
        if task is None:
            task_queue.put(None)
            return

        attack_name, sample = task

        if attack_name is None:
            writer_queue.put(sample)
            continue

        try:
            setup_randomness(config)
            attack = attack_list[attack_name]

            src_text = sample.get("watermarked", None)
            out = deepcopy(sample)
            out["attack"] = attack_name

            if src_text is None:
                out["perturbed_watermarked"] = None
                out["_status"] = "missing_watermarked"
                writer_queue.put(out)
                continue

            perturbed_text = attack.warp(src_text, None)
            out["perturbed_watermarked"] = perturbed_text
            out["_status"] = "ok"
            writer_queue.put(out)

        except Exception as e:
            out = deepcopy(sample)
            out["attack"] = attack_name
            out["perturbed_watermarked"] = None
            out["_status"] = f"error: {type(e).__name__}: {e}"
            writer_queue.put(out)


def writer_process(queue, outfilepath, w_count):
    with open(outfilepath, "a", encoding="utf-8") as outfile:
        for _ in tqdm(range(w_count), total=w_count, desc="Perturb process"):
            task = queue.get(block=True)
            if task is None:
                queue.put(None)
                return
            outfile.write(json.dumps(task, ensure_ascii=False) + "\n")


def run(samples=None):
    import torch
    # from watermark_benchmark.utils.apis import (
    #     dipper_server,
    #     openai_process,
    #     translate_process,
    # )

    config = LocalConfig()

    setup_randomness(config)
    setup(misspellings_path=MISSPELLINGS_PATH)

    input_file = config.input_file
    output_file = config.output_file

    if input_file is None or output_file is None:
        raise ValueError("INPUT_FILE and OUTPUT_FILE must be set")

    if samples is None:
        samples = load_jsonl(input_file)

    if not os.path.exists(output_file):
        open(output_file, "w", encoding="utf-8").close()

    existing = build_existing_keys(output_file)
    attack_list = init_attacks(config, names_only=True)

    global_manager = multiprocessing.Manager()
    processes = []
    tasks = []
    task_count = 0

    for sample in samples:
        sid = sample.get("_sample_id")
        watermarked_text = sample.get("watermarked")

        original_key = make_dedup_key(sample, None)
        if original_key not in existing:
            raw_out = deepcopy(sample)
            raw_out["perturbed_watermarked"] = watermarked_text
            raw_out["attack"] = None
            raw_out["_status"] = "original"
            tasks.append((None, raw_out))
            task_count += 1

        if sid % 100 > 33:
            continue

        if not sample.get("is_watermarked", True):
            continue

        if watermarked_text is None:
            continue

        for attack in attack_list:
            key = make_dedup_key(sample, attack)
            if key not in existing:
                tasks.append((attack, sample))
                task_count += 1

    if not task_count:
        return load_jsonl(output_file)

    # dipper_queue = global_manager.Queue()
    # translate_queue = global_manager.Queue()

    # if config.paraphrase:
    #     devices = config.get_devices()
    #     for i in range(config.dipper_processes):
    #         processes.append(
    #             multiprocessing.Process(
    #                 target=dipper_server,
    #                 args=(
    #                     dipper_queue,
    #                     [devices[i % len(devices)]] if len(devices) else [],
    #                 ),
    #             )
    #         )
    #         processes[-1].start()

    # devices = config.get_devices()
    # if not devices:
    #     devices = ["cpu"]

    # for d in devices:
    #     if d == "cpu":
    #         process_count = 1
    #     else:
    #         process_count = int(
    #             math.floor(
    #                 torch.cuda.get_device_properties(d).total_memory
    #                 / 2000000000
    #             )
    #         )

    #     for _ in range(process_count):
    #         processes.append(
    #             multiprocessing.Process(
    #                 target=translate_process,
    #                 args=(translate_queue, ["en", "fr", "ru"], d),
    #             )
    #         )
    #         processes[-1].start()
    #         time.sleep(1)

    # synonym_cache = global_manager.dict()

    # openai_queue = global_manager.Queue()
    # openai_cache = global_manager.dict()
    # if config.paraphrase and config.openai_processes > 0:
    #     for _ in range(config.openai_processes):
    #         processes.append(
    #             multiprocessing.Process(
    #                 target=openai_process,
    #                 args=(openai_queue, config.openai_key, openai_cache),
    #             )
    #         )
    #         processes[-1].start()

    # dispatch_queues = {
    #     "dipper": dipper_queue,
    #     "translate": translate_queue,
    #     "openai": openai_queue,
    # }
    synonym_cache = global_manager.dict()
    dispatch_queues = None

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

    writer = multiprocessing.Process(
        target=writer_process,
        args=(writer_queue, output_file, task_count),
    )
    writer.start()

    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in processes:
            p.terminate()
        writer.terminate()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in processes:
        p.terminate()

    return load_jsonl(output_file)


def main():
    multiprocessing.set_start_method("spawn", force=True)
    run()


def perturb(samples=None):
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    return run(samples)


if __name__ == "__main__":
    main()