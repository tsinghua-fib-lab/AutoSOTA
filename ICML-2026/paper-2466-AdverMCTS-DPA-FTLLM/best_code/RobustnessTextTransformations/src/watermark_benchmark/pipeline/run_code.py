import json
import multiprocessing
import os
import re
import sys
from dataclasses import replace

from apps import load_generations

from watermark_benchmark.servers.server import Server
from watermark_benchmark.utils import load_config
from watermark_benchmark.utils.classes import Generation, WatermarkSpec
from watermark_benchmark.utils.generation_prompts import (
    code_prompt_prefix,
    raw_low_entropy_prompts,
)

from .summarize import run as summary_run


def gen_wrapper(config, watermarks, custom_builder=None):
    config.baseline = True
    from .generate import run as gen_run

    # Load datasets
    prompt_texts, _, folder_names = load_generations(
        config.code_config, Server.get_tokenizer(config.model)
    )

    prompts = [
        (
            code_prompt_prefix
            + prompt,  ## re.sub(r"ANSWER: ", "", prompt) + "\n\n" + code_prompt_suffix,
            None,
        )
        for prompt in prompt_texts
    ]

    response_id_to_folder = {i: folder for i, folder in enumerate(folder_names)}

    gen_run(config, watermarks, custom_builder, raw_prompts=prompts)

    # Filter response
    generations = Generation.from_file(config.output_file)
    for idx, generation in enumerate(generations):
        try:
            response = (
                generation.response.split("[[SOLUTION]]")[-1]
                .split("[[END]]")[0]
                .strip()
            )
            if "```" in response:
                response = re.split(r"```(?:python)?", response)[-2]
        except Exception:
            print(f"Cannot parse response {generation.id}")
            response = generation.response
        generations[idx] = replace(generation, response=response)

    with open(
        os.path.join(config.results, "response_id_to_folder.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(response_id_to_folder, f)
    Generation.to_file(config.output_file, generations)


def detect_wrapper(config, generations, custom_builder=None):
    from .detect import run as detect_run

    detect_run(config, generations, custom_builder)


def rate_wrapper(config, generations):
    from .quality import run as rate_run

    code_generations = generations

    # Now rate the code generations
    config.quality_metric = "code"
    config.output_file = config.results + "/rated{}.tsv".format(
        "_val" if config.validation else ""
    )
    rate_run(config, code_generations)


def run(
    config,
    watermarks,
    custom_builder=None,
    GENERATE=True,
    RATE=True,
    DETECT=True,
):
    # Generation
    generations = []

    # Create output dir:
    try:
        os.mkdir(config.results)
    except Exception:
        pass

    print("### GENERATING ###")

    if GENERATE:
        config.input_file = None
        config.output_file = config.results + "/generations{}.tsv".format(
            "_val" if config.validation else ""
        )
        gen_wrapper(config, watermarks, custom_builder)

    print("### RATING ###")

    # Rate
    if RATE:
        config.input_file = config.results + "/generations{}.tsv".format(
            "_val" if config.validation else ""
        )
        config.output_file = config.results + "/rated{}.tsv".format(
            "_val" if config.validation else ""
        )
        generations = Generation.from_file(config.input_file)
        rate_wrapper(config, generations)

    print("### DETECTING ###")

    # Detect
    if DETECT:
        config.input_file = config.results + "/rated{}.tsv".format(
            "_val" if config.validation else ""
        )
        config.output_file = config.results + "/detect{}.tsv".format(
            "_val" if config.validation else ""
        )
        generations = Generation.from_file(config.input_file)
        detect_wrapper(config, generations, custom_builder)
        generations = Generation.from_file(config.output_file)
    else:
        try:
            generations = Generation.from_file(
                config.results
                + "/detect{}.tsv".format("_val" if config.validation else "")
            )
        except:
            generations = []

    return generations


def main():
    if sys.argv[2] == "gen":
        multiprocessing.set_start_method("spawn")
        config = load_config(sys.argv[1])
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(l.strip()), tokenizer=config.model
                )
                for l in infile.read().split("\n")
                if len(l)
            ]
        run(config, watermarks, GENERATE=True, RATE=False, DETECT=False)
    elif sys.argv[2] == "detect":
        config = load_config(sys.argv[1])
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(l.strip()), tokenizer=config.model
                )
                for l in infile.read().split("\n")
                if len(l)
            ]
        run(config, watermarks, GENERATE=False, RATE=False, DETECT=True)
    else:
        config = load_config(sys.argv[1])
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(l.strip()), tokenizer=config.model
                )
                for l in infile.read().split("\n")
                if len(l)
            ]
        run(config, watermarks, GENERATE=False, RATE=True, DETECT=False)

    # summary_run(config, generations)
