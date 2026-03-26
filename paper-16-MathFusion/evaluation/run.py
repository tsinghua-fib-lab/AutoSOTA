#!/usr/bin/env python
# coding: utf-8

# # Generation
#
# > Generate with specified stopping criteria
#

import argparse
import logging
import os
import time

import torch
from vllm import LLM, SamplingParams

from dart_math.data import RespSampleVLLM, load_query_dps
from dart_math.eval import EvaluatorMathBatch
from dart_math.exec import CodeExecCfg
from dart_math.gen import Generator, is_dp_dars_finished
from dart_math.utils import (
    PROJ_HOME,
    PromptTemplate,
    get_pathname_from_name_or_path,
    init_logging,
)

if __name__ == "__main__":

    init_logging()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="vLLM generation", allow_abbrev=False)

    parser.add_argument(
        "--gen_save_path",
        type=str,
        default=os.path.join(PROJ_HOME, "data/res/gen.jsonl"),
        help="Path save results of generation (and evaluation).",
    )

    # Device
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM.",
    )

    parser.add_argument(
        "--swap_space", type=float, default=4, help="CPU swap space in GB for vLLM."
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-math-7b-rl",
        help="HF-style model name or path.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for the model.",
    )

    # Data
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["math-test"],
        help="Dataset(s) to generate on.",
    )

    # Generation configurations
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--max_new_toks",
        type=int,
        default=2048,
        help="Maximum number of new tokens.",
    )
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        default=False,
        help="Ignore EOS token in generation. Llama-3-8B(-Base) tends to decode EoS immediately. Try this if you encounter this issue.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=-1,
        help="Number of shots for prompting. -1 means adaptive to datasets.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="cot",
        help="ID / Path to the file of prompt template.",
    )
    parser.add_argument(
        "--n_paths",
        type=int,
        default=1,
        help="Number of generated completions per request. NOTE: might cause bug in vLLM (0.4.2).",
    )
    parser.add_argument(
        "--save_gen_path_bs",
        type=int,
        default=2**14,
        help="# Completions = # Paths per request * # Requests. Values <= 0 mean adaptive.",
    )
    parser.add_argument(
        "--inf_seed",
        type=int,
        default=0,
        help="Random seed for inference. -1 means using us timestamp mod 2^32.",
    )

    # Stopping criteria
    parser.add_argument(
        "--max_n_trials",
        nargs="+",
        type=int,
        default=1,
        help="(List of) maximum number of trials for each query. Non-positive means no limit.",
    )
    parser.add_argument(
        "--gen_only",
        action="store_true",
        help="Whether to only generate reponses and not evaluate the generated completions.",
    )
    parser.add_argument(
        "--min_n_corrects",
        nargs="+",
        type=int,
        default=0,
        help="(List of) minimum number of correct completions per query needed to stop generation. Non-positive means no goal.",
    )
    parser.add_argument(
        "--strict_extract",
        action="store_true",
        help="Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.",
    )

    # Code execution
    parser.add_argument(
        "--code_exec_cfg",
        type=str,
        default="",
        help="ID / Path to file of the code execution configuration.",
    )
    parser.add_argument(
        "--max_n_workers",
        type=int,
        default=None,
        help="The maximum number of CPU core workers to execute the code with multi-processing. Default as `None`, meaning using default value of `code_exec_cfg`. ",
    )
    parser.add_argument(
        "--max_n_calls",
        type=int,
        default=None,
        help="The maximum number of calls to the code execution function.\nThis could be large because there is token length limit already.\nDefault as `None`, meaning using default value of `code_exec_cfg`.  Non-positive values mean no limit.",
    )
    parser.add_argument(
        "--trunc_len",
        type=int,
        nargs=2,
        default=None,
        help="The maximum lengths to truncate the output into the beginning and end.\nDefault as `None`, meaning using default value of `code_exec_cfg`. Double non-positive values like `(0, 0)` mean no truncation. ",
    )

    args, unk_args = parser.parse_known_args()

    for arg_str in unk_args:
        if arg_str.startswith("--f="):
            continue  # For Jupyter notebook
        else:
            raise ValueError(f"Unknown arguments: {unk_args}")

    if args.inf_seed == -1:
        args.inf_seed = int(time.time() * 10**6) % 2**32
        logging.warning(f"args.inf_seed=-1 -> Setting {args.inf_seed=}")

    if "tool" in args.prompt_template and args.code_exec_cfg == "":
        args.code_exec_cfg = "python"
        logging.warning(f"{args.prompt_template=} -> Setting {args.code_exec_cfg=}")

    query_dps = load_query_dps(
        args.datasets,
        args.max_n_trials,
        args.min_n_corrects,
        n_shots=args.n_shots,
    )
    logging.info(f"Loaded {len(query_dps)} query data points.")

    model_dirname = get_pathname_from_name_or_path(args.model_name_or_path)
    logging.info(f"{model_dirname=}")

    prompt_template = (
        PromptTemplate.get_prompt_template_from_prompt_type_and_model(
            prompt_type=args.prompt_template, model_dirname=model_dirname
        )
        if args.prompt_template in ["cot", "tool"]
        else PromptTemplate.load_from_id_or_path(args.prompt_template)
    )
    logging.info(f"{prompt_template.id=}")
    # TODO: response-wise prompt template
    for query_dp in query_dps:
        query_dp.prompt_template = prompt_template

    if args.temperature <= 1e-5:
        args.temperature = 0
        args.n_paths = 1
        args.top_p = 1
        logging.warning(
            f"args.temperature<=1e-5 -> Setting {args.temperature=}, {args.n_paths=}, {args.top_p=} for vLLM."
        )

    sampling_params = SamplingParams(
        n=args.n_paths,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_toks,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=True,
        seed=args.inf_seed,
    )

    sampling_params.stop = [
        prompt_template.query_prompt.strip(),
        prompt_template.resp_prompt.strip(),
    ]
    if 'refaug' in args.model_name_or_path:
        sampling_params.stop.append("Reflection:")
    logging.info(f"sampling_params = {sampling_params}")

    llm = LLM(
        model=args.model_name_or_path,
        revision=args.revision,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=args.dtype,
        seed=args.inf_seed,
        gpu_memory_utilization=args.gpu_mem_util,
        swap_space=args.swap_space,
        trust_remote_code=True,
    )
    logging.info("LLM loaded!")

    code_exec_cfg = (
        CodeExecCfg.load_from_id_or_path(args.code_exec_cfg)
        if args.code_exec_cfg
        else None
    )
    if code_exec_cfg:
        if args.max_n_workers is not None:
            code_exec_cfg.max_n_workers = args.max_n_workers
        if args.max_n_calls is not None:
            code_exec_cfg.max_n_calls = args.max_n_calls
        if args.trunc_len is not None:
            code_exec_cfg.trunc_len = args.trunc_len

        print(f"{code_exec_cfg.__dict__=}")

    generator = Generator(
        llm,
        sampling_params,
        resp_sample_cls=RespSampleVLLM,
        batch_evaluator=(
            EvaluatorMathBatch(strict_extract=args.strict_extract, use_qwen25_math_extract=args.datasets[0] in ['gsm8k/test', 'math/test'])
            if not args.gen_only
            else None
        ),
        code_exec_cfg=code_exec_cfg,
    )
    generator.gen(
        query_dps=query_dps,
        dp_stop_criteria=is_dp_dars_finished,
        save_path=args.gen_save_path,
        n_paths_per_save=args.save_gen_path_bs,
    )

    logging.info("Generation done!")
