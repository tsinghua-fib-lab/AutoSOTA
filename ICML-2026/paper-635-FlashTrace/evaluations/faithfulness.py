import os
import sys
# Ensure project root is importable regardless of CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import numpy as np
from transformers import utils
import math
from tqdm import tqdm
import random
import argparse
import csv
from itertools import islice
from typing import Tuple
from huggingface_hub import login

from attribution_datasets import (
    AttributionDataset,
    FactsAttributionDataset,
    MathAttributionDataset,
    MoreHopQAAttributionDataset,
)

utils.logging.set_verbosity_error()  # Suppress standard warnings

import llm_attr
import llm_attr_eval


def _resolve_indices_to_explain_token_span(
    attr_result: llm_attr.LLMAttributionResult, indices_to_explain: list[int] | None
) -> list[int]:
    if (
        isinstance(indices_to_explain, list)
        and len(indices_to_explain) == 2
        and all(isinstance(x, int) and x >= 0 for x in indices_to_explain)
        and indices_to_explain[0] <= indices_to_explain[1]
    ):
        return indices_to_explain

    gen_len = int(attr_result.attribution_matrix.shape[0])
    if gen_len <= 0:
        return [0, 0]

    # Default: explain the full generation excluding the appended EOS token.
    end_tok = max(0, gen_len - 2)
    return [0, end_tok]


def run_attribution(testing_dict, prompt, batch_size, indices_to_explain = [1], target = None) -> tuple[list[torch.Tensor], dict | None]:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]

    # Now we create an attribution for the full response
    if "IG" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)

        if testing_dict["attr_func"] == "IG":
            attr = llm_attributor.calculate_IG_per_generation(prompt, 20, tokenizer.eos_token_id, batch_size = batch_size, target = target)

        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    elif "perturbation" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMPerturbationAttribution(model, tokenizer)

        if testing_dict["attr_func"] == "perturbation_all":
            attr = llm_attributor.calculate_feature_ablation_sentences(prompt, baseline = tokenizer.eos_token_id, measure="log_loss", target = target)
        elif testing_dict["attr_func"] == "perturbation_CLP":
            attr = llm_attributor.calculate_feature_ablation_sentences(prompt, baseline = tokenizer.eos_token_id, measure="KL", target = target)
        elif testing_dict["attr_func"] == "perturbation_REAGENT":
            attr = llm_attributor.calculate_feature_ablation_sentences_mlm(prompt, target = target)

        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))
        
    elif "attention" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMAttentionAttribution(model, tokenizer)
        llm_attributor_ig = llm_attr.LLMGradientAttribtion(model, tokenizer)

        if testing_dict["attr_func"] == "attention_I_G":
            attr = llm_attributor.calculate_attention_attribution(prompt, target = target)
            attr_b = llm_attributor_ig.calculate_IG_per_generation(prompt, 20, tokenizer.eos_token_id, batch_size = batch_size, target = target)
            attr.attribution_matrix = attr.attribution_matrix * attr_b.attribution_matrix

        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))       
        
    elif "ifr" in testing_dict["attr_func"].lower():
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        attr_func = testing_dict["attr_func"].lower()
        renorm_threshold = testing_dict.get("renorm_threshold")

        if attr_func == "ifr_all_positions":
            attr = llm_attributor.calculate_ifr_for_all_positions(prompt, target=target, renorm_threshold=renorm_threshold)
        elif attr_func == "ifr_all_positions_output_only":
            attr = llm_attributor.calculate_ifr_for_all_positions_output_only(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                renorm_threshold=renorm_threshold,
            )
        elif attr_func == "ifr_span":
            span = testing_dict.get("sink_span")
            attr = llm_attributor.calculate_ifr_span(
                prompt,
                target=target,
                span=tuple(span) if span is not None else None,
                renorm_threshold=renorm_threshold,
            )
        elif attr_func == "ifr_multi_hop":
            attr = llm_attributor.calculate_ifr_multi_hop(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        elif attr_func == "ifr_in_all_gen":
            import ft_ifr_improve

            llm_attributor = ft_ifr_improve.LLMIFRAttributionInAllGen(model, tokenizer)
            attr = llm_attributor.calculate_ifr_in_all_gen(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        elif attr_func == "ifr_multi_hop_stop_words":
            import ft_ifr_improve

            llm_attributor = ft_ifr_improve.LLMIFRAttributionImproved(model, tokenizer)
            attr = llm_attributor.calculate_ifr_multi_hop_stop_words(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        elif attr_func == "ifr_multi_hop_both":
            import ft_ifr_improve

            llm_attributor = ft_ifr_improve.LLMIFRAttributionBoth(model, tokenizer)
            attr = llm_attributor.calculate_ifr_multi_hop_both(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        elif attr_func == "ifr_multi_hop_split_hop":
            import ft_ifr_improve

            llm_attributor = ft_ifr_improve.LLMIFRAttributionSplitHop(model, tokenizer)
            attr = llm_attributor.calculate_ifr_multi_hop_split_hop(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        else:
            raise ValueError(f"Unsupported IFR attribution function '{testing_dict['attr_func']}'.")

        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    elif "basic" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMBasicAttribution(model, tokenizer)
        attr = llm_attributor.calculate_basic_attribution(prompt, target = target)
        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    elif testing_dict["attr_func"] == "attnlrp":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_ft_hop0(prompt, target=target)
        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    elif testing_dict["attr_func"] == "attnlrp_aggregated":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated(prompt, target=target)
        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    elif testing_dict["attr_func"] == "attnlrp_aggregated_multi_hop":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated_multi_hop(
            prompt,
            target=target,
            sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
            thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, indices_to_explain)
        attributions = list(attr.get_all_token_attrs(token_span))

    else:
        raise ValueError(f"Unsupported attribution function '{testing_dict['attr_func']}'.")

    extra = None
    if testing_dict["attr_func"].lower() in ("ifr_multi_hop_stop_words", "ifr_multi_hop_both"):
        import ft_ifr_improve

        extra = {
            "keep_prompt_token_indices": ft_ifr_improve.keep_token_indices(list(attr.prompt_tokens)),
            "user_prompt_indices": list(getattr(llm_attributor, "user_prompt_indices", []) or []),
        }

    return attributions, extra

def faithfulness_test(testing_dict, llm_evaluator, prompt, indices_to_explain, target = None) -> np.ndarray[float]:
    tokenizer = testing_dict["tokenizer"]
    faithfulness_k = int(testing_dict.get("faithfulness_k", 20))

    scores = []

    # batch size is set based on the max_input_len in main(). Currently set to fully fill a 196GB GPU.
    if target is None:
        generation, full_output = llm_evaluator.response(prompt)
        batch_size = math.floor((testing_dict["max_input_len"] - 100) / len(tokenizer(full_output).input_ids))
    else:
        generation = target
        batch_size = math.floor(
            (testing_dict["max_input_len"] - 100)
            / len(tokenizer(llm_evaluator.format_prompt(" " + prompt) + generation).input_ids)
        )

    # We run an attribution on the input
    # A list of attribution tensors will be returned and scored individually.
    attr_list, extra = run_attribution(testing_dict, prompt, batch_size, indices_to_explain = indices_to_explain, target = target)

    seq_attr = attr_list[0]
    prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G

    for i in range(len(attr_list)):
        attr = attr_list[i][:, :prompt_len]
        if testing_dict["attr_func"].lower() in ("ifr_multi_hop_stop_words", "ifr_multi_hop_both") and extra is not None:
            import ft_ifr_improve

            scores.append(
                ft_ifr_improve.faithfulness_test_skip_tokens(
                    llm_evaluator,
                    attr,
                    prompt,
                    generation,
                    keep_prompt_token_indices=extra.get("keep_prompt_token_indices") or [],
                    user_prompt_indices=extra.get("user_prompt_indices"),
                    k=faithfulness_k,
                )
            )
        else:
            scores.append(llm_evaluator.faithfulness_test(attr, prompt, generation, k=faithfulness_k)) # [3 scores]

    return np.array(scores)

def clean_trailing_space(text) -> str:
    if text[-1] == ' ':
        return text[:-1]
    else:
        return text
    
def evaluate_attribution(testing_dict) -> None:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]

    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)

    scores = []

    description = "Faithfulness " + testing_dict["model_name"] + " " + testing_dict["dataset_name"] + " " + testing_dict["attr_func"]

    dataset: AttributionDataset = testing_dict["dataset"]
    num_examples = testing_dict["num_examples"]
    total = min(len(dataset), num_examples) if hasattr(dataset, "__len__") else num_examples
    example_iterator = islice(dataset, num_examples)

    for example in tqdm(example_iterator, desc=description, total=total):
        indices_to_explain = example.indices_to_explain if example.indices_to_explain is not None else [-2]
        scores.append(
            faithfulness_test(
                testing_dict,
                llm_evaluator,
                example.prompt,
                indices_to_explain=indices_to_explain,
                target=example.target,
            )
        )

    scores = np.array(scores) # [num_examples, num_attrs, 3 scores]
    scores_mean = scores.mean(0) # [num_attrs, 3 scores]
    scores_var = scores.std(0) # [num_attrs, 3 scores]

    # make the test folder if it doesn't exist
    folder = "./test_results/faithfulness/" + testing_dict["dataset_name"] + "/" + testing_dict["model_name"] + "/" 
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save all data
    file_name = testing_dict["attr_func"] + "_" + str(testing_dict["num_examples"]) + "_examples"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)

        write.writerow(["Method", "RISE", "MAS", "RISE + AP"])

        write.writerow(["Seq Attr Scores Mean"] + scores_mean[0].tolist())
        write.writerow(["Row Attr Scores Mean"] + scores_mean[1].tolist())
        write.writerow(["Recursive Attr Scores Mean"] + scores_mean[2].tolist())

        write.writerow(["Seq Attr Scores Var"] + scores_var[0].tolist())
        write.writerow(["Row Attr Scores Var"] + scores_var[1].tolist())
        write.writerow(["Recursive Attr Scores Var"] + scores_var[2].tolist())

    return

def load_model(model_name, device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

    # Respect three modes:
    # - device == 'auto'               -> multi-GPU sharding across all visible devices
    # - device startswith('cuda:IDX')   -> place entire model on a single GPU IDX (relative to visible devices)
    # - device == 'cpu'                -> CPU
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except Exception:
            gpu_idx = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main(args) -> None:
    # login(token = "")
    
    # Device selection policy (mirrors attribution_recovery):
    # - If --cuda is a comma-separated list (e.g. "0,1"), set visibility to that list and shard with device_map='auto'.
    # - If --cuda is a single index (e.g. "0"), do NOT override CUDA_VISIBLE_DEVICES; place model on cuda:{index}.
    # - Else (no --cuda), use --cuda_num as single-device index relative to current visibility.
    if args.cuda is not None and isinstance(args.cuda, str) and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = "auto"
    elif args.cuda is not None and isinstance(args.cuda, str) and args.cuda.strip() != "":
        try:
            idx = int(args.cuda)
        except Exception:
            idx = 0
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    else:
        device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # set up model
    if args.model == "llama-1B":
        model_name = "meta-llama/Llama-3.2-1B-Instruct" 
        max_input_len = 5500
    elif args.model == "llama-3B":
        model_name = "meta-llama/Llama-3.2-3B-Instruct" 
        max_input_len = 4800
    elif args.model == "llama-8B":
        model_name = "meta-llama/Llama-3.1-8B-Instruct" 
        max_input_len = 3500
    elif args.model == "qwen-1.7B":
        model_name = "Qwen/Qwen3-1.7B"
        max_input_len = 5500
    elif args.model == "qwen-4B":
        model_name = "Qwen/Qwen3-4B-Instruct-2507" 
        max_input_len = 3500
    elif args.model == "qwen-8B":
        model_name = "Qwen/Qwen3-8B" 
        max_input_len = 3000
    elif args.model == "qwen-32B":
        model_name = "Qwen/Qwen3-32B"
        max_input_len = 1500
    elif args.model == "gemma-12B":
        model_name = "gemma/gemma-3-12b-it"
        max_input_len = 1500
    elif args.model == "gemma-27B":
        model_name = "gemma/gemma-3-27b-it"
        max_input_len = 2000
    else:
        model_name = args.model_path if args.model_path is not None else args.model
        max_input_len = 2000

    model, tokenizer = load_model(model_name if args.model_path is None else args.model_path, device)

    dataset_registry = {
        "math": lambda: MathAttributionDataset("./data/math_mine.json", tokenizer),
        "facts": lambda: FactsAttributionDataset("./data/10000_facts_9_choose_3.json"),
        "morehopqa": lambda: MoreHopQAAttributionDataset("./data/with_human_verification.json"),
    }
    dataset_loader = dataset_registry.get(args.dataset)
    if dataset_loader is None:
        print("You have not specified an acceptable dataset. Exiting.")
        exit()
    dataset = dataset_loader()

    testing_dict = {
        "model" : model,
        "model_name": args.model,
        "tokenizer" : tokenizer,
        "dataset" : dataset,
        "dataset_name" : args.dataset,
        "max_input_len": max_input_len,
        "attr_func": args.attr_func,
        "num_examples": args.num_examples,
        "device": device,
        "faithfulness_k": args.faithfulness_k,
    }

    # call the test function
    evaluate_attribution(testing_dict)

    return

if __name__ == "__main__":    
    parser = argparse.ArgumentParser('')
    parser.add_argument('--num_examples',
                        type = int, default = 100,
                        help='How many dataset examples to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "llama",
                        help='Model to use: llama or qwen')
    parser.add_argument('--model_path',
                        type=str, default=None,
                        help='Optional local model path to load (overrides model repo id only).')
    parser.add_argument('--attr_func',
                        type = str,
                        default = "IG",
                        help="attr to use: \
                            grad, IG, IG_captum, contextcite, attention, rollout, perturbation \
                        ")
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--cuda',
                        type=str, default=None,
                        help='GPU selection: use comma-separated ids for multi-GPU sharding (e.g. "0,1"); use a single index for one GPU relative to current CUDA_VISIBLE_DEVICES (e.g. "0").')
    parser.add_argument('--dataset',
            type = str, default = "math",
            help = 'The dataset to evaluate on: math, facts, or morehopqa')
    parser.add_argument(
        "--faithfulness_k",
        type=int,
        default=20,
        help="Total perturbation steps k for MAS/RISE (each step perturbs ~1/k of prompt tokens).",
    )
    
    args, unparsed = parser.parse_known_args()
    
    main(args)
