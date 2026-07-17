import numpy as np
import pandas as pd
import time
import evaluate

# from markllm.evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer

from data_processing import cached_function2
from activation_gathering import activation_gathering_data_ordering
from llm_wrapper import LLMWrapper
from quality_model import QualityModelWrapper



def quality_data_ordering(params):
    # Implement the data ordering function for quality evaluation
    return activation_gathering_data_ordering(params)


def evaluate_quality(df, params):
    # Curate the usefull parameters for the caching system
    curated_params = {
        "parameters_type": "quality_evaluation",
        "data_ordering_function": quality_data_ordering(params),
        "model_arguments": params["model_arguments"],
        "data_arguments": params["data_arguments"],
        "generation_arguments": params["generation_arguments"],
        "steering_arguments": params["steering_arguments"],
        "quality_evaluation_arguments": params["quality_evaluation_arguments"],
        "verbose": params["verbose"],
        "hf_token": params["hf_token"],
    }
    # Optionally add paraphrasing parameters
    if params["robustness_arguments"]["paraphrasing"]["enabled"]:
        curated_params["robustness_arguments"] = {"paraphrasing": params["robustness_arguments"]["paraphrasing"]}
    return evaluate_quality_cached(df, curated_params)


@cached_function2()
def evaluate_quality_cached(df, params):
    # Populate a df containing generated text with a quality assessment of the given text.
    time.sleep(1)  # To avoid cache collision

    out_texts = df["output_text"].tolist()
    # If paraphrasing is enabled, evaluate on the paraphrased texts
    if "robustness_arguments" in params and params["robustness_arguments"]["paraphrasing"]["enabled"]:
        print("Evaluating quality of paraphrased texts...")
        out_texts = df["paraphrased_text"].tolist()

    print("Evaluation of these texts quality...")
    for i in range(len(out_texts)):
        if len(out_texts[i]) == 0:
            out_texts[i] = " "  # Avoid empty texts

    # Perplexity calculation
    ppl_list = perplexity_calculation(out_texts, params)
    df["perplexity"] = ppl_list

    # Log diversity calculation TODO: Disabled for speed
    ld_list = log_diversity_calculation(out_texts, params)
    df["log_diversity"] = ld_list

    # Quality classifier calculation
    quality_list = quality_classifier_calculation(out_texts, params)
    df["quality"] = quality_list

    return df

def perplexity_calculation(output_texts, params):
    # Perplexity calculation Should use an external model as reference
    ref_model_arguments = {
        "model_id": params["quality_evaluation_arguments"]["perplexity_calculation"]["model_id"],
        # "load_in_4bit": params["quality_evaluation_arguments"]["perplexity_calculation"].get("load_in_4bit", False),
        # "load_in_8bit": params["quality_evaluation_arguments"]["perplexity_calculation"].get("load_in_8bit", True),
        # "torch_dtype": params["quality_evaluation_arguments"]["perplexity_calculation"].get("torch_dtype", "auto"),
        "batch_size": params["quality_evaluation_arguments"]["perplexity_calculation"].get("batch_size", 4),
        "hf_token": params['hf_token'],
    }

    perplexity = evaluate.load("perplexity", module_type="metric")

    ppl_df = perplexity.compute(
        model_id=ref_model_arguments["model_id"],
        predictions=output_texts,
        batch_size=ref_model_arguments.get("batch_size", 4),        # Adjust for memory
        add_start_token=True                                        # If your outputs lack BOS
    )

    ppl_list = ppl_df['perplexities']

    if params["verbose"]:
        total_perplexity = np.mean(ppl_list)
        print("Average perplexity of the generated text:", total_perplexity)

    return ppl_list


def log_diversity_calculation(output_texts, params):
    # Log diversity calculation
    # ld = LogDiversityAnalyzer()
    print("TODO:  Skipping log diversity calculation for now.")
    return [0] * len(output_texts)

    diversity_list = []
    for gen_text in df["output_text"].tolist():
        diversity_list.append(ld.analyze(gen_text))

    if params["verbose"]:
        avg_diversity = np.mean(diversity_list)
        print("Average log diversity of the generated text:", avg_diversity)

    df["log_diversity"] = diversity_list

    return df


def quality_classifier_calculation(output_texts, params):
    # Quality classifier calculation Should use an external model as reference
    qlm = QualityModelWrapper(
        params["quality_evaluation_arguments"]["quality_classifier_calculation"]["model_id"],
    )

    quality_list = []
    for gen_text in output_texts:
        quality_list.append(qlm.forward(gen_text).detach().cpu().numpy())

    del qlm # Free memory

    if params["verbose"]:
        avg_quality = np.mean(quality_list)
        print("Average quality of the generated text:", avg_quality)

    return quality_list