
import os
import pandas as pd
import torch
from tqdm import tqdm

from data_processing import cached_function2
from text_generation import generate_text_data_ordering
from llm_wrapper import LLMWrapper, ActivationHook, SteeringHook

def activation_gathering_data_ordering(params):
    # Get the parameters as input and give back the path to the according data
    text_generation_path = generate_text_data_ordering(params)
    gathering_layer = params['gathering_arguments']['gathering_layers']

    path = f"{text_generation_path}/{gathering_layer}/"
    return path


def gather_data(df, params):
    # Curate the usefull parameters for the caching system
    curated_params = {
        "parameters_type": "activation_gathering",
        "data_ordering_function": activation_gathering_data_ordering(params),
        "model_arguments": params["model_arguments"],
        "data_arguments": params["data_arguments"],
        "generation_arguments": params["generation_arguments"],
        "steering_arguments": params["steering_arguments"],
        "gathering_arguments": params["gathering_arguments"],
        "verbose": params["verbose"],
        "hf_token": params["hf_token"],
    }
    # Optionally add paraphrasing parameters
    if params["robustness_arguments"]["paraphrasing"]["enabled"]:
        curated_params["robustness_arguments"] = {"paraphrasing": params["robustness_arguments"]["paraphrasing"]}
    return gather_data_cached(df, curated_params)



@cached_function2()
def gather_data_cached(data, curated_params):
    verbose = curated_params.get("verbose", True)

    use_robustness = False
    if curated_params.get("robustness_arguments", {}).get("paraphrasing", {}).get("enabled", False):
        print("> Gathering Robustness activations too...")
        use_robustness = True

    # Prepare the LLM with hooks
    llm = LLMWrapper(
        hf_token = curated_params["hf_token"],
        **curated_params["model_arguments"]
    )
    saving_hooks = llm.register_hooks("gather", curated_params["gathering_arguments"]["gathering_layers"])

    # Same parameters as generation, but max_new_tokens = 1. Temp, top_k, top_p, etc have no impact on the first activations
    gathering_kwargs = curated_params["generation_arguments"].copy()
    gathering_kwargs.pop("generation_batch_size")
    gathering_kwargs["max_new_tokens"] = 1

    # Remove prompt from generated text if specified
    if curated_params["gathering_arguments"].get("remove_prompt", True):
        gathering_text_list = data["output_text"].tolist()
    else:
        gathering_text_list = (data["input_text"] + data["output_text"]).tolist()


    expended_data = []
    # Loop on the generated texts to gather activations 
    for i, generated_text in tqdm(enumerate(gathering_text_list)):

        # if curated_params.get("gathering_arguments", {}).get("remove_prompt", False):
        #     # Remove the prompt from the generated text if specified
        #     prompt_string_length = len(data["input_text"][i])
        #     generated_text = generated_text[prompt_string_length:]

        # Run the model to gather activations
        _ = llm.gathering_forward([generated_text], **gathering_kwargs)

        # Collect activations from hooks
        activations = {}
        for hook in saving_hooks:
            for layer in curated_params["gathering_arguments"]["gathering_layers"]:
                if hook.layer_name == layer:
                    activations[layer] = hook.activations

        if use_robustness:
            paraphrased_text = data["paraphrased_text"][i]
            _ = llm.gathering_forward([paraphrased_text], **gathering_kwargs)
            paraphrased_activations = {}
            for hook in saving_hooks:
                for layer in curated_params["gathering_arguments"]["gathering_layers"]:
                    if hook.layer_name == layer:
                        paraphrased_activations[layer] = hook.activations

        # Remove activations for tokens beyond max_token_seq if specified
        max_token_seq = curated_params["gathering_arguments"].get("max_token_seq", None)
        if max_token_seq is not None:
            for layer in activations:
                trimed_size = min(max_token_seq, activations[layer].shape[0])
                activations[layer] = activations[layer][:trimed_size]
                if use_robustness:
                    paraphrased_activations[layer] = paraphrased_activations[layer][:trimed_size]
        
        # Store the activations
        expended_data.append({
            "activations": activations,
            "paraphrased_activations": paraphrased_activations if use_robustness else None
        })
    
    for hook in saving_hooks:
        hook.remove()

    data = pd.concat([data, pd.DataFrame(expended_data)], axis=1)
    data["params"] = [curated_params] * len(data)

    return data