from html import parser
from pathlib import Path
import json
import hashlib
import functools
import random as pyrandom
import os
import pickle
from random import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import petname

######################
#   Load Text Data   #
######################


def load_text_list(file_path: str, delimiter: str = "\n") -> list:
    """
    Load text data from a file and return it as a list of strings.
    
    Args:
        file_path (str): Path to the text file.
        delimiter (str): Delimiter used to split the text into lines. Default is newline.
        
    Returns:
        list: A list of strings, each representing a line in the file.
    """
    with open(Path("text_data") / Path(file_path), 'r', encoding='utf-8') as file:
        data = file.read().strip().split(delimiter)
    return data


########################
#   Parameters tools   #
########################

def format_labels(df):
    # Should format the labels to be consecutive integers starting from 0
    unique_labels = sorted(df["classification_label"].unique())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    df["classification_label"] = df["classification_label"].map(label_mapping)
    return df


def params_to_vanilla(params):
    vanilla_params = copy.deepcopy(params)
    vanilla_params["steering_arguments"]["noise_max"] = 0.0
    vanilla_params["steering_arguments"]["steering_layers"] = [0]
    vanilla_params["steering_arguments"]["noise_type"] = "vanilla"
    return vanilla_params


def hash_params(params, human_readable=False):
    # Remove parameters independent from the processing
    params_to_ignore = ["verbose", "hf_token"]
    curated_params = {k: v for k, v in params.items() if k not in params_to_ignore} 

    # Serialize curated params into a JSON string
    params_str = json.dumps(curated_params, sort_keys=True, default=str)

    # Compute hash
    param_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()[:8]

    if human_readable:
        nb_words = 2
        seed = int(hashlib.sha256(param_hash.encode()).hexdigest(), 16)
        rng = pyrandom.Random(seed)
        return param_hash, "-".join(rng.choice(petname.adjectives + petname.names)for _ in range(nb_words))

    return param_hash


#############################
#   Decorators for Caching  #
#############################


def cached_function2(root_dir="./data"):
    Path(root_dir).mkdir(exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(data, params, additional_data=None):
            # Remove parameters independent from the processing
            params_to_ignore = ["verbose", "hf_token"]
            curated_params = {k: v for k, v in params.items() if k not in params_to_ignore} 

            # Serialize curated params into a JSON string
            params_str = json.dumps(curated_params, sort_keys=True, default=str)

            print(f"\t >> {func.__name__} [in Cached_function2]")

            # Compute hash
            param_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()[:8]

            print(f"Parameter hash: {param_hash}")

            step_dir = curated_params["data_ordering_function"]
            Path(step_dir).mkdir(parents=True, exist_ok=True)

            # Create folder for this step and hash
            # _dir = os.path.join(root_dir, step_dir, param_hash)
            # os.makedirs(step_dir, exist_ok=True)
            
            # Save params for reproducibility
            param_name = f"{curated_params['parameters_type']}_params_{param_hash}.json"
            result_name = f"{curated_params['parameters_type']}_{param_hash}.pkl"

            if not os.path.exists(os.path.join(step_dir, param_name)):
                # Run the actual function
                if additional_data is not None:
                    result = func(data, params, additional_data)
                else:
                    result = func(data, params)

                # Optionally, save result if needed
                with open(os.path.join(step_dir, result_name), "wb") as f:
                    # Use pickle to save the result
                    pickle.dump(result, f)

                with open(os.path.join(step_dir, param_name), "w") as f:
                    f.write(params_str)
                
                print(f"[{param_name}] Saved parameters and [{result_name}] saved results to {step_dir}.")

            else:
                print(f"[{param_name}] Parameters already exist in {step_dir}, loading existing results.")
                # Load existing result if needed
                if os.path.exists(os.path.join(step_dir, result_name)):
                    with open(os.path.join(step_dir, result_name), "rb") as f:
                        result = pickle.load(f)
                else:
                    raise FileNotFoundError("Result file not found, re-running the function.")

            return result
        return wrapper
    return decorator

    


def cached_function(root_dir="./data"):
    Path(root_dir).mkdir(exist_ok=True)
    
    def decorator(func):
        def wrapper(wrapped_llm, text_list, **gen_kwargs):
            # Hash the model generation arguments to create a unique cache key
            gen_dic = gen_kwargs["model_arguments"].copy()
            gen_dic.update({"input_version": gen_kwargs["input_version"]})
            gen_dic.update(gen_kwargs["generation_arguments"])
            print(f"Cached_function: Running {func.__name__} with arguments: {gen_dic}")
            generation_key = hashlib.md5(json.dumps(gen_dic, sort_keys=True).encode()).hexdigest()
            print(f"Generation full key: {generation_key}")
            # The generation_key holds info of:
            #   - The dataset
            #   - The model
            #   - The generation parameters


            # Add some human readable information to the cache directory
            generation_key = f"{gen_kwargs['input_version']}_{gen_kwargs['model_arguments']['model_id'].split('/')[1]}_{generation_key[:8]}"

            # Create a directory for the cache if it doesn't exist
            cache_dir = (Path(root_dir) / Path(generation_key))
            cache_dir.mkdir(exist_ok=True)
            with open(cache_dir / Path("model_arguments.json"), "w") as fp:
                json.dump(gen_dic , fp)

            # For different functions, create different cache file names
            if func.__name__ == "vanilla_generation":
                # The vanilla generation is only determinated by the previous listed elements
                cache_file = Path(cache_dir) / f"vanilla.pkl"
            
            elif func.__name__ == "steering_generation":
                steering_key = hashlib.md5(json.dumps(gen_kwargs["steering_kwargs"], sort_keys=True).encode()).hexdigest()
                cache_file = Path(cache_dir) / f"steering_{steering_key}.pkl"
                # Save the steering arguments in a separate json
                if not cache_file.exists():
                    with open(Path(cache_dir) / f"steering_params_{steering_key}.json", "w") as fp:
                        json.dump(gen_kwargs["steering_kwargs"] , fp)

            # For gathering_generation, we need to check the steering flag
            elif func.__name__ == "gather_generation":
                if text_list[0]["steering"] == "yes":
                    steering_key = hashlib.md5(json.dumps(gen_kwargs["steering_kwargs"], sort_keys=True).encode()).hexdigest()
                    cache_file = Path(cache_dir) / f"gathering_detection_steered_{steering_key}.pkl"
                else:
                    steering_key = hashlib.md5(json.dumps(gen_kwargs["steering_kwargs"], sort_keys=True).encode()).hexdigest()
                    cache_file = Path(cache_dir) / f"gathering_detection_vanilla_{steering_key}.pkl"
            
            print(f"Cache file: {cache_file}")
            print(f"Existance of cache file: {cache_file.exists()}")

            # If the cache file exists, load and return the cached result
            if cache_file.exists():
                return pickle.load(open(cache_file, 'rb'))
            
            # Otherwise, run the function and save the result to cache
            result = func(wrapped_llm, text_list, **gen_kwargs)
            pickle.dump(result, open(cache_file, 'wb'))
            return result
        return wrapper
    return decorator


#######################
#   Text generation   #
#######################

@cached_function()
def vanilla_generation(wrapped_llm, text_list, **kwargs):
    verbose = kwargs.get("verbose", True)
    generation_data_vanilla = []
    # Batch size for generation
    batch_size = kwargs.get("generation_batch_size", 8)
    for batch_start in range(0, len(text_list), batch_size):
        batch_texts = text_list[batch_start:batch_start + batch_size]
        # Run generation in batch
        output_texts = wrapped_llm.pipeline(batch_texts, **kwargs["generation_arguments"])
        encoded_inputs = [wrapped_llm.pipeline.tokenizer(text, return_tensors="pt", padding=False, truncation=False).to(wrapped_llm.pipeline.model.device) for text in batch_texts]
        input_lengths = [len(ids["input_ids"][0]) for ids in encoded_inputs]

        for i, (input_text, output, input_len) in enumerate(zip(batch_texts, output_texts, input_lengths)):
            if verbose:
                print("output_text:", output[0]['generated_text'])
            generation_data_vanilla.append({
                "input_text": input_text,
                "input_text_id": batch_start + i,
                "input_token_length": input_len,
                "output_text": output[0]['generated_text'],
                "steering": "none",
            })
    return generation_data_vanilla

@cached_function()
def steering_generation(wrapped_llm, text_list, verbose=True,**kwargs):
    verbose = kwargs.get("verbose", True)
    # Trully random noise key vector
    if kwargs["steering_kwargs"]["noise_type"] == "uniform":
        key_vector = torch.rand(wrapped_llm.embedding_dim, dtype=torch.bfloat16).to(wrapped_llm.pipeline.model.device)
        key_vector = 2*key_vector - 1  # Scale to [-1, 1]
    elif kwargs["steering_kwargs"]["noise_type"] == "sparse_0.05":
        key_vector = torch.zeros(wrapped_llm.embedding_dim, dtype=torch.bfloat16).to(wrapped_llm.pipeline.model.device)
        num_nonzero = int(0.05 * wrapped_llm.embedding_dim)
        nonzero_indices = torch.randperm(wrapped_llm.embedding_dim)[:num_nonzero]
        gap_noise = torch.rand(num_nonzero, dtype=torch.bfloat16).to(wrapped_llm.pipeline.model.device)
        gap_noise = 2*gap_noise - 1  # Scale to [-1, 1]
        key_vector[nonzero_indices] = gap_noise
        if verbose:
            plt.plot(key_vector.detach().type(torch.float16).cpu().numpy())
            plt.show()
    else:
        raise ValueError("Unsupported noise type. Only 'uniform' is currently implemented.")
    key_vector = key_vector * kwargs["steering_kwargs"]["noise_max"]

    generation_data_steered = []
    steering_hooks = wrapped_llm.register_hooks("steering", kwargs["steering_kwargs"]["layers"], key_vector)

    print("Number of registered hooks:", len(steering_hooks))

    batch_size = kwargs.get("generation_batch_size", 8)
    for batch_start in range(0, len(text_list), batch_size):
        batch_texts = text_list[batch_start:batch_start + batch_size]
        # Run generation in batch
        output_texts = wrapped_llm.pipeline(batch_texts, **kwargs["generation_arguments"])
        encoded_inputs = [wrapped_llm.pipeline.tokenizer(text, return_tensors="pt", padding=False, truncation=False).to(wrapped_llm.pipeline.model.device) for text in batch_texts]
        input_lengths = [len(ids["input_ids"][0]) for ids in encoded_inputs]
        for i, (input_text, output, input_len) in enumerate(zip(batch_texts, output_texts, input_lengths)):
            if verbose:
                print("output_text:", output[0]['generated_text'])
            generation_data_steered.append({
                "input_text": input_text,
                "input_text_id": batch_start + i,
                "input_token_length": input_len,
                "output_text": output[0]['generated_text'],
                "steering": "yes",
            })

    for hook in steering_hooks:
        hook.remove()

    return generation_data_steered, key_vector

#################################
#   Gathering Activation Data   #
#################################

@cached_function()
def gather_generation(wrapped_llm, generation_data, **generate_kwargs):
    """
    Inpout:
        wrapped_llm: type WrappedLLM
        generation_data: list of dictionaries, each containing:
            - "input_text": The input text for generation.
            - "input_text_id": The ID of the input text.
            - "input_token_length": The length of the input text in tokens.
            - "output_text": The generated output text.
            - "steering": The steering type (e.g., "none" or "yes").
        generate_kwargs: Dictionary containing:
            - "generation_arguments": Arguments for the generation pipeline.
            - "steering_kwargs": Dictionary containing:
                - "layers": List of layer names to gather activations from.
    """
    detection_data = []

    # saved_activation = {}
    saving_hooks = wrapped_llm.register_hooks("gather", generate_kwargs["steering_kwargs"]["layers"])


    for gen_data in tqdm(generation_data):

        generated_text = gen_data["output_text"]

        # Generation arguments in context of gathering: No need for long sequence generation, just 1 token is enough
        gathering_gen_kwargs = generate_kwargs["generation_arguments"].copy()
        gathering_gen_kwargs["max_new_tokens"] = 1

        # Run generation
        output_text = wrapped_llm.pipeline(generated_text, **gathering_gen_kwargs)

        token_ids = wrapped_llm.pipeline.tokenizer.encode(generated_text)
        # Convert back to tokens
        tokens = wrapped_llm.pipeline.tokenizer.convert_ids_to_tokens(token_ids)
        
        # gather in order the activations memorries from the hooks
        activations = {}
        for hook in saving_hooks:
            for layer in generate_kwargs["steering_kwargs"]["layers"]:
                if hook.layer_name == layer:
                    activations[layer] = hook.activations

        detection_data.append({
            "input": gen_data,
            "input_tokens": tokens,
            "output_text": output_text[0]['generated_text'],
            "activations": activations,
        })

    # Always remove hooks after you're done
    for hook in saving_hooks:
        hook.remove()

    return detection_data




##########################
#   Dataset Management   #
##########################

def split_data_accoring_to_sentence_id(data, val_size=0.1, test_size=0.2, seed=0, token_aggregation=False, sentence_array=False):
    """
    Split the data according to the sentence id.
    """
    np.random.seed(seed)

    if type(data) == pd.DataFrame:
        print(data.columns)
        total_sentences = len(set(data["input_token_ids"].tolist()))
    elif type(data) == list:
        total_sentences = len(set([d["input"]["input_text_id"] for d in data]))
    else:
        raise ValueError("Data should be a pandas DataFrame or a list of dictionaries.")

    val_nb = int(total_sentences * val_size)
    test_nb = int(total_sentences * test_size)
    split_labels = [0] * (total_sentences - val_nb - test_nb) + [1] * val_nb + [2] * test_nb
    np.random.shuffle(split_labels)

    df = pd.DataFrame(data)
    # ENTRIES: 'input', 'input_tokens', 'output_text', 'activations'
    # ENTRIES in 'input': 'input_text', 'input_text_id', 'input_token_length', 'output_text', 'steering'

    # Convert steering type into labels for classification
    # df["label"] = df["input"].apply(lambda x: x["steering"]== "none")

    # Convert the sentence id into a boolean for test/train split
    df["split_label"] = df["input"].apply(lambda x: split_labels[x["input_text_id"]])

    # Convert the activations into numpy arrays
    # if not aggregation:
    if True:
        dict_col_name = 'activations'
        # Collect flattened rows
        rows = []
        for _, row in df.iterrows():
            base_data = row.drop(dict_col_name).to_dict()
            dict_data = row[dict_col_name]
            if isinstance(dict_data, dict):
                # Flatten the dictionary of activations according to the layer
                for k, v in dict_data.items():
                    layer_row = base_data.copy()
                    layer_row['layer'] = k

                    # Remove the activations from the input text
                    nb_input_token = row['input']["input_token_length"]
                    v = v[nb_input_token:]

                    # To create 1 data point per sentence. For example for transformer input
                    if sentence_array:
                        layer_row['fwd_data'] = np.array(v)
                        rows.append(layer_row)

                    # To create 1 data point per token
                    else:
                        # If token aggregation is enabled, aggregate the activations over tokens
                        if token_aggregation:
                            
                            layer_row['fwd_data'] = np.array([np.mean(v, axis=0)], dtype=np.float32)  # Aggregate over tokens
                            rows.append(layer_row)
                        # Otherwise, keep the activations for each token
                        else:
                            for i in range(len(v)):
                                token_row = layer_row.copy()
                                token_row['token_id'] = i
                                token_row['fwd_data'] = np.array(v[i], dtype=np.float32)
                                rows.append(token_row)
        df = pd.DataFrame(rows)

    # else :
    #     # Average the activations over the layers
    #     df["fwd_data"] = df["activations"].apply(lambda x: np.array([v for k, v in x.items()]).mean(axis=0))



    data_train = df[df["split_label"] == 0]
    data_val = df[df["split_label"] == 1]
    data_test = df[df["split_label"] == 2]

    return data_train, data_val, data_test



# New version of the function for the broader data version
def split_data_accoring_to_sentence_id2(data, val_size=0.1, test_size=0.2, seed=0, token_aggregation=False, sentence_array=False, max_token_seq=None, split_labels=None):
    """
    Split the data according to the sentence id.
    Take the dataframe of all sentences, to output dataframes for each split with 1 row per token or per sentence
    """
    np.random.seed(seed)

    # print(data.columns)
    # 'input_text', 'input_text_id', 'input_token_length', 'input_token_ids', 'output_text', 'steering_noise', 'steering_type', 'steering_layers', 'params', 'activations


    if split_labels is None:
        total_sentences = len(set(data["input_text_id"].tolist()))
        val_nb = int(total_sentences * val_size)
        test_nb = int(total_sentences * test_size)
        split_labels = [0] * (total_sentences - val_nb - test_nb) + [1] * val_nb + [2] * test_nb
        np.random.shuffle(split_labels)

    print("Split sizes:", "Train:", split_labels.count(0), "Val:", split_labels.count(1), "Test:", split_labels.count(2))
    print("Split labels:", split_labels)
    # print(data["steering_type"])


    # Convert steering type into labels for classification
    # TODO: This is not in use anymore, classification label is given in the main script, ex: robustness_pipeline.py
    # list_of_label_zeros = ["vanilla", "human"]
    # data["label"] = data["steering_type"].apply(lambda x: not x in list_of_label_zeros)

    # Convert the sentence id into a boolean for test/train split
    data["split_label"] = data["input_text_id"].apply(lambda x: split_labels[x])

    # Convert the activations into numpy arrays
    dict_col_name = 'activations'

    # Check if robustness data is present and if yes, use it
    use_robustness = False
    if "paraphrased_activations" in data.columns:
        robustness_col_name = "paraphrased_activations"
        if data[robustness_col_name].notnull().any():
            use_robustness = True

    print("Use robustness data:", use_robustness)

    # Collect flattened rows
    rows = []
    for _, row in tqdm(data.iterrows()):
        base_data = row.drop(dict_col_name).to_dict()
        dict_data = row[dict_col_name]

        if use_robustness:
            base_data = row.drop([robustness_col_name, dict_col_name]).to_dict()

            dict_data_robust = row[robustness_col_name]
            if dict_data_robust is None:    
                # TODO: Make this cleaner / Try with paraphrasing vanilla too
                # The case where we mix Vanilla (Non paraphrased) and steered (paraphrased) data
                # This if concerns only the vanilla data points
                # So we just use the original activations as a shortcute..
                # NOTE: This is actually used usefully in case no paraphrasing compared to paraphrased part.
                dict_data = {layer_key: (dict_data[layer_key], dict_data[layer_key]) for layer_key in dict_data.keys()}
            else :
                dict_data = {layer_key: (dict_data[layer_key], dict_data_robust[layer_key]) for layer_key in dict_data.keys()}

        if isinstance(dict_data, dict):
            # Flatten the dictionary of activations according to the layer
            # In case of robustness, dict_data[layer_key] = (original_activations, robust_activations)
            for k, v in dict_data.items():
                layer_row = base_data.copy()
                layer_row['layer'] = k

                # # Remove the activations from the input text
                # nb_input_token = row["input_token_length"]
                if use_robustness:
                    v, v_robust = v

                # To create 1 data point per sentence. For example for transformer input
                if sentence_array:
                    layer_row['token_id'] = 0
                    layer_row['fwd_data'] = np.array(v)
                    if use_robustness:
                        layer_row['fwd_data_robust'] = np.array(v_robust)
                    rows.append(layer_row)

                # To create 1 data point per token
                else:
                    # If token aggregation is enabled, aggregate the activations over tokens
                    if token_aggregation:

                        layer_row['token_id'] = 0
                        layer_row['fwd_data'] = np.mean(v, axis=0)  # Aggregate over tokens
                        if use_robustness:
                            layer_row['fwd_data_robust'] = np.mean(v_robust, axis=0)
                        rows.append(layer_row)
                    # Otherwise, keep the activations for each token
                    else:
                        for i in range(len(v)):
                            # Only keep tokens up to max_token_seq if specified
                            if (max_token_seq is not None) and (i < max_token_seq):
                                # Cutoff to the min sequence length between paraphrased data and the original one.
                                # if use_robustness and i >= len(v_robust):
                                #     continue
                                # else:                                
                                token_row = layer_row.copy()
                                token_row['token_id'] = i       # Actually the token position in the output sequence...
                                token_row['fwd_data'] = np.array(v[i], dtype=np.float16)
                                if use_robustness:
                                    if i < len(v_robust):
                                        token_row['fwd_data_robust'] = np.array(v_robust[i], dtype=np.float16)

                                rows.append(token_row)
    df = pd.DataFrame(rows)


    data_train = df[df["split_label"] == 0]
    data_val = df[df["split_label"] == 1]
    data_test = df[df["split_label"] == 2]

    # This split is on the token level
    # split_label_list = df["split_label"].apply(lambda x: ["train", "val", "test"][x]).tolist()

    # For rest of the evaluation we need the split at sentence level
    return data_train, data_val, data_test, split_labels






































if __name__ == "__main__":
    from llm_wrapper import LLMWrapper
    
    text_text_list = ["This is a test text"]

    # GENERATION ARGUMENTS
    generate_kwargs = {
        "generation_batch_size": 8,

        # FOR LLMS
        "model_arguments": {
            # "pad_token_id": pipeline.tokenizer.eos_token_id,
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "load_in_8bit": False,
            "torch_dtype": "torch.bfloat16",
            "hf_token": os.environ.get("HF_TOKEN", ""),
        },
        
        # FOR GENERATION
        "generation_arguments": {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 35,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": None, 
            "eos_token_id": None,
        },

        # FOR TEXT DATA
        "input_version": "home_made_v1.210", # "test_v1.10", #"home_made_v1.210",

        # FOR STEERING
        "steering_kwargs": {
            "layers": [16],
            "noise_max": 0.00001,
            "noise_type": "uniform",
        },
    }

    wrapped_llm = LLMWrapper(**generate_kwargs["model_arguments"])

    detection_data = gather_generation(wrapped_llm, text_text_list, **generate_kwargs)

    print(detection_data)
