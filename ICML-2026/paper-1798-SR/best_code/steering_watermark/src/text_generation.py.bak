from llm_wrapper import LLMWrapper
from data_processing import load_text_list, cached_function2
import torch
import pandas as pd
from datasets import load_dataset, Dataset




def generate_noise(embedding_dim, params):
    torch.random.manual_seed(params["steering_arguments"]["noise_seed"])
    if params["steering_arguments"]["noise_type"] == "uniform":
        key_vector = torch.rand(embedding_dim, dtype=torch.bfloat16)
        key_vector = 2*key_vector - 1  # Scale to [-1, 1]

    elif params["steering_arguments"]["noise_type"].startswith("sparse"):
        sparse_level = float(params["steering_arguments"]["noise_type"].split("_")[-1])
        key_vector = torch.zeros(embedding_dim, dtype=torch.bfloat16)
        num_nonzero = int(sparse_level * embedding_dim)
        nonzero_indices = torch.randperm(embedding_dim)[:num_nonzero]
        gap_noise = torch.rand(num_nonzero, dtype=torch.bfloat16)
        gap_noise = 2*gap_noise - 1  # Scale to [-1, 1]
        key_vector[nonzero_indices] = gap_noise
    
    elif params["steering_arguments"]["noise_type"] == "vanilla":
        key_vector = torch.zeros(embedding_dim, dtype=torch.bfloat16)

    else:
        raise ValueError("Unsupported noise type. Only 'uniform' is currently implemented.")

    key_vector = key_vector * params["steering_arguments"]["noise_max"]
    return key_vector


def generate_text_data_ordering(params):
    # Get the parameters as input and give back the path to the according data
    data_name = params['data_arguments']['input_version']
    model_name = params['model_arguments']['model_id'].split('/')[-1]
    generation_style = f"gen_{params['generation_arguments']['max_new_tokens']}_temp{params['generation_arguments']['temperature']}"
    if float(params['steering_arguments']['noise_max']) == 0.0:
        steering_style = params["comparison_arguments"]["compared_text_type"]
    else:
        steering_style = f"steering_noise{params['steering_arguments']['noise_type']}_{params['steering_arguments']['noise_max']}_layers{'-'.join(map(str, params['steering_arguments']['steering_layers']))}"
    path = f"data/{data_name}/{model_name}/{generation_style}/{steering_style}/"
    return path


def generate_text(params):
    # Curate the usefull parameters for the caching system
    curated_params = {
        "parameters_type": "text_generation",
        "data_ordering_function": generate_text_data_ordering(params),
        "model_arguments": params["model_arguments"],
        "data_arguments": params["data_arguments"],
        "generation_arguments": params["generation_arguments"],
        "steering_arguments": params["steering_arguments"],
        "verbose": params.get("verbose", True),
        "hf_token": params["hf_token"],
    }
    return generate_text_cached(None, curated_params)

def truncate_text_by_words(text, max_words):
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    else:
        return ' '.join(words[:max_words])


def get_input_text_dataset(params):

    # Load from hugging face dataset
    if params["data_arguments"]["input_version"] == "Hello-SimpleAI/HC3":
        # No distinction between different sub cathegories for now
        dataset = load_dataset(
            params["data_arguments"]["input_version"], 
            "all", #"reddit_eli5"
        ) 
        input_questions = []
        for i in range(len(dataset["train"])):
            # Skip edited questions (In reddit dataset, Edits contain modifications that may not be suitable for our use case)
            filtered_out_elements = ["edit", "url"]
            # Only consider question that dont contain any of the filtered out elements
            if all(elem not in dataset["train"][i]["question"].lower() for elem in filtered_out_elements):
                
                # # Create prompt in chat format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Write only in plain text, without formatting using * or #."},
                    {"role": "user", "content": dataset["train"][i]["question"]}
                ]
                
                input_questions.append(messages)
                if params["data_arguments"].get("max_loaded_samples", None) is not None and len(input_questions) >= params["data_arguments"]["max_loaded_samples"]:
                    break

        text_list = input_questions
                    
    # Load from local file
    else:
        text_list = load_text_list(
            params["data_arguments"]["input_version"],
        )
        # Truncate by words if specified
        if params["data_arguments"].get("truncate_input_words", None) is not None:
            text_list = [truncate_text_by_words(text, params["data_arguments"]["truncate_input_words"]) for text in text_list]
        # Limit to max loaded samples
        if params["data_arguments"].get("max_loaded_samples", None) is not None and len(text_list) >= params["data_arguments"]["max_loaded_samples"]:
            text_list = text_list[:params["data_arguments"]["max_loaded_samples"]]

    return text_list

def get_human_written_text(params):
    # Filter only human written text
    classification_label = 0  # Human written texts are labeled as 0

    llm = LLMWrapper(
        hf_token = params["hf_token"],
        **params["model_arguments"]
    )

    if params["data_arguments"]["input_version"] == "Hello-SimpleAI/HC3":
        # No distinction between different sub cathegories for now
        dataset = load_dataset(
            params["data_arguments"]["input_version"], 
            "all", #"reddit_eli5"
        ) 
        text_list = []
        for i in range(len(dataset["train"])):
            # Skip edited questions (In reddit dataset, Edits contain modifications that may not be suitable for our use case)
            filtered_out_elements = ["edit", "url"]
            # Only consider question that dont contain any of the filtered out elements
            if all(elem not in dataset["train"][i]["question"].lower() for elem in filtered_out_elements):

                text_list.append(dataset["train"][i]["question"] + " " + dataset["train"][i]["human_answers"][0])

    else :  # news dataset
        # Load dataset
        text_list = load_text_list(
                params["data_arguments"]["input_version"],
            )
    
    human_texts = []
    for i in range(params["data_arguments"]["max_loaded_samples"]):

        text_token_ids = llm.tokenizer([text_list[i]], return_tensors="pt")["input_ids"][0]
        truncated_token_ids = text_token_ids[:params["data_arguments"].get("max_input_tokens", 512)]
        truncated_text = llm.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

        human_texts.append({
            "classification_label": classification_label,
            "input_text": "",
            "input_text_id": i,
            "input_token_length": 0,
            "input_token_ids": [],
            "output_text": truncated_text,
            "output_token_ids": truncated_token_ids,
            "output_token_strings": llm.tokenizer.decode(truncated_token_ids),
            "steering_noise": 0,
            "steering_type": "human",
            "steering_layers": params["steering_arguments"]["steering_layers"],
            "key_vector": torch.zeros(llm.embedding_dim).float().detach().cpu().numpy(),
        })

    del llm # Free up memory

    data = pd.DataFrame(human_texts)
    data["params"] = [params] * len(data)

    return data





@cached_function2()
def generate_text_cached(data, params):
    # data is unused, only for caching purposes
    verbose = params["verbose"]

    llm = LLMWrapper(
        hf_token = params["hf_token"],
        **params["model_arguments"]
    )

    text_dataset = get_input_text_dataset(params)

    ### IF STEERING HERE
    key_vector = generate_noise(llm.embedding_dim, params).to(llm.device)
    steering_hooks = llm.register_hooks("steering", params["steering_arguments"]["steering_layers"], key_vector)

    # Format arguments for generation
    formated_gen_args = params["generation_arguments"].copy()
    formated_gen_args.pop("generation_batch_size") 
    generated_text = []
    # Batch size for generation
    batch_size = params["generation_arguments"]["generation_batch_size"]

    # Create dataset for efficient batching and generation
    output_dict = llm(
        text_dataset,
        rich_output=True,
        batch_size=batch_size,
        **formated_gen_args
    )

    # Classification labels: 0 for non-watermarked, else the noise seed for multi-bit steering
    if params["steering_arguments"]["noise_max"] == 0.0:
        classification_label = 0
    else:
        classification_label = params["steering_arguments"]["noise_seed"]

    for i, (output_dict_elmt) in enumerate(output_dict):
        # if verbose:
            # print(" >> Output text:", output_dict_elmt)
        generated_text.append({
            "classification_label": classification_label,
            "input_text": output_dict_elmt["input_text"],
            "input_text_id": i,
            "input_token_length": output_dict_elmt["input_lengths"],
            "input_token_ids": output_dict_elmt["encoded_inputs"],
            "output_text": output_dict_elmt["generated_texts"],
            "output_token_ids": output_dict_elmt["encoded_outputs"],
            "output_token_strings": output_dict_elmt["output_token_strings"],
            "steering_noise": params["steering_arguments"]["noise_max"],
            "steering_type": params["steering_arguments"]["noise_type"],
            "steering_layers": params["steering_arguments"]["steering_layers"],
            "key_vector": key_vector.float().detach().cpu().numpy(),
        })

    for hook in steering_hooks:
        hook.remove()

    del llm # Free up memory

    data = pd.DataFrame(generated_text)
    data["params"] = [params] * len(data)

    return data


    