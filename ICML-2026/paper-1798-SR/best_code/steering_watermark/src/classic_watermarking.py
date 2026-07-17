"""
# Test quality measures of traditional Watermark

- Test the hugging face interface of the watermark
- Check the version used
- Check the **accuracy** in simple setup
- Check the **Quality drop**
- Check the **robustness** to paraphrasing for example

"""

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config, pipeline, MistralCommonBackend
from transformers import WatermarkDetector
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
import numpy as np
import os
import time
import json
import copy
import flatdict

from clearml import Task

import torch
import sys
from datasets import load_dataset, Dataset

sys.path.append("src")
from llm_wrapper import LLMWrapper
from data_processing import load_text_list, cached_function2, hash_params
from text_generation import generate_text_cached, get_input_text_dataset, generate_noise
from paraphrasing import paraphrase_texts
from data_processing import cached_function2

def chunks(lst, n):
    # Gives successive n-sized chunks from lst. Last part may be smaller.
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def paraphrasing_data_ordering(params):
    # Get the parameters as input and give back the path to the according data
    data_name = params["human_hash"]
    path = f"data/classic_watermarking/{data_name}/"
    return path

def get_human_generated_text_dataset(params, tokenizer, classification_label=0):
    # Load dataset
    if params["data_arguments"]["input_version"].startswith("guardian"):
        text_list = load_text_list(
                params["data_arguments"]["input_version"],
            )

    elif params["data_arguments"]["input_version"].startswith("Hello-SimpleAI"):
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




        
    kept_text_list = []
    for i in range(params["data_arguments"]["max_loaded_samples"]):

        text_token_ids = tokenizer([text_list[i]], return_tensors="pt")["input_ids"][0]

        truncated_token_ids = text_token_ids[:params["data_arguments"].get("max_input_tokens", 512)]
        truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        print("Loaded human text:", i, truncated_text)
        kept_text_list.append({
            "classification_label": classification_label,
            "input_text": "",
            "output_text": truncated_text,
            "output_token_ids": truncated_token_ids,
            "input_text_id": i,
            "output_token_strings": tokenizer.convert_ids_to_tokens(truncated_token_ids),
        })

    data = pd.DataFrame(kept_text_list)
    data["params"] = [params] * len(data)
    return data



@cached_function2()
def generate_watermarked_text(data, params, additional_data):
    model = additional_data["model"]
    tokenizer = additional_data["tokenizer"]
    classification_label = additional_data["classification_label"]
    # Set up watermarking config
    if params["watermarking_arguments"]["watermarking_turned_on"]:
        watermarking_config = WatermarkingConfig(
            bias=params["watermarking_arguments"]["bias"], 
            context_width=params["watermarking_arguments"]["context_width"], 
            seeding_scheme=params["watermarking_arguments"]["seeding_scheme"]
        )
    else:
        watermarking_config = None

    # Load dataset
    text_dataset = get_input_text_dataset(params)

    # Generate watermarked text
    output_list = []
    # Batched generation    
    for batch in tqdm(chunks(text_dataset, params["generation_arguments"].get("generation_batch_size", 1))) :
        
        if isinstance(batch[0], list) :
            # Prepare inputs with chat template if needed. When generating on questions. Not when gathering activations

            text_prompt = [batch[i][1]["content"] for i in range(len(batch))]

            # 1) Chat template → strings
            inputs = tokenizer.apply_chat_template(
                batch,
                return_tensors="pt",
                tokenize=True,
                padding=True,
                truncation=True,
                add_generation_prompt=True,
            ).to(params["device"])
        
        else :
            text_prompt = batch
            inputs = tokenizer(batch, padding=True, padding_side="left", return_tensors="pt").to(params["device"])

        out = model.generate(
            inputs["input_ids"], 
            watermarking_config=watermarking_config, 
            max_length=params["generation_arguments"]["max_new_tokens"], 
            temperature= params["generation_arguments"].get("temperature", 0.7),
            top_p= params["generation_arguments"].get("top_p", 0.9),
            top_k= params["generation_arguments"].get("top_k", 50),
            repetition_penalty= params["generation_arguments"].get("repetition_penalty", 1.1),
            # do_sample=False
        )
        out_text = tokenizer.batch_decode(out, skip_special_tokens=True)
        
        for i, (input_text, output_text, out_ids)  in enumerate(zip(text_prompt, out_text, out)):
            # Store output
            output_list.append({
                "classification_label": classification_label,
                "input_text": input_text,
                "input_text_id": i,
                "input_token_length": inputs["input_ids"].shape[1],
                "input_token_ids": inputs["input_ids"],
                "output_text": output_text[len(input_text):],     # TODO: check if correct
                "output_token_ids": out_ids,
                "output_token_strings": tokenizer.convert_ids_to_tokens(out_ids),
            })

    data = pd.DataFrame(output_list)
    data["params"] = [params] * len(data)
    return data




def detecting_watermark(df, tokenizer, model_config, params):
    watermarking_config = WatermarkingConfig(
        bias=params["watermarking_arguments"]["bias"], 
        context_width=params["watermarking_arguments"]["context_width"], 
        seeding_scheme=params["watermarking_arguments"]["seeding_scheme"]
    )

    detector = WatermarkDetector(model_config=model_config, device=params["device"], watermarking_config= watermarking_config)

    results = []

    for row in df.itertuples():
        # print("Detecting watermark for text:", row.output_text)

        token_ids = tokenizer([row.output_text], return_tensors="pt")["input_ids"].to(params["device"])
        detection_preds = detector(token_ids, return_dict=True)

        accuracy = (int(detection_preds.prediction.item()) == row.classification_label)

        results.append({
            "label": row.classification_label,
            "predictions": int(detection_preds.prediction.item()),
            "confidences": detection_preds.confidence,
            "accuracies": accuracy,
            "z_scores": detection_preds.z_score,
            "p_values": detection_preds.p_value,
            "row": row,
        })

    return results


def calculate_metrics(results, experiment_name="", logger=None):
    total = len(results)
    df_results = pd.DataFrame(results)

    # Evaluate detection results
    detection_results = {}
    detection_results["accuracy"] = float(df_results["accuracies"].sum() / total)
    # detection_results["results"] = results
    # detection_results["labels"] = list(df_results["label"].array)
    # detection_results["predictions"] = list(df_results["predictions"].array)
    detection_results["confusion_matrix"] = confusion_matrix(list(df_results["label"].array), list(df_results["predictions"].array))
    detection_results["f1_score"] = f1_score(list(df_results["label"].array), list(df_results["predictions"].array))
    detection_results["total_samples"] = total
    detection_results["false_positives"] = int(detection_results["confusion_matrix"][0,1])
    detection_results["false_negatives"] = int(detection_results["confusion_matrix"][1,0])
    detection_results["true_positives"] = int(detection_results["confusion_matrix"][1,1])
    detection_results["true_negatives"] = int(detection_results["confusion_matrix"][0,0])

    if logger is not None:
        logger.report_scalar("detection/accuracy", "overall", iteration=0, value=detection_results["accuracy"])
        logger.report_scalar("detection/f1_score", "overall", iteration=0, value=detection_results["f1_score"])
        logger.report_scalar("detection/false_positives", "overall", iteration=0, value=detection_results["false_positives"])
        logger.report_scalar("detection/false_negatives", "overall", iteration=0, value=detection_results["false_negatives"])
        logger.report_scalar("detection/true_positives", "overall", iteration=0, value=detection_results["true_positives"])
        logger.report_scalar("detection/true_negatives", "overall", iteration=0, value=detection_results["true_negatives"])

    # compute confusion matrix
    print("Confusion matrix:\n", detection_results["confusion_matrix"])
    print("F1 score:", detection_results["f1_score"])
    print("Accuracy:", detection_results["accuracy"])

    return detection_results


def save_dict_as_yaml(d, filepath):
    # Convert numpy arrays to lists
    result_dict_clean = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            result_dict_clean[key] = value.tolist()
        else:
            result_dict_clean[key] = value

    with open(filepath, 'w') as f:
        yaml.dump(result_dict_clean, f, default_flow_style=False)










if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classic_watermarking.py <param_file.yaml>")
        sys.exit(1)

    param_file = sys.argv[1]
    start_time = time.time()


    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    test_setup = False  # set to False to run the full pipeline
    # param_file = "param.yaml"
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    a_hash, human_hash = hash_params(params, human_readable=True)
    params["human_hash"] = human_hash  
    params["data_ordering_function"] = paraphrasing_data_ordering(params)
    params["parameters_type"] = "classic_watermarking"

    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # params["model_arguments"]["model_id"] = "openai-community/gpt2"
    # params["data_arguments"]["input_version"] = "guardian_from_nov2025_articles_v1.10k"
    # params["data_arguments"]["max_loaded_samples"] = 31  # keep small for testing
    # params["generation_arguments"]["max_new_tokens"] = 512
    # params["generation_arguments"]["generation_batch_size"] = 64

    print("** Running Robustness pipeline with parameters from", param_file)
    flat_params = dict(flatdict.FlatDict(params, delimiter='-'))
    # pprint.pprint(flat_params)

    unique_param_hash, human_readable_hash = hash_params(params, human_readable=True)
    print("-> Unique parameter hash:", human_readable_hash, unique_param_hash)

    task = Task.init(project_name='Watermarking steering', task_name='Robustness_' + human_readable_hash + "_" +str(unique_param_hash))
    task.set_parameters(flat_params)

    
    params["watermarking_arguments"] = {
        "watermarking_turned_on": True,
        "bias": 2.5, 
        "context_width": 2, 
        "seeding_scheme": "selfhash"
    }

    if not test_setup:
        #######
        # Model
        #######
        
        if params["model_arguments"]["model_id"].startswith("meta-llama"):
            model = AutoModelForCausalLM.from_pretrained(params["model_arguments"]["model_id"]).to(params["device"])
            tokenizer = AutoTokenizer.from_pretrained(params["model_arguments"]["model_id"])

            # Token stuff
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            model_config = model.config


        elif params["model_arguments"]["model_id"].startswith("mistralai/Ministral"):

            model = torch.compile(
                Mistral3ForConditionalGeneration.from_pretrained(
                    params["model_arguments"]["model_id"],
                    device_map="auto",
                    # torch_dtype=torch.bfloat16,
                    quantization_config=FineGrainedFP8Config(dequantize=True)
                )
            )
            tokenizer = MistralCommonBackend.from_pretrained(params["model_arguments"]["model_id"])

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id


            model_config = model.config 

            model_config.vocab_size = tokenizer.vocab_size


        # It is all test
        params["split_labels"] = [2] * params["data_arguments"]["max_loaded_samples"]

        wat_params = copy.deepcopy(params)
        vanilla_params = copy.deepcopy(params)
        wat_params["watermarking_arguments"]["watermarking_turned_on"] = True
        wat_params["generation_arguments"]["watermark"] = True

        vanilla_params["watermarking_arguments"]["watermarking_turned_on"] = False
        vanilla_params["generation_arguments"]["watermark"] = False

        ###############
        ## Generation
        ###############
        # Watermarked generation
        print("<< Generating text with watermark...")
        df_watermarked = generate_watermarked_text(None, wat_params, {"model": model, "tokenizer": tokenizer, "classification_label": 1})
        # Vanilla generation
        print("<< Generating text without watermark...")
        df_vanilla = get_human_generated_text_dataset(params, tokenizer, classification_label=0)
        #generate_watermarked_text(None, vanilla_params, {"model": model, "tokenizer": tokenizer, "classification_label": 0})
        print(">> Generation done!")

        del model  # free some memory

        ###############
        ## Paraphrasing
        ###############
        print("<< Paraphrasing watermarked texts...")
        df_paraphrased_watermarked = paraphrase_texts(df_watermarked.copy(), wat_params)

        df_paraphrased_watermarked["output_token_ids"] = df_paraphrased_watermarked["paraphrased_text"].apply(
            lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0]
        )
        df_paraphrased_watermarked["output_text"] = df_paraphrased_watermarked["paraphrased_text"]

        # print(" << Paraphrasing vanilla texts...")
        # df_paraphrased_vanilla = paraphrase_texts(df_vanilla.copy(), vanilla_params)
        # print(">> Paraphrasing done!")
        # df_paraphrased_vanilla["output_token_ids"] = df_paraphrased_vanilla["paraphrased_text"].apply(
        #     lambda x: tokenizer([x], return_tensors="pt")["input_ids"][0]
        # )
        # df_paraphrased_vanilla["output_text"] = df_paraphrased_vanilla["paraphrased_text"]


        ###############
        ## Detection
        ###############
        vanilla_results = detecting_watermark(df_vanilla, tokenizer, model_config, vanilla_params)
        # paraphrased_vanilla_results = detecting_watermark(df_paraphrased_vanilla, tokenizer, model_config, vanilla_params)
        watermarked_results = detecting_watermark(df_watermarked, tokenizer, model_config, wat_params)
        paraphrased_results = detecting_watermark(df_paraphrased_watermarked, tokenizer, model_config, wat_params)

        df_vanilla_results = pd.DataFrame(vanilla_results)
        # df_paraphrased_vanilla_results = pd.DataFrame(paraphrased_vanilla_results)
        df_watermarked_results = pd.DataFrame(watermarked_results)
        df_paraphrased_results = pd.DataFrame(paraphrased_results)

        print("=== Detection results summary individual ===")

        print("Vanilla detection results:")
        print(df_vanilla_results["accuracies"].mean())
        print(df_vanilla_results["confidences"].mean())
        print(df_vanilla_results["p_values"].mean())

        print("Watermarked detection results:")
        print(df_watermarked_results["accuracies"].mean())
        print(df_watermarked_results["confidences"].mean())
        print(df_watermarked_results["p_values"].mean())

        print("Paraphrased detection results:")
        print(df_paraphrased_results["accuracies"].mean())
        print(df_paraphrased_results["confidences"].mean())
        print(df_paraphrased_results["p_values"].mean())
        print()
        # print("Paraphrased vanilla detection results:")
        # print(df_paraphrased_vanilla_results["accuracies"].mean())
        # print(df_paraphrased_vanilla_results["confidences"].mean())
        # print(df_paraphrased_vanilla_results["p_values"].mean())

        ############
        ## Calculate metrics
        ############
        print("=== Detection results summary original ===")
        original_metrics = calculate_metrics(watermarked_results + vanilla_results, experiment_name="watVShuman", logger=task.get_logger())
        print()

        print("=== Detection results summary paraphrased ===")
        paraphrased_metrics = calculate_metrics(paraphrased_results + vanilla_results, experiment_name="paraphrasedWatVShuman", logger=task.get_logger())
        # print("=== Detection results summary paraphrased vanilla ===")
        # paraphrased_vanilla_metrics = calculate_metrics(paraphrased_results + paraphrased_vanilla_results)

        # Save results as dictionary of df
        output_folder = "./data/classic_watermarking/"
        os.makedirs(output_folder, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"watres_{human_hash}_{timestamp}"

        print("Saving results to:", output_folder + file_name)

        save_dict_as_yaml(params, output_folder + file_name + "_params.yaml")
        save_dict_as_yaml(original_metrics, output_folder + file_name + "_original_metrics.yaml")
        save_dict_as_yaml(paraphrased_metrics, output_folder + file_name + "_paraphrased_metrics.yaml")
        # save_dict_as_yaml(paraphrased_vanilla_metrics, output_folder + file_name + "_paraphrased_vanilla_metrics.yaml")

    # TIME
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total elapsed time: {minutes}:{seconds} minutes.")