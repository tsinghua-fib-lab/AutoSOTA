import os
from dotenv import load_dotenv
import json
import argparse
from src.inference import Inference
from data import get_dataset_generator
from huggingface_hub import login

load_dotenv()

# Login to Hugging Face using token from environment
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')

    parser.add_argument('--data_name', type=str, default='gsm8k', 
                        help='Choose among: gpqa, gsm8k, mmlu, hle, umwp, mc, mip')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--prompt_intervention', action='store_true',
                        help='Whether to use prompt intervention.')
    parser.add_argument('--prompt_template', type=int, default=1,
                        help='The prompt template to use for prompt intervention. Choose 1 for "Answer only if you are confident. Otherwise, say "I am not sure."`; choose 2 for "Please solve these problems with criticism. If the problem is reasonable, please think step by step and put your final answer within boxed. If the problem are unreasonable, highlight these issues clearly in your response and provide a succinct explanation."')

    return parser

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    model_name = args.model_name
    data_name = args.data_name

    results_dir = "results"

    if args.prompt_intervention:
        if args.prompt_template == 2:
            additional_format_prompt = "Please solve these problems with criticism. If the problem is reasonable, please think step by step and put your final answer within boxed. If the problem are unreasonable, highlight these issues clearly in your response and provide a succinct explanation."
            data_generator = get_dataset_generator(data_name, additional_format_prompt=additional_format_prompt)
            results_dir += "_prompt_intervention_criticism"
        else:
            additional_format_prompt = "Answer only if you are confident. Otherwise, say \"I am not sure.\""
            data_generator = get_dataset_generator(data_name, additional_format_prompt=additional_format_prompt)
            results_dir += "_prompt_intervention"
    else:
        data_generator = get_dataset_generator(data_name)

    begin_index = 0
    output_file = f"{results_dir}/{model_name}/{data_name}_results.jsonl"
    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        # get the index of the last line
        with open(output_file, "r") as f:
            lines = f.readlines()
            # for each line, if "ref_answer" does not exist, we add the ref_answer from the dataset
            for i, line in enumerate(lines):
                row = json.loads(line)
                #if "ref_answer" not in row:
                row["ref_answer"] = data_generator.dataset[i]["ref_answer"]
                lines[i] = json.dumps(row) + "\n"
            # rewrite the file
        with open(output_file, "w") as f:
            f.writelines(lines)
        with open(output_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                last_row = json.loads(last_line)
                begin_index = last_row.get("index", 0) + 1
    else:
        with open(output_file, "w") as f:
            f.write("")
    
    # if begin_index > 0:
    #     print(f"Some thread has already been processing the data. Exiting for now. ")
    #     exit(0)

    if begin_index >= len(data_generator.dataset):
        print(f"All data has been processed. No new data to process from index {begin_index}.")
        exit(0)

    inference_instance = Inference(model_name=model_name)
    data_generator.run_inference(inference_instance, output_file, begin_index, batch_size=args.batch_size)
    exit(0)