import os
import re
import json
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from utils import calculate_f1_over_questions, calculate_exact_match_for_questions
import argparse
import csv
import random 

def extract_answer_option(model_output, options):
    # Define regex pattern to match "Answer: A" where A is any letter A-D
    pattern = r"Answer:\s*([A-D])"
    
    # Search for the pattern in the model output
    match = re.search(pattern, model_output)
    
    if match:
        return ord(match.group(1)) - ord('A')  # Returns the letter (A, B, C, or D)
    else:
        model_output = model_output.lower()
        for i, option in enumerate(options):
            if option.lower() in model_output:
                return i
        return random.choice([0,1,2,3])
    

def extract_answer(model_output):
    if "Answer:" in model_output:
        _, res = model_output.split("Answer:")
        res = res.strip()
    else:
        res = ""
    return res
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default="./dataset/ifqa")
    parser.add_argument('--data_split', type=str, help='data split', default="test") # validation
    parser.add_argument('--folder_path', type=str, help='model output folder', default="./outputs/ifqa/test")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    dataset = load_from_disk(args.dataset_name)
    for file_name in os.listdir(args.folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(args.folder_path, file_name)
            with open(file_path, 'r') as f:
                model_outputs = json.load(f)
            
            preds, golds = [], []
            for data, model_output in zip(dataset[args.data_split], model_outputs):
                final_res = extract_answer(model_output)
                preds.append(final_res)
                golds.append(data['answers'])
            
            f1 = calculate_f1_over_questions(preds, golds)
            em = calculate_exact_match_for_questions(preds, golds)
            print(file_name)
            print(f'F1: {f1}  EM: {em}')
            print()

