import json
import argparse
import re
import os

def negate_conjecture(conjecture):
    if isinstance(conjecture, str):
        if re.search(r'True', conjecture):
            updated_conjecture = re.sub(r'True', 'False', conjecture)
        elif re.search(r'False', conjecture):
            updated_conjecture = re.sub(r'False', 'True', conjecture)
        else:
            updated_conjecture = "false"
    else:
        updated_conjecture = "Invalid input: conjecture must be a string"
        print(f"Invalid input {conjecture}: conjecture must be a string")

    return updated_conjecture

def main(args):
    input_path = f'./results/{args.dataset_name}/{args.dataset_name}_{args.model}_trans_decompose_no_negation.json'
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    for item in data:
        item['sos_list'] = negate_conjecture(item['sos_list'])
        item['negated_label'] = "True"
    
    save_path = f'./results/{args.dataset_name}/{args.dataset_name}_{args.model}_trans_decompose_negated_data.json'
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)