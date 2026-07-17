# Open parameters file

from turtle import pd
import pandas as pd
import yaml
import copy
import sys
import pprint
import numpy as np

import flatdict
from clearml import Task

from data_processing import params_to_vanilla, hash_params, format_labels
from text_generation import generate_text, get_human_written_text
from activation_gathering import gather_data
from detection import detect_watermark
from quality_evaluation import evaluate_quality
from evaluation import evaluate_detection
from dataset_reader import get_human_texts
from paraphrasing import paraphrase_texts

if len(sys.argv) < 2:
    print("Usage: python robustness_pipeline.py <param_file.yaml>")
    sys.exit(1)

param_file = sys.argv[1]

with open(param_file, "r") as f:
    params = yaml.safe_load(f)

print("** Running Robustness pipeline with parameters from", param_file)
flat_params = dict(flatdict.FlatDict(params, delimiter='-'))
pprint.pprint(flat_params)

unique_param_hash, human_readable_hash = hash_params(params, human_readable=True)
print("-> Unique parameter hash:", human_readable_hash, unique_param_hash)

task = Task.init(project_name='Watermarking steering', task_name='Robustness_' + human_readable_hash + "_" +str(unique_param_hash))
task.set_parameters(flat_params)
# In this pipeline, we slowly enrich the data until we have the full dataframe, from Input, output to detection and evaluation

number_of_bits = params["detection_arguments"]["number_of_bits"]
text_generation_seeds = 1
all_gathered_list = []


# The vanilla process
print(">>>>>>> Vanilla generation")
vanilla_params = params_to_vanilla(copy.deepcopy(params))
df_vanilla_generated = get_human_written_text(vanilla_params)

# Generate the split between test, val, train
total_sentences = len(set(df_vanilla_generated["input_text_id"].tolist()))
val_ratio = 0.1
test_ratio = 0.2
val_nb = int(total_sentences * val_ratio)
test_nb = int(total_sentences * test_ratio)
split_labels = [0] * (total_sentences - val_nb - test_nb) + [1] * val_nb + [2] * test_nb
np.random.shuffle(split_labels)
print("Split sizes:", "Train:", split_labels.count(0), "Val:", split_labels.count(1), "Test:", split_labels.count(2))
vanilla_params["split_labels"] = split_labels
params["split_labels"] = split_labels


df_vanilla_generated = evaluate_quality(df_vanilla_generated, vanilla_params)
vanilla_params["robustness_arguments"]["paraphrasing"]["enabled"] = True

df_vanilla_generated = paraphrase_texts(df_vanilla_generated, vanilla_params)
# vanilla_params["robustness_arguments"]["paraphrasing"]["enabled"] = False

df_vanilla_gathering = gather_data(df_vanilla_generated, vanilla_params)
df_vanilla_generated["classification_label"] = 0

# The Steered process
print(">>>>>>> Steered generation")
steer_params = copy.deepcopy(params)
steer_params["eval_code_name"] = "steered"
steer_params["steering_arguments"]["noise_seed"] = text_generation_seeds + 10*steer_params["steering_arguments"]["noise_offset"]
df_steered_generated = generate_text(steer_params)

# Paraphrased process
print(">>>>>>> Paraphrased generation")
paraph_params = copy.deepcopy(params)
paraph_params["eval_code_name"] = "paraphrased"
paraph_params["steering_arguments"]["noise_seed"] = 1 + text_generation_seeds + 10*paraph_params["steering_arguments"]["noise_offset"]
paraph_params["robustness_arguments"]["paraphrasing"]["enabled"] = True

df_paraphrased = paraphrase_texts(df_steered_generated.copy(), paraph_params)
df_paraphrased = evaluate_quality(df_paraphrased, paraph_params)
df_paraphrased_gathering = gather_data(df_paraphrased, paraph_params)
df_paraphrased_gathering["classification_label"] = 1

# Finishing the steered generation
print(">>>>>>> Steered Gathering")
df_steered_generated = evaluate_quality(df_steered_generated, steer_params)
df_steered_gathering = gather_data(df_steered_generated, steer_params)
df_steered_gathering["classification_label"] = 1
# df_steered_generated.iloc[:, "classification_label"] = 1

# Combined data for detection

# Print the generated texts for debugging
print()
print("** Sample generated texts print:")
print("Input prompt:", df_vanilla_gathering['input_text'].iloc[0])
print(f"Vanilla : {df_vanilla_gathering['output_text'].iloc[0]}")
# print(f"Vanilla : {df_vanilla_gathering['paraphrased_text'].iloc[0]}")
print("Average length of vanilla text:", df_vanilla_gathering['output_text'].apply(len).mean())
# print("Average length of vanilla paraphrased text:", df_vanilla_gathering['paraphrased_text'].apply(len).mean())
print("----") 
print(f"Steered : {df_steered_gathering['output_text'].iloc[0]}")
print("Average length of steered text:", df_steered_gathering['output_text'].apply(len).mean())
print("----") 
# print(f"Paraphrased : {df_paraphrased_gathering['output_text'].iloc[0]}")
print(f"Paraphrased : {df_paraphrased_gathering['paraphrased_text'].iloc[0]}")
print("Average length of paraphrased paraphrased text:", df_paraphrased_gathering['paraphrased_text'].apply(len).mean())
print("----")

print("** Now detecting watermark...")

df_vanilla_and_steered = pd.concat([df_vanilla_gathering, df_steered_gathering], ignore_index=True)
df_vanilla_and_paraphrased = pd.concat([df_vanilla_gathering, df_paraphrased_gathering], ignore_index=True)
    
df_vanilla_and_steered = format_labels(df_vanilla_and_steered)
df_vanilla_and_paraphrased = format_labels(df_vanilla_and_paraphrased)

detection_vanilla_and_steered = detect_watermark(df_vanilla_and_steered, steer_params)
detection_vanilla_and_paraphrased = detect_watermark(df_vanilla_and_paraphrased, paraph_params)

print()
print("** Now evaluating original vanilla VS steered ...")
df_evaluation_steered = evaluate_detection(df_vanilla_and_steered, steer_params, detection_vanilla_and_steered, task.get_logger())
print()
print("** Now evaluating original vanilla VS paraphrased ...")
df_evaluation_paraphrased = evaluate_detection(df_vanilla_and_paraphrased, paraph_params, detection_vanilla_and_paraphrased, task.get_logger())
print("** Pipeline", unique_param_hash, "finished")