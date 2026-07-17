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
from text_generation import generate_text 
from activation_gathering import gather_data
from detection import detect_watermark
from quality_evaluation import evaluate_quality
from evaluation import evaluate_detection
from dataset_reader import get_human_texts
from paraphrasing import paraphrase_texts


if len(sys.argv) < 2:
    print("Usage: python multibit_pipeline.py <param_file.yaml>")
    sys.exit(1)

param_file = sys.argv[1]

with open(param_file, "r") as f:
    params = yaml.safe_load(f)

print("** Running Multi-bit pipeline with parameters from", param_file)
flat_params = dict(flatdict.FlatDict(params, delimiter='-'))
pprint.pprint(flat_params)

unique_param_hash, human_readable_hash = hash_params(params, human_readable=True)
print("-> Unique parameter hash:", human_readable_hash, unique_param_hash)

task = Task.init(project_name='Watermarking steering', task_name=params.get("run_name", "") + "_" + human_readable_hash + "_" +str(unique_param_hash))
task.set_parameters(flat_params)
# In this pipeline, we slowly enrich the data until we have the full dataframe, from Input, output to detection and evaluation

# Generate the split between test, val, train
total_sentences = params["data_arguments"]["max_loaded_samples"]
val_ratio = 0.1
test_ratio = 0.2
val_nb = int(total_sentences * val_ratio)
test_nb = int(total_sentences * test_ratio)
split_labels = [0] * (total_sentences - val_nb - test_nb) + [1] * val_nb + [2] * test_nb
np.random.shuffle(split_labels)
print("Split sizes:", "Train:", split_labels.count(0), "Val:", split_labels.count(1), "Test:", split_labels.count(2))
params["split_labels"] = split_labels


number_of_bits = params["detection_arguments"]["number_of_bits"]
text_generation_seeds = list(range(number_of_bits))
all_gathered_list = []

for n in range(number_of_bits):
    # Steered generation
    params["steering_arguments"]["noise_seed"] = text_generation_seeds[n] + 10*params["steering_arguments"]["noise_offset"]
    df_generated = generate_text(params)

    df_generated = paraphrase_texts(df_generated, params)
    df_generated = evaluate_quality(df_generated, params)

    if params["gathering_arguments"]["gathering_truncation"]:
        df_generated = df_generated[df_generated["input_text_id"] <= params["gathering_arguments"]["gathering_truncation"]]

    # params["robustness_arguments"]["paraphrasing"]["enabled"] = True
    df_gathering = gather_data(df_generated, params)
    # params["robustness_arguments"]["paraphrasing"]["enabled"] = False
    all_gathered_list.append(df_gathering)

df_all_gathering = pd.concat(all_gathered_list, ignore_index=True)
df_all_gathering = format_labels(df_all_gathering)

# params["robustness_arguments"]["paraphrasing"]["enabled"] = True


print("** Now detecting watermark...")
detection_dictionary = detect_watermark(df_all_gathering, params)

df_evaluation = evaluate_detection(df_all_gathering, params, detection_dictionary, task.get_logger())

print("** Pipeline", unique_param_hash, "finished")