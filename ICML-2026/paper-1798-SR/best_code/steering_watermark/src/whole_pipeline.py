# Open parameters file

from turtle import pd
import pandas as pd
import yaml
import copy
import sys
import pprint

import flatdict
from clearml import Task

from data_processing import params_to_vanilla, hash_params
from text_generation import generate_text 
from activation_gathering import gather_data
from detection import detect_watermark
from quality_evaluation import evaluate_quality
from evaluation import evaluate_detection
from dataset_reader import get_human_texts

if len(sys.argv) < 2:
    print("Usage: python whole_pipeline.py <param_file.yaml>")
    sys.exit(1)

param_file = sys.argv[1]

with open(param_file, "r") as f:
    params = yaml.safe_load(f)

print("** Running pipeline with parameters from", param_file)
flat_params = dict(flatdict.FlatDict(params, delimiter='-'))
pprint.pprint(flat_params)

unique_param_hash, human_readable_hash = hash_params(params, human_readable=True)
print("-> Unique parameter hash:", human_readable_hash, unique_param_hash)

task = Task.init(project_name='Watermarking steering', task_name='P1_' + human_readable_hash + "_" +str(unique_param_hash))
task.set_parameters(flat_params)
# In this pipeline, we slowly enrich the data until we have the full dataframe, from Input, output to detection and evaluation

# Steered generation
df_generated = generate_text(params)
print(len(df_generated), "texts generated.")
df_generated = evaluate_quality(df_generated, params)
df_gathering = gather_data(df_generated, params)

# Comparison generation
if params.get("comparison_arguments", {}).get("compared_text_type", None) == "vanilla":
    # Vanilla generation
    vanilla_params = params_to_vanilla(params)
    df_generated_vanilla = generate_text(vanilla_params)
    df_generated_vanilla = evaluate_quality(df_generated_vanilla, vanilla_params)
    df_gathering_vanilla = gather_data(df_generated_vanilla, vanilla_params)
    df_all_gathering = pd.concat([df_gathering, df_gathering_vanilla], ignore_index=True)

elif params.get("comparison_arguments", {}).get("compared_text_type", None) == "human":
    # Human text gathering
    df_human, human_params = get_human_texts(params)
    print(len(df_human), "human texts gathered.")
    df_human = evaluate_quality(df_human, human_params)
    df_gathering_human = gather_data(df_human, human_params)
    df_all_gathering = pd.concat([df_gathering, df_gathering_human], ignore_index=True)

elif params.get("comparison_arguments", {}).get("compared_text_type", None) == "steered_bis":
    # Steered bis generation
    steering_params_bis = copy.deepcopy(params)
    steering_params_bis["steering_arguments"]["noise_seed"] = 12345  # Different seed for different noise
    df_generated_bis = generate_text(steering_params_bis)
    print(len(df_generated_bis), "texts generated with bis steering.")
    steering_params_bis["steering_arguments"]["noise_type"] = "steered_bis"
    df_generated_bis = evaluate_quality(df_generated_bis, steering_params_bis)
    df_gathering_bis = gather_data(df_generated_bis, steering_params_bis)
    df_gathering_bis["steering_type"] = "steered_bis"
    df_all_gathering = pd.concat([df_gathering, df_gathering_bis], ignore_index=True)


print("** Now detecting watermark...")
detection_dictionary = detect_watermark(df_all_gathering, params)

df_evaluation = evaluate_detection(df_all_gathering, params, detection_dictionary, task.get_logger())

print("** Pipeline", unique_param_hash, "finished")