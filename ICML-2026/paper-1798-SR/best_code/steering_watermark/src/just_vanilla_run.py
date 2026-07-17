# Open parameters file

from turtle import pd
import pandas as pd
import yaml
import copy
import sys
import pprint

import flatdict
from clearml import Task

from data_processing import params_to_vanilla, hash_params, format_labels
from text_generation import generate_text 
from activation_gathering import gather_data
from detection import detect_watermark
from quality_evaluation import evaluate_quality
from evaluation import evaluate_detection
from dataset_reader import get_human_texts

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
human_readable_hash = "Just vanilla" + human_readable_hash 
print("-> Unique parameter hash:", human_readable_hash, unique_param_hash)

task = Task.init(project_name='Watermarking steering', task_name=params.get("run_name", "") + "_" + human_readable_hash + "_" +str(unique_param_hash))
task.set_parameters(flat_params)
# In this pipeline, we slowly enrich the data until we have the full dataframe, from Input, output to detection and evaluation

# Steered generation
params["steering_arguments"]["noise_seed"] = 0

# Define the vanilla parameters
setparams = params_to_vanilla(params)
print(">>>>>>> Vanilla parameters")
pprint.pprint(setparams)

df_generated = generate_text(setparams)
print(len(df_generated), "texts generated.")
df_generated = evaluate_quality(df_generated, setparams)
print("Quality evaluation done.")