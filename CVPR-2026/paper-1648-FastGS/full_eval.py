#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, time
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Scene-specific budgets for "big" mode (final_count)
big_budgets = {
    "bicycle": 5987095,
    "flowers": 3618411,
    "garden": 5728191,
    "stump": 4867429,
    "treehill": 3770257,
    "room": 1548960,
    "counter": 1190919,
    "kitchen": 1803735,
    "bonsai": 1252367,
    "truck": 2584171,
    "train": 1085480,
    "playroom": 2326100,
    "drjohnson": 3273600
}

# Scene-specific budgets for "budget" mode (multiplier)
budget_multipliers = {
    "bicycle": 15,
    "flowers": 15,
    "garden": 15,
    "stump": 15,
    "treehill": 15,
    "room": 2,
    "counter": 2,
    "kitchen": 2,
    "bonsai": 2,
    "truck": 2,
    "train": 2,
    "playroom": 5,
    "drjohnson": 5
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--mode", type=str, default="big", choices=["budget", "big"])
parser.add_argument("--optimizer_type", type=str, default="default")
parser.add_argument("--sh_lower", action="store_true")
parser.add_argument("--dry_run", action="store_true")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

def run_cmd(CMD, args):
    print(CMD)
    if not args.dry_run:
        os.system(CMD)

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    common_args += " --optimizer_type {}".format(args.optimizer_type)
    
    if args.sh_lower:
        common_args += " --sh_lower"
    
    if args.mode == "big":
        mode_param = " --densification_interval 100 --mode final_count"
        start_time = time.time()
        for scene in mipnerf360_outdoor_scenes:
            source = args.mipnerf360 + "/" + scene
            budget_param = " --budget {} ".format(big_budgets[scene])
            CMD = "python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        for scene in mipnerf360_indoor_scenes:
            source = args.mipnerf360 + "/" + scene
            budget_param = " --budget {} ".format(big_budgets[scene])
            CMD = "python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        m360_timing = (time.time() - start_time)/60.0

        start_time = time.time()
        for scene in tanks_and_temples_scenes:
            source = args.tanksandtemples + "/" + scene
            budget_param = " --budget {} ".format(big_budgets[scene])
            CMD = "python train.py -s " + source + " -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        tandt_timing = (time.time() - start_time)/60.0

        start_time = time.time()
        for scene in deep_blending_scenes:
            source = args.deepblending  + "/" + scene
            budget_param = " --budget {} ".format(big_budgets[scene])
            CMD = "python train.py -s " + source + " -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        db_timing = (time.time() - start_time)/60.0

    elif args.mode == "budget":
        mode_param = " --densification_interval 500 --mode multiplier"
        start_time = time.time()
        for scene in mipnerf360_outdoor_scenes:
            source = args.mipnerf360 + "/" + scene
            budget_param = " --budget {} ".format(budget_multipliers[scene])
            CMD = "python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + f"{scene}_budget" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        for scene in mipnerf360_indoor_scenes:
            source = args.mipnerf360 + "/" + scene
            budget_param = " --budget {} ".format(budget_multipliers[scene])
            CMD = "python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + f"{scene}_budget" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        m360_timing = (time.time() - start_time)/60.0

        start_time = time.time()
        for scene in tanks_and_temples_scenes:
            source = args.tanksandtemples + "/" + scene
            budget_param = " --budget {} ".format(budget_multipliers[scene])
            CMD = "python train.py -s " + source + " -m " + args.output_path + "/" + f"{scene}_budget" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        tandt_timing = (time.time() - start_time)/60.0

        start_time = time.time()
        for scene in deep_blending_scenes:
            source = args.deepblending  + "/" + scene
            budget_param = " --budget {} ".format(budget_multipliers[scene])
            CMD = "python train.py -s " + source + " -m " + args.output_path + "/" + f"{scene}_budget" + common_args + budget_param + mode_param
            run_cmd(CMD, args)
        db_timing = (time.time() - start_time)/60.0

if not args.dry_run:
    with open(os.path.join(args.output_path, "timing.txt"), 'w') as file:
        file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

if not args.skip_rendering:
    if args.mode == "big":
        for scene in all_scenes:
            output_path = args.output_path + "/" + scene + "_big"
            CMD = f"python render.py -m {output_path}"
            run_cmd(CMD, args)
    
    elif args.mode == "budget":
        for scene in all_scenes:
            output_path = args.output_path + "/" + scene + "_budget"
            CMD = f"python render.py -m {output_path}"
            run_cmd(CMD, args)

if not args.skip_metrics:
    if args.mode == "big":
        for scene in all_scenes:
            output_path = args.output_path + "/" + scene + "_big"
            CMD = f"python metrics.py -m {output_path}"
            run_cmd(CMD, args)
    
    elif args.mode == "budget":
        for scene in all_scenes:
            output_path = args.output_path + "/" + scene + "_budget"
            CMD = f"python metrics.py -m {output_path}"
            run_cmd(CMD, args)