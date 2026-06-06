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

"""
python ./full_eval.py --save_images \
    --mipnerf360 xxx \
    --tanksandtemples xxx \
    --deepblending xxx \
    --enable_dash --lambda_entropy 0.015 --scale_reset_factor 0.2
"""


import os
from argparse import ArgumentParser

# 3.3M
MAX_N_GAUSSIAN = {
    "bicycle": 5987095,#54275
    "flowers": 3618411,#38347
    "garden": 5728191,#138766
    "stump": 4867429,#32049
    "treehill": 3770257,#52363
    "room": 1548960,#112627
    "counter": 1190919,#155767
    "kitchen": 1803735,#241367
    "bonsai": 1252367,#206613
    "truck": 2584171,#136029
    "train": 1085480,#182686
    "drjohnson": 3273600,#80861
    "playroom": 2326100#37005
}

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

def build_scene_output_path(output_path, dataset, scene, scale_reset_factor, enable_dash, lambda_entropy):
    path_suffix = f'-{dataset}-litegs'
    if enable_dash:
        path_suffix += '+dash'
    if scale_reset_factor > 0.0:
        path_suffix += f'+reset.{scale_reset_factor}'
    if lambda_entropy > 0:
        path_suffix += f'+entropy.{lambda_entropy}'
    return os.path.join(output_path+path_suffix, scene)

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./output")
parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
parser.add_argument("--deepblending", "-db", required=True, type=str)
parser.add_argument('--colmap_subfolder', default=".", type=str)
parser.add_argument('--save_images', action="store_true", help="Save rendered and ground truth images")
parser.add_argument('--enable_dash', action="store_true", help="Enable dash mode")
parser.add_argument('--scale_reset_factor', type=float, default=0.0, help="Scale reset factor")
parser.add_argument('--lambda_entropy', type=float, default=0.0, help="Entropy regularization weight")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training:
    # Build parameter strings based on command line arguments
    scale_reset_factor_param = f" --scale_reset_factor {args.scale_reset_factor}"
    lambda_entropy_param = f" --lambda_entropy {args.lambda_entropy}"

    for scene in mipnerf360_outdoor_scenes:
        scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
        scene_output_path=build_scene_output_path(args.output_path, 'm360', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
        test_epochs = " --test_epochs " + " ".join(map(str, range(0, 150)))
        if args.enable_dash:
            final_gaussian_count_param = f' --max_n_gaussian {int(MAX_N_GAUSSIAN[scene])}'
        else:
            final_gaussian_count_param = f' --final_gaussian_count {int(MAX_N_GAUSSIAN[scene])}'
        res = os.system("python example_train.py -s " + scene_input_path + " -i images_4 -m " + scene_output_path + " --eval --sh_degree 3 " + final_gaussian_count_param
                        + scale_reset_factor_param + lambda_entropy_param
                        )
        if res != 0:
            print(f"Training failed for scene {scene}")
    
    for scene in mipnerf360_indoor_scenes:
        scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
        scene_output_path=build_scene_output_path(args.output_path, 'm360', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
        test_epochs = " --test_epochs " + " ".join(map(str, range(0, 150)))
        if args.enable_dash:
            final_gaussian_count_param = f' --max_n_gaussian {int(MAX_N_GAUSSIAN[scene])}'
        else:
            final_gaussian_count_param = f' --final_gaussian_count {int(MAX_N_GAUSSIAN[scene])}'
        res = os.system("python example_train.py -s " + scene_input_path + " -i images_2 -m " + scene_output_path + " --eval --sh_degree 3 " + final_gaussian_count_param
                        + scale_reset_factor_param + lambda_entropy_param
                        )
        if res != 0:
            print(f"Training failed for scene {scene}")

    for scene in tanks_and_temples_scenes:
        scene_input_path=os.path.join(args.tanksandtemples,scene,args.colmap_subfolder)
        scene_output_path=build_scene_output_path(args.output_path, 'tat', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
        test_epochs = " --test_epochs " + " ".join(map(str, range(0, 150)))
        if args.enable_dash:
            final_gaussian_count_param = f' --max_n_gaussian {int(MAX_N_GAUSSIAN[scene])}'
        else:
            final_gaussian_count_param = f' --final_gaussian_count {int(MAX_N_GAUSSIAN[scene])}'
        res = os.system("python example_train.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --eval --sh_degree 3 " + final_gaussian_count_param
                        + scale_reset_factor_param + lambda_entropy_param
                        )
        if res != 0:
            print(f"Training failed for scene {scene}")

    for scene in deep_blending_scenes:
        scene_input_path=os.path.join(args.deepblending,scene,args.colmap_subfolder)
        scene_output_path=build_scene_output_path(args.output_path, 'db', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
        test_epochs = " --test_epochs " + " ".join(map(str, range(0, 150)))
        if args.enable_dash:
            final_gaussian_count_param = f' --max_n_gaussian {int(MAX_N_GAUSSIAN[scene])}'
        else:
            final_gaussian_count_param = f' --final_gaussian_count {int(MAX_N_GAUSSIAN[scene])}'
        res = os.system("python example_train.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --eval --sh_degree 3 " + final_gaussian_count_param
                        + scale_reset_factor_param + lambda_entropy_param
                        )
        if res != 0:
            print(f"Training failed for scene {scene}")

save_images_flag = " --save_images" if args.save_images else ""
for scene in mipnerf360_outdoor_scenes:
    scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
    scene_output_path=build_scene_output_path(args.output_path, 'm360', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
    res = os.system("python example_metrics.py -s " + scene_input_path + " -i images_4 -m " + scene_output_path + " --sh_degree 3" + save_images_flag)
    if res != 0:
        print(f"Evaluation failed for scene {scene}")

for scene in mipnerf360_indoor_scenes:
    scene_input_path=os.path.join(args.mipnerf360,scene,args.colmap_subfolder)
    scene_output_path=build_scene_output_path(args.output_path, 'm360', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
    res = os.system("python example_metrics.py -s " + scene_input_path + " -i images_2 -m " + scene_output_path + " --sh_degree 3" + save_images_flag)
    if res != 0:
        print(f"Evaluation failed for scene {scene}")

for scene in tanks_and_temples_scenes:
    scene_input_path=os.path.join(args.tanksandtemples,scene,args.colmap_subfolder)
    scene_output_path=build_scene_output_path(args.output_path, 'tat', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
    res = os.system("python example_metrics.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --sh_degree 3" + save_images_flag)
    if res != 0:
        print(f"Evaluation failed for scene {scene}")

for scene in deep_blending_scenes:
    scene_input_path=os.path.join(args.deepblending,scene,args.colmap_subfolder)
    scene_output_path=build_scene_output_path(args.output_path, 'db', scene, args.scale_reset_factor, args.enable_dash, args.lambda_entropy)
    res = os.system("python example_metrics.py -s " + scene_input_path + " -i images -m " + scene_output_path + " --sh_degree 3" + save_images_flag)
    if res != 0:
        print(f"Evaluation failed for scene {scene}")
