import json
import os

import numpy as np
import scipy.io as sio


def convert_json_to_mat(json_dir="ppo_logs", output_file="ppo_results.mat"):
    generation = 500
    ind_runs = 10

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    json_files.sort(key=lambda x: int(x.split("_")[0][1:]))

    results_dict = {}
    all_tasks_list = []

    for json_file in json_files:
        task_id = int(json_file.split("_")[0][1:])
        env_name = json_file.split("_")[1]
        print(f"Processing task {task_id}, environment {env_name}")

        with open(os.path.join(json_dir, json_file), "r") as f:
            data = json.load(f)

        if len(data["all_rewards"]) < ind_runs:
            print(
                f"Warning: {env_name} has only {len(data['all_rewards'])} runs, skipping"
            )
            continue

        runs_data = []
        for run in data["all_rewards"][:ind_runs]:
            processed_run = [np.nan if x is None else x for x in run]

            if len(processed_run) < generation:
                padded_run = processed_run + [processed_run[-1]] * (
                    generation - len(processed_run)
                )
                runs_data.append(padded_run[:generation])
            else:
                runs_data.append(processed_run[:generation])

        env_matrix = np.array(runs_data)

        key = f"task_{task_id}_{env_name}"
        results_dict[key] = env_matrix
        all_tasks_list.append(env_matrix)

    if all_tasks_list:
        all_tasks_matrix = np.stack(all_tasks_list)
    else:
        all_tasks_matrix = np.array([])

    sio.savemat(
        output_file,
        {
            "all_tasks": all_tasks_matrix,
            **results_dict,
        },
    )

    print(f"Successfully saved results to {output_file}")
    print(f"Total tasks processed: {len(json_files)}")
    if all_tasks_list:
        print(f"Matrix dimensions: {all_tasks_matrix.shape}")
    else:
        print("No valid data to save")


if __name__ == "__main__":
    convert_json_to_mat(json_dir="ppo_logs", output_file="ppo_results.mat")
    convert_json_to_mat(json_dir="a2c_logs", output_file="a2c_results.mat")
