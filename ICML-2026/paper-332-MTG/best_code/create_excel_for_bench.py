import os
import pandas as pd

graph_dataset_name = ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'COX2', 'BZR', 'DD']
node_dataset_name = ['PubMed', 'CiteSeer', 'Cora', 'Wisconsin', 'Texas', 'ogbn-arxiv', 'Actor']
shot_nums = [1, 3, 5]
for dataset_name in node_dataset_name:
    for shot_num in shot_nums:
        pre_train_types = ['None','DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
        prompt_types = ['None', 'GPPT', 'Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'MTG']

        column_names = [f"{pre_train}+{prompt}" for prompt in prompt_types for pre_train in pre_train_types if
                        pre_train != 'None' or prompt == 'None']

        data = pd.DataFrame(columns=column_names, index=['Final Accuracy', 'Final F1', 'Final AUROC'])

        gnn_type = 'GCN'

        file_name = gnn_type + "_total_results.xlsx"
        file_path = os.path.join('./Experiment/ExcelResults/Node/' + str(shot_num) + 'shot/' + dataset_name + '/',
                                 file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_excel(file_path)

        print(f"Data saved to {file_path} successfully.")

for dataset_name in graph_dataset_name:
    for shot_num in shot_nums:
        pre_train_types = ['None','DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
        prompt_types = ['None', 'GPPT', 'Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'MTG']

        column_names = [f"{pre_train}+{prompt}" for prompt in prompt_types for pre_train in pre_train_types if
                        pre_train != 'None' or prompt == 'None']

        data = pd.DataFrame(columns=column_names, index=['Final Accuracy', 'Final F1', 'Final AUROC'])

        gnn_type = 'GCN'

        file_name = gnn_type + "_total_results.xlsx"
        file_path = os.path.join('./Experiment/ExcelResults/Graph/' + str(shot_num) + 'shot/' + dataset_name + '/',
                                 file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_excel(file_path)

        print(f"Data saved to {file_path} successfully.")
