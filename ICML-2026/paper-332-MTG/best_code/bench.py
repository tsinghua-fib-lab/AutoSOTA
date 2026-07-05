import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from prompt_graph.utils import get_args
from prompt_graph.data import load4node, load4graph, split_induced_graphs

from prompt_graph.utils.report_data import ConfigBenchResult


def load_induced_graph(dataset_name, data, device):
    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print('loading induced graph...')
            graphs_list = pickle.load(f)
            print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


"""
Auto bench function. Using predefined param grid to search best results
for 1 pretrained model.
You need to provide at least 3 arguments
pretrain_task,
dataset_name,
prompt_type
"""


def do_config_bench(args: argparse.Namespace):
    seed_everything(args.seed)

    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64, 128],
    }

    num_iter = 10
    print('args.dataset_name', args.dataset_name)

    # Define special num_iter cases
    if args.prompt_type in ['MultiGprompt', 'GPPT']:
        print('num_iter = 1')
        num_iter = 1
    if args.dataset_name in ['ogbn-arxiv', 'Flickr']:
        print('num_iter = 1')
        num_iter = 1
    best_params = {}
    best_loss = float('inf')
    final_acc_mean = 0
    final_acc_std = 0
    final_f1_mean = 0
    final_f1_std = 0
    final_roc_mean = 0
    final_roc_std = 0

    if args.downstream_task == 'NodeTask':
        data, input_dim, output_dim = load4node(args.dataset_name)
        data = data.to(args.device)
        if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'MTG']:
            graphs_list = load_induced_graph(args.dataset_name, data, args.device)
        else:
            graphs_list = None

    if args.downstream_task == 'GraphTask':
        input_dim, output_dim, dataset = load4graph(args.dataset_name)

    print('num_iter', num_iter)
    for a in range(num_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(params)

        if args.downstream_task == 'NodeTask':
            tasker = NodeTask(pre_train_model_path=args.pre_train_model_path,
                              dataset_name=args.dataset_name, num_layer=args.num_layer,
                              gnn_type=args.gnn_type, hid_dim=args.hid_dim, prompt_type=args.prompt_type,
                              epochs=args.epochs, shot_num=args.shot_num, device=args.device,
                              lr=params['learning_rate'], wd=params['weight_decay'],
                              batch_size=int(params['batch_size']), data=data, input_dim=input_dim,
                              output_dim=output_dim, graphs_list=graphs_list)


        elif args.downstream_task == 'GraphTask':
            tasker = GraphTask(pre_train_model_path=args.pre_train_model_path,
                               dataset_name=args.dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type,
                               hid_dim=args.hid_dim, prompt_type=args.prompt_type, epochs=args.epochs,
                               shot_num=args.shot_num, device=args.device, lr=params['learning_rate'],
                               wd=params['weight_decay'],
                               batch_size=int(params['batch_size']), dataset=dataset, input_dim=input_dim,
                               output_dim=output_dim)
        else:
            raise ValueError(f"Unexpected pretrain_task: {args.pretrain_task}.")
        pre_train_type = tasker.pre_train_type

        # Return the average loss
        avg_best_loss, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc = tasker.run()

        # Convert each metric to Python float
        avg_best_loss = float(avg_best_loss)
        mean_test_acc = float(mean_test_acc)
        std_test_acc = float(std_test_acc)
        mean_f1 = float(mean_f1)
        std_f1 = float(std_f1)
        mean_roc = float(mean_roc)
        std_roc = float(std_roc)
        mean_prc = float(mean_prc)
        std_prc = float(std_prc)
        print(f"For {a}th searching, Tested Params: {params}, Avg Best Loss: {avg_best_loss}")

        if avg_best_loss < best_loss:
            best_loss = avg_best_loss
            best_params = params
            final_acc_mean = mean_test_acc
            final_acc_std = std_test_acc
            final_f1_mean = mean_f1
            final_f1_std = std_f1
            final_roc_mean = mean_roc
            final_roc_std = std_roc

    if isinstance(best_params, dict):
        best_params = {k: float(v) for k, v in best_params.items()}
    return ConfigBenchResult(
        pretrain_task_type=args.pretrain_task,
        dataset_name=args.dataset_name,
        prompt_type=args.prompt_type,
        best_params=best_params,
        best_loss=best_loss,
        final_acc_mean=final_acc_mean,
        final_acc_std=final_acc_std,
        final_f1_mean=final_f1_mean,
        final_f1_std=final_f1_std,
        final_roc_mean=final_roc_mean,
        final_roc_std=final_roc_std,
    )


# pre_train_types = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
# prompt_types = ['None', 'GPPT', 'All-in-one', 'Gprompt', 'GPF', 'GPF-plus', 'MTG']
if __name__ == "__main__":
    args = get_args()

    cbr_result = do_config_bench(args=args)

    file_name = args.gnn_type + "_total_results.xlsx"
    if args.downstream_task == 'NodeTask':
        file_path = os.path.join(
            './Experiment/ExcelResults/Node/' + str(args.shot_num) + 'shot/' + args.dataset_name + '/', file_name)
    if args.downstream_task == 'GraphTask':
        file_path = os.path.join(
            './Experiment/ExcelResults/Graph/' + str(args.shot_num) + 'shot/' + args.dataset_name + '/', file_name)
    data = pd.read_excel(file_path, index_col=0)

    col_name = f"{cbr_result.pretrain_task_type}+{args.prompt_type}"
    print('col_name', col_name)
    data.at['Final Accuracy', col_name] = f"{cbr_result.final_acc_mean:.4f}±{cbr_result.final_acc_std:.4f}"
    data.at['Final F1', col_name] = f"{cbr_result.final_f1_mean:.4f}±{cbr_result.final_f1_std:.4f}"
    data.at['Final AUROC', col_name] = f"{cbr_result.final_roc_mean:.4f}±{cbr_result.final_roc_std:.4f}"
    data.to_excel(file_path)

    print("Data saved to " + file_path + " successfully.")

    print("After searching, Final Accuracy {:.4f}±{:.4f}(std)".format(cbr_result.final_acc_mean,
                                                                      cbr_result.final_acc_std))
    print("After searching, Final F1 {:.4f}±{:.4f}(std)".format(cbr_result.final_f1_mean, cbr_result.final_f1_std))
    print("After searching, Final AUROC {:.4f}±{:.4f}(std)".format(cbr_result.final_roc_mean, cbr_result.final_roc_std))
    print('best_params ', cbr_result.best_params)
    print('best_loss ', cbr_result.best_loss)
