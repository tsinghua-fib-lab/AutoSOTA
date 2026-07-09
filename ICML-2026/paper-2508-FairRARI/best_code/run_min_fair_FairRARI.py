import os, sys

import copy
import torch
import numpy as np
import networkx as nx

from init_graph import *
import fairPageRank

import argparse

def main():
    
    logs_source_path = "/esat/augustijn1/scratch/ekariota/python_scripts/pagerank/logs/"

    parser = argparse.ArgumentParser(description='Run NX PR Iters')
    parser.add_argument('--dataset-name', type=str, default='karate', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--alph_', type=float, default=0.0, metavar='R', help='Target Fairness Level')
    parser.add_argument('--protected-class', type=float, default=0.0, metavar='R', help='Protected Class Label')
    parser.add_argument('--gamma', type=float, default=0.15, metavar='R', help='Gamma value')
    parser.add_argument('--max-iters', type=int, default=200, metavar='R', help='Maximum Number of Iterations')
    
    args = parser.parse_args()
    dataset_name = copy.deepcopy(args.dataset_name)
    alph_ = copy.deepcopy(args.alph_)
    protected_class = copy.deepcopy(args.protected_class)
    gamma = copy.deepcopy(args.gamma)
    max_iters = copy.deepcopy(args.max_iters)

    print()
    print("------------------------------")
    print("Dataset Name:", dataset_name)

    source_path = "datasets/"
    G, protected_nodes, blue_nodes, red_nodes = init_graph(dataset_name, source_path)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print('Number of Nodes:', n)
    print('Number of Edges:', m)
    print('Number of Protected Nodes:', len(protected_nodes))

    # Create Protected and Un-Protected Set Vectors
    S_p = torch.zeros(n).int()
    S_p[protected_nodes] = 1
    S_up = torch.ones(n).int()
    S_up[protected_nodes] = 0

    opr = nx.pagerank(G)
    opr_scores = torch.FloatTensor(list(opr.values()))
    opr_scores_S_p = opr_scores[S_p==1]
    phi_opr = torch.sum(opr_scores_S_p)

    if alph_==0:
        alph_ = float(torch.min(opr_scores_S_p))
    else:
        alph_ = alph_/sum(S_p).item()

    fair_nx_pr_ = fairPageRank.min_fair_FairRARI(G, S_p, S_up, alph_, alpha=1-gamma, max_iter=max_iters)
    
    fair_nx_pr = fair_nx_pr_[0]
    fair_nx_pr_scores = torch.FloatTensor(list(fair_nx_pr.values()))
    nx_x_diff = fair_nx_pr_[1]
    nx_loss = fair_nx_pr_[2]

    fair_pr_S_p = fair_nx_pr_scores[S_p==1]
    fairness = torch.sum(fair_pr_S_p)

    variables_dict = {
        'loss': nx_loss,
        'x_diff': nx_x_diff,
        'fairness': fairness,
        'x_opt': fair_nx_pr_scores,
        'opr_scores': opr_scores
    }

    save_folder = logs_source_path+'min_fair_FairRARI/'+dataset_name
    save_path = save_folder+'/'+dataset_name+'_alpha'+'{:.8f}'.format(alph_)
    save_path = save_path+'_gamma'+str(gamma)
    save_path = save_path+'_iters'+str(max_iters)+'_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)


if __name__ == '__main__':
    main()