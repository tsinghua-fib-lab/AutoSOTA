import os, sys

import copy
import torch
import numpy as np
import networkx as nx

from init_graph_4c import *
import fairPageRank_4c

import argparse

def main():
    
    logs_source_path = "/esat/augustijn1/scratch/ekariota/python_scripts/pagerank/logs/"

    parser = argparse.ArgumentParser(description='Run NX PR Iters')
    parser.add_argument('--dataset-name', type=str, default='karate', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--phi', type=float, default=0.0, metavar='R', help='Target Fairness Level')
    parser.add_argument('--protected-class', type=float, default=0.0, metavar='R', help='Protected Class Label')
    parser.add_argument('--gamma', type=float, default=0.15, metavar='R', help='Gamma value')
    parser.add_argument('--max-iters', type=int, default=200, metavar='R', help='Maximum Number of Iterations')
    
    args = parser.parse_args()
    dataset_name = copy.deepcopy(args.dataset_name)
    phi = copy.deepcopy(args.phi)
    protected_class = copy.deepcopy(args.protected_class)
    gamma = copy.deepcopy(args.gamma)
    max_iters = copy.deepcopy(args.max_iters)

    print()
    print("------------------------------")
    print("Dataset Name:", dataset_name)

    source_path = "datasets/"
    G, protected_nodes_0, protected_nodes_1, protected_nodes_2, protected_nodes_3 = init_graph_4c(dataset_name, source_path)
    n0, n1, n2, n3 = len(protected_nodes_0), len(protected_nodes_1), len(protected_nodes_2), len(protected_nodes_3)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print('Number of Nodes:', n)
    print('Number of Edges:', m)
    print('Number of Nodes Per Group:' + str(n0) +'-'+ str(n1) +'-'+ str(n2) +'-'+ str(n3))

    # Create Protected and Un-Protected Set Vectors
    S_0 = torch.zeros(n).int()
    S_0[protected_nodes_0] = 1
    S_1 = torch.zeros(n).int()
    S_1[protected_nodes_1] = 1
    S_2 = torch.zeros(n).int()
    S_2[protected_nodes_2] = 1
    S_3 = torch.zeros(n).int()
    S_3[protected_nodes_3] = 1
        
    if phi==0:
        phi_ = [n0/n, n1/n, n2/n, n3/n]
    else:
        phi_ = [phi, (1-phi)/3, (1-phi)/3, (1-phi)/3]

    opr = nx.pagerank(G)
    opr_scores = torch.FloatTensor(list(opr.values()))
    fair_nx_pr_ = fairPageRank_4c.sum_fair_FairRARI(G, S_0, S_1, S_2, S_3, phi_, alpha=1-gamma, max_iter=max_iters)
    
    fair_nx_pr = fair_nx_pr_[0]
    fair_nx_pr_scores = torch.FloatTensor(list(fair_nx_pr.values()))
    nx_x_diff = fair_nx_pr_[1]
    nx_loss = fair_nx_pr_[2]

    fair_pr_S_0 = fair_nx_pr_scores[S_0==1]
    fairness_0 = torch.sum(fair_pr_S_0)
    fair_pr_S_1 = fair_nx_pr_scores[S_1==1]
    fairness_1 = torch.sum(fair_pr_S_1)
    fair_pr_S_2 = fair_nx_pr_scores[S_2==1]
    fairness_2 = torch.sum(fair_pr_S_2)
    fair_pr_S_3 = fair_nx_pr_scores[S_3==1]
    fairness_3 = torch.sum(fair_pr_S_3)

    variables_dict = {
        'loss': nx_loss,
        'x_diff': nx_x_diff,
        'fairness_0': fairness_0,
        'fairness_1': fairness_1,
        'fairness_2': fairness_2,
        'fairness_3': fairness_3,
        'x_opt': fair_nx_pr_scores,
        'opr_scores': opr_scores
    }

    save_folder = logs_source_path+'sum_fair_FairRARI_4c/'+dataset_name
    save_path = save_folder+'/'+dataset_name+'_phi'+'{:.2f}'.format(phi)
    save_path = save_path+'_gamma'+str(gamma)
    save_path = save_path+'_iters'+str(max_iters)+'_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)


if __name__ == '__main__':
    main()