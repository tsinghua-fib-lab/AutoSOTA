import sys; sys.path.insert(0, "/repo")
import numpy as np
import networkx as nx
import torch
from init_graph import init_graph
import fairPageRank

G, protected_nodes, blue_nodes, red_nodes = init_graph("polbooks", "datasets/")
n = G.number_of_nodes()
S_p = torch.zeros(n).int()
S_p[protected_nodes] = 1
S_up = torch.ones(n).int()
S_up[protected_nodes] = 0

opr = nx.pagerank(G)
opr_scores = torch.FloatTensor(list(opr.values()))

# Post-processing with uniform teleportation
pp_pr, pp_err, pp_loss = fairPageRank.sum_fair_post_processing(
    G, S_p, S_up, 0.8, alpha=0.85, max_iter=10000
)
pp_scores = torch.FloatTensor(list(pp_pr.values()))
pp_tv = 0.5 * np.sum(np.abs(opr_scores.numpy() - pp_scores.numpy()))
pp_fair = torch.sum(pp_scores[S_p == 1]).item()

# Post-processing with personalized teleportation
opr_dict = dict(zip(G.nodes(), opr_scores.numpy()))
pp_pr2, pp_err2, pp_loss2 = fairPageRank.sum_fair_post_processing(
    G, S_p, S_up, 0.8, alpha=0.85, max_iter=10000, personalization=opr_dict
)
pp_scores2 = torch.FloatTensor(list(pp_pr2.values()))
pp_tv2 = 0.5 * np.sum(np.abs(opr_scores.numpy() - pp_scores2.numpy()))
pp_fair2 = torch.sum(pp_scores2[S_p == 1]).item()

print(f"Post-processing (uniform p):     TV={pp_tv:.6f}, AchievedFairness={pp_fair:.6f}")
print(f"Post-processing (personalized p): TV={pp_tv2:.6f}, AchievedFairness={pp_fair2:.6f}")
print(f"FairRARI (personalized p):        TV=0.328995, AchievedFairness=0.799979")
