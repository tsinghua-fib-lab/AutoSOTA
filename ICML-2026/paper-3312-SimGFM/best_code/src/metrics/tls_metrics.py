# import copy
# from collections import Counter
# from typing import List

import networkx as nx
# import numpy as np
# import scipy.sparse as sp
# import wandb
# import torch
# from torch import Tensor
# from torchmetrics import MeanMetric

# from datasets.tls_dataset import CellGraph
# from analysis.spectre_utils import PlanarSamplingMetrics
# from analysis.spectre_utils import is_planar_graph


# class TLSSamplingMetrics(PlanarSamplingMetrics):
#     def __init__(self, datamodule):
#         super().__init__(datamodule)
#         self.train_cell_graphs = self.loader_to_cell_graphs(
#             datamodule.train_dataloader()
#         )
#         self.val_cell_graphs = self.loader_to_cell_graphs(datamodule.val_dataloader())
#         self.test_cell_graphs = self.loader_to_cell_graphs(datamodule.test_dataloader())

#     def loader_to_cell_graphs(self, loader):
#         cell_graphs = []
#         for batch in loader:
#             for tg_graph in batch.to_data_list():
#                 cell_graph = CellGraph.from_torch_geometric(tg_graph)
#                 cell_graphs.append(cell_graph)

#         return cell_graphs

#     def is_cell_graph_valid(self, cg: CellGraph):
#         # connected and planar
#         return is_planar_graph(cg)

#     def forward(
#         self,
#         generated_graphs: list,
#         ref_metrics,
#         name,
#         current_epoch,
#         val_counter,
#         local_rank,
#         test=False,
#         labels=None,
#         log_dict : dict =None,
#     ):

#         # Unattributed graphs specific
#         if log_dict is None:
#             prefix = ""
#         elif test:
#             prefix = "test"
#         else:
#             prefix = "val"
#         super().forward(
#             generated_graphs,
#             ref_metrics,
#             name,
#             current_epoch,
#             val_counter,
#             local_rank,
#             test,
#             labels,
#             log_dict,
#         )

#         # Cell graph specific
#         reference_cgs = self.test_cell_graphs if test else self.val_cell_graphs
#         generated_cgs = []
#         if local_rank == 0:
#             print("Building generated cell graphs...")
#         for graph in generated_graphs:
#             generated_cgs.append(CellGraph.from_dense_graph(graph))

#         # TODO: Implement these metrics with torchmetrics for parallelization
#         generated_labels = torch.tensor([cg.to_label() for cg in generated_cgs])
#         ambiguous_gen_cgs = sum(
#             [(cg_label == -1).int() for cg_label in generated_labels]
#         ).item()
#         if labels is not None:
#             true_labels = torch.tensor(labels)
#             high_tls_idxs = true_labels == 1
#             low_tls_idxs = true_labels == 0
#             total_tls_acc = (generated_labels == true_labels).float().mean().item()
#             high_tls_acc = (
#                 (generated_labels[high_tls_idxs] == true_labels[high_tls_idxs])
#                 .float()
#                 .mean()
#                 .item()
#             )
#             low_tls_acc = (
#                 (generated_labels[low_tls_idxs] == true_labels[low_tls_idxs])
#                 .float()
#                 .mean()
#                 .item()
#             )
#         else:
#             total_tls_acc = -1
#             high_tls_acc = -1
#             low_tls_acc = -1

#         # Compute novelty and uniqueness
#         if local_rank == 0:
#             print("Computing uniqueness, novelty and validity for cell graphs...")
#             frac_novel = eval_fraction_novel_cell_graphs(
#                 generated_cell_graphs=generated_cgs,
#                 train_cell_graphs=self.train_cell_graphs,
#             )
#             (
#                 frac_unique,
#                 frac_unique_and_novel,
#                 frac_unique_and_novel_valid,
#             ) = eval_fraction_unique_novel_valid_cell_graphs(
#                 generated_cell_graphs=generated_cgs,
#                 train_cell_graphs=self.train_cell_graphs,
#                 valid_cg_fn=self.is_cell_graph_valid,
#             )

#         tls_to_log = {
#             f"{prefix}/tls_metrics/total_tls_acc": total_tls_acc,
#             f"{prefix}/tls_metrics/high_tls_acc": high_tls_acc,
#             f"{prefix}/tls_metrics/low_tls_acc": low_tls_acc,
#             f"{prefix}/tls_metrics/num_ambiguous_tls": ambiguous_gen_cgs,
#             f"{prefix}/tls_metrics/frac_novel": frac_novel,
#             f"{prefix}/tls_metrics/frac_unique": frac_unique,
#             f"{prefix}/tls_metrics/frac_unique_and_novel": frac_unique_and_novel,
#             f"{prefix}/tls_metrics/frac_unique_and_novel_valid": frac_unique_and_novel_valid,
#         }

#         print(f"TLS sampling metrics: {tls_to_log}")
#         # if wandb.run:
#         #     # only log TLS sampling metrics because others are already logged by planar sampling metrics
#         #     wandb.log(tls_to_log, commit=False)

#         if log_dict is not None:
#             log_dict.update(tls_to_log)

#         return tls_to_log


# specific for cell graphs (isomorphism function is of cell graphs)
def eval_fraction_novel_cell_graphs(generated_cell_graphs, train_cell_graphs):
    count_non_novel = 0
    for gen_cg in generated_cell_graphs:
        for train_cg in train_cell_graphs:
            if nx.faster_could_be_isomorphic(train_cg, gen_cg):
                if gen_cg.is_isomorphic(train_cg):
                    count_non_novel += 1
                    break
    return 1 - count_non_novel / len(generated_cell_graphs)


# specific for cell graphs (isomorphism function is of cell graphs)
def eval_fraction_unique_novel_valid_cell_graphs(
    generated_cell_graphs,
    train_cell_graphs,
    valid_cg_fn,
):
    count_non_unique = 0
    count_not_novel = 0
    count_not_valid = 0
    for cg_idx, gen_cg in enumerate(generated_cell_graphs):
        is_unique = True
        for gen_cg_seen in generated_cell_graphs[:cg_idx]:
            # test =gen_cg.is_isomorphic(gen_cg_seen)
            # print(test)
            # breakpoint()
            if nx.faster_could_be_isomorphic(gen_cg_seen, gen_cg):
                # we also need to consider phenotypes of nodes
                if gen_cg.is_isomorphic(gen_cg_seen):
                    count_non_unique += 1
                    is_unique = False
                    break
        if is_unique:
            is_novel = True
            for train_cg in train_cell_graphs:
                if nx.faster_could_be_isomorphic(train_cg, gen_cg):
                    if gen_cg.is_isomorphic(train_cg):
                        count_not_novel += 1
                        is_novel = False
                        break
            if is_novel:
                if not valid_cg_fn(gen_cg):
                    count_not_valid += 1

    frac_unique = 1 - count_non_unique / len(generated_cell_graphs)
    frac_unique_non_isomorphic = frac_unique - count_not_novel / len(
        generated_cell_graphs
    )
    frac_unique_non_isomorphic_valid = (
        frac_unique_non_isomorphic - count_not_valid / len(generated_cell_graphs)
    )

    return (
        frac_unique,
        frac_unique_non_isomorphic,
        frac_unique_non_isomorphic_valid,
    )
