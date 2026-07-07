# Copyright 2023 Weiyifan <weiyifan@buaa.edu.cn>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-20.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import math
import itertools
import collections
import networkx as nx
import hashlib
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

device = "cuda" if torch.cuda.is_available() else "cpu"


class CommunityNode:
    def __init__(self, node_ids, parent=None):
        self.node_ids = node_ids
        self.parent = parent
        self.children = []
        self.volume = 0.0
        self.cut = 0.0
        self.se_term = 0.0
        # Use a unique ID for each community node for easier lookup
        self.ID = hashlib.md5(str(sorted(node_ids)).encode()).hexdigest()



class StructEntropy:
    def __init__(self, edges: torch.Tensor, weights: torch.Tensor, relations: list = None):
        self.edges = edges.to(device)
        self.weights = weights.to(device)
        self.num_nodes = int(self.edges.max() + 1) if self.edges.numel() > 0 else 0
        self.relations = relations
        self.node_to_comm = torch.arange(self.num_nodes, device=device)
        self.community_tree = None
        self.graph_node_to_leaf_map = {}  # Map graph nodes to the leaf nodes of coding tree
        self.vol = self._calc_graph_volume()
        self.degrees = self._get_degrees(self.edges, self.weights, self.num_nodes)
        self.node_se = None
        self.all_community_nodes = {}  # Store all community nodes by their ID

    def _calc_graph_volume(self):
        """
        Calculates the sum of all node degrees in the Graph G (2m in the theory).
        (Fix: Ensures the volume is 2 * sum of weights for undirected graph theory)
        """
        if self.weights.numel() == 0:
            return 0.0
        # Correctly return 2m
        return 2 * torch.sum(self.weights).item()

    def _get_degrees(self, edges, weights, num_nodes):
        """Calculates the degree of each node in the subgraph."""
        degrees = torch.zeros(num_nodes, dtype=weights.dtype, device=device)
        degrees.scatter_add_(0, edges[:, 0], weights)
        degrees.scatter_add_(0, edges[:, 1], weights)
        return degrees

    def _get_community_properties(self, edges, weights, node_to_comm, num_nodes, graph_vol):
        """
        NP-hard: Greedy merging is used to approximate the global optimal solution.
        Calculates the volume and cut for each community in a given partition.

        Args:
            edges (torch.Tensor): Edges of the subgraph.
            weights (torch.Tensor): Weights of the subgraph edges.
            node_to_comm (torch.Tensor): Mapping from node IDs to community IDs.
            num_nodes (int): Number of nodes in the subgraph.
            graph_vol (float): Total volume (sum of degrees) of the subgraph.

        Returns:
            tuple: A tuple containing community IDs, volumes, cuts, and SE terms.
        """
        comm_ids, node_to_comm_idx = torch.unique(node_to_comm, return_inverse=True)
        num_comms = comm_ids.size(0)

        degrees = self._get_degrees(edges, weights, num_nodes)

        comm_degrees = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        comm_degrees.scatter_add_(0, node_to_comm_idx, degrees)

        edge_comm_s = node_to_comm_idx[edges[:, 0]]
        edge_comm_t = node_to_comm_idx[edges[:, 1]]

        internal_edges_mask = edge_comm_s == edge_comm_t
        internal_edge_weights = weights[internal_edges_mask]
        internal_edge_comms = edge_comm_s[internal_edges_mask]

        in_comm_weights = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        if internal_edge_weights.numel() > 0:
            in_comm_weights.scatter_add_(0, internal_edge_comms, internal_edge_weights)

        volumes = comm_degrees
        cuts = volumes - 2 * in_comm_weights

        comm_node_se = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        valid_comms_mask = volumes > 0
        if torch.sum(valid_comms_mask) > 0:
            comm_vol_nonzero = volumes[valid_comms_mask]
            # SE term: -(cut / Vol(G)) * log2(Vol(comm) / Vol(G))
            comm_node_se_valid = - (cuts[valid_comms_mask] / graph_vol) * torch.log2(comm_vol_nonzero / graph_vol)
            comm_node_se[valid_comms_mask] = comm_node_se_valid

        leaf_node_se = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        comm_vol_by_node = volumes[node_to_comm_idx]
        valid_nodes_mask = (degrees > 0) & (comm_vol_by_node > 0)
        if torch.sum(valid_nodes_mask) > 0:
            # Leaf SE term: -(deg(i) / Vol(G)) * log2(deg(i) / Vol(comm(i)))
            leaf_node_se_contribs = - (degrees[valid_nodes_mask] / graph_vol) * torch.log2(
                degrees[valid_nodes_mask] / comm_vol_by_node[valid_nodes_mask])
            leaf_node_se.scatter_add_(0, node_to_comm_idx[valid_nodes_mask], leaf_node_se_contribs)

        return comm_ids, volumes, cuts, comm_node_se, leaf_node_se

    def _calc_delta_se_recursive(self, edges, weights, node_to_comm, num_nodes, graph_vol):
        """
        Calculates the change in structural entropy if two communities are merged.
        This is a core component of the greedy agglomerative clustering algorithm.
        """
        comm_ids, volumes, cuts, comm_node_se, leaf_node_se \
            = self._get_community_properties(edges, weights, node_to_comm, num_nodes, graph_vol)
        num_comms = comm_ids.size(0)
        if num_comms <= 1:
            return None, None

        comm_pairs = torch.combinations(torch.arange(num_comms, device=device), 2)
        idx1, idx2 = comm_pairs[:, 0], comm_pairs[:, 1]

        v1_vols, v2_vols = volumes[idx1], volumes[idx2]
        v1_cuts, v2_cuts = cuts[idx1], cuts[idx2]

        v1_comm_se, v2_comm_se = comm_node_se[idx1], comm_node_se[idx2]
        v1_leaf_se, v2_leaf_se = leaf_node_se[idx1], leaf_node_se[idx2]

        node_to_comm_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        comm_id_map = {id.item(): i for i, id in enumerate(comm_ids)}
        node_to_comm_idx = torch.tensor([comm_id_map[c.item()] for c in node_to_comm], device=device)

        edge_comm_s = node_to_comm_idx[edges[:, 0]]
        edge_comm_t = node_to_comm_idx[edges[:, 1]]

        comm_adj_matrix = torch.zeros((num_comms, num_comms), dtype=weights.dtype, device=device)
        comm_adj_matrix.index_put_((edge_comm_s, edge_comm_t), weights, accumulate=True)
        weights_between_comms = comm_adj_matrix[idx1, idx2] + comm_adj_matrix[idx2, idx1]

        original_se = v1_comm_se + v1_leaf_se + v2_comm_se + v2_leaf_se
        merged_comm_vols = v1_vols + v2_vols
        merged_comm_cuts = v1_cuts + v2_cuts - 2 * weights_between_comms

        vol_tensor = torch.tensor(graph_vol, device=device)
        merged_comm_vols_valid = merged_comm_vols > 0
        merged_comm_se_term = torch.zeros_like(merged_comm_vols)
        merged_comm_se_term[merged_comm_vols_valid] = - (
                merged_comm_cuts[merged_comm_vols_valid] / vol_tensor) * torch.log2(
            merged_comm_vols[merged_comm_vols_valid] / vol_tensor)

        merged_node_se_term = v1_leaf_se + v2_leaf_se

        v1_vols_valid = v1_vols > 0
        merged_node_se_term[v1_vols_valid] -= (v1_vols[v1_vols_valid] / vol_tensor) * torch.log2(
            v1_vols[v1_vols_valid] / merged_comm_vols[v1_vols_valid])

        v2_vols_valid = v2_vols > 0
        merged_node_se_term[v2_vols_valid] -= (v2_vols[v2_vols_valid] / vol_tensor) * torch.log2(
            v2_vols[v2_vols_valid] / merged_comm_vols[v2_vols_valid])

        merged_se = merged_comm_se_term + merged_node_se_term
        delta_SEs = merged_se - original_se
        return delta_SEs, comm_pairs

    def _get_community_node_map(self, root_node: CommunityNode) -> Dict[int, CommunityNode]:
        """
        辅助方法：遍历编码树，将 Community ID 映射到 CommunityNode 对象。
        这是 SeRAG coarse-grained matching 所需的。
        """
        node_map = {}
        nodes_to_visit = [root_node]
        
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            # 节点的 ID 是基于其包含的 node_ids 哈希生成的
            node_map[current_node.ID] = current_node
            nodes_to_visit.extend(current_node.children)
            
        return node_map
    
    def _build_node_to_community_map(self):
        """
        Builds a map from each original graph node ID to its corresponding
        leaf community node object in the tree.
        """
        self.graph_node_to_leaf_map = {}
        nodes_to_visit = [self.community_tree]
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if not current_node.children:
                if len(current_node.node_ids) == 1:
                    node_id = current_node.node_ids[0]
                    self.graph_node_to_leaf_map[node_id] = current_node
            else:
                for child in current_node.children:
                    nodes_to_visit.append(child)

    def calc_se_from_tree(self):
        if self.community_tree is None:
            print("Please run find_k_dim_entropy_tree() first.")
            return 0.0

        total_se = 0.0
        nodes_to_visit = [(self.community_tree, None)]

        while nodes_to_visit:
            current_node, parent_node = nodes_to_visit.pop(0)

            if parent_node is not None:
                if current_node.volume > 0 and parent_node.volume > 0:
                    se_term = -(current_node.cut / self.vol) * math.log2(current_node.volume / parent_node.volume)
                    total_se += se_term

            for child in current_node.children:
                nodes_to_visit.append((child, current_node))
        return total_se

    def calc_node_se_from_tree(self):
        if self.community_tree is None:
            print("Please run find_k_dim_entropy_tree() first.")
            return

        node_se_dict = {i: 0.0 for i in range(self.num_nodes)}
        nodes_to_visit = [(self.community_tree, None)]

        while nodes_to_visit:
            current_node, parent_node = nodes_to_visit.pop(0)
            if parent_node is not None:
                if current_node.volume > 0 and parent_node.volume > 0:
                    se_term = -(current_node.cut / self.vol) * math.log2(current_node.volume / parent_node.volume)
                    for node_id in current_node.node_ids:
                        node_se_dict[node_id] += se_term

            for child in current_node.children:
                nodes_to_visit.append((child, current_node))

        self.node_se = torch.tensor([node_se_dict[i] for i in range(self.num_nodes)], device=device)
        return self.node_se

    def _get_se_of_node_set(self, node_ids: list[int]) -> float:
        if self.node_se is None:
            self.calc_node_se_from_tree()

        se_sum = 0.0
        for node_id in node_ids:
            if node_id < self.num_nodes:
                se_sum += self.node_se[node_id].item()
        return se_sum





    def _find_common_ancestor_community_nodes(self, node_ids: list[int]) -> list[CommunityNode]:
        """
        Finds the highest-level community node that contains all given node_ids.
        Returns a list of such community nodes, ordered from top to bottom (root to leaf).
        """
        if not node_ids or not self.community_tree:
            return []

        # Start from the root and go down
        current_community = self.community_tree

        # Keep track of the path of community nodes that contain all x_set nodes
        common_ancestor_path = []

        nodes_to_check = set(node_ids)

        while True:
            # Check if all x_set nodes are in the current community
            if not nodes_to_check.issubset(set(current_community.node_ids)):
                # This community doesn't contain all nodes, so the previous one was the highest common ancestor
                break

            common_ancestor_path.append(current_community)

            found_next_level = False
            for child in current_community.children:
                if nodes_to_check.issubset(set(child.node_ids)):
                    current_community = child
                    found_next_level = True
                    break

            if not found_next_level:
                break

        return common_ancestor_path


    def find_k_dim_entropy_tree(self, k_dim: int = 2):
        if k_dim <= 1:
            self.community_tree = CommunityNode(list(range(self.num_nodes)))
            self.all_community_nodes[self.community_tree.ID] = self.community_tree
            self.graph_node_to_leaf_map = {node_id: self.community_tree for node_id in range(self.num_nodes)}
            return self.community_tree

        root_comm_node = CommunityNode(list(range(self.num_nodes)))
        self.community_tree = root_comm_node
        self.all_community_nodes[root_comm_node.ID] = root_comm_node
        current_level_comms = [root_comm_node]

        for current_level in range(1, k_dim + 1):
            next_level_comms = []
            print(f"Finding communities for dimension {current_level} (tree level {current_level})...")

            for parent_comm in current_level_comms:
                subgraph_nodes_orig_ids = torch.tensor(parent_comm.node_ids, device=device)
                if len(subgraph_nodes_orig_ids) <= 1:
                    child_comm_node = CommunityNode(parent_comm.node_ids, parent=parent_comm)
                    parent_comm.children.append(child_comm_node)
                    next_level_comms.append(child_comm_node)
                    self.all_community_nodes[child_comm_node.ID] = child_comm_node
                    if parent_comm.node_ids:
                        node_id = parent_comm.node_ids[0]
                        node_deg = self.degrees[node_id].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                node_map = {node_id.item(): i for i, node_id in enumerate(subgraph_nodes_orig_ids)}
                edges_mask = torch.isin(self.edges, subgraph_nodes_orig_ids).all(dim=1)
                sub_edges_orig_ids = self.edges[edges_mask]
                sub_weights = self.weights[edges_mask]

                if sub_edges_orig_ids.numel() == 0:
                    for node_id in subgraph_nodes_orig_ids:
                        child_comm_node = CommunityNode([node_id.item()], parent=parent_comm)
                        parent_comm.children.append(child_comm_node)
                        next_level_comms.append(child_comm_node)
                        self.all_community_nodes[child_comm_node.ID] = child_comm_node
                        node_deg = self.degrees[node_id.item()].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                sub_edges_relative_ids = torch.tensor(
                    [[node_map[u.item()], node_map[v.item()]] for u, v in sub_edges_orig_ids], dtype=torch.long,
                    device=device)
                sub_num_nodes = len(subgraph_nodes_orig_ids)
                sub_graph_vol = torch.sum(
                    self._get_degrees(sub_edges_relative_ids, sub_weights, sub_num_nodes)).item()
                if sub_graph_vol == 0:
                    for node_id in subgraph_nodes_orig_ids:
                        child_comm_node = CommunityNode([node_id.item()], parent=parent_comm)
                        parent_comm.children.append(child_comm_node)
                        next_level_comms.append(child_comm_node)
                        self.all_community_nodes[child_comm_node.ID] = child_comm_node
                        node_deg = self.degrees[node_id.item()].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                sub_node_to_comm = torch.arange(sub_num_nodes, device=device)

                while True:
                    sub_comm_ids, _, _, _, _ = self._get_community_properties(
                        sub_edges_relative_ids, sub_weights, sub_node_to_comm, sub_num_nodes, sub_graph_vol
                    )
                    delta_SEs, comm_pairs = self._calc_delta_se_recursive(
                        sub_edges_relative_ids, sub_weights, sub_node_to_comm, sub_num_nodes, sub_graph_vol
                    )
                    if delta_SEs is None:
                        break
                    min_delta_SE, min_idx = torch.min(delta_SEs, dim=0)
                    if min_delta_SE < 0:
                        best_comm_idx1, best_comm_idx2 = comm_pairs[min_idx]
                        best_comm1 = sub_comm_ids[best_comm_idx1].item()
                        best_comm2 = sub_comm_ids[best_comm_idx2].item()
                        mask_comm2 = sub_node_to_comm == best_comm2
                        sub_node_to_comm[mask_comm2] = best_comm1
                    else:
                        break

                sub_comm_ids_unique, _ = torch.unique(sub_node_to_comm, return_inverse=True)
                for sub_comm_id in sub_comm_ids_unique:
                    sub_comm_id = sub_comm_id.item()
                    member_relative_ids = (sub_node_to_comm == sub_comm_id).nonzero(
                        as_tuple=False).squeeze().cpu().tolist()
                    if not isinstance(member_relative_ids, list):
                        member_relative_ids = [member_relative_ids]

                    member_orig_ids = [subgraph_nodes_orig_ids[i].item() for i in member_relative_ids]
                    child_comm_node = CommunityNode(member_orig_ids, parent=parent_comm)
                    parent_comm.children.append(child_comm_node)
                    next_level_comms.append(child_comm_node)
                    self.all_community_nodes[child_comm_node.ID] = child_comm_node

                    subgraph_nodes_orig_ids_tensor = torch.tensor(member_orig_ids, device=device)
                    child_comm_node.volume = torch.sum(self.degrees[subgraph_nodes_orig_ids_tensor]).item()

                    edges_mask = torch.isin(self.edges, subgraph_nodes_orig_ids_tensor).all(dim=1)
                    internal_sum = torch.sum(self.weights[edges_mask]).item()
                    child_comm_node.cut = child_comm_node.volume - 2 * internal_sum

            if not next_level_comms:
                break
            current_level_comms = next_level_comms

        self._build_node_to_community_map()

        return self.community_tree

    def _get_networkx_graph(self):
        edges_cpu = self.edges.cpu().tolist()
        weights_cpu = self.weights.cpu().tolist()
        relations_cpu = self.relations
        G = nx.DiGraph()
        for i, ((u, v), w) in enumerate(zip(edges_cpu, weights_cpu)):
            relation = relations_cpu[i] if relations_cpu and i < len(relations_cpu) else None
            G.add_edge(u, v, weight=w, relation=relation)
        if hasattr(self, 'node_se') and self.node_se is not None:
            node_se_cpu = self.node_se.cpu().tolist()
            for i, se in enumerate(node_se_cpu):
                if i in G.nodes:
                    G.nodes[i]['se'] = se
        return G


    def calc_first_order_se(self) -> float:
        """
        Calculates the total first-order entropy H^1(G) of the graph G.
        H^1(G) = -sum_{i in V} p_i log_2 p_i, where p_i = deg(i) / (2m).
        """

        # Calculate p_i (normalized degrees)
        p_i = self.degrees / self.vol

        # Filter out nodes with zero degree (p_i=0) to avoid log(0)
        valid_mask = p_i > 0
        p_i_valid = p_i[valid_mask]

        # Calculate H^1(G) = -sum(p_i * log2(p_i))
        h1_G = -torch.sum(p_i_valid * torch.log2(p_i_valid))

        return h1_G.item()




