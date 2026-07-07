import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def _safe_normalize_tensor(x: torch.Tensor):
    mn = x.min()
    mx = x.max()
    denom = (mx - mn).item()
    if denom <= 1e-12:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-12)

class CrossDomainMixup:
    def __init__(self, num_domains=3, alpha=0.2, hop=2, boundary_ratio=0.3, 
                 similarity_threshold=0.7, device=None, seed=39):
        self.alpha = alpha
        self.hop = hop
        self.num_domains = num_domains
        self.boundary_ratio = boundary_ratio
        self.similarity_threshold = similarity_threshold
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        # set random seeds
        set_seed(seed)
        
        # cache
        self.subgraphs_cache = {}
        self.domain_centers = {}
        self.boundary_nodes = {}
        self.center_nodes = {}
        
    def compute_domain_centers(self, domain_graphs: Dict[int, Data]):
        print("Computing domain centers...")
        self.domain_centers = {}
        
        for domain_id, graph in domain_graphs.items():
            if graph.x is not None and len(graph.x) > 0:

                domain_center = graph.x.mean(dim=0)
                self.domain_centers[domain_id] = domain_center
                print(f"Domain {domain_id} center computed: {domain_center.shape}")
    
    def identify_boundary_nodes(self, domain_graphs: Dict[int, Data]):
        print("Identifying boundary nodes using multi-domain consensus...")
        self.boundary_nodes = {}
        self.center_nodes = {}
        
        domain_ids = list(domain_graphs.keys())
        
        #compute the boundary nodes for each domain relative to all other domains
        domain_boundary_candidates = {}
        
        for domain_id, graph in domain_graphs.items():
            if graph.x is None or len(graph.x) == 0:
                continue
                
            features = graph.x
            all_centers = torch.stack([self.domain_centers[did] for did in domain_ids])

            
            # Calculate the distance to all domain centers
            distances = torch.cdist(features.unsqueeze(0), all_centers.unsqueeze(0)).squeeze(0)
            
            # Multi-domain Boundary Consensus
            boundary_candidates_per_domain = []
            
            # For the current domain, compute the boundary nodes with each of the other domains
            current_domain_idx = domain_ids.index(domain_id)
            
            for other_domain_idx in range(len(domain_ids)):
                if other_domain_idx == current_domain_idx:
                    continue
                    
                # Calculate the boundary score between the current domain and another specified domain.
                domain_pair_distances = distances[:, [current_domain_idx, other_domain_idx]]
             
                sorted_pair_dists, _ = domain_pair_distances.sort(dim=1)
                margin_scores = sorted_pair_dists[:, 1] - sorted_pair_dists[:, 0]

                
                # Min-Max Normalization
                min_margin = margin_scores.min()
                max_margin = margin_scores.max()
                if max_margin > min_margin:
                    normalized_margins = (margin_scores - min_margin) / (max_margin - min_margin)
                else:
                    normalized_margins = torch.zeros_like(margin_scores)
          
                confidence_scores = 1.0 - normalized_margins  # The smaller the boundary score, the higher the confidence level.

                # Select the boundary node between these two domains.
                num_boundary = max(1, int(len(features) * self.boundary_ratio))

                _, boundary_idx = torch.topk(confidence_scores, num_boundary, largest=True)
                boundary_candidates_per_domain.append(set(boundary_idx.tolist()))
            
            # The intersection of the current domain and the boundary nodes of all other domains as the consensus boundary nodes.
            if boundary_candidates_per_domain:
                consensus_boundary = set.intersection(*boundary_candidates_per_domain)
                # If the intersection is empty, it degenerates into the union.
                if not consensus_boundary:
                    consensus_boundary = set.union(*boundary_candidates_per_domain)
            else:
                consensus_boundary = set()
            
            domain_boundary_candidates[domain_id] = consensus_boundary

            # domain_center = self.domain_centers[domain_id]
            # dist_to_own_center = F.pairwise_distance(features, domain_center.unsqueeze(0))
            # num_center = max(1, int(len(features) * (1 - self.boundary_ratio)))
            # _, center_indices = torch.topk(dist_to_own_center, num_center, largest=False)
            
            # self.center_nodes[domain_id] = center_indices
        for domain_id, graph in domain_graphs.items():

            candidate_indices = list(domain_boundary_candidates[domain_id])
            candidate_tensor = torch.tensor(candidate_indices)
            self.boundary_nodes[domain_id] = candidate_tensor


            
    def generate_positive_pairs(self, domain_graphs: Dict[int, Data], 
                              nodes_per_domain: int = 50,
                              pairs_per_domain: int = 20) -> List[Data]:
        """Generate same-domain positive sample pairs"""

        set_seed(self.seed)
        
        positive_pairs = []
        
        for domain_id, graph in domain_graphs.items():
            if not self._validate_graph_data(graph, domain_id) or graph.num_nodes < 2:
                continue
                
            # sample center node
            if graph.num_nodes < nodes_per_domain:
                center_nodes = torch.arange(graph.num_nodes)
            else:
                center_nodes = torch.randperm(graph.num_nodes)[:nodes_per_domain]
            
            print(f"Domain {domain_id}: generating positive pairs from {len(center_nodes)} center nodes")
            
            # same domain pair
            for i in range(0, len(center_nodes), 2):
                if i + 1 >= len(center_nodes):
                    break
                    
                node_i = center_nodes[i]
                node_j = center_nodes[i + 1]
                
                try:
                    # sample subgraph
                    subgraph_i = self.get_k_hop_subgraph(graph, node_i, domain_id)
                    subgraph_j = self.get_k_hop_subgraph(graph, node_j, domain_id)
                    
                    pos_pair = self._mix_subgraphs_safe_positive(subgraph_i, subgraph_j, graph, graph)
                    if pos_pair is not None:
                        positive_pairs.append(pos_pair)
                        if len(positive_pairs) >= pairs_per_domain:
                            break
                            
                except Exception as e:
                    print(f"Error generating positive pair for nodes ({node_i}, {node_j}): {e}")
                    continue
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs

    def generate_negative_pairs(self, domain_graphs: Dict[int, Data], 
                              num_samples: int = 50) -> List[Data]:
        """Generate cross-domain negative sample pairs"""
        set_seed(self.seed)
        
        # Initialize domain centers and boundary nodes
        if not self.domain_centers:
            self.compute_domain_centers(domain_graphs)
        if not self.boundary_nodes:
            self.identify_boundary_nodes(domain_graphs)

        
        print(f"Starting to generate {num_samples} cross-domain negative mixtures...")
        
        mixtures = []
        domain_ids = list(domain_graphs.keys())
        
        # Phase 1: Select node pairs with high similarity from boundary nodes for blending.
        high_similarity_mixtures = self._generate_high_similarity_mixtures(
            domain_graphs, domain_ids, num_samples
        )
        mixtures.extend(high_similarity_mixtures)
        
        # Phase Two: If the quantity is insufficient, employ the contingency strategy to make up the shortfall.
        if len(mixtures) < num_samples:
            remaining_samples = num_samples - len(mixtures)
            fallback_mixtures = self._generate_fallback_mixtures(
                domain_graphs, domain_ids, remaining_samples
            )
            mixtures.extend(fallback_mixtures)
            print(f"Added {len(fallback_mixtures)} fallback mixtures")
        
        print(f"Successfully generated {len(mixtures)} cross-domain negative mixtures")
        return mixtures

    def _mix_subgraphs_safe_positive(self, subgraph_i: Dict, subgraph_j: Dict, 
                                graph_i: Data, graph_j: Data) -> Optional[Data]:
        """merge two subgraphs within the same domain"""
        try:
            nodes_i, edges_i, x_i, center_idx_i = (
                subgraph_i['nodes'], subgraph_i['edge_index'], 
                subgraph_i['x'], subgraph_i['center_node_idx']
            )
            nodes_j, edges_j, x_j, center_idx_j = (
                subgraph_j['nodes'], subgraph_j['edge_index'],
                subgraph_j['x'], subgraph_j['center_node_idx']
            )
            
            # Verification Center Node Index
            if center_idx_i >= len(x_i) or center_idx_j >= len(x_j):
                return None
            
            # fixed lambda=0.5
            lam = 0.5
            
            num_nodes_i = len(x_i)
            num_nodes_j = len(x_j) - 1  
            
            # Initialize the mixed feature
            mixed_x = torch.zeros((num_nodes_i + num_nodes_j, x_i.shape[1]), dtype=x_i.dtype)
            
            # Copy all node features from the first subgraph
            mixed_x[:num_nodes_i] = x_i.clone()
            
            # mixup center node features
            mixed_x[center_idx_i] = lam * x_i[center_idx_i] + (1 - lam) * x_j[center_idx_j]
            
            # Copy the features of other nodes in the second subgraph (excluding the central node).
            if num_nodes_j > 0:
                other_nodes_j = [i for i in range(len(x_j)) if i != center_idx_j]
                mixed_x[num_nodes_i:num_nodes_i + num_nodes_j] = x_j[other_nodes_j]
            
            # Remap Edge Index
            mixed_edges_list = []
            
            #  the edges of the first subgraph
            if edges_i.numel() > 0:
                mixed_edges_list.append(edges_i)
            
            #  the edges of the second subgraph
            if edges_j.numel() > 0:
                node_mapping_j = {}
                current_idx = num_nodes_i
                
                for old_idx in range(len(x_j)):
                    if old_idx == center_idx_j:
                        node_mapping_j[old_idx] = center_idx_i
                    else:
                        node_mapping_j[old_idx] = current_idx
                        current_idx += 1
                
                edges_j_remapped = edges_j.clone()
                for old_idx, new_idx in node_mapping_j.items():
                    mask = (edges_j_remapped == old_idx)
                    edges_j_remapped[mask] = new_idx
                
                mixed_edges_list.append(edges_j_remapped)
            
            # merge all edges
            if mixed_edges_list:
                mixed_edges = torch.cat(mixed_edges_list, dim=1)
            else:
                mixed_edges = torch.empty((2, 0), dtype=torch.long)
            
            if not self._validate_mixed_graph(mixed_x, mixed_edges):
                return None
            
            # Create a one-hot probability vector composed of domains
            domain_id = subgraph_i['domain_id']
            composition_prob = torch.zeros(self.num_domains, dtype=torch.float32)
            composition_prob[domain_id] = 1.0
            
            # mix graph data
            mixed_graph = Data(
                x=mixed_x,
                edge_index=mixed_edges,
                domain_list=None,
                domain_i=torch.tensor([domain_id], dtype=torch.long),
                domain_j=torch.tensor([domain_id], dtype=torch.long),
                center_node_i=torch.tensor([subgraph_i['center_node_orig']], dtype=torch.long),
                center_node_j=torch.tensor([subgraph_j['center_node_orig']], dtype=torch.long),
                lam=torch.tensor([lam], dtype=torch.float32),
                y=composition_prob,
                center_id=center_idx_i,
                is_cross=torch.tensor([0], dtype=torch.long)  # 正样本标识
            )
            
            return mixed_graph
            
        except Exception as e:
            print(f"Error mixing positive subgraphs: {e}")
            return None

    def _mix_subgraphs_cross_domain(self, subgraph_i: Dict, subgraph_j: Dict, 
                                  graph_i: Data, graph_j: Data, lam: float = 0.5) -> Optional[Data]:
        """Mixup two cross-domain subgraphs"""
        try:
            nodes_i, edges_i, x_i, center_idx_i = (
                subgraph_i['nodes'], subgraph_i['edge_index'], 
                subgraph_i['x'], subgraph_i['center_node_idx']
            )
            nodes_j, edges_j, x_j, center_idx_j = (
                subgraph_j['nodes'], subgraph_j['edge_index'],
                subgraph_j['x'], subgraph_j['center_node_idx']
            )
            
            if center_idx_i >= len(x_i) or center_idx_j >= len(x_j):
                return None
            
            num_nodes_i = len(x_i)
            num_nodes_j = len(x_j) - 1  
            
            
            mixed_x = torch.zeros((num_nodes_i + num_nodes_j, x_i.shape[1]), dtype=x_i.dtype)
            
            
            mixed_x[:num_nodes_i] = x_i.clone()
            
            
            mixed_x[center_idx_i] = lam * x_i[center_idx_i] + (1 - lam) * x_j[center_idx_j]
            
            if num_nodes_j > 0:
                other_nodes_j = [i for i in range(len(x_j)) if i != center_idx_j]
                mixed_x[num_nodes_i:num_nodes_i + num_nodes_j] = x_j[other_nodes_j]
            
            
            mixed_edges_list = []
            
            
            if edges_i.numel() > 0:
                mixed_edges_list.append(edges_i)
            
            
            if edges_j.numel() > 0:
                node_mapping_j = {}
                current_idx = num_nodes_i
                
                for old_idx in range(len(x_j)):
                    if old_idx == center_idx_j:
                        node_mapping_j[old_idx] = center_idx_i
                    else:
                        node_mapping_j[old_idx] = current_idx
                        current_idx += 1
                
                edges_j_remapped = edges_j.clone()
                for old_idx, new_idx in node_mapping_j.items():
                    mask = (edges_j_remapped == old_idx)
                    edges_j_remapped[mask] = new_idx
                
                mixed_edges_list.append(edges_j_remapped)
            
            
            if mixed_edges_list:
                mixed_edges = torch.cat(mixed_edges_list, dim=1)
            else:
                mixed_edges = torch.empty((2, 0), dtype=torch.long)
            
            
            if not self._validate_mixed_graph(mixed_x, mixed_edges):
                return None
            
            
            domain_i = subgraph_i['domain_id']
            domain_j = subgraph_j['domain_id']
            composition_prob = torch.zeros(self.num_domains, dtype=torch.float32)
            composition_prob[domain_i] = lam
            composition_prob[domain_j] = 1 - lam
            
            mixed_graph = Data(
                x=mixed_x,
                edge_index=mixed_edges,
                domain_list=torch.tensor([domain_i, domain_j], dtype=torch.long),
                domain_i=torch.tensor([domain_i], dtype=torch.long),
                domain_j=torch.tensor([domain_j], dtype=torch.long),
                center_node_i=torch.tensor([subgraph_i['center_node_orig']], dtype=torch.long),
                center_node_j=torch.tensor([subgraph_j['center_node_orig']], dtype=torch.long),
                lam=torch.tensor([lam], dtype=torch.float32),
                y=composition_prob,
                center_id=center_idx_i,
                is_cross=torch.tensor([1], dtype=torch.long)  
            )
            
            return mixed_graph
            
        except Exception as e:
            print(f"Error mixing cross-domain subgraphs: {e}")
            return None
    
    def _generate_high_similarity_mixtures(self, domain_graphs: Dict[int, Data],
                                         domain_ids: List[int], 
                                         target_count: int) -> List[Data]:
        """Select node pairs with high similarity from the boundary nodes for blending"""
        mixtures = []
        max_attempts = target_count * 10
        attempts = 0
        
        while len(mixtures) < target_count and attempts < max_attempts:
            attempts += 1
            try:
                # Randomly select two different domains
                selected_domains = random.sample(domain_ids, 2)
                domain_i, domain_j = selected_domains
                
                # Identify pairs of nodes with high similarity from the boundary nodes of two domains
                node_pair = self._find_high_similarity_boundary_pair(
                    domain_graphs, domain_i, domain_j
                )
                
                if node_pair is not None:
                    node_i, node_j, similarity = node_pair
                    
                    subgraph_i = self.get_k_hop_subgraph(domain_graphs[domain_i], node_i, domain_i)
                    subgraph_j = self.get_k_hop_subgraph(domain_graphs[domain_j], node_j, domain_j)
                    
                    lam = self._compute_mixing_weight_from_similarity(similarity)
                    
                    mixed_graph = self._mix_subgraphs_cross_domain(
                        subgraph_i, subgraph_j, 
                        domain_graphs[domain_i], domain_graphs[domain_j],
                        lam
                    )
                    
                    if mixed_graph is not None:
                        mixed_graph.domain_similarity = torch.tensor([similarity], dtype=torch.float32)
                        mixtures.append(mixed_graph)
                        print(f"Generated cross-domain mixture with similarity: {similarity:.3f}")
                        
            except Exception as e:
                continue
        
        print(f"Generated {len(mixtures)} high-similarity mixtures (after {attempts} attempts)")
        return mixtures

    def _find_high_similarity_boundary_pair(self, domain_graphs: Dict[int, Data],
                                          domain_i: int, domain_j: int) -> Optional[Tuple[int, int, float]]:
        """Identify node pairs with high similarity among boundary nodes in two domains"""
        if (domain_i not in self.boundary_nodes or domain_j not in self.boundary_nodes or
            len(self.boundary_nodes[domain_i]) == 0 or len(self.boundary_nodes[domain_j]) == 0):
            return None
        
        graph_i = domain_graphs[domain_i]
        graph_j = domain_graphs[domain_j]
        
        boundaries_i = self.boundary_nodes[domain_i]
        boundaries_j = self.boundary_nodes[domain_j]
        
        # Perform similarity calculations by randomly sampling boundary nodes.
        sampled_i = boundaries_i[torch.randperm(len(boundaries_i))[:min(10, len(boundaries_i))]]
        sampled_j = boundaries_j[torch.randperm(len(boundaries_j))[:min(10, len(boundaries_j))]]
        
        best_similarity = -1.0
        best_pair = None
        
        # Calculate the similarity between all pairs of sampling nodes
        for node_i in sampled_i:
            feature_i = graph_i.x[node_i]
            for node_j in sampled_j:
                feature_j = graph_j.x[node_j]
                
                similarity = F.cosine_similarity(
                    F.normalize(feature_i.unsqueeze(0), dim=1, p=2),
                    F.normalize(feature_j.unsqueeze(0), dim=1, p=2)
                ).item()
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_pair = (node_i.item(), node_j.item(), similarity)
        
        return best_pair

    def _compute_mixing_weight_from_similarity(self, similarity: float) -> float:
        """Adaptive mixuo"""
        # distance = 1 - similarity  # 
        # gamma = 0.5
        # weight = 0.5 * torch.exp(-gamma * distance ** 2)
        # return weight
        # return np.random.beta(self.alpha, self.alpha)
        # return 0.5
        if similarity > 0.8:
            return 0.5  
        else:
            # 使用相似度影响权重分布
            base_weight = np.random.beta(self.alpha, self.alpha)
            return float(base_weight)
           
    def _generate_fallback_mixtures(self, domain_graphs: Dict[int, Data],
                                  domain_ids: List[int],
                                  num_samples: int) -> List[Data]:
        
        mixtures = []
        
        for i in range(num_samples):
            try:
                
                selected_domains = random.sample(domain_ids, 2)
                domain_i, domain_j = selected_domains
                
                
                node_i = self._sample_boundary_node(domain_i)
                node_j = self._sample_boundary_node(domain_j)
                
                if node_i is not None and node_j is not None:
                    subgraph_i = self.get_k_hop_subgraph(domain_graphs[domain_i], node_i, domain_i)
                    subgraph_j = self.get_k_hop_subgraph(domain_graphs[domain_j], node_j, domain_j)
                    
                    lam = float(np.random.beta(self.alpha, self.alpha))
                    
                    mixed_graph = self._mix_subgraphs_cross_domain(
                        subgraph_i, subgraph_j,
                        domain_graphs[domain_i], domain_graphs[domain_j],
                        lam
                    )
                    
                    if mixed_graph is not None:
                        mixtures.append(mixed_graph)
                        
            except Exception as e:
                continue
        
        return mixtures

    def _sample_boundary_node(self, domain_id: int) -> Optional[int]:
        """Sample a node from the boundary nodes of the specified domain"""
        if (domain_id in self.boundary_nodes and 
            len(self.boundary_nodes[domain_id]) > 0):
            boundaries = self.boundary_nodes[domain_id]
            return boundaries[torch.randint(0, len(boundaries), (1,))].item()
        return None

    def get_k_hop_subgraph(self, graph: Data, node_index: int, domain_id: int, smallest_size=5):
        """k-hop subgraph"""
        cache_key = (domain_id, node_index)
        
        if cache_key in self.subgraphs_cache:
            return self.subgraphs_cache[cache_key]
        
        if isinstance(node_index, int):
            node_idx_tensor = torch.tensor([node_index], dtype=torch.long)
        else:
            node_idx_tensor = node_index.unsqueeze(0) if node_index.dim() == 0 else node_index
        
        try:
            current_label = graph.y[node_idx_tensor].item()

            sub_nodes, _, mapping, _ = k_hop_subgraph(
                node_idx=node_idx_tensor, num_hops=self.hop,
                edge_index=graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
            )
            
            if len(sub_nodes) < smallest_size:
                need_node_num = smallest_size - len(sub_nodes)
                pos_nodes = torch.argwhere(graph.y == int(current_label))
                pos_nodes = pos_nodes.to('cpu')
                sub_nodes = sub_nodes.to('cpu')
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), sub_nodes.numpy()))
                candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
                sub_nodes = torch.cat([torch.flatten(sub_nodes), torch.flatten(candidate_nodes)]).to(self.device)

            sub_edge_index, _ = subgraph(sub_nodes, graph.edge_index, relabel_nodes=True)

        except Exception as e:
            print(f"Error in k_hop_subgraph for node {node_index} in domain {domain_id}: {e}")
            raise e
        
        sub_x = graph.x[sub_nodes] if graph.x is not None else torch.ones((len(sub_nodes), 1))
        
        if isinstance(mapping, torch.Tensor) and mapping.numel() == 1:
            mapping = mapping.item()
        elif isinstance(mapping, torch.Tensor) and mapping.numel() > 1:
            mapping = mapping[0].item()
        
        subgraph_data = {
            'nodes': sub_nodes,
            'edge_index': sub_edge_index,
            'x': sub_x,
            'center_node_idx': mapping,
            'center_node_orig': node_index,
            'domain_id': domain_id
        }
        
        self.subgraphs_cache[cache_key] = subgraph_data
        return subgraph_data

    def _validate_graph_data(self, graph: Data, domain_id: int) -> bool:
        if graph.edge_index is None:
            print(f"Warning: Domain {domain_id} graph has no edge_index")
            return False
        
        if graph.edge_index.numel() == 0:
            print(f"Warning: Domain {domain_id} graph has empty edge_index")
            return False
        
        if graph.num_nodes == 0:
            print(f"Warning: Domain {domain_id} graph has no nodes")
            return False
        
        if graph.edge_index.max() >= graph.num_nodes:
            print(f"Warning: Domain {domain_id} graph has invalid edge indices")
            return False
            
        return True

    def _validate_mixed_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> bool:
        try:
            num_nodes = x.shape[0]
            
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                if max_idx >= num_nodes:
                    print(f"Invalid graph: edge index {max_idx} >= num nodes {num_nodes}")
                    return False
                
                min_idx = edge_index.min().item()
                if min_idx < 0:
                    print(f"Invalid graph: negative edge index {min_idx}")
                    return False
            
            return True
        except Exception as e:
            print(f"Error validating graph: {e}")
            return False

    def clear_cache(self):
        self.subgraphs_cache.clear()
        self.domain_centers.clear()
        self.boundary_nodes.clear()
        self.center_nodes.clear()



def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

