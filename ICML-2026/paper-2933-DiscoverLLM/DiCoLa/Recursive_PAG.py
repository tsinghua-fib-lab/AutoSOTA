from itertools import combinations
from Graph.GraphClass_PAG import MixGraph, Mark, Node
import pandas as pd
import networkx as nx
from DiCoLa.Markov_Blanket_Learner import MB_learn
import time
from DiCoLa.CI_test import CI_test,CI_Test_
from Graph.Sepset import Separation_Set



"""
NOTE: The 'Node' only be used in  the Graph learning process, we don't use it in the other process, e.g., MB learning.
"""


def DiCola_learner(observed_data: pd.DataFrame, leaf_node_learner=None, alpha:float=0.01, ci_type="Fisher_Z", **kwargs) -> dict:

    time_start = time.time()
    learner = Recursive_Learner(observed_data, leaf_node_learner=leaf_node_learner, ci_type=ci_type, alpha=alpha, **kwargs)
    pag = learner.pag_learner()
    time_end = time.time()

    return {'PAG.DataFrame': pag.MG_to_pandas_adjacency(), 'CI_num': learner.get_ci_test_number(), "runtime_sec": time_end - time_start}

class Recursive_Learner:

    def __init__(self, data, leaf_node_learner=None, alpha = 0.01, ci_type = "Fisher_Z", **kwargs):
        
    
        ### For the orcale case, ci_type should be "D_sep"
        self.latent_nodes = kwargs.get("latent_nodes", None)  # List of latent nodes list[str]
        if (self.latent_nodes is not None)\
            and ci_type != "D_sep":
            raise ValueError("Latent nodes or selection bias nodes are only supported with D_sep CI type.")
        
        if ci_type=="D_sep": 
            self.G = nx.DiGraph(data)  # Create a directed graph from the adjacency matrix
            self.ancList = {}
            self.vars_list = data.columns.tolist()
            for var in self.vars_list:
                self.ancList[var] = set(nx.ancestors(self.G, var))
        self.observed_data = data.drop(columns=self.latent_nodes) if self.latent_nodes is not None else data
        ### End for the oracle case
        self.alpha = alpha


        if ci_type=="D_sep":
            self.uig_type = "uig_dsep"
            self.ci_test = CI_test(data, method_type=ci_type, alpha=self.alpha, **kwargs)
        elif ci_type=="Fisher_Z":
            self.uig_type = "uig_gaussian_mb"
            self.ci_test = CI_Test_(data)
            
        else:
            raise ValueError(f"Unsupported ci_type: {ci_type}. Supported types are 'D_sep' and 'Fisher_Z'.")

        self.leaf_node_learner = leaf_node_learner
        if ci_type=="Fisher_Z":
            assert callable(self.leaf_node_learner), "leaf_node_learner must be a callable function."
            assert self.leaf_node_learner.__code__.co_argcount >= 2, "leaf_node_learner must accept at least two arguments: data and alpha."
            assert 'data' in self.leaf_node_learner.__code__.co_varnames, "leaf_node_learner must have a 'data' parameter."
            assert 'alpha' in self.leaf_node_learner.__code__.co_varnames, "leaf_node_learner must have an 'alpha' parameter."
        self._init_info(self.observed_data,**kwargs)
        self.sepsets = Separation_Set(set(self.Nodes_list))
        self.pag = MixGraph(incoming_graph_data=self.Nodes_list) # Nodes_list only includes observed nodes
   
        
    def get_ci_test_number(self) -> int:
        """
        Get the number of CI tests performed.
        """
        return self.ci_test.get_ci_num()  
    
    def _init_info(self, data: pd.DataFrame, **kwargs):
        

        if isinstance(data, pd.DataFrame):
            self.Nodes_list = [Node(node_name, index) for index, node_name in enumerate(data.columns)]
            self.Nodes_dict = {node.name: node for node in self.Nodes_list}
        else:
            raise TypeError("Data must be a pandas DataFrame.")
        self.min_leaf_size = kwargs.get("min_leaf_size", int(len(self.Nodes_list)/3))
        self.max_recursion_depth = kwargs.get("max_recursion_depth", 3 if len(self.Nodes_list) <= 200 else 5)


    def oracle_learn_uig(self, sub_nodes: list[Node],  
                  uig_pa: "MixGraph | None" = None, 
                  C_nodes: "set[Node] | None" = None) -> MixGraph:
        """
        Learn undirected independence graph for a subset of nodes.
        """
        uig = MixGraph(incoming_graph_data=sub_nodes) # empty graph
        sub_vars = list(node.name for node in sub_nodes)
        if uig_pa is None or C_nodes is None:
            for u, v in combinations(sub_vars, 2):
                u_node = self.Nodes_dict[u]
                v_node = self.Nodes_dict[v]
                if uig.has_edge(u_node, v_node):
                    continue
                S = set(sub_vars) - {u, v}
                if not self.ci_test(u, v, list(S))[0]:
                    uig.add_circ_Edge(u_node, v_node)
                else:
                    self.sepsets._add(u_node, v_node, set(self.Nodes_dict[name] for name in S)) # add the sepset info
            
            return uig
        else:
            A_nodes = set(sub_nodes) - C_nodes
            for u_node, v_node in combinations(sub_nodes, 2):
                if uig.has_edge(u_node, v_node):
                    continue
                if u_node in A_nodes or v_node in A_nodes:
                    if uig_pa.has_edge(u_node, v_node):
                        uig.add_circ_Edge(u_node, v_node)
                else:
                    S = set(sub_vars) - {u_node.name, v_node.name}
                    if not self.ci_test(u_node.name, v_node.name, list(S))[0]:
                        uig.add_circ_Edge(u_node, v_node)
                    else:
                        self.sepsets._add(u_node, v_node, set(self.Nodes_dict[name] for name in S)) # add the sepset info

            return uig

    
    def learn_uig_base_gaussian_mb_fast(self, sub_nodes: list[Node], 
                            uig_pa: "nx.Graph | None" = None, 
                            C_nodes: "set[Node] | None" = None) -> nx.Graph:
        """
        Learn undirected independence graph for a subset of nodes using Gaussian MB.
        """

        if C_nodes is None or len(C_nodes) > 1:
            sub_observed_data = self.observed_data[[node.name for node in sub_nodes]]
            mb_learner = MB_learn(sub_observed_data, mb_method_type='gaussian_MB')  
            uig = nx.from_pandas_adjacency((mb_learner.bool_mb_df > 0).astype(int))
            # sub_vars = list(node.name for node in sub_nodes)
            # for u, v in combinations(sub_vars, 2):
            #     if mb_learner.bool_mb_df.loc[u, v] != 1:
            #         S = set(sub_vars) - {u, v}
            #         self.sepsets._add(self.Nodes_dict[u], self.Nodes_dict[v], set(self.Nodes_dict[name] for name in S)) # add the sepset info
        else:
            sub_nodes_name = [node.name for node in sub_nodes]
            uig= uig_pa.subgraph(sub_nodes_name).copy()

        return uig
    

    def split_by_connected_components_balanced(slef, G: nx.Graph):

        if G.number_of_nodes() <= 1 or nx.is_connected(G):
            return None

        components = sorted(
            nx.connected_components(G),
            key=len,
            reverse=True
        )

        part1, part2 = set(), set()
        size1, size2 = 0, 0

        for comp in components:
            if size1 <= size2:
                part1 |= comp
                size1 += len(comp)
            else:
                part2 |= comp
                size2 += len(comp)

        return part1, part2

    def find_decomposition(self, uig:" MixGraph | nx.Graph ") -> tuple:
        """
        Find A CI B | C decomposition in the undirected independence graph.
        """

        if isinstance(uig, MixGraph):
            if uig.is_complete_graph():
                return None
            nxGraph = uig.to_networkx_Graph()
        elif isinstance(uig, nx.Graph):
            uig_node_num = uig.number_of_nodes()
            if uig.number_of_edges() == (uig_node_num * (uig_node_num - 1) // 2):
                return None
            nxGraph = uig
        else:
            raise TypeError(f"Unsupported graph type: {type(uig)}")
        
        # if uig is disconnected, split by connected components
        res = self.split_by_connected_components_balanced(uig)
        if res is not None:
            A, B = res
            A = set(self.Nodes_dict[str(name)] for name in A)
            B = set(self.Nodes_dict[str(name)] for name in B)
            C = set()
            return (A, B, C)


        JT = nx.junction_tree(nxGraph)
        best = None
        best_score = None

        for u, v in JT.edges():
            C = set(u) & set(v)
            if len(C) == 0:
                continue

            # Remove edge and find connected components
            JT_minus = JT.copy()
            JT_minus.remove_edge(u, v)
            comps = list(nx.connected_components(JT_minus))

            if len(comps) != 2:
                continue  # should not happen in a tree

            comp1, comp2 = comps[0], comps[1]

            # Union variables in components
            U1 = set().union(*comp1)
            U2 = set().union(*comp2)

            A = U1 - C
            B = U2 - C

            if len(A) == 0 or len(B) == 0:
                continue

            # Good decomposition criteria
            score = len(C) / (min(len(A), len(B)) if min(len(A), len(B)) > 0 else 1)

            if (best_score is None) or (score < best_score):
                best_score = score
                best = (A, B, C)

        if best is not None:
            A, B, C = best
            A = set(self.Nodes_dict[str(name)] for name in A)
            B = set(self.Nodes_dict[str(name)] for name in B)
            C = set(self.Nodes_dict[str(name)] for name in C)
            best = (A, B, C)

        return best
    
    def combine_subgraphs(self, skel1: MixGraph, skel2: MixGraph) -> MixGraph:

        
        """
        Combine two local skeletons (MixGraphs) skel1=(U,E_U) and skel2=(V,E_V)
        into a skeleton over K = U \cup V following the provided pseudocode:
        - start with vertex set K and edge set E_K = E_U \cup E_V
        - for each unordered pair in INTER = U ∩ V, if the pair is absent
          from both E_U and E_V then remove it from E_K

        Returns a new MixGraph defined on K with the combined edges.
        """

        U = set(skel1.node_list)
        V = set(skel2.node_list)
        K = U | V

        # sort by index for deterministic ordering
        K_list = sorted(list(K), key=lambda n: n.index if n.index is not None else 0)
        combined = MixGraph(incoming_graph_data=K_list)

        def _copy_edges_from(skel: MixGraph):
            for u, v, data in skel.edges(data=True):
                edge_obj = data.get('edge')
                if edge_obj is None:
                    continue
                try:
                    combined.add_Edge(edge_obj.start, edge_obj.end, edge_obj.copy())
                except ValueError:
                    # edge may already exist in combined, ignore
                    pass

        # 1. merge edge sets
        _copy_edges_from(skel1)
        _copy_edges_from(skel2)

        # 2. for pairs inside the intersection, remove if absent in both originals
        INTER = U & V
        for a, b in combinations(list(INTER), 2):
            if (not skel1.has_edge(a, b)) or (not skel2.has_edge(a, b)):
                if combined.has_edge(a, b):
                    combined.remove_Edge(a, b)

        combined.clear_cache()
        return combined
    
   

    """
    skeleton learning.
    """

    def leaf_node_skeleton_learner(self, sub_nodes: list[Node]) -> MixGraph:
        """
        Learn the skeleton of the graph based on the provided subset of all nodes.
        Using the skeleton learner method.
        """

        sub_graph = MixGraph(incoming_graph_data=sub_nodes)
        if len(sub_nodes) <= 1:
            return sub_graph
        sub_graph._init_complete_graph()
   
        res = self.leaf_node_learner(self.observed_data[[node.name for node in sub_nodes]], alpha=self.alpha)
        adj_df = res['PAG.DataFrame']
        ci_num = res['CI_num']
        self.ci_test._ci_num += ci_num  # update the ci_num in the main ci_test object
        # print(f'learning  leaf_node_learner time = {res["runtime_sec"]} seconds')

        for u_node, v_node in combinations(sub_nodes, 2):
            if adj_df.loc[u_node.name, v_node.name] == 0:
                sub_graph.remove_Edge(u_node, v_node)
        
        return sub_graph

    
    def oracle_leaf_node_learning(self, sub_nodes: list[Node]) -> MixGraph:
        """
        Oracle skeleton learning using D-separation.
        """

        sub_graph = MixGraph(incoming_graph_data=sub_nodes)
        sub_graph._init_complete_graph()
   
        sub_vars = set(node.name for node in sub_nodes)
        for x_node, y_node in sub_graph.edges():
      
            sepset_candidates = (self.ancList[x_node.name] | self.ancList[y_node.name]) - {x_node.name, y_node.name}
            sepset = sepset_candidates & sub_vars  ## consider sub_vars is the observed variables in the current sub_graph

            if self.ci_test(x_node.name, y_node.name, list(sepset))[0]:
                self.sepsets._add(x_node, y_node, set(self.Nodes_dict[name] for name in sepset))
                sub_graph.remove_Edge(x_node, y_node) 
      
        return sub_graph


    def decomp_recovery_skeleton(self, sub_nodes: list[Node], 
                                uig_pa: MixGraph=None, C_nodes: set[Node]=None,
                                depth: int=0) -> MixGraph:

        if len(sub_nodes) <= self.min_leaf_size or depth >= self.max_recursion_depth:
            if self.uig_type == "uig_gaussian_mb":
                skel = self.leaf_node_skeleton_learner(sub_nodes) 
            elif self.uig_type == "uig_dsep":
                skel = self.oracle_leaf_node_learning(sub_nodes)
            return skel

        # --- UIG ---
        if self.uig_type == "uig_dsep":
            uig = self.oracle_learn_uig(sub_nodes, uig_pa=uig_pa, C_nodes=C_nodes)
        elif self.uig_type == "uig_gaussian_mb":
            uig = self.learn_uig_base_gaussian_mb_fast(sub_nodes, uig_pa=uig_pa, C_nodes=C_nodes)

        # --- Decomposition ---
        decomposition = self.find_decomposition(uig)

        if decomposition is None:
            if self.uig_type == "uig_gaussian_mb":
                skel = self.leaf_node_skeleton_learner(sub_nodes)
            elif self.uig_type == "uig_dsep":
                skel = self.oracle_leaf_node_learning(sub_nodes)
            return skel

        # Recursive Case
        A, B, C = decomposition
        # record sepsets
        for a in A:
            for b in B:
                self.sepsets._add(a, b, C)
        for node in sub_nodes:
            if node not in (A | B | C):
                for a in list(A | B | C):
                    self.sepsets._add(node, a, set())

        # ---------- Left child: A ∪ C ----------
        AC = list(A | C)
        skel_left = self.decomp_recovery_skeleton(AC, uig, C, depth=depth+1)
        # ---------- Right child: B ∪ C ----------
        BC = list(B | C)
        skel_right = self.decomp_recovery_skeleton(BC, uig, C, depth=depth+1)
        # ---------- MERGE ----------
        combined = self.combine_subgraphs(skel_left, skel_right)

        return combined

    
    def pag_learner(self) -> MixGraph:

        skel = self.decomp_recovery_skeleton(self.pag.node_list,  depth=0)

        # handle isolated nodes
        isolated_nodes = []
        for node in self.pag.node_list:
            if node not in skel.node_list:
                skel.add_Node(node)
                isolated_nodes.append(node)
        
        for node in isolated_nodes:
            for a in skel.node_list:
                if node != a:
                    self.sepsets._add(a, node, set())

        return skel
    

