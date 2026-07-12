import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.Graph import Node
import graphviz
from causallearn.graph.Endpoint import Endpoint

class CausalGraphLearner:
    """
    A class to learn causal graphs using the PC algorithm from the 'causal-learn' library.
    
    Attributes:
        alpha (float): Significance level for conditional independence tests.
        indep_test (str): The type of independence test to use. Default is 'fisherz' for continuous data.
        uc_rule (int): Uniform cost rule parameter for the PC algorithm.
        uc_priority (int): Uniform cost priority parameter for the PC algorithm.
        graph: The learned causal graph (after fit is called).
        nodes: List of Node objects corresponding to the DataFrame's columns.
        column_names: List of column names from the DataFrame.
        name_to_node: A dictionary mapping from column names to Node objects.
    """
    
    def __init__(self, alpha=0.05, indep_test='fisherz', uc_rule=0, uc_priority=0):
        self.alpha = alpha
        self.indep_test = indep_test
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.graph = None
        self.nodes = None
        self.column_names = None
        self.name_to_node = {}

    def fit(self, df):
        """
        Learns the causal graph structure from the given DataFrame using the PC algorithm.
        
        Parameters:
            df (pd.DataFrame): The preprocessed DataFrame containing your features.
                               Columns should be named (e.g., ['F0', 'F1', ..., 'Z', ...]).
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        self.column_names = df.columns.tolist()
        data = df.values
        
        # Run the PC algorithm (let it create nodes internally)
        pc_result = pc(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            uc_rule=self.uc_rule,
            uc_priority=self.uc_priority,
            show_progress=False
        )
        
        self.graph = pc_result.G
        self.nodes = self.graph.get_nodes()
        
        # Create a mapping from node names (X1, X2...) to original column names
        # and from original column names to nodes
        self.name_to_node = {}
        for i, name in enumerate(self.column_names):
            node = self.nodes[i]
            self.name_to_node[name] = node
            # Store mapping from "Xi" format name to node object as well
            self.name_to_node[node.get_name()] = node

    def get_graph(self):
        """
        Returns the learned causal graph.
        
        Returns:
            The causal graph (Graph object) as produced by the PC algorithm.
        """
        if self.graph is None:
            raise ValueError("Graph not learned yet. Call 'fit' with your data first.")
        return self.graph

    def get_adjacent_nodes(self, target_node_name):
        """
        Returns a list of node names that are directly connected (conditionally dependent) with the target node.
        
        Parameters:
            target_node_name (str): The name of the target node (e.g., 'Z').
            
        Returns:
            List of node names that are adjacent to the target node in the learned graph.
        """
        if self.graph is None or self.nodes is None:
            raise ValueError("Graph not learned yet. Call 'fit' with your data first.")
        
        # Get the node by its name (either original column name or Xi format)
        target_node = self.name_to_node.get(target_node_name)
        if target_node is None:
            raise ValueError(f"Node '{target_node_name}' not found in the data.")
        
        adjacent_nodes = self.graph.get_adjacent_nodes(target_node)
        
        # Map node names back to original column names
        adjacent_node_names = []
        for node in adjacent_nodes:
            idx = self.nodes.index(node)
            if idx < len(self.column_names):
                adjacent_node_names.append(self.column_names[idx])
            else:
                adjacent_node_names.append(node.get_name())
                
        return adjacent_node_names

    def get_non_adjacent_nodes(self, target_node_name):
        """
        Returns a list of node names that are NOT directly connected to the target node.
        
        Parameters:
            target_node_name (str): The name of the target node (e.g., 'Z').
            
        Returns:
            List of node names that are not adjacent to the target node.
        """
        if self.nodes is None:
            raise ValueError("Graph not learned yet. Call 'fit' with your data first.")
        
        all_nodes = set(self.column_names)
        adjacent_nodes = set(self.get_adjacent_nodes(target_node_name))
        non_adjacent = all_nodes - adjacent_nodes - {target_node_name}
        return list(non_adjacent)
    
    def plot_graph(self):
        """
        Plots the learned causal graph using the 'graphviz' library.
        Returns a graphviz.Digraph object.
        """
        if self.graph is None:
            raise ValueError("Graph not learned yet. Call 'fit' with your data first.")
        
        # Create a new directed graph
        dot = graphviz.Digraph(comment='Causal Graph')
        
        # Add nodes with original column names
        for i, name in enumerate(self.column_names):
            dot.node(name, name)
        
        # Add edges with appropriate styles based on endpoints
        for i in range(len(self.nodes)):
            node_i = self.nodes[i]
            name_i = self.column_names[i]
            
            for j in range(i+1, len(self.nodes)):
                node_j = self.nodes[j]
                name_j = self.column_names[j]
                
                # Get the edge between nodes, if any
                edge = self.graph.get_edge(node_i, node_j)
                
                if edge is None:
                    continue  # No edge between these nodes
                
                # Get endpoints
                endpoint1 = self.graph.get_endpoint(node_i, node_j)
                endpoint2 = self.graph.get_endpoint(node_j, node_i)
                
                # Determine edge type based on endpoints
                if endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
                    # i -> i
                    dot.edge(name_i, name_j)
                elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
                    # i <- j
                    dot.edge(name_j, name_i)
                elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
                    # i -- j (undirected)
                    dot.edge(name_i, name_j, dir='none', style='dashed')
                elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:
                    # i <-> j (bidirected)
                    dot.edge(name_i, name_j, dir='both')
                else:
                    # Other types (partially directed, etc.)
                    dot.edge(name_i, name_j, dir='none', style='dotted')
        
        return dot
