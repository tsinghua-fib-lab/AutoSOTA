from itertools import combinations
import logging
import networkx as nx
from typing import Optional, Set, Tuple, Union, List
import numpy as np
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger("GraphClass")
logger.setLevel(logging.DEBUG) 



from enum import Enum
from typing import Optional
class Mark(Enum):
    "causal-learn package mark definition"
    # TAIL = -1
    # ARROW = 1
    # CIRCLE = 2
    # NULL = 0


    "pcalg package mark definition"
    TAIL = 3   # "-"
    ARROW = 2  # ">"
    CIRCLE = 1 # "o"
    NULL = 0   # " "

    def __eq__(self, other):
        if isinstance(other, Mark):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o", Mark.NULL: " "}[self]

    def __repr__(self):
        return f"Mark.{self.name}"


from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    name: Optional[str] = None
    index: Optional[int] = None

    def __repr__(self):
        return f"{self.name}"
    def __str__(self):
        return f"{self.name}"
    # def __eq__(self, value):
    #     if not isinstance(value, Node):
    #         return NotImplemented
    #     return self.name == value.name 

    
class Edge:
    def __init__(self, start: Node, lmark: Mark, rmark: Mark, end: Node):  # start lmark-rmark end
        self.start = start
        self.end = end
        self.lmark = lmark  # The mark here can be Mark.TAIL, Mark.ARROW, or Mark.CIRCLE
        self.rmark = rmark
    def __repr__(self):
        rmark_symbol = {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o"}
        lmark_symbol = {Mark.TAIL: "-", Mark.ARROW: "<", Mark.CIRCLE: "o"}
        return f"{self.start}{lmark_symbol[self.lmark]}-{rmark_symbol[self.rmark]} {self.end}"
    def _invert(self): 
        self.start, self.end = self.end, self.start
        self.lmark, self.rmark = self.rmark, self.lmark

    def copy(self):
        return Edge(self.start, self.lmark, self.rmark, self.end)



class MixGraph(nx.Graph):
    def __init__(self, incoming_graph_data: Optional[Union[pd.DataFrame, np.ndarray, list]] = None):
        """
        Initialize a Mix Graph based on networkx.Graph.
        If incoming_graph_data is provided, it will be used to create the graph.
        pd.DataFrame or np.ndarray means the adjacency matrix. if list, it means the node list.
        """
        super().__init__()
        self._cache_path = defaultdict(dict)  # for caching results of get...paths functions
        self._cache_nodes = defaultdict(dict)  # for caching results of get...nodes functions
        if incoming_graph_data is not None:
            if isinstance(incoming_graph_data, pd.DataFrame):
                self.from_pandas_adjacency(incoming_graph_data)
            elif isinstance(incoming_graph_data, np.ndarray):
                self.from_numpy_array(incoming_graph_data)
            elif isinstance(incoming_graph_data, list):
                self.from_node_list(incoming_graph_data)
            else:
                raise TypeError("Invalid graph data type.")

    def clear_cache(self):
        """Clear the cache."""
        self._cache_path.clear()
        self._cache_nodes.clear()

    ####################
    ## Edge functions ##
    ####################

    def _ensure_order(self, node1: Node, node2: Node) -> tuple[Node, Node]:
        """
        Ensure node1 < node2 based on the integer mapping.
        """
        if node1.index > node2.index:
            return node2, node1
        return node1, node2


    def add_Edge(self, node1: Node, node2: Node, edge: Edge):
        """
        Add an edge to the graph. 
        The parent function that for add different types of edges.
        """
        self.clear_cache()
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise TypeError("node1 and node2 must be of type Node.")
        if not isinstance(edge, Edge):
            raise TypeError("edge must be of type Edge.")
        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before adding an edge.")
        
        # If an edge exists, remove it first
        if self.has_edge(node1, node2):
            raise ValueError(f"Edge between {node1} and {node2} already exists. Use update_Edge() to update it.")
        
        # Ensure node1.index < node2.index
        n1, n2 = self._ensure_order(node1, node2)
        if (n1, n2) == (node2, node1):
            edge._invert()  # Invert the edge if the order is reversed

        self.add_edge(node1, node2, edge=edge)
        

    def remove_Edge(self, node1: Node, node2: Node):
        """
        Remove an edge between two nodes if it exists.
        """
        self.clear_cache()
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise TypeError("node1 and node2 must be of type Node.")
        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before removing an edge.")
        
        if self.has_edge(node1, node2):
            self.remove_edge(node1, node2)
        else:
            raise ValueError(f"Edge between {node1} and {node2} does not exist.")

        


    def add_circ_Edge(self, node1: Node, node2: Node):
        """
        Set an Undirected edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-o node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        # Add the new edge with the specified marks
        edge = Edge(node1, Mark.CIRCLE, Mark.CIRCLE, node2)
        self.add_Edge(node1, node2, edge=edge)

        

    def add_directed_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes if it does not already exist. If the edge already exists, it will not be replaced.
        node1 -> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.  
        """
        self.clear_cache()
        edge = Edge(node1, Mark.TAIL, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)

        

    def add_bidirected_Edge(self, node1: Node, node2: Node):
        """
        Add a bidirected edge between two nodes if it does not already exist. If the edge already exists, it will not be replaced.
        node1 <-> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        edge = Edge(node1, Mark.ARROW, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)

        

    def add_circ_arrow_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        edge = Edge(node1, Mark.CIRCLE, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)

        

    def add_circ_tail_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-- node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()

        edge = Edge(node1, Mark.CIRCLE, Mark.TAIL, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)

        

    def add_tail_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with TAIL marks. If the edge already exists, it will not be replaced.
        node1 --- node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()

        edge = Edge(node1, Mark.TAIL, Mark.TAIL, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)

    def clear_all_orientations(self):
        """
        Clear all edge orientations in the graph.
        """
        for u, v in self.edges():
            self.update_Edge(u, Mark.CIRCLE, Mark.CIRCLE, v)
 


    def update_Edge(self, node1: Node, lmark: Mark, rmark: Mark, node2: Node):
        """
        Orient an edge between two nodes with specific marks. If the edge does not exist, it will not be created.

        Parameters:
        - node1: The first node.
        - node2: The second node.
        - lmark: The left mark.
        - rmark: The right mark.
        """
        self.clear_cache()

        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

        if self.has_edge(node1, node2):

            # Ensure node1 < node2
            n1, n2 = self._ensure_order(node1, node2)
            if (n1, n2) == (node2, node1):
                lmark, rmark = rmark, lmark  # Swap marks if the order is reversed
                node1, node2 = node2, node1  # Swap nodes

            # Update the existing edge with new marks
            edge = self[node1][node2]['edge'].copy()  # Copy the existing edge object
            if lmark is not None: 
                edge.lmark = lmark
            if rmark is not None:
                edge.rmark = rmark
            self.add_edge(node1, node2, edge=edge)  # Update the edge in the graph
        else:
            raise ValueError(f"Edge between {node1} and {node2} does not exist. Use add_xx_edge() to create it.")

        

    def get_Edge(self, start: Node, end: Node) -> Optional[Edge]:
        """
        Retrieve the edge object between two nodes if it exists.
        start --- end

        Parameters:
        - start: The starting node of the edge.
        - end: The ending node of the edge.

        Returns:
        - The Edge object (start--edge--end) if the edge exists, otherwise None.
        """
        # logger.debug(f"get_Edge function, start:{start}, end:{end}")
        if self.has_edge(start, end):
            # logger.debug(f"get_Edge function, start:{start}, end:{end}, edge exists")
            if self[start][end]['edge'] is not None:
                edge = self[start][end]['edge'].copy()
            else:
                raise ValueError(f"get_Edge error, 'edge' between {start} and {end} does not exist.")
            # print(f"in function get_edge:{edge}")
            if edge.start == end and edge.end == start:
                # logger.debug(f"get_Edge function, edge:{edge}, edge._invert()")
                
                edge._invert()
            if edge.start != start or edge.end != end:
                raise ValueError(f"get_Edge error, Edge between {start} and {end} is not in the correct order.")
            return  edge # Invert the edge if the order is reversed
        return None

    def has_directed_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a directed edge from the start node to the end node.  
        start -> end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if a directed edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.TAIL and edge.rmark == Mark.ARROW)
        return False

    def has_bidirected_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a bidirected edge between two nodes.  
        start <-> end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a bidirected edge exists.
        - False otherwise.
        """
        
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.ARROW and edge.rmark == Mark.ARROW)
        return False

    def has_into_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is an edge into the start node from the end node.  
        start *-> end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if a into edge exists.
        - False otherwise.
        """
        # logger.debug(f"has_into_edge function, start:{start}, end:{end}")
        edge = self.get_Edge(start, end)
        # logger.debug(f"get into edge:{edge}")
        if edge is not None:
            return (edge.rmark == Mark.ARROW)
        return False
    
    def has_pd_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a potential directed edge between two nodes.
        start (o-)-- (o>) end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if a potential directed edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark in {Mark.CIRCLE, Mark.TAIL} and edge.rmark in {Mark.CIRCLE, Mark.ARROW})
        return False

    def has_out_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is an edge out of the start node to the end node.  
        start --* end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if an out edge exists.
        - False otherwise.
        """
        # logger.debug(f"has_out_edge function, start:{start}, end:{end}")
        edge = self.get_Edge(start, end)
        # logger.debug(f"get_Edge output, edge:{edge}")
        if edge is not None:
            return (edge.lmark == Mark.TAIL)
        return False
    
    def has_circ_star_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a circ-star edge between two nodes.  
        start o-* end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a circ-star edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.CIRCLE)
        return False
    
    def has_tail_circ_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a tail-circ edge between two nodes.  
        start --o end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a tail-circ edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.TAIL and edge.rmark == Mark.CIRCLE)
        return False

    def has_tail_tail_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a tail-tail edge between two nodes.  
        start --- end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a tail-tail edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.TAIL and edge.rmark == Mark.TAIL)
        return False


    def has_circ_circ_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a circ-circ edge between two nodes.  
        start o-o end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a circ-circ edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.CIRCLE and edge.rmark == Mark.CIRCLE)
        return False
    
    def has_circ_arrow_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a circ-arrow edge between two nodes.  
        start o-> end

        Parameters:
        - start: The first node.
        - end: The second node.

        Returns:
        - True if a circ-arrow edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.CIRCLE and edge.rmark == Mark.ARROW)
        return False

    def _check_Edge(self):
        """
        Check if the edges in the graph are node1 < node2.
        """
        # logger.debug("Checking edge order...")
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            node1 = edge_obj.start
            node2 = edge_obj.end
            # logger.debug(f"Checking edge: {node1} - {node2}")
            if node1.index > node2.index:
                raise ValueError(f"Edge {node1} - {node2} is not in the correct order. Edge:{edge_obj}: {node1}:{node1.name}, {node1.index} - {node2}:{node2.name}, {node2.index}")

    


    def get_circ_star_Edge(self) -> List[tuple[Node, Node]]:
        """
        Get all edges formed by o-* in the graph.
        Returns:
            A list of tuples representing the edges formed by o-* in the graph.
            In the list, each tuple is of the form (start_node, end_node), start_node o-* end_node.
        """
        circ_star_edges = []
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            if edge_obj.lmark == Mark.CIRCLE:  #  start o-* end
                circ_star_edges.append((edge_obj.start, edge_obj.end))
            if edge_obj.rmark == Mark.CIRCLE:  #  end o-* start
                circ_star_edges.append((edge_obj.end, edge_obj.start))

        return circ_star_edges
    
    def get_circ_circ_Edge(self) -> List[tuple[Node, Node]]:
        """
        Get all edges formed by o-o in the graph.
        Returns:
            A list of tuples representing the edges formed by o-o in the graph.
            In the list, each tuple is of the form (start_node, end_node), start_node o-o end_node.
        """
        circ_circ_edges = []
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            if edge_obj.lmark == Mark.CIRCLE and edge_obj.rmark == Mark.CIRCLE:  #  start o-o end
                circ_circ_edges.append((edge_obj.start, edge_obj.end))
            if edge_obj.rmark == Mark.ARROW and edge_obj.lmark == Mark.CIRCLE:
                circ_circ_edges.append((edge_obj.end, edge_obj.start))   #  end o-o start

        return circ_circ_edges
    
    def get_circ_arrow_Edge(self) -> List[tuple[Node, Node]]:
        """
        Get all directed edges in the graph.
        Returns:
            A list of tuples representing the directed edges in the graph.
            In the list, each tuple is of the form (start_node, end_node), start_node o-> end_node.
        """
        circ_arrow_edges = []
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            if edge_obj.lmark == Mark.CIRCLE and edge_obj.rmark == Mark.ARROW:  #  start o-> end
                circ_arrow_edges.append((edge_obj.start, edge_obj.end))
            if edge_obj.lmark == Mark.ARROW and edge_obj.rmark == Mark.CIRCLE:
                circ_arrow_edges.append((edge_obj.end, edge_obj.start))   #  start <-o end

        return circ_arrow_edges

    def get_directed_Edge(self) -> List[tuple[Node, Node]]:
        """
        Get all directed edges in the graph.
        Returns:
            A list of tuples representing the directed edges in the graph.
            In the list, each tuple is of the form (start_node, end_node), start_node -> end_node.
        """
        directed_edges = []
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            if edge_obj.lmark == Mark.TAIL and edge_obj.rmark == Mark.ARROW:  #  start --> end
                directed_edges.append((edge_obj.start, edge_obj.end))
            if edge_obj.lmark == Mark.ARROW and edge_obj.rmark == Mark.TAIL:  # start <-- end
                directed_edges.append((edge_obj.end, edge_obj.start))
        
        return directed_edges

    def max_degree(self) -> int:
        """
        Get the maximum degree of the graph.
        """
        return max(dict(self.degree()).values(), default=0)

    def visible_Edge(self, X: Node, Y: Node) -> bool:
        """
        Purpose: Check if the directed edge from X to Y in a MAG or in a PAG
        is visible or not.
        A visible edge X → Y means that there are no latent confounders between X and Y.

        Parameters:
        - X: The first node.
        - Y: The second node.

        Returns:
        - True if a directed edge exists and visible.
        - False otherwise.
        """

        if not self.has_node(X) or not self.has_node(Y):
            raise ValueError("Both nodes must exist in the graph before checking visibility.")
        
        if not self.has_directed_Edge(X, Y):
            return False

        # 1. scenario: there exists a vertex V not adjacent to Y with *--> X
        for V in self.get_into_nodes(X):
            if not self.has_edge(V, Y):
                return True


        # 2. scenario: there is a collider path between V and X that is into X and every non-endpoint node on the path is a parent of Y

        Parents_Y = self.get_parents(Y)
        District_X = self.get_district(X)
        discriminator = (Parents_Y & District_X)  # The discriminator set includes all parents of Y that are in the district of X

        Cand_Vs= set()
        for node in discriminator:
            S = self.get_into_nodes(node) - {X}.update(discriminator)    # V*-> node
            if S:
                for s in S:
                    if not self.has_edge(s, Y):
                        Cand_Vs.add(s)        # Add all candidates to the set Cand_Vs
                # Cand_Vs |= S  

        if len(Cand_Vs) == 0:
            return False
        else:
            for V in Cand_Vs:
                for path in self.get_all_paths(source=V, end=X):
                    if all(node in discriminator for node in path[1:-1]) and self.is_collider_path(path):  # path[1:-1] are all non-endpoint nodes
                        return True
        
        return False



    def clear_all_Edges(self):
        """
        Clear all edges in the graph.
        """
        self.clear_edges()

    ########################
    ## End Edge functions ##
    ########################

    ####################
    ## Path functions ##
    ####################

    def has_path(self, source: Node, end: Node) -> bool:
        """
        Check if there is a path between two nodes in the graph.

        Parameters:
        - source: The source node.
        - end: The end node.

        Returns:
        - True if a path exists, False otherwise.
        """
        if not self.has_node(source) or not self.has_node(end):
            raise ValueError("Both nodes must exist in the graph before checking for a path.")

        return nx.has_path(self, source, end)

    def has_pd_path(self, source: Node, end: Node) -> bool:
        """
        Efficiently check if there is a potentially directed path (p.d. path) from source to end.
        Uses DFS, only traversing edges that satisfy has_pd_Edge.
        """
        if not self.has_node(source) or not self.has_node(end):
            raise ValueError("Both nodes must exist in the graph before checking for a p.d. path.")

        visited = set()

        def dfs(current):
            if current == end:
                return True
            visited.add(current)
            for neighbor in self.get_adj_nodes(current):
                if neighbor not in visited and self.has_pd_Edge(current, neighbor):
                    if dfs(neighbor):
                        return True
            return False

        return dfs(source)
    
    
    
    def has_directed_path(self, source: Node, end: Node) -> bool:
        """
        Efficiently check if there is a directed path from source to end.
        Uses DFS, only traversing edges that satisfy has_directed_Edge.
        """
        if not self.has_node(source) or not self.has_node(end):
            raise ValueError("Both nodes must exist in the graph before checking for a directed path.")

        visited = set()

        def dfs(current):
            if current == end:
                return True
            visited.add(current)
            for neighbor in self.get_adj_nodes(current):
                if neighbor not in visited and self.has_directed_Edge(current, neighbor):
                    if dfs(neighbor):
                        return True
            return False

        return dfs(source)
    


    def get_all_paths(self, source: Node, end: Node) -> List[List[Node]]:
        """
        Get all paths between two nodes in the graph.

        Parameters:
        - source: The source node.
        - end: The end node.

        Returns:
        - A list of all paths, where each path is represented as a list of nodes.
        """
        if not self.has_node(source) or not self.has_node(end):
            raise ValueError("Both nodes must exist in the graph before getting all paths.")
        
        si, ei = source.index, end.index
        paths_by_src = self._cache_path['paths']   # If not exists, automatically create an empty dict
        src_map = paths_by_src.setdefault(si, {})  # Ensure src_map exists, create if not
        cached = src_map.get(ei)
        if cached is not None:
            return cached.copy()


        paths = list(nx.all_simple_paths(self, source=source, target=end))
        src_map[ei] = paths
        return paths.copy()



    def get_all_uncovered_pd_path(self, source: Node, end: Node) -> List[List[Node]]:
        """
        Efficiently generate all uncovered potentially directed paths (p.d. paths) from source to end using DFS.
        - Paths contain no repeated nodes.
        - Each step must be a potentially directed edge.
        - For every triple (Vi-1, Vi, Vi+1) in the path, Vi-1 and Vi+1 must not be adjacent (uncovered).
        """

        si, ei = source.index, end.index
        paths_by_src = self._cache_path['uncovered_pd_paths']   # If not exists, automatically create an empty dict
        src_map = paths_by_src.setdefault(si, {})  # Ensure src_map exists, create if not
        cached = src_map.get(ei)
        if cached is not None:
            return cached.copy()

        results = []

        def dfs(path: List[Node], visited: set):
            current = path[-1]
            if current == end:
                results.append(path.copy())
                return
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited:
                    continue
                # Must be a potentially directed edge
                if not self.has_pd_Edge(current, neighbor):
                    continue
                # Ensure uncovered: path[-2] and neighbor are not adjacent (if length >= 2)
                if len(path) >= 2 and self.has_edge(path[-2], neighbor):
                    continue
                path.append(neighbor)
                visited.add(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

        visited = {source}
        dfs([source], visited)
        src_map[ei] = results
        return results.copy()
    
    def get_all_chordless_pd_path(self, source: Node, end: Node) -> List[List[Node]]:
        """
        Efficiently generate all chordless potentially directed paths (p.d. paths) from source to end using DFS.
        - Paths contain no repeated nodes.
        - Each step must be a potentially directed edge.
        - No chords allowed: For every triple (Vi-1, Vi, Vi+1) in the path, Vi-1 and Vi+1 must not be adjacent.
        """

        si, ei = source.index, end.index
        paths_by_src = self._cache_path['chordless_pd_paths']   # If not exists, automatically create an empty dict
        src_map = paths_by_src.setdefault(si, {})  # Ensure src_map exists, create if not
        cached = src_map.get(ei)
        if cached is not None:
            return cached.copy()

        results = []

        def dfs(path: List[Node], visited: set):
            current = path[-1]
            if current == end:
                results.append(path.copy())
                return
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited:
                    continue
                # Must be a potentially directed edge
                if not self.has_pd_Edge(current, neighbor):
                    continue
                # Ensure no chords: path[0:-2] and neighbor are not adjacent (if length >= 2)
                if len(path) >= 2 and any(self.has_edge(path[i], neighbor) for i in range(len(path)-1)):
                    continue
                path.append(neighbor)
                visited.add(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

        visited = {source}
        dfs([source], visited)
        src_map[ei] = results
        return results.copy()


    def get_all_uncovered_collider_paths_from_target(self, source: Node) -> List[List[Node]]:
        """
        Get all uncovered collider paths starting from a specific source node.
        """
        si = source.index
        paths_by_src = self._cache_path['uncovered_collider_paths_from_t']   # If not exists, automatically create an empty dict
        uncovered_collider_paths_from_target = paths_by_src.setdefault(si, {})  # Ensure uncovered_collider_paths_from_target exists, create if not
        if uncovered_collider_paths_from_target:
            return uncovered_collider_paths_from_target.copy()


        results = []

        def dfs(path: List[Node], visited: set):
            current = path[-1]
            if current == source:
                if path not in results:
                    results.append(path.copy())
                return
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited:
                    continue                
                prev = path[-2]
                # Current node must be collider: prev *-> current <-* neighbor
                if self.has_into_Edge(neighbor, current) and (not self.has_edge(prev, neighbor)):
                    if self.has_into_Edge(current, neighbor): # if current <-> neighbor
                        path.append(neighbor)
                        visited.add(neighbor)
                        dfs(path, visited)
                        path.pop()
                        visited.remove(neighbor)
                    else:                             # if not current <-> neighbor
                        path.append(neighbor)
                        if path not in results and len(path) >= 3:
                            results.append(path.copy())
                        path.pop()
                    
            
        candidate_paths = []
        for neighbor in self.get_adj_nodes(source):
            if self.has_into_Edge(source, neighbor):
                candidate_paths.append([source, neighbor])

        for path in candidate_paths:
            dfs(path, visited=set(path))

        uncovered_collider_paths_from_target = results
        return results.copy()

    def get_all_arrow_collider_paths(self, source: Node, end: Node) -> List[List[Node]]:
        """
        Efficiently search for all arrow-collider paths:
        - The first edge must point to the source (i.e., source <-* neighbor)
        - Every non-endpoint node must be a collider (i.e., Vi-1 *-> Vi <-* Vi+1)
        - The path must not contain repeated nodes
        """

        si, ei = source.index, end.index
        paths_by_src = self._cache_path['arrow_collider_paths']   # If not exists, automatically create an empty dict
        src_map = paths_by_src.setdefault(si, {})  # Ensure src_map exists, create if not
        cached = src_map.get(ei)
        if cached is not None:
            return cached.copy()

        results = []

        def dfs(path: List[Node], visited: set):
            current = path[-1]
            if current == end:
                if len(path) == 2 or all(
                    self.has_into_Edge(path[i-1], path[i]) and self.has_into_Edge(path[i+1], path[i])
                    for i in range(1, len(path)-1)
                ):
                    results.append(path.copy())
                return
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited:
                    continue
                # The first edge must point to source
                if len(path) == 1 and not self.has_into_Edge(neighbor, current):
                    continue
                # Non-endpoint must be collider
                if len(path) >= 2:
                    prev = path[-2]
                    # Current node must be collider: prev *-> current <-* neighbor
                    if not (self.has_into_Edge(prev, current) and self.has_into_Edge(neighbor, current)):
                        continue
                path.append(neighbor)
                visited.add(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

        visited = {source}
        dfs([source], visited)
        # self._cache[key] = results
        src_map[ei] = results
        return results.copy()
   
    def get_all_circle_collider_paths(self, source: Node, end: Node) -> List[List[Node]]:
        """
        Efficiently search for all circle-collider paths:
        - The first edge must circle relative to the source (i.e., source o-* neighbor)
        - Every non-endpoint node must be a collider (i.e., Vi-1 *-> Vi <-* Vi+1)
        - The path must not contain repeated nodes
        """

        si, ei = source.index, end.index
        paths_by_src = self._cache_path['circle_collider_paths']   # If not exists, automatically create an empty dict
        src_map = paths_by_src.setdefault(si, {})  # Ensure src_map exists, create if not
        cached = src_map.get(ei)
        if cached is not None:
            return cached.copy()

        results = []

        def dfs(path: List[Node], visited: set):
            current = path[-1]
            if current == end:
                if len(path) == 2 or all(
                    self.has_into_Edge(path[i-1], path[i]) and self.has_into_Edge(path[i+1], path[i])
                    for i in range(1, len(path)-1)
                ):
                    results.append(path.copy())
                return
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited:
                    continue
                # The first edge must be circle relative to the source
                if len(path) == 1 and not self.has_circ_star_Edge(current, neighbor):
                    continue
                # Non-endpoint must be collider
                if len(path) >= 2:
                    prev = path[-2]
                    # Current node must be collider: prev *-> current <-* neighbor
                    if not (self.has_into_Edge(prev, current) and self.has_into_Edge(neighbor, current)):
                        continue
                path.append(neighbor)
                visited.add(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

        visited = {source}
        dfs([source], visited)

        src_map[ei] = results
        return results.copy()


    def is_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        """
        Check if the path node1 -> node2 <- node3 is a collider path.

        Parameters:
        - node1: The first node.
        - node2: The second node (the collider).
        - node3: The third node.

        Returns:
        - True if the path is a collider path, False otherwise.
        """
        if not self.has_node(node1) or not self.has_node(node2) or not self.has_node(node3):
            raise ValueError("All nodes must exist in the graph before checking for a collider.")
        
        return self.has_into_Edge(node1, node2) and self.has_into_Edge(node3, node2)


    def is_collider_path(self, path: List[Node]) -> bool:
        """
        Check if a given path is a collider path.

        Parameters:
        - path: The path to check.

        Returns:
        - True if the path is a collider path, False otherwise.
        """
        if len(path) < 3:
            return False

        # Check if the middle node is a collider
        for i in range(1, len(path) - 1):
            if not self.is_collider(path[i - 1], path[i], path[i + 1]):
                return False
        return True

    def is_uncovered_path(self, path: List[Node]) -> bool:
        """
        Check if a given path is an uncovered path.

        An uncovered path <V0, ..., Vn> is one where for every consecutive triple (Vi-1, Vi, Vi+1),
        Vi-1 and Vi+1 are not adjacent.

        Parameters:
        - path: The path to check (list of Node).

        Returns:
        - True if the path is uncovered, False otherwise.
        """
        if len(path) < 3:
            return True  # Trivially uncovered

        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            next_node = path[i + 1]
            if self.has_edge(prev_node, next_node):
                return False
        return True
    
    def is_potential_directed_path(self, path: List[Node]) -> bool:
        """
        Check if a given path is a potentially directed path (p.d. path).

        A path p = <V0, ..., Vn> is potentially directed if for every 0 <= i <= n-1,
        the edge between Vi and Vi+1 is not into Vi or out of Vi+1.

        Parameters:
        - path: The path to check (list of Node).

        Returns:
        - True if the path is potentially directed, False otherwise.
        """
        if len(path) < 2:
            return False  # A path must have at least two nodes

        for i in range(len(path) - 1):
            Vi = path[i]
            Vi1 = path[i + 1]
            edge = self.get_Edge(Vi, Vi1)
            if edge is None:
                return False
            # Not into Vi: 
            if edge.lmark == Mark.ARROW:
                return False
            # Not out of Vi+1: 
            if edge.rmark == Mark.TAIL:
                return False
        return True
 

    ########################
    ## End Path functions ##
    ########################



    ###########################################
    ## Initializing Graphs in Different Ways ##
    ###########################################
    def from_numpy_array(self, adj_matrix: np.ndarray, node_list: List[Node] = None):
        """
        Create a Mixed Graph from a NumPy adjacency matrix.
        """
        if node_list is None:
            self.node_list = [Node(name=str(i), index=i) for i in range(adj_matrix.shape[0])]  # [Node(str, 0), Node(str, 1), ...]
        else:
            self.node_list = node_list

        self.add_nodes_from(self.node_list)
        for i, j in combinations(self.node_list, 2):
            mark_ij = adj_matrix[i.index, j.index]  # i markji --- markij j
            mark_ji = adj_matrix[j.index, i.index]

            # print(f"Mark.ARROW:{Mark.ARROW.value}, Mark.CIRCLE:{Mark.CIRCLE}, Mark.TAIL:{Mark.TAIL}, Mark.NULL:{Mark.NULL}")

            if mark_ij == Mark.ARROW.value:
                if mark_ji == Mark.ARROW.value:    # i <-> j
                    self.add_bidirected_Edge(i, j)  
                elif mark_ji == Mark.CIRCLE.value:  # i o-> j
                    self.add_circ_arrow_Edge(i, j)
                elif mark_ji == Mark.TAIL.value:     # i --> j
                    self.add_directed_Edge(i, j)
                else:  
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.CIRCLE.value:      
                if mark_ji == Mark.ARROW.value:    # i <-o j
                    self.add_circ_arrow_Edge(j, i)
                elif mark_ji == Mark.CIRCLE.value:  # i o-o j
                    self.add_circ_Edge(j, i)
                elif mark_ji == Mark.TAIL.value:  # i --o j
                    self.add_circ_tail_Edge(j, i)
                else:  
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.TAIL.value:
                if mark_ji == Mark.ARROW.value:  # i <-- j
                    self.add_directed_Edge(j, i)
                elif mark_ji == Mark.CIRCLE.value: # i o-- j
                    self.add_circ_tail_Edge(i, j)
                elif mark_ji == Mark.TAIL.value:  # i --- j
                    self.add_tail_Edge(i, j)
                else:  
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.NULL.value:
                pass  # No edge between i and j
            else:
                raise ValueError(f"Invalid edge mark: {mark_ij} between {i} and {j}." "Invalid edge mark in the adjacency matrix.")
            
    def DAG_from_numpy_array(self, adj_matrix: np.ndarray, node_list: List[Node] = None):
        """
        Create a Directed Acyclic Graph (DAG) from a NumPy adjacency matrix.
        """
        if node_list is None:
            self.node_list = [Node(name=str(i), index=i) for i in range(adj_matrix.shape[0])]  # [Node(str, 0), Node(str, 1), ...]
        else:
            self.node_list = node_list


        self.add_nodes_from(self.node_list)
        for i, j in combinations(self.node_list, 2):
            mark_ij = adj_matrix[i.index, j.index]  # i markji --- markij j
            mark_ji = adj_matrix[j.index, i.index]

            # print(f"Mark.ARROW:{Mark.ARROW.value}, Mark.CIRCLE:{Mark.CIRCLE}, Mark.TAIL:{Mark.TAIL}, Mark.NULL:{Mark.NULL}")

            if mark_ij == 1:
                self.add_directed_Edge(i, j)     # i --> j
            elif mark_ji == 1:
                self.add_directed_Edge(j, i)     # j --> i
            elif mark_ij == 0 and mark_ji == 0:
                pass  # No edge between i and j
            else:
                raise ValueError(f"Invalid edge mark: {mark_ij} between {i} and {j}." "Invalid edge mark in the adjacency matrix.")

    def from_pandas_adjacency(self, adj_matrix: pd.DataFrame, graph_type: str = 'MG'):

        """
        Create a Mixed Graph from a Pandas DataFrame adjacency matrix.
        """
        # logger.debug(f"from_pandas_adjacency function")
        Node_list = [Node(name=node, index=i) for i, node in enumerate(adj_matrix.columns.to_list())]  # [Node(str, 0), Node(str, 1), ...]
        if graph_type == 'MG':
            self.from_numpy_array(adj_matrix.to_numpy(), node_list=Node_list)
        elif graph_type == 'DAG':
            self.DAG_from_numpy_array(adj_matrix.to_numpy(), node_list=Node_list)
            

    def from_node_list(self, node_list: List[Union[str, Node]]):
        """
        Create a empty graph from a list of nodes.
        """
        if all(isinstance(node, Node) for node in node_list):
            # If node_list is already a list of Node objects, use it directly
            Node_list = node_list
        else:
            Node_list = [Node(name=node, index=i) for i, node in enumerate(node_list)] # [Node(str, 0), Node(str, 1), ...]
        self.node_list = Node_list
        self.add_nodes_from(self.node_list)
        self.clear_cache()

    ###############################################
    ## End Initializing Graphs in Different Ways ##
    ###############################################



    ####################
    ## Node functions ##
    ####################
    def add_Node(self, node: Node):
        """
        Add a node to the graph.

        Parameters:
        - node: The node to be added.
        """
        if not isinstance(node, Node):
            raise TypeError("node must be of type Node.")
        self.add_node(node)
        self.node_list.append(node)
        # Keep node_list ordered by node.index (ascending)
        self.node_list.sort(key=lambda n: n.index if n.index is not None else 0)
        self.clear_cache()

    def remove_Node(self, node: Node):
        """
        Remove a node from the graph.

        Parameters:
        - node: The node to be removed.
        """
        
        if not isinstance(node, Node):
            raise TypeError("node must be of type Node.")
        if not self.has_node(node):
            raise ValueError(f"Node {node} does not exist in the graph.")
        
        self.remove_node(node)
        # Remove all edges connected to the node
        self.remove_edges_from(list(self.edges(node)))
        self.clear_cache()

    def get_adj_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node in the graph.  
        *-* node
        """
        t = node.index
        nodes_by_src = self._cache_nodes['adj_nodes']   # If not exists, automatically create an empty dict
        adj_nodes = nodes_by_src.setdefault(t, set())
        if adj_nodes:  # Cache exists and the cached set is not empty
            return adj_nodes.copy()
        
        adj = set(self.neighbors(node))
        nodes_by_src[t] = adj

        return adj.copy()

    def get_into_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent into the given node in the graph. 
        node <-*
        """
        t = node.index
        nodes_by_src = self._cache_nodes['into_nodes']   # If not exists, automatically create an empty dict
        into_nodes_t = nodes_by_src.setdefault(t, set())
        if into_nodes_t:  # Cache exists and the cached set is not empty
            return into_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        into_nodes = set()
        for adj_node in adj_nodes:
            if self.has_into_Edge(adj_node, node):
                into_nodes.add(adj_node)

        nodes_by_src[t] = into_nodes
        return into_nodes.copy()
    
    def get_no_into_nodes(self, node: Node) -> Set[Node]: 
        """
        Get the set of nodes that are adjacent not into the given node in the graph. 
        node -* or node o-*
        """
        t = node.index
        nodes_by_src = self._cache_nodes['no_into_nodes']   # If not exists, automatically create an empty dict
        no_into_nodes_t = nodes_by_src.setdefault(t, set())
        if no_into_nodes_t:  # Cache exists and the cached set is not empty
            return no_into_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        into_nodes = self.get_into_nodes(node)
        no_into_nodes = adj_nodes - into_nodes

        nodes_by_src[t] = no_into_nodes
        return no_into_nodes.copy()
    
    def get_nondirect_adj_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nondirected neighbors of the given node in the graph.
        node o-o adj_node
        """

        t = node.index
        nodes_by_src = self._cache_nodes['nondirect_adj_nodes']   # If not exists, automatically create an empty dict
        nondirect_adj_nodes_t = nodes_by_src.setdefault(t, set())
        if nondirect_adj_nodes_t:  # Cache exists and the cached set is not empty
            return nondirect_adj_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        nondirect_adj_nodes = set()
        for adj_node in adj_nodes:
            if self.has_circ_circ_Edge(node, adj_node):
                nondirect_adj_nodes.add(adj_node)

        nodes_by_src[t] = nondirect_adj_nodes
        return nondirect_adj_nodes.copy()

    def get_circ_star_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of adjacent nodes such that node (o) --* adj_node.
        node o-* adj_node
        """

        t = node.index
        nodes_by_src = self._cache_nodes['circ_star_nodes']   # If not exists, automatically create an empty dict
        circ_star_nodes_t = nodes_by_src.setdefault(t, set())
        if circ_star_nodes_t:  # Cache exists and the cached set is not empty
            return circ_star_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        circ_star_nodes = set()
        for adj_node in adj_nodes:
            if self.has_circ_star_Edge(node, adj_node):
                circ_star_nodes.add(adj_node)


        nodes_by_src[t] = circ_star_nodes
        return circ_star_nodes.copy()
    
    def max_pds_size(self) -> int:
        """
        Get the maximum size of the possible d-separation set for all nodes in the graph.
        """
        return max(len(self.get_possible_d_sep(node)) for node in self.node_list) if self.node_list else 0

    

    def get_possible_d_sep(self, node: Node) -> Set[Node]:
        """
        Optimized Possible-D-SEP:
        Faster version that avoids storing full paths and avoids repeated
        triple-checks using (prev, curr) state deduplication.
        """
        t = node.index
        cache = self._cache_nodes['possible_d_sep']
        if t in cache and cache[t]:
            return cache[t].copy()

        results = set()
        visited = set()   # store visited (prev, curr) states
        queue = deque()

        # initialize: (prev=node, curr=neighbor)
        for neigh in self.get_adj_nodes(node):
            queue.append((node, neigh))
            visited.add((node, neigh))
            results.add(neigh)

        while queue:
            prev, curr = queue.popleft()

            for nxt in self.get_adj_nodes(curr):
                if nxt is prev:
                    continue

                # avoid revisiting the same (curr, nxt) triple head
                state = (curr, nxt)
                if state in visited:
                    continue

                # FCI condition: collider or triangle
                if self.is_collider(prev, curr, nxt) or self.has_edge(prev, nxt):
                    visited.add(state)
                    queue.append(state)
                    results.add(nxt)

        results.discard(node)
        cache[t] = results
        return results.copy()



    
    def get_pd_path_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of adjacent nodes such that node (o or -) -- (o or >) adj_node.
        node o-o adj_node or node o-> adj_node or node --o adj_node or node --> adj_node
        """
        t = node.index
        nodes_by_src = self._cache_nodes['pd_path_nodes']   
        pd_path_nodes_t = nodes_by_src.setdefault(t, set())
        if pd_path_nodes_t:  
            return pd_path_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        pd_path_nodes = set()
        for adj_node in adj_nodes:
            edge = self.get_Edge(node, adj_node)
            if (edge.lmark in {Mark.CIRCLE, Mark.TAIL}) and (edge.rmark in {Mark.CIRCLE, Mark.ARROW}):
                pd_path_nodes.add(adj_node)

        nodes_by_src[t] = pd_path_nodes
        return pd_path_nodes.copy()

    def get_parents(self, node: Node) -> Set[Node]:
        """
        Get the set of parent nodes of the given node in the graph.  
        --> node
        """

        t = node.index
        nodes_by_src = self._cache_nodes['parents']   
        parents_t = nodes_by_src.setdefault(t, set())
        if parents_t:  
            return parents_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        parents = set()
        for adj_node in adj_nodes:
            if self.has_directed_Edge(adj_node, node):
                parents.add(adj_node)

        nodes_by_src[t] = parents
        return parents.copy()

    def get_children(self, node: Node) -> Set[Node]:
        """
        Get the set of child nodes of the given node in the graph.  
        node -> *
        """

        t = node.index
        nodes_by_src = self._cache_nodes['children']   
        children_t = nodes_by_src.setdefault(t, set())
        if children_t:  
            return children_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        children = set()
        for adj_node in adj_nodes:
            if self.has_directed_Edge(node, adj_node):
                children.add(adj_node)

        nodes_by_src[t] = children
        return children.copy()

    def get_spouse(self, node: Node) -> Set[Node]:
        """
        Get the set of spouse nodes of the given node in the graph.  
        node <-> *
        """
        t = node.index
        nodes_by_src = self._cache_nodes['spouses']   
        spouses_t = nodes_by_src.setdefault(t, set())
        if spouses_t:  
            return spouses_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        spouses = set()
        for adj_node in adj_nodes:
            if self.has_bidirected_Edge(node, adj_node):
                spouses.add(adj_node)

        nodes_by_src[t] = spouses
        return spouses.copy()

    def get_district(self, node: Node) -> Set[Node]:
        """
        Get the district of the given node in the graph.  
        The district includes all nodes reachable from the given node via only bidirected edges (<->).
        """
        t = node.index
        nodes_by_src = self._cache_nodes['district']   # If not exists, automatically create an empty dict
        district_t = nodes_by_src.setdefault(t, set())
        if district_t:  # Cache exists and the cached set is not empty
            return district_t.copy()

        district = set()
        stack = [node]  # Use a stack for DFS
        visited = set()

        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                district.add(current_node)
                # Add all spouses (nodes connected by bidirected edges) to the stack
                spouses = self.get_spouse(current_node)
                stack.extend(spouses - visited)

        # Remove the node itself from the district
        district.discard(node)

        nodes_by_src[t] = district
        return district.copy()
    
    

    def get_PossibleDe(self, node: Union[str, int]) -> Set[Union[str, int]]:
        """
        A possibly directed path or possibly causal path from X to Y is a path from X to Y that does not contain an arrowhead pointing in the direction of X.
        If there is a directed (possibly directed) path from X to Y, then X is a ancestor (possible ancestor) of Y, and Y is a descendant (possible descendant) of X.
        """


        possible_de = set()
        stack = [node]
        visited = set()

        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                possible_de.add(current_node)
                # Add all adjacent nodes to the stack
                not_into_nodes = self.get_no_into_nodes(current_node)
                stack.extend(not_into_nodes - visited)

        # Remove the node itself from the possible_de set
        possible_de.discard(node)


        return possible_de



    def find_unique_triplets(self) -> List[Tuple[Node, Node, Node]]:
        """
        Efficiently find unique triplets <z, y, x> in MixGraph, avoiding symmetric duplicates.
        z.index < y.index < x.index, avoid <z, y, x> and <x, y, z>
        Returns:
            List [Tuple[Node, Node, Node]]: List of unique triplets. 
        """


        triplets = []

        for y in self.node_list:
            # Get neighbors of y
            neighbors_y = self.get_adj_nodes(y)

            if len(neighbors_y) >= 2:
                for z, x in combinations(neighbors_y, 2):
                    if z.index < x.index:
                        (z, y, x) = (z, y, x)
                    else:
                        (z, y, x) = (x, y, z)
                    
                    if (z, y, x) not in triplets:
                        triplets.append((z, y, x))


        return triplets


    ########################
    ## End Node functions ##
    ########################

    def _init_complete_graph(self):
        """
        Initialize a complete undirected graph with the given node set.
        """
        for node1, node2 in combinations(self.node_list, 2):
            self.add_circ_Edge(node1, node2)
        self.clear_cache()

    def is_complete_graph(self) -> bool:
        """
        Check if the graph is a complete graph.
        A complete graph is one where every pair of distinct nodes is connected by a unique edge.
        """
        n = self.number_of_nodes()
        expected_edges = n * (n - 1) // 2
        return self.number_of_edges() == expected_edges
    
    #############################
    ## Visualization functions ##
    #############################
    """
    Install Graphviz from https://graphviz.org/download/ (12.2.1)
    and make sure to add the Graphviz bin directory to your PATH environment variable.
    Use 'dot -V' in the command line to check if Graphviz is installed correctly.
    NOTE: (1) pip install graphviz==0.20.3 pydot==3.0.2
          (2) add the graphviz bin directory (default path: C:\Program Files\Graphviz\bin) to your PATH environment variable
    """

    def to_pydot(self, filename: str = "example_file_pydot", view: bool = False, **kwargs):
        """
        Convert the PAG to a PyDot graph object.

        Reference:
        https://github.com/py-why/causal-learn/blob/9689c1bdc468847729eacf0921b76f598161ae16/causallearn/utils/GraphUtils.py#L512
        """
        import sys
        if not (sys.version_info.major == 3 and sys.version_info.minor == 9 and sys.version_info.micro == 19):
            raise RuntimeError("Python version must be 3.9.19")

        import pydot

        # Create a new PyDot graph
        # graph_pydot = pydot.Dot(graph_type='digraph', fontsize=18)
        graph_pydot = pydot.Dot(
            graph_type='digraph',
            fontsize=10,
            # size="20,5!",       
            # size="10,0",
            # ratio="expand",     
            # dpi="300"           
        )
        latent_nodes = kwargs.get('latent_nodes', list())
        selection_bias_nodes = kwargs.get('selection_bias_nodes', list())
        is_undirected = kwargs.get('is_undirected', False)

        for node in self.nodes:
            if node.name in latent_nodes:
                graph_pydot.add_node(pydot.Node(node.name, shape='circle', style='filled', color='lightgray'))
            elif node.name in selection_bias_nodes:
                graph_pydot.add_node(pydot.Node(node.name, shape='box', style='filled', color='lightgray'))
            else:
                graph_pydot.add_node(pydot.Node(node.name, shape='circle', style='', color='black'))
            

        map_mark = {Mark.CIRCLE.value: "odot", 
                    Mark.TAIL.value: "none", 
                    Mark.ARROW.value: "normal"}

        # Add edges
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            node1 = edge_obj.start
            node2 = edge_obj.end
            head = edge_obj.rmark
            tail = edge_obj.lmark

            if is_undirected:
                # For UIG, use undirected edges
                graph_pydot.add_edge(pydot.Edge(node1.name, node2.name, 
                                      arrowtail='none',
                                      arrowhead='none',
                                      dir='none'))
            else:
                # Add directed edges with appropriate arrowheads and arrowtails
                graph_pydot.add_edge(pydot.Edge(node1.name, node2.name, 
                                        arrowtail=map_mark[tail.value],
                                        arrowhead=map_mark[head.value],
                                        dir='both'))
        # # Save the graph to a file
        # pdf_path = filename + '.pdf'
        # graph_pydot.write_pdf(pdf_path)
        # Save the graph to a file in SVG format
        svg_path = filename + '.svg'
        graph_pydot.write_svg(svg_path)
        # if view:
        #     # Automatically open the PDF file if view is True
        #     import os
        #     os.startfile(pdf_path)
        

        return graph_pydot



    #################################
    ## End Visualization functions ##
    #################################

    #####################
    ## Graph functions ##
    #####################

    def MG_to_numpy_array(self) -> np.ndarray:
        """
        Convert the graph to a NumPy adjacency matrix.
        """
        length = len(self.node_list)
        graph_matrix = np.zeros((length, length), dtype=int)
        for u, v in self.edges():
            edge = self.get_Edge(u, v)
            lmark_index = edge.lmark.value
            rmark_index = edge.rmark.value
            graph_matrix[u.index, v.index] = rmark_index
            graph_matrix[v.index, u.index] = lmark_index

        return graph_matrix
    
    def MG_to_pandas_adjacency(self) -> pd.DataFrame:
        """
        Convert the graph to a Pandas DataFrame adjacency matrix.
        """
        graph_matrix = self.MG_to_numpy_array()
        return pd.DataFrame(graph_matrix, index=[node.name for node in self.node_list], columns=[node.name for node in self.node_list])

    def to_networkx_Graph(self) -> nx.Graph:
        """
        Convert the CPDAG to a NetworkX graph that not has Node and Edge classes.
        """
        G = nx.Graph()
        edges = []
        for u, v in self.edges():
            edges.append((u.name, v.name))
        G.add_edges_from(edges)

        return G
    
    def induced_subgraph(self, nodes: List[Node]) -> "MixGraph":
        """
        Get the induced subgraph of the MixGraph on the specified nodes.

        Parameters:
        - nodes: List of Node objects to include in the induced subgraph.

        Returns:
        - A new MixGraph instance representing the induced subgraph.
        """
        subgraph = MixGraph()
        node_names = {node.name for node in nodes}
        subgraph.from_node_list(node_names)

        for u, v in combinations(subgraph.node_list, 2):
            if self.has_edge(u, v):
                edge = self.get_Edge(u, v)
                subgraph.add_Edge(u, v, edge)

        return subgraph
    
    
    def average_degree(self) -> float:
        """
        Calculate the average degree of the graph.
        """
        n = self.number_of_nodes()
        if n == 0:
            return 0.0
        total_degree = sum(dict(self.degree()).values())
        return total_degree / n

    def is_equal(self, other: "MixGraph") -> bool:
        """
        Compare this MixGraph with another MixGraph for structural and edge-mark equivalence using networkx.is_isomorphic.

        Parameters:
        - other: Another MixGraph instance.

        Returns:
        - True if both graphs are isomorphic with respect to edge marks.
        - False otherwise.
        """
        if not isinstance(other, MixGraph):
            return False

        return nx.is_isomorphic(self, other)
    
    def copy(self) -> "MixGraph":
        """
        Create a deep copy of the MixGraph, including the node_list attribute.
        """
        new_graph = MixGraph()
        # Copy the graph structure from networkx
        new_graph.add_nodes_from(self.nodes())
        new_graph.add_edges_from((u, v, attr.copy()) for (u, v, attr) in self.edges(data=True))
        # Copy the node_list
        new_graph.node_list = self.node_list.copy() if hasattr(self, 'node_list') else []
        # Copy the cache
        new_graph._cache_path = self._cache_path.copy()
        new_graph._cache_nodes = self._cache_nodes.copy()
        return new_graph
    
    #########################
    ## End Graph functions ##
    #########################


    