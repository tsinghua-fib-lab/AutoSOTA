from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edges import Edges
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import numpy as np
from itertools import combinations
from causallearn.search.ConstraintBased.FCI import get_color_edges

def FCI_orientations(graph, dataset, sep_sets, nodes, 
                     independence_test_method,
                     background_knowledge, max_path_length = -1,
                        ambiguous_triple = [],
                     alpha = 0.05, verbose=False):
    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, ambiguous_triple, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, ambiguous_triple, sep_sets, background_knowledge,change_flag, verbose)

        # if change_flag or (first_time and background_knowledge is not None and
        #                    len(background_knowledge.forbidden_rules_specs) > 0 and
        #                    len(background_knowledge.required_rules_specs) > 0 and
        #                    len(background_knowledge.tier_map.keys()) > 0):
        #     change_flag = ruleR4B(graph, ambiguous_triple, max_path_length, dataset, independence_test_method, alpha, sep_sets,
        #                           change_flag,
        #                           background_knowledge, verbose)

        #     first_time = False

        #     if verbose:
        #         print("Epoch")
        
                    # rule 8
        change_flag = rule8(graph,nodes)
        # rule 9
        change_flag = rule9(graph, nodes)
        # rule 10
        change_flag = rule10(graph)

    # graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges

def fci_orient_bk(bk: BackgroundKnowledge | None, graph: Graph):
    if bk is None:
        return
    print("Starting BK Orientation.")
    edges = graph.get_graph_edges()
    for edge in edges:
        if bk.is_forbidden(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_forbidden(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_required(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_required(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
    print("Finishing BK Orientation.")


def is_arrow_point_allowed(node_x: Node, node_y: Node, graph: Graph, knowledge: BackgroundKnowledge | None) -> bool:
    if graph.get_endpoint(node_x, node_y) == Endpoint.ARROW:
        return True
    if graph.get_endpoint(node_x, node_y) == Endpoint.TAIL:
        return False
    if graph.get_endpoint(node_y, node_x) == Endpoint.ARROW:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    if graph.get_endpoint(node_y, node_x) == Endpoint.TAIL:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    return graph.get_endpoint(node_x, node_y) == Endpoint.CIRCLE


def rule8(graph: Graph, nodes: List[Node]):
    """
    [R8] If A ---> B ---> C or A ---o B ---> C and A o--> C then orient tail A ---> C

    :return: True if the graph was modified; otherwise False
    """
    nodes = graph.get_nodes()
    changeFlag = False
    for node_B in nodes:
        adj = graph.get_adjacent_nodes(node_B)
        if len(adj) < 2:
            continue

        cg = ChoiceGenerator(len(adj), 2)
        combination = cg.next()

        while combination is not None:
            node_A = adj[combination[0]]
            node_C = adj[combination[1]]
            combination = cg.next()

            # if (node_A, node_B, node_C) in  ambiguous_triple or (node_C, node_B, node_A) in ambiguous_triple:
            #     continue
            
            # we want either 1.) A->B->C and A o->C  or 2.) A-oB->C  and Ao->C 
            if(graph.get_endpoint(node_A, node_B) == Endpoint.ARROW and graph.get_endpoint(node_B, node_A) == Endpoint.TAIL and \
                graph.get_endpoint(node_B, node_C) == Endpoint.ARROW and graph.get_endpoint(node_C, node_B) == Endpoint.TAIL and \
                    graph.is_adjacent_to(node_A, node_C) and \
                        graph.get_endpoint(node_A, node_C) == Endpoint.ARROW and graph.get_endpoint(node_C, node_A)== Endpoint.CIRCLE) or \
                        (graph.get_endpoint(node_A, node_B) == Endpoint.CIRCLE and graph.get_endpoint(node_B, node_A) == Endpoint.TAIL and \
                graph.get_endpoint(node_B, node_C) == Endpoint.ARROW and graph.get_endpoint(node_C, node_B) == Endpoint.TAIL and \
                    graph.is_adjacent_to(node_A, node_C) and \
                        graph.get_endpoint(node_A, node_C) == Endpoint.ARROW and graph.get_endpoint(node_C, node_A)== Endpoint.CIRCLE):
                edge1 = graph.get_edge(node_A, node_C)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_A, node_C,Endpoint.TAIL, Endpoint.ARROW))
                changeFlag = True

    return changeFlag


def is_possible_parent(graph: Graph, potential_parent_node, child_node):
    """
    Test if a node can possibly serve as parent of the given node.
    Make sure that on the connecting edge
        (a) there is no head edge-mark (->) at the tested node and
        (b) there is no tail edge-mark (--) at the given node,
    where variant edge-marks (o) are allowed.
    :param potential_parent_node: the node that is being tested
    :param child_node: the node that serves as the child
    :return:
     """
    if graph.node_map[potential_parent_node] == graph.node_map[child_node]:
        return False
    if not graph.is_adjacent_to(potential_parent_node, child_node):
        return False

    if graph.get_endpoint(child_node, potential_parent_node) == Endpoint.ARROW:
        return False
    else:
        return True

        
def find_possible_children(graph: Graph, parent_node, en_nodes=None):
    if en_nodes is None:
        nodes = graph.get_nodes()
        en_nodes = [node for node in nodes if graph.node_map[node] != graph.node_map[parent_node]]

    potential_child_nodes = set()
    for potential_node in en_nodes:
        if is_possible_parent(graph, potential_parent_node=parent_node, child_node=potential_node):
            potential_child_nodes.add(potential_node)

    return potential_child_nodes



def find_possible_parents(graph: Graph, child_node, en_nodes=None):
    if en_nodes is None:
        nodes = graph.get_nodes()
        en_nodes = [node for node in nodes if graph.node_map[node] != graph.node_map[child_node]]

    possible_parents = [parent_node for parent_node in en_nodes if is_possible_parent(graph, parent_node, child_node)]
    
    return possible_parents


def traverseSemiDirected(node: Node, edge: Edge) -> Node | None:
    if node == edge.get_node1():
        if edge.get_endpoint1() == Endpoint.TAIL or edge.get_endpoint1() == Endpoint.CIRCLE:
            return edge.get_node2()
    elif node == edge.get_node2():
        if edge.get_endpoint2() == Endpoint.TAIL or edge.get_endpoint2() == Endpoint.CIRCLE:
            return edge.get_node1()
    return None

def existsSemidirectedPath(node_from: Node, node_to: Node, G: Graph) -> bool:
    Q = Queue()
    V = set()
    Q.put(node_from)
    V.add(node_from)
    while not Q.empty():
        node_t = Q.get_nowait()
        if node_t == node_to:
            return True

        for node_u in G.get_adjacent_nodes(node_t):
            edge = G.get_edge(node_t, node_u)
            node_c = traverseSemiDirected(node_t, edge)

            if node_c is None:
                continue

            if V.__contains__(node_c):
                continue

            V.add(node_c)
            Q.put(node_c)

    return False

def rule9(graph: Graph, nodes: List[Node]):
    """
    [R9] If A o--> C and there is a possibly directed uncovered path <A, B, ..., D, C>, B and C are not connected
            then orient tail A ---> C.

    :return: True if the graph was modified; otherwise False
    """

    changeFlag = False
    nodes = graph.get_nodes()
    for node_C in nodes:
        intoCArrows = graph.get_nodes_into(node_C, Endpoint.ARROW)
        for node_A in intoCArrows:
            # we want A o--> C
            if not graph.get_endpoint(node_C, node_A) == Endpoint.CIRCLE:
                continue
        
            # look for a possibly directed uncovered path s.t. B and C are not connected (for the given A o--> C
            a_node_idx = graph.node_map[node_A]
            c_node_idx = graph.node_map[node_C]
            a_adj_nodes = graph.get_adjacent_nodes(node_A)
            nodes_set = [node for node in a_adj_nodes if graph.node_map[node] != a_node_idx and graph.node_map[node]!= c_node_idx]
            possible_children = find_possible_children(graph, node_A, nodes_set)
            for node_B in possible_children:
                if graph.is_adjacent_to(node_B, node_C):
                    continue
                if existsSemidirectedPath(node_from=node_B, node_to=node_C, G=graph):
                    edge1 = graph.get_edge(node_A, node_C)
                    graph.remove_edge(edge1)
                    graph.add_edge(Edge(node_A, node_C, Endpoint.TAIL, Endpoint.ARROW))
                    changeFlag = True
                    break #once we found it, break out since we have already oriented Ao->C to A->C, we want to find the next A 
    return changeFlag


def rule10(graph: Graph):
    """
    [R10] If A o--> C and B ---> C <---D, and if there are two possibly directed uncovered paths
    <A, E, ..., B>, <A, F, ..., D> s.t. E, F are disconnected, and any of these paths can be a single-edge path,
    A o--> B or A o--> D, then orient tail A ---> C.

    :return: True if the graph was modified; otherwise False
    """
    changeFlag = False
    nodes = graph.get_nodes()
    for node_C in nodes:
        intoCArrows = graph.get_nodes_into(node_C, Endpoint.ARROW)
        if len(intoCArrows) < 2:
                continue
        # get all A where A o-> C
        Anodes = [node_A for node_A in intoCArrows if graph.get_endpoint(node_C, node_A) == Endpoint.CIRCLE]
        if len(Anodes) == 0:
            continue
        
        for node_A in Anodes:
            A_adj_nodes = graph.get_adjacent_nodes(node_A)
            en_nodes = [i for i in A_adj_nodes if i is not node_C]
            A_possible_children = find_possible_children(graph, parent_node=node_A, en_nodes=en_nodes)
            if len(A_possible_children) < 2:
                continue

            gen = ChoiceGenerator(len(intoCArrows), 2)
            choice = gen.next()
            while choice is not None:
                node_B = intoCArrows[choice[0]]
                node_D = intoCArrows[choice[1]]

                choice = gen.next()
                # we want B->C<-D 
                if graph.get_endpoint(node_C, node_B) != Endpoint.TAIL:
                    continue

                if graph.get_endpoint(node_C, node_D) != Endpoint.TAIL:
                    continue

                for children in combinations(A_possible_children, 2):
                    child_one, child_two = children
                    if not existsSemidirectedPath(node_from=child_one, node_to=node_B, G=graph) or \
                        not existsSemidirectedPath(node_from=child_two, node_to=node_D, G=graph):
                        continue

                    if not graph.is_adjacent_to(child_one, child_two):
                        edge1 = graph.get_edge(node_A, node_C)
                        graph.remove_edge(edge1)
                        graph.add_edge(Edge(node_A, node_C, Endpoint.TAIL, Endpoint.ARROW))
                        changeFlag = True
                        break #once we found it, break out since we have already oriented Ao->C to A->C, we want to find the next A 

    return changeFlag


def rule0(graph: Graph, nodes: List[Node], sep_sets: Dict[Tuple[int, int], Set[int]],
          ambiguous_triple: List[Tuple], 
          knowledge: BackgroundKnowledge | None,
          verbose: bool):
    reorientAllWith(graph, Endpoint.CIRCLE)
    fci_orient_bk(knowledge, graph)
    for node_b in nodes:
        adjacent_nodes = graph.get_adjacent_nodes(node_b)
        if len(adjacent_nodes) < 2:
            continue

        cg = ChoiceGenerator(len(adjacent_nodes), 2)
        combination = cg.next()
        while combination is not None:
            node_a = adjacent_nodes[combination[0]]
            node_c = adjacent_nodes[combination[1]]

            combination = cg.next()

            if graph.is_adjacent_to(node_a, node_c):
                continue
            if graph.is_def_collider(node_a, node_b, node_c):
                continue
            
            if (node_a, node_c, node_b) in ambiguous_triple or (node_b, node_c, node_a) in ambiguous_triple:
                continue

            # check if is collider
            if not (graph.get_node_map()[node_a], graph.get_node_map()[node_c]) in sep_sets.keys():
                sep_set = None
            else:
                sep_set = sep_sets.get((graph.get_node_map()[node_a], graph.get_node_map()[node_c]))
            if sep_set is not None and not sep_set.__contains__(graph.get_node_map()[node_b]):
                if not is_arrow_point_allowed(node_a, node_b, graph, knowledge):
                    continue
                if not is_arrow_point_allowed(node_c, node_b, graph, knowledge):
                    continue

                edge1 = graph.get_edge(node_a, node_b)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

                edge2 = graph.get_edge(node_c, node_b)
                graph.remove_edge(edge2)
                graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

                if verbose:
                    print(
                        "Orienting collider: " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())


def reorientAllWith(graph: Graph, endpoint: Endpoint):
    # reorient all edges with CIRCLE Endpoint
    ori_edges = graph.get_graph_edges()
    for ori_edge in ori_edges:
        graph.remove_edge(ori_edge)
        ori_edge.set_endpoint1(endpoint)
        ori_edge.set_endpoint2(endpoint)
        graph.add_edge(ori_edge)


def ruleR1(node_a: Node, node_b: Node, node_c: Node, graph: Graph, ambiguous_triple: List[Tuple], bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose: bool = False) -> bool:
    if graph.is_adjacent_to(node_a, node_c):
        return changeFlag
    
    if (node_a, node_b, node_c) in  ambiguous_triple or (node_c, node_b, node_a) in ambiguous_triple:
        return changeFlag

    if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and graph.get_endpoint(node_c, node_b) == Endpoint.CIRCLE:
        if not is_arrow_point_allowed(node_b, node_c, graph, bk):
            return changeFlag

        edge1 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_c, node_b, Endpoint.ARROW, Endpoint.TAIL))

        changeFlag = True

        if verbose:
            print("Orienting edge (Away from collider):" + graph.get_edge(node_b, node_c).__str__())

    return changeFlag


def ruleR2(node_a: Node, node_b: Node, node_c: Node, graph: Graph, bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose=False) -> bool:
    if graph.is_adjacent_to(node_a, node_c) and graph.get_endpoint(node_a, node_c) == Endpoint.CIRCLE:
        if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and \
                graph.get_endpoint(node_b, node_c) == Endpoint.ARROW and \
                (graph.get_endpoint(node_b, node_a) == Endpoint.TAIL or
                 graph.get_endpoint(node_c, node_b) == Endpoint.TAIL):
            if not is_arrow_point_allowed(node_a, node_c, graph, bk):
                return changeFlag

            edge1 = graph.get_edge(node_a, node_c)
            graph.remove_edge(edge1)
            graph.add_edge(Edge(node_a, node_c, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

            if verbose:
                print("Orienting edge (Away from ancestor): " + graph.get_edge(node_a, node_c).__str__())

            changeFlag = True

    return changeFlag


def rulesR1R2cycle(graph: Graph, ambiguous_triple: List[Tuple],
                   bk: BackgroundKnowledge | None, changeFlag: bool, verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        adj = graph.get_adjacent_nodes(node_B)

        if len(adj) < 2:
            continue

        cg = ChoiceGenerator(len(adj), 2)
        combination = cg.next()

        while combination is not None:
            node_A = adj[combination[0]]
            node_C = adj[combination[1]]
            combination = cg.next()

            changeFlag = ruleR1(node_A, node_B, node_C, graph, ambiguous_triple, bk, changeFlag, verbose)
            changeFlag = ruleR1(node_C, node_B, node_A, graph, ambiguous_triple, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_C, node_B, node_A, graph, bk, changeFlag, verbose)

    return changeFlag


def isNoncollider(graph: Graph, sep_sets: Dict[Tuple[int, int], Set[int]], node_i: Node, node_j: Node,
                  node_k: Node) -> bool:
    x = (graph.get_node_map()[node_i], graph.get_node_map()[node_k])
    y = (graph.get_node_map()[node_k], graph.get_node_map()[node_i])
    if sep_sets.keys().__contains__(x):
        sep_set = sep_sets[x]
        return sep_set is not None and sep_set.__contains__(graph.get_node_map()[node_j])
    elif sep_sets.keys().__contains__(y):
        sep_set = sep_sets[y]
        return sep_set is not None and sep_set.__contains__(graph.get_node_map()[node_j])
    else:
        print(sep_sets)
        print("Something is wrong; there is no separating set for the nonAdj pair:{} or {}!".format(x, y))



def ruleR3(graph: Graph, ambiguous_triple: List[Tuple], 
           sep_sets: Dict[Tuple[int, int], Set[int]], 
           bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        intoBArrows = graph.get_nodes_into(node_B, Endpoint.ARROW)
        intoBCircles = graph.get_nodes_into(node_B, Endpoint.CIRCLE)

        for node_D in intoBCircles:
            if len(intoBArrows) < 2:
                continue
            gen = ChoiceGenerator(len(intoBArrows), 2)
            choice = gen.next()

            while choice is not None:
                node_A = intoBArrows[choice[0]]
                node_C = intoBArrows[choice[1]]
                choice = gen.next()

                if graph.is_adjacent_to(node_A, node_C):
                    continue

                if not graph.is_adjacent_to(node_A, node_D) or graph.is_adjacent_to(node_C, node_D):
                    continue

                if not isNoncollider(graph, sep_sets, node_A, node_D, node_C):
                    continue

                if graph.get_endpoint(node_A, node_D) != Endpoint.CIRCLE:
                    continue

                if graph.get_endpoint(node_C, node_D) != Endpoint.CIRCLE:
                    continue

                if not is_arrow_point_allowed(node_D, node_B, graph, bk):
                    continue
                
                if (node_A, node_D, node_C) in ambiguous_triple or (node_C, node_D, node_A) in ambiguous_triple:
                    continue 

                edge1 = graph.get_edge(node_D, node_B)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_D, node_B, edge1.get_proximal_endpoint(node_D), Endpoint.ARROW))

                if verbose:
                    print("Orienting edge (Double triangle): " + graph.get_edge(node_D, node_B).__str__())

                changeFlag = True
    return changeFlag


def getPath(node_c: Node, previous) -> List[Node]:
    l = []
    node_p = previous[node_c]
    if node_p is not None:
        l.append(node_p)
    while node_p is not None:
        node_p = previous.get(node_p)
        if node_p is not None:
            l.append(node_p)
    return l


def doDdpOrientation(node_d: Node, node_a: Node, node_b: Node, node_c: Node, ambiguous_triple: List[Tuple], previous, graph: Graph, data,
                     independence_test_method, alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]],
                     change_flag: bool, bk, verbose: bool = False, partial_order = None) -> (bool, bool):
    if graph.is_adjacent_to(node_d, node_c):
        raise Exception("illegal argument!")
    path = getPath(node_d, previous)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn] for nn in path])
    p_value = independence_test_method(X, Y, condSet)
    ind = p_value > alpha

    path2 = list(path)
    path2.remove(node_b)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn2] for nn2 in path2])
    p_value2 = independence_test_method(X, Y, condSet)
    ind2 = p_value2 > alpha

    if not ind and not ind2:
        sep_set = sep_sets.get((graph.get_node_map()[node_d], graph.get_node_map()[node_c]))
        if verbose:
            message = "Sepset for d = " + node_d.get_name() + " and c = " + node_c.get_name() + " = [ "
            if sep_set is not None:
                for ss in sep_set:
                    message += graph.get_nodes()[ss].get_name() + " "
            message += "]"
            print(message)

        if sep_set is None:
            if verbose:
                print(
                    "Must be a sepset: " + node_d.get_name() + " and " + node_c.get_name() + "; they're non-adjacent.")
            return False, change_flag

        ind = sep_set.__contains__(graph.get_node_map()[node_b])

    if ind:
        if (node_a, node_b, node_c) in ambiguous_triple or (node_c, node_b, node_a) in ambiguous_triple:
            return False, change_flag
        
        edge = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge)
        graph.add_edge(Edge(node_c, node_b, edge.get_proximal_endpoint(node_c), Endpoint.TAIL))

        if verbose:
            print(
                "Orienting edge (Definite discriminating path d = " + node_d.get_name() + "): " + graph.get_edge(node_b,
                                                                                                                 node_c).__str__())

        change_flag = True
        return True, change_flag
    else:
        if not is_arrow_point_allowed(node_a, node_b, graph, partial_order):
            return False, change_flag

        if not is_arrow_point_allowed(node_c, node_b, graph, partial_order):
            return False, change_flag
        
        if (node_a, node_b, node_c) in ambiguous_triple or (node_c, node_b, node_a) in ambiguous_triple:
            return False, change_flag
        edge1 = graph.get_edge(node_a, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

        edge2 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge2)
        graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

        if verbose:
            print(
                "Orienting collider (Definite discriminating path.. d = " + node_d.get_name() + "): " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())

        change_flag = True
        return True, change_flag


def ddpOrient(node_a: Node, node_b: Node, node_c: Node, ambiguous_triple: List[Tuple], graph: Graph, maxPathLength: int, data: ndarray,
              independence_test_method, alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]], change_flag: bool,
              bk: BackgroundKnowledge | None, verbose: bool = False) -> bool:
    Q = Queue()
    V = set()
    e = None
    distance = 0
    previous = {}

    cParents = graph.get_parents(node_c)

    Q.put(node_a)
    V.add(node_a)
    V.add(node_b)
    previous[node_a] = node_b

    while not Q.empty():
        node_t = Q.get_nowait()

        if e is None or e == node_t:
            e = node_t
            distance += 1
            if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                return change_flag

        nodesInTo = graph.get_nodes_into(node_t, Endpoint.ARROW)

        for node_d in nodesInTo:
            if V.__contains__(node_d):
                continue

            previous[node_d] = node_t
            node_p = previous[node_t]

            if not graph.is_def_collider(node_d, node_t, node_p):
                continue

            previous[node_d] = node_t

            if not graph.is_adjacent_to(node_d, node_c) and node_d != node_c:
                res, change_flag = \
                    doDdpOrientation(node_d, node_a, node_b, node_c, ambiguous_triple, previous, graph, data,
                                     independence_test_method, alpha, sep_sets, change_flag, bk, verbose)

                if res:
                    return change_flag

            if cParents.__contains__(node_d):
                Q.put(node_d)
                V.add(node_d)
    return change_flag


def ruleR4B(graph: Graph, ambiguous_triple: List[Tuple], maxPathLength: int, data: ndarray, independence_test_method, alpha: float,
            sep_sets: Dict[Tuple[int, int], Set[int]],
            change_flag: bool, bk: BackgroundKnowledge | None,
            verbose: bool = False) -> bool:
    nodes = graph.get_nodes()

    for node_b in nodes:
        possA = graph.get_nodes_out_of(node_b, Endpoint.ARROW)
        possC = graph.get_nodes_into(node_b, Endpoint.CIRCLE)

        for node_a in possA:
            for node_c in possC:
                if not graph.is_parent_of(node_a, node_c):
                    continue

                if graph.get_endpoint(node_b, node_c) != Endpoint.ARROW:
                    continue

                change_flag = ddpOrient(node_a, node_b, node_c, ambiguous_triple, graph, maxPathLength, data, independence_test_method,
                                        alpha, sep_sets, change_flag, bk, verbose)
    return change_flag


def existsBoundedSemiDirectedPath(node_from: Node, node_to: Node, bound: int, graph: Graph) -> bool:
    Q = Queue()
    V = set()
    Q.put(node_from)
    V.add(node_from)
    node_e = None
    distance = 0

    while not Q.empty():
        node_t = Q.get_nowait()
        if node_t == node_to:
            return True

        if node_e == node_t:
            node_e = None
            distance += 1
            if distance > (1000 if bound == -1 else bound):
                return False

        for node_u in graph.get_adjacent_nodes(node_t):
            edge = graph.get_edge(node_t, node_u)
            node_c = traverseSemiDirected(node_t, edge)

            if node_c is None:
                continue

            if node_c == node_to:
                return True

            if not V.__contains__(node_c):
                V.add(node_c)
                Q.put(node_c)

                if node_e is None:
                    node_e = node_u

    return False




def kPC_orientations(G,n):
    if n is None:
        n = G.graph.shape[0]

    adj = G.graph
    #################################

    # Find nodes with no incoming edges:
    # parse rows
    candidate_nodes = [i for i in range(n) if not (adj[i, :] == 1).any()]

    # print(candidate_nodes)

    # Now for each one of those nodes, find B and C sets:
    B = {my_node: [i for i in np.where(adj[my_node, :] == 2)[0] if adj[i, my_node] == 1] for my_node in candidate_nodes}
    # print(B)
    C = {my_node: [i for i in np.where(adj[my_node, :] == 2)[0] if adj[i, my_node] == 2] for my_node in candidate_nodes}

    # print(C)

    # C* is subset of C that only contains nodes that are non-adjacent to the other nodes in C
    C_star = {}
    for my_node in candidate_nodes:
        C_star[my_node] = []
        C_my_node = C[my_node]
        for i in C_my_node:
            neigh = np.where(adj[i, :] != 0)[0]
            if all(j == my_node or j not in C_my_node for j in neigh):
                C_star[my_node].append(i)

    # print(C_star)

    # B* is subset of B that only contains nodes that are non-adjacent to any of the nodes in C
    B_star = {}
    for my_node in candidate_nodes:
        B_star[my_node] = []
        B_my_node = B[my_node]
        C_my_node = C[my_node]
        for i in B_my_node:
            neigh = np.where(adj[i, :] != 0)[0]
            if all(j == my_node or j not in C_my_node for j in neigh):
                B_star[my_node].append(i)
    # print(B_star)


    # re-orient them
    new_adj=adj
    for node in candidate_nodes:
        for i in B_star[node]:
            new_adj[node,i]=-1
        for i in C_star[node]:
            new_adj[node,i]=-1
            new_adj[i,node]=-1
    return new_adj  
#######################

def make_kess_graph(new_adj,n, data_names = None, nodes=None):
    if nodes is None:
        nodes=[]
        if data_names:
            for name in data_names:
                nodes.append(GraphNode(name))
        else:
            for i in np.arange(1,n+1):
                nodes.append(GraphNode('X'+str(i)))
    D = GeneralGraph(nodes)  
    for i in range(n):
        #neigh=new_adj[i,:]
        for j in range(n):
            if new_adj[i,j]==0:
                pass
            else:
                D.add_edge(Edge(nodes[i],nodes[j],Endpoint(new_adj[i,j]),Endpoint(new_adj[j,i])))

    D.pag=True
    return D

