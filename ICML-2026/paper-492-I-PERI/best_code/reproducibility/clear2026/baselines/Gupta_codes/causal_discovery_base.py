import numpy as np
import pandas as pd
import networkx as nx
import numpy
import pandas
import networkx
from itertools import combinations, permutations
import logging
#from causallearn.utils.cit import CIT
from sklearn import linear_model
import collections
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import copy

from causallearn.utils.cit import CIT


from .utils_exp import *


# Adapted from https://github.com/keiichishima/pcalg/blob/master/pcalg.py.

class CausalDiscoveryBase:

  def __init__(self, treatment_node, outcome_node, alpha, use_ci_oracle, 
               graph_true, enable_logging, max_tests):
    """
    Args:
      treatment_node: The label of the treatment node.
      outcome_node: The label of the outcome node.
      alpha: The value used in conditional independence (CI) testing.
      use_ci_oracle: Whether to use an oracle for CI testing.
      graph_true: Required when `use_ci_oracle = True` to simulate the
        CI testing oracle. Else, this is not used and can be `None`.
      enable_logging: Whether to print logging info.
      max_tests: Restrict the maximum number of CI tests if not `None`.
    """
    self.alpha = alpha
    self.tmt_node = treatment_node
    self.outcome_node = outcome_node
    self.use_ci_oracle = use_ci_oracle
    self.enable_logging = enable_logging
    self.max_tests = max_tests
    self.max_tests_done = False
    
    # used to simulate an oracle for conditional independence testing.
    self.graph_true = graph_true
    if self.use_ci_oracle and self.graph_true is None:
      raise ValueError("graph_true cannot be None when use_ci_oracle=True.")

    # These variables store information about the number of CI tests performed.
    self.ci_test_calls = {
        "total": 0,
        "cov_size": collections.defaultdict(lambda: 0),
    }
    self.tests_done = []
    self.tests_done_by_size = collections.defaultdict(list)
  
  def log(self, *args):
    if self.enable_logging:
      print(args)
  
  def _partial_corr_independence_test(self, data, x, y, covar):
    """CI test for Gaussian data: tests if (x ⫫ y | covar) using Fisher's Z test."""

    self.ci_test_calls["total"] += 1
    self.ci_test_calls["cov_size"][len(covar)] += 1

    if self.max_tests is not None and self.ci_test_calls["total"] > self.max_tests:
        self.max_tests_done = True

    if self.ci_test_calls["total"] % 500 == 0:
        self.log("Tests done: %d" % self.ci_test_calls["total"])

    nodes_true = self.cols

    if self.use_ci_oracle:
        p_val = 1 if nx.d_separated(self.graph_true, {nodes_true[x]}, {nodes_true[y]},
                                    {nodes_true[i] for i in covar}) else 0
    else:
        data_np = data.to_numpy()
        fisherz_obj = CIT(data_np, "fisherz")
        p_val = fisherz_obj(x, y, covar)

    self.tests_done.append((self.cols[x], self.cols[y], 
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))
    self.tests_done_by_size[len(covar)].append((self.cols[x], self.cols[y], 
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))

    return p_val

  
  def _gsq_independence_test(self, data, x, y, covar):
    """CI test for categorical data: tests if (x ⫫ y | covar) using G² test."""

    self.ci_test_calls["total"] += 1
    self.ci_test_calls["cov_size"][len(covar)] += 1

    if self.max_tests is not None and self.ci_test_calls["total"] > self.max_tests:
        self.max_tests_done = True

    if self.ci_test_calls["total"] % 500 == 0:
        self.log("Tests done: %d" % self.ci_test_calls["total"])

    nodes_true = self.cols

    if self.use_ci_oracle:
        p_val = 1 if nx.d_separated(self.graph_true, {nodes_true[x]}, {nodes_true[y]},
                                    {nodes_true[i] for i in covar}) else 0
    else:
        # Use G² test
        data_np = data.to_numpy()
        gsq_obj = CIT(data_np, "gsq")
        p_val = gsq_obj(x, y, covar)

    self.tests_done.append((self.cols[x], self.cols[y], 
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))
    self.tests_done_by_size[len(covar)].append((self.cols[x], self.cols[y], 
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))

    return p_val
  
  def _kci_independence_test(self, data, x, y, covar):
    """
    CI test using the Kernel-based Conditional Independence (KCI) test.
    Tests if (x ⫫ y | covar) using nonparametric kernels.

    Args:
        data: pandas DataFrame with all variables.
        x (int): index of first variable (column).
        y (int): index of second variable.
        covar (list[int]): list of indices to condition on.

    Returns:
        float: p-value of the test.
    """

    # Track number of CI test calls
    self.ci_test_calls["total"] += 1
    self.ci_test_calls["cov_size"][len(covar)] += 1

    if self.max_tests is not None and self.ci_test_calls["total"] > self.max_tests:
        self.max_tests_done = True

    if self.ci_test_calls["total"] % 500 == 0:
        self.log("Tests done: %d" % self.ci_test_calls["total"])

    nodes_true = self.cols

    if self.use_ci_oracle:
        p_val = 1 if nx.d_separated(self.graph_true, {nodes_true[x]}, {nodes_true[y]},
                                    {nodes_true[i] for i in covar}) else 0
    else:
        # Use the causallearn CIT interface with the "kci" method
        data_np = data.to_numpy()
        kci_obj = CIT(data_np, "kci")
        p_val = kci_obj(x, y, covar)

    # Log the test result
    self.tests_done.append((self.cols[x], self.cols[y],
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))
    self.tests_done_by_size[len(covar)].append((self.cols[x], self.cols[y],
                            {self.cols[i] for i in covar}, p_val > self.alpha, p_val))

    return p_val

  
  def _create_complete_graph(self, node_ids):
    """Creates a complete graph from the list of node ids."""
    nx = networkx

    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
      g.add_edge(i, j)
    return g


class SequentialLocalCausalDiscoveryBase(CausalDiscoveryBase):
  """Base class for the SD and MB-by-MB algorithms."""

  def __init__(self, treatment_node, outcome_node, alpha, use_ci_oracle, 
               graph_true, enable_logging, max_tests, local_structure_learning_func):
    """
    Args:
      local_structure_learning_func: The function used to perform the local
        structure learning in Sequential Discovery.
    """
    super(SequentialLocalCausalDiscoveryBase, self).__init__(
        treatment_node=treatment_node, outcome_node=outcome_node, alpha=alpha, 
        use_ci_oracle=use_ci_oracle, graph_true=graph_true, 
        enable_logging=enable_logging, max_tests=max_tests)
    self.local_structure_learning_func = local_structure_learning_func
    
  def run(self, data):
    nx = networkx
    pd = pandas
    
    def relabel_graph(g, labels):
      return nx.relabel_nodes(g, {i: labels[i] for i in range(len(labels))})
    
    def to_labels(nodes):
      return {self.cols[i] for i in nodes}
    
    def get_parents_using_nbrs(skeleton, sep_set):
      neighbors_X = set(skeleton.neighbors(tmt_id))
      parents = set()

      for (i, j) in combinations(neighbors_X, 2):
        if skeleton.has_edge(i, j):
          continue
        
        if tmt_id not in sep_set[i][j]:
          self.log("Marking parents: (%s, %s) | %s" % (cols[i], cols[j], to_labels(sep_set[i][j])))
          parents |= {i, j}
        
      return parents
    
    def get_non_collider_nodes(neighbors, non_colliders):
      res = set()
      for n in neighbors:
        res |= non_colliders[n]
      return res
    
    def update_subgraph(g, sub_g):
      g = g.copy()
      for (i, j) in sub_g.edges:
        if g.has_edge(i, j):
          g.remove_edge(i, j)
        if g.has_edge(j, i):
          g.remove_edge(j, i)
      
      for (i, j) in sub_g.edges:
        g.add_edge(i, j)
      
      return g
    
    tmt = self.tmt_node
    cols = data.columns
    self.cols = cols
    tmt_id = list(cols).index(tmt)
    self.tmt_id = tmt_id
    sep_set_dict = collections.defaultdict(dict)
    data_matrix = pd.DataFrame(data=data.values, columns=range(data.values.shape[1]))

    wait_queue = [tmt_id]
    all_queued = set(wait_queue)
    to_be_oriented = set()
    tests_done = set()
    nodes_done = set()
    non_colliders = collections.defaultdict(set)
    skeleton = None
    sep_set = None
    cpdag = None
    while len(wait_queue) > 0:
      node_to_check = wait_queue.pop(0)
      skeleton, sep_set = self.local_structure_learning_func(skeleton, sep_set,
                                                             node_to_check,
                                                             data_matrix)
      cmb = set(skeleton.neighbors(node_to_check))
      nodes_done |= {node_to_check}
      
      self.log("Tests after MB(%s): %s" % (self.cols[node_to_check], str(self.ci_test_calls)))
      self.log("CMB: %s" % str(to_labels(cmb)))

      for n in cmb:
        if n in all_queued:
          continue
        
        wait_queue.append(n)
        all_queued |= {n}

      for (i, j) in combinations(list(skeleton.neighbors(tmt_id)), 2):
        if tmt_id in sep_set[i][j]:
          self.log("Marking non collider: %s - %s - %s" % (self.cols[i], self.cols[tmt_id], self.cols[j]))
          non_colliders[i] |= {j}
          non_colliders[j] |= {i}

      cpdag_sub = orient_colliders(skeleton.subgraph(nodes_done).to_directed(), sep_set, tmt=tmt_id)
      cpdag_sub = apply_meek_rules(cpdag_sub)
      cpdag = update_subgraph(skeleton.to_directed(), cpdag_sub)
      
      parents = set(cpdag.predecessors(tmt_id)) - set(cpdag.successors(tmt_id))
      children = set(cpdag.successors(tmt_id)) - set(cpdag.predecessors(tmt_id))
      children |= get_non_collider_nodes(parents, non_colliders)

      to_be_oriented = set(skeleton.neighbors(tmt_id)) - parents - children
      
      self.log("parents: %s" % (str(to_labels(parents))))
      self.log("children: %s" % (str(to_labels(children))))
      self.log("to_be_oriented: %s" % (str(to_labels(to_be_oriented))))

      if len(to_be_oriented) == 0:
        break

      if self.max_tests_done:
        break
      
      tmt_chain_component = get_connected_component_with_node(cpdag, tmt_id)
      wait_queue = [w for w in wait_queue if w in tmt_chain_component]

      self.log("Wait queue: %s" % str(to_labels(wait_queue)))

    parents |= get_parents_using_nbrs(skeleton, sep_set)
    to_be_oriented -= parents
    
    self.log("parents: %s" % (str(to_labels(parents))))
    self.log("children: %s" % (str(to_labels(children))))
    self.log("to_be_oriented: %s" % (str(to_labels(to_be_oriented))))
    
    return {
        "tmt_parents": set(self.cols[o] for o in parents),
        "tmt_children": set(self.cols[o] for o in children),
        "unoriented": set(self.cols[o] for o in to_be_oriented),
        "non_colliders": (
            {self.cols[key]: {self.cols[v] for v in non_colliders[key]} 
             for key in parents | children | to_be_oriented}),
    }