import numpy as np
import pandas as pd
import networkx as nx
import numpy
import pandas
import networkx
from itertools import combinations, permutations
import logging
#from causallearn.utils.cit import CIT
import pingouin as pg

from sklearn import linear_model
import collections
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import copy


from causallearn.utils.cit import CIT



from scipy.stats import pearsonr
import statsmodels.api as sm



def fisherz(data, X, Y, S):
    """
    Tests whether X is conditionally independent of Y given a conditioning set S,
    using partial correlation and a Pearson test.

    Args:
        data: pandas DataFrame containing the variables.
        X (str): name of the first variable.
        Y (str): name of the second variable.
        S (list[str] or None): list of conditioning variables.
        alpha (float): significance level threshold (default is 0.05).

    Returns:
        bool: True if X is conditionally independent of Y given S (p > alpha), False otherwise.
    """
    S = list(S)
    if S is None or len(S) == 0:
        r, pval = pearsonr(data[X], data[Y])
    else:
        # Regress X ~ S
        model_X = sm.OLS(data[X], sm.add_constant(data[S])).fit()
        resid_X = model_X.resid

        # Regress Y ~ S
        model_Y = sm.OLS(data[Y], sm.add_constant(data[S])).fit()
        resid_Y = model_Y.resid

        # Correlation of residuals = partial correlation
        r, pval = pearsonr(resid_X, resid_Y)

    return pval 

def gsq(data, X, Y, S):
    """
    Tests whether X is conditionally independent of Y given a conditioning set S,
    using the G² (likelihood ratio chi-squared) test for categorical variables.

    Args:
        data: pandas DataFrame containing the variables (assumed categorical or binary).
        X (str): name of the first variable.
        Y (str): name of the second variable.
        S (list[str] or None): list of conditioning variables.

    Returns:
        float: p-value of the conditional independence test.
    """
    # Ensure S is a list
    S = list(S) if S else []

    # Subset of variables to check for missing data
    variables_of_interest = [X, Y] + S

    # Drop rows with missing data only in the variables of interest
    data_clean = data.dropna(subset=variables_of_interest)

    # Convert cleaned DataFrame to numpy array
    data_np = data_clean.to_numpy()

    # Map variable names to column indices (in the cleaned data)
    var_idx = {var: idx for idx, var in enumerate(data_clean.columns)}

    # Create G² CIT object
    gsq_obj = CIT(data_np, "gsq")

    # Convert variable names to indices
    x_idx = var_idx[X]
    y_idx = var_idx[Y]
    s_idx = [var_idx[s] for s in S]

    # Compute p-value
    pval = gsq_obj(x_idx, y_idx, s_idx)

    return pval



def get_noise_matrix(noise_vars, num_samples):
  np = numpy

  noise_vars = np.array(noise_vars)
  num_nodes = noise_vars.shape[0]
  noise_matrix = np.zeros(shape=(num_nodes, num_samples))

  for i in range(num_nodes):
    noise_matrix[i, :] = np.random.normal(scale=np.sqrt(noise_vars[i]),
                                          size=(num_samples,))
  
  return noise_matrix

def get_dataset(mixing_matrix, noise_vars, num_samples, labels):
  np = numpy
  pd = pandas

  A = mixing_matrix
  # shape: (NUM_NODES, NUM_SAMPLES)
  noise_matrix = get_noise_matrix(noise_vars, num_samples)
  # shape: (NUM_NODES, NUM_NODES)
  B = np.linalg.inv(np.eye(A.shape[0]) - A)

  # shape: (NUM_SAMPLES, NUM_NODES)
  data = (B @ noise_matrix).T

  return pd.DataFrame(data, columns=labels)


def remove_directed_only_edges(g):
  g_ud = g.to_undirected()

  for n1, n2 in g.edges:
    if not g.has_edge(n2, n1):
      g_ud.remove_edge(n1, n2)
  
  return g_ud

def get_connected_component_with_node(g, node):
  nx = networkx
  return nx.node_connected_component(remove_directed_only_edges(g), node)


def orient_colliders(cpdag, sep_set, tmt=None):
  dag = cpdag
  node_ids = dag.nodes()
  if tmt is not None:
    tmt_neighbors = set(cpdag.to_undirected().neighbors(tmt))
    node_ids = sorted(node_ids, key=lambda n:0 if n in tmt_neighbors else 1)
  
  for (i, j) in combinations(node_ids, 2):
    adj_i = set(dag.successors(i))
    if j in adj_i:
      continue
    adj_j = set(dag.successors(j))
    if i in adj_j:
      continue
    if sep_set[i][j] is None:
      continue
    common_k = adj_i & adj_j
    for k in common_k:
      if k not in sep_set[i][j]:
        if dag.has_edge(k, i):
          dag.remove_edge(k, i)
        if dag.has_edge(k, j):
          dag.remove_edge(k, j)
  return dag

def apply_meek_rules(dag):
  nx = networkx
  
  def _has_both_edges(dag, i, j):
      return dag.has_edge(i, j) and dag.has_edge(j, i)

  def _has_any_edge(dag, i, j):
      return dag.has_edge(i, j) or dag.has_edge(j, i)

  def _has_one_edge(dag, i, j):
      return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
              (not dag.has_edge(i, j)) and dag.has_edge(j, i))

  def _has_no_edge(dag, i, j):
      return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

  # For all the combination of nodes i and j, apply the following
  # rules.
  node_ids = dag.nodes()
  old_dag = dag.copy()
  while True:
      # for (i, j) in combinations(node_ids, 2):
      for (i, j) in permutations(node_ids, 2):
          # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
          # such that k and j are nonadjacent.
          #
          # Check if i-j.
          if _has_both_edges(dag, i, j):
              # Look all the predecessors of i.
              for k in dag.predecessors(i):
                  # Skip if there is an arrow i->k.
                  if dag.has_edge(i, k):
                      continue
                  # Skip if k and j are adjacent.
                  if _has_any_edge(dag, k, j):
                      continue
                  # Make i-j into i->j
                  # self._logger.debug('R1: remove edge (%s, %s)' % (j, i))
                  dag.remove_edge(j, i)
                  break

          # Rule 2: Orient i-j into i->j whenever there is a chain
          # i->k->j.
          #
          # Check if i-j.
          if _has_both_edges(dag, i, j):
              # Find nodes k where k is i->k.
              succs_i = set()
              for k in dag.successors(i):
                  if not dag.has_edge(k, i):
                      succs_i.add(k)
              # Find nodes j where j is k->j.
              preds_j = set()
              for k in dag.predecessors(j):
                  if not dag.has_edge(j, k):
                      preds_j.add(k)
              # Check if there is any node k where i->k->j.
              if len(succs_i & preds_j) > 0:
                  # Make i-j into i->j
                  # self._logger.debug('R2: remove edge (%s, %s)' % (j, i))
                  dag.remove_edge(j, i)

          # Rule 3: Orient i-j into i->j whenever there are two chains
          # i-k->j and i-l->j such that k and l are nonadjacent.
          #
          # Check if i-j.
          if _has_both_edges(dag, i, j):
              # Find nodes k where i-k.
              adj_i = set()
              for k in dag.successors(i):
                  if dag.has_edge(k, i):
                      adj_i.add(k)
              # For all the pairs of nodes in adj_i,
              for (k, l) in combinations(adj_i, 2):
                  # Skip if k and l are adjacent.
                  if _has_any_edge(dag, k, l):
                      continue
                  # Skip if not k->j.
                  if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                      continue
                  # Skip if not l->j.
                  if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                      continue
                  # Make i-j into i->j.
                  # self._logger.debug('R3: remove edge (%s, %s)' % (j, i))
                  dag.remove_edge(j, i)
                  break

          # Rule 4: Orient i-j into i->j whenever there are two chains
          # i-k->l and k->l->j such that k and j are nonadjacent.
          #
          # However, this rule is not necessary when the PC-algorithm
          # is used to estimate a DAG.

      if nx.is_isomorphic(dag, old_dag):
          break
      old_dag = dag.copy()
    
  return dag


def to_dict(current_dict):
  new_dict = {}
  for k, v in current_dict.items():
    if isinstance(v, dict):
      new_dict[k] = dict(v)
    else:
      new_dict[k] = v
  
  return new_dict

def generate_local_dags_from(cpdag):

  def get_X_undir_edges():
    return list(set(cpdag.predecessors("X")) & set(cpdag.successors("X")))
  
  undir_nodes_X = get_X_undir_edges()
  if len(undir_nodes_X) == 0:
    yield cpdag
    return
  
  cpdag1 = cpdag.copy()
  cpdag2 = cpdag.copy()

  node = undir_nodes_X[0]

  cpdag1.remove_edge("X", node)
  apply_meek_rules(cpdag1)
  yield from generate_local_dags_from(cpdag1)

  cpdag2.remove_edge(node, "X")
  apply_meek_rules(cpdag2)
  yield from generate_local_dags_from(cpdag2)

def get_ATE_using_cov(dag, cov_matrix):
  np = numpy

  label_to_idx = {j: i for i, j in enumerate(dag.nodes())}

  nodes = ["X"] + list(dag.predecessors("X"))
  nodes = [label_to_idx[l] for l in nodes]
  
  ols_coef = np.linalg.inv(cov_matrix[nodes][:, nodes]) @ cov_matrix[label_to_idx["Y"], nodes]
  return ols_coef[0]

def get_ATE_using_nodes_and_cov(all_nodes, parents_tmt, cov_matrix):
  np = numpy
  
  label_to_idx = {j: i for i, j in enumerate(all_nodes)}

  regressors = ["X"] + list(parents_tmt)
  regressors = [label_to_idx[l] for l in regressors]
  
  ols_coef = np.linalg.inv(cov_matrix[regressors][:, regressors]) @ cov_matrix[label_to_idx["Y"], regressors]
  return ols_coef[0]

def get_all_combinations(input, non_colliders=None):
  
  def is_valid_parent_set(par):
    par = set(par)
    for p in par:
      if len(par & non_colliders[p]) != 0:
        return False
    
    return True

  res = []

  for l in range(len(input) + 1):
    for c in combinations(input, l):
      if non_colliders is not None and not is_valid_parent_set(c):
        continue
        
      res.append(list(c))
  
  return res

def get_neighbor_types(cpdag, tmt):
  parents = set()
  children = set()
  unoriented = set()

  successors = set(cpdag.successors(tmt))
  predecessors = set(cpdag.predecessors(tmt))
  
  for n in successors:
    if n in predecessors:
      unoriented |= {n}
    else:
      children |= {n}
  
  for n in predecessors:
    if n in successors:
      unoriented |= {n}
    else:
      parents |= {n}
  
  return {
      "parents": parents,
      "children": children,
      "unoriented": unoriented,
  }

def dag_to_cpdag(dag):

  def is_v_structure(dag, i, j, k):
    adj_i = set(dag.successors(i))
    adj_j = set(dag.successors(j))

    return k in adj_i and k in adj_j

  skeleton = dag.to_undirected().to_directed().copy()
  node_ids = dag.nodes()

  for (i, j) in combinations(node_ids, 2):
    adj_i = set(dag.successors(i))
    if j in adj_i:
        continue
    adj_j = set(dag.successors(j))
    if i in adj_j:
        continue
    
    common_k = adj_i & adj_j
    for k in common_k:
      if is_v_structure(dag, i, j, k):
        if skeleton.has_edge(k, i):
          skeleton.remove_edge(k, i)
        if skeleton.has_edge(k, j):
          skeleton.remove_edge(k, j)

  cpdag = apply_meek_rules(skeleton)
  return cpdag


def draw_graph(graph, highlight_nodes=set()):
  edge_color = ["black" for i, j in graph.edges()]
  node_color = ["red" if i in highlight_nodes
                else "blue" for i in graph.nodes()]
  if "edge_removed" in graph.graph:
    print("Edge removed: %s" % str(graph.graph["edge_removed"]))

  nx.draw(graph, with_labels=True,
          edge_color=edge_color,
          node_color=node_color,
          pos=nx.drawing.nx_agraph.graphviz_layout(graph),
          node_size=1000, font_color="white", arrowsize=20)
  plt.show()


def adj_matrix_to_nx_graph(A, labels, draw=False):
  nx = networkx
  
  graph = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
  graph = nx.relabel.relabel_nodes(graph, {i: labels[i] for i in range(len(labels))})

  if draw:
    draw_graph(graph)
  
  return graph