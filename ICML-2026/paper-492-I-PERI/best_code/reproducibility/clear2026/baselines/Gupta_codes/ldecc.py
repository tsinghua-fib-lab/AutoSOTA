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

from .utils_exp import *
from .causal_discovery_base import CausalDiscoveryBase



class MNS:
  """Instantiates a Minimal Neighbor Separator (MNS)."""

  def __init__(self, is_valid, mns=None):
    self._is_valid = is_valid
    self._mns = mns
  
  def __str__(self):
    if not self._is_valid:
      return "Invalid MNS"
    else:
      return str(to_labels(self._mns))
  
  def is_valid(self):
    return self._is_valid
  
  def mns(self):
    if not self.is_valid:
      raise ValueError("MNS is not valid")
    
    return self._mns
  
  @staticmethod
  def equals(m1, m2):
    if (not m1.is_valid()) or (not m2.is_valid()):
      return False
    
    return m1.mns() == m2.mns()


class LDECCAlgorithm(CausalDiscoveryBase):
  def __init__(self, treatment_node="X", outcome_node="Y", alpha=0.05, 
                use_ci_oracle=True, graph_true=None, enable_logging=False,
                ldecc_do_checks=False, max_tests=None, ci_test="fisherz"):
      super(LDECCAlgorithm, self).__init__(treatment_node=treatment_node,
                                          outcome_node=outcome_node, alpha=alpha,
                                          use_ci_oracle=use_ci_oracle,
                                          graph_true=graph_true,
                                          enable_logging=enable_logging,
                                          max_tests=max_tests)
      self.nb_ci_tests = 0
      self.mns_cache = collections.defaultdict(lambda: None)
      self.ldecc_do_checks = ldecc_do_checks
      # New attribute: function for conditional independence test
      if ci_test == "fisherz":
        self.ci_test_func = self._partial_corr_independence_test
      elif ci_test == "gsq":
        self.ci_test_func = self._gsq_independence_test
      elif ci_test == "kci":
        self.ci_test_func = self._kci_independence_test

  
  def relabel_graph(g, labels):
      return nx.relabel_nodes(g, {i: labels[i] for i in range(len(labels))})

  def run(self, data):
    cols = data.columns
    self.cols = cols
    return self._run_ldecc(self.ci_test_func,
                           pd.DataFrame(data=data.values, columns=range(data.values.shape[1])),
                           alpha=self.alpha,
                           treatment_node_id=list(cols).index(self.tmt_node),
                           outcome_node_id=list(cols).index(self.outcome_node))

    
  
  def _run_ldecc(self, indep_test_func, data_matrix, alpha, treatment_node_id,
                 outcome_node_id):
    
      nx = networkx

      tmt = treatment_node_id
      outcome = outcome_node_id
      self.tmt_id = tmt
      
      def to_labels(nodes):
        return {self.cols[i] for i in nodes}

      def is_independent(i, j, k, ret_p_val=False):
        p_val = indep_test_func(data_matrix, i, j, k)
        self.nb_ci_tests += 1
        is_indep = p_val > alpha

        if ret_p_val:
          return is_indep, p_val
        else:
          return is_indep
      
      def find_markov_blanket(g, sep_set, node_ids, main_node_id):
        g = g.copy()
        sep_set = copy.deepcopy(sep_set)
        markov_blanket = set()

        # forward pass.
        cont = True
        while cont:
          cont = False
          mb_copy = set(markov_blanket)
          nodes_to_check = (node_ids - {main_node_id} - mb_copy)
          for n in nodes_to_check:
            if not is_independent(n, main_node_id, markov_blanket - {n}):
              markov_blanket |= {n}
              cont = True
            elif g.has_edge(n, main_node_id):
              g.remove_edge(n, main_node_id)
              sep_set[n][main_node_id] = markov_blanket - {n}
              sep_set[main_node_id][n] = markov_blanket - {n}
          
        # backward pass.
        mb_copy = set(markov_blanket)
        for n in mb_copy:
          if is_independent(n, main_node_id, markov_blanket - {n}):
            if g.has_edge(n, main_node_id):
              g.remove_edge(n, main_node_id)
              sep_set[n][main_node_id] = markov_blanket - {n}
              sep_set[main_node_id][n] = markov_blanket - {n}
            
            markov_blanket -= {n}
        
        return g, sep_set, markov_blanket

      def do_local_pc(g, sep_set, markov_blanket):
        g = g.copy()
        sep_set = copy.deepcopy(sep_set)
        tmt = treatment_node_id

        def get_sepset(node, neighbors_X, size):
          for possible_mns in combinations(neighbors_X, size):
            possible_mns = set(possible_mns)
            if is_independent(tmt, node, possible_mns):
              return possible_mns

        markov_blanket = set(markov_blanket)
        mb_copy = set(markov_blanket)

        if self.use_ci_oracle:
          corr = np.zeros((len(self.graph_true.nodes()),))
        else:
          corr = data_matrix.corr()[treatment_node_id]
        size = 0
        while len(mb_copy) > size:
          nodes_to_check = sorted(list(mb_copy), key=lambda n:(corr[n]))
          for n in nodes_to_check:
            sep = get_sepset(n, mb_copy - {n}, size)
            if sep is not None:
              mb_copy -= {n}
              if g.has_edge(n, tmt):
                g.remove_edge(n, tmt)
                sep_set[n][tmt] = sep
                sep_set[tmt][n] = sep

          size += 1

        return g, sep_set

      def get_mns(g, sep_set, node):
        # For convenience, for N \in Ne(X),  mns(N) = {N}.

        def cache_and_get(node, mns):
          self.mns_cache[node] = mns
          return mns

        if self.mns_cache[node] is not None:
          return self.mns_cache[node]

        neighbors_X = set(g.neighbors(tmt))

        if node == tmt:
          raise ValueError("Can't get MNS for treatment node.")
        
        if node in neighbors_X:
          return cache_and_get(node, MNS(is_valid=True, mns={node}))
        
        for size in range(0, len(neighbors_X) + 1):
          for possible_mns in combinations(neighbors_X, size):
            possible_mns = set(possible_mns)
            if is_independent(tmt, node, possible_mns):
              return cache_and_get(node, MNS(is_valid=True, mns=possible_mns))

        return cache_and_get(node, MNS(is_valid=False))

      def get_all_neighbor_separators(g, node):
        neighbors_to_check = set(g.neighbors(tmt))

        for size in range(0, len(neighbors_to_check) + 1):
          for possible_sep in combinations(neighbors_to_check, size):
            possible_sep = set(possible_sep)
            if is_independent(tmt, node, possible_sep):
              yield possible_sep

      def mark_children_unshielded_colliders(g, sep_set, node_ids,
          to_be_oriented, markov_blanket, tmt_neighbors):
        g = g.copy()
        sep_set = copy.deepcopy(sep_set)
        node_ids = set(node_ids)

        tmt = treatment_node_id
        tmt_children = set()
        
        spouses_tmt = markov_blanket - tmt_neighbors
        for n in to_be_oriented:
          for nA in spouses_tmt:
            mark_as_child = False
            for separator in get_all_neighbor_separators(g, nA):
              cond_set = separator
              if n in cond_set or is_independent(n, nA, cond_set):
                mark_as_child = False
                break
              else:
                mark_as_child = True

            if mark_as_child:
              tmt_children |= {n}
              self.log("Collider detected: (%s -> %s <- %s) | %s" % 
                    (self.cols[tmt], self.cols[n], self.cols[nA], str(to_labels(cond_set))))
              break
        
        return g, sep_set, tmt_children

      
      def eager_collider_check(g, i, j, sep_set, tmt_neighbors):
        g = g.copy()
        detected_parents = set()

        if tmt in sep_set[i][j]:
          return g, detected_parents
        
        if i in tmt_neighbors and j in tmt_neighbors:
          
          if tmt not in sep_set[i][j]:
            
            if not is_independent(i, j, ({tmt} | sep_set[i][j])):
              # This means that there is an unshielded collider i -> X <- j and
              # so mark i, j as parents.
              detected_parents = {i, j}
              # This is to account for Meek rule 3.
              detected_parents |= (sep_set[i][j] & tmt_neighbors)
            
          return g, detected_parents
        
        # Run an Eager collider check.
        is_indep, p_val = is_independent(i, j, ({tmt} | sep_set[i][j]), ret_p_val=True)
        if not is_indep:
          c = self.cols
          self.log("Dependent: (%s \notindep %s) | %s; p_val = %f" % (
            c[i], c[j], str(to_labels({tmt} | sep_set[i][j])), p_val))
          
          mns_i = get_mns(g, sep_set, i)
          mns_j = get_mns(g, sep_set, j)

          if mns_i.is_valid() and mns_j.is_valid():
            
            if len({i, j} & tmt_neighbors) == 1:
              if i in tmt_neighbors:
                nbr = i
                mns_non_nbr = mns_j.mns()
              else:
                nbr = j
                mns_non_nbr = mns_i.mns()
              
              # In case only one of {i, j} is a neighbor of X, the ECC leverages
              # an unshielded collider of the form {nbr} -> V <- {non_nbr},
              # where where V is a parent of X (i.e., there is a V -> X edge).
              # Here, MNS(non_nbr) must contain both V and {nbr}.
              if len(mns_non_nbr) < 2 or (nbr not in mns_non_nbr):
                return g, detected_parents

              detected_parents = mns_i.mns() | mns_j.mns()
            else:
              mns_check_pass = True
              if self.ldecc_do_checks:
                mns_check_pass = (mns_i.mns() == mns_j.mns())
                
              if mns_check_pass:
                detected_parents = ((mns_i.mns() | mns_j.mns()) | {i, j}) & tmt_neighbors

        return g, detected_parents
      
      def is_test_already_done(i, j, cond, tests_done):
        return frozenset([i, j, frozenset(cond)]) in tests_done
      
      def get_non_collider_nodes(neighbors, non_colliders):
        res = set()
        for n in neighbors:
          res |= non_colliders[n]
        return res
      

      tests_done = set()
      non_colliders = collections.defaultdict(set)

      node_ids = set(range(data_matrix.shape[1]))
      node_size = data_matrix.shape[1]
      sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
      g = self._create_complete_graph(node_ids)

      # We begin by finding neighbors Ne(X). For this, we first find the Markov 
      # blanket MB(X) and then run the PC algorithm locally within MB(X) to find
      # Ne(X).

      g, sep_set, markov_blanket = (
          find_markov_blanket(g, sep_set, node_ids, treatment_node_id)
      )
      self.log("Markov blanket of X: %s" % str(to_labels(markov_blanket)))

      g, sep_set = do_local_pc(g, sep_set, markov_blanket)
      tmt_neighbors = set(g.neighbors(tmt))
      self.log("Tests after finding Ne(X): %s" % str(self.ci_test_calls))
      self.log("Treatment neighbors: %s" % str(to_labels(tmt_neighbors)))

      tmt_parents = set()
      tmt_children = set()
      
      to_be_oriented = tmt_neighbors - tmt_children - tmt_parents

      # Mark children `C` of X that form an unshielded collider of the form
      # X -> C <- V.
      g, sep_set, tmt_children = mark_children_unshielded_colliders(
          g, sep_set, node_ids, to_be_oriented, markov_blanket, tmt_neighbors)
      
      to_be_oriented = tmt_neighbors - tmt_children - tmt_parents

      # Start running conditional independence tests in the same way as the PC
      # algorithm along with Eager Collider Checks.
      l = 0
      debug_skel = g.copy()
      while True:
          cont = False
          remove_edges = []
          node_ids_sorted = sorted(node_ids, key=lambda n:0 if n in tmt_neighbors else 1)
          for (i, j) in permutations(node_ids_sorted, 2):
              if len(to_be_oriented) == 0:
                break

              if self.max_tests_done:
                break
              
              # Skip the treatment node because we have already discovered the
              # local structure around the treatment X.
              if i == tmt or j == tmt:
                continue

              adj_i = list(g.neighbors(i))
              if j not in adj_i:
                  continue
              else:
                  adj_i.remove(j)
              
              adj_i = set(g.neighbors(i)) - {j}

              if len(adj_i) >= l:
                  adj_i_sorted = sorted(adj_i, key=lambda n: 0 if n == tmt else 1)
                  for k in combinations(adj_i_sorted, l):
                      k = set(k)
                      
                      if len(to_be_oriented) == 0:
                        break
                        
                      if is_test_already_done(i, j, k, tests_done):
                        continue
                      
                      tests_done |= {frozenset([i, j, frozenset(k)])}
                      is_indep, p_val = is_independent(i, j, k, ret_p_val=True)
                      if is_indep:
                          if g.has_edge(i, j):
                              g.remove_edge(i, j)
                              debug_skel.remove_edge(i, j)
                                  
                          sep_set[i][j] |= set(k)
                          sep_set[j][i] |= set(k)

                          self.log("Removed edge (%s - %s) | %s; p_val=%f" % (
                            self.cols[i], self.cols[j], str(to_labels(sep_set[i][j])), p_val))

                          mark_non_collider = False
                          if tmt in k and i in tmt_neighbors and j in tmt_neighbors:
                            mark_non_collider = True
                            self.log("Marking non collider: %s - %s - %s" % (
                              self.cols[i], self.cols[tmt], self.cols[j]))
                            non_colliders[i] |= {j}
                            non_colliders[j] |= {i}

                            for nbr in (to_be_oriented - {i, j} - k):
                              if not is_independent(i, j, k | {nbr}):
                                self.log("Marking child: %s; %s \notindep %s | %s" % (
                                    self.cols[nbr], self.cols[i], self.cols[j],
                                    str(to_labels(k | {nbr}))
                                ))
                                tmt_children |= {nbr}

                          edges_to_remove = []
                          detected_parents = set()
                          debug_skel, detected_parents = eager_collider_check(
                            debug_skel, i, j, sep_set, tmt_neighbors)
                          
                          if mark_non_collider or len(detected_parents) > 0:
                            tmt_parents |= detected_parents
                            tmt_children |= get_non_collider_nodes(tmt_parents, non_colliders)
                            to_be_oriented -= (detected_parents | tmt_children)
                            
                            self.log("Parents: %s" % str(set(self.cols[o] for o in tmt_parents)))
                            self.log("Children: %s" % str(set(self.cols[o] for o in tmt_children)))
                          
                          break

                  cont = True
          
          if len(node_ids) < 2:
            break

          l += 1
          if cont is False:
              break
          

      # arbitrarily resolve contradictions.
      tmt_parents -= tmt_children

      self.log("Parents: %s" % str(set(self.cols[o] for o in tmt_parents)))
      self.log("Children: %s" % str(set(self.cols[o] for o in tmt_children)))
      self.log("Unoriented: %s" % str(set(self.cols[o] for o in to_be_oriented)))
      
      return {
          "tmt_parents": set(self.cols[o] for o in tmt_parents),
          "tmt_children": set(self.cols[o] for o in tmt_children),
          "unoriented": set(self.cols[o] for o in to_be_oriented),
          "non_colliders": (
              {self.cols[key]: {self.cols[v] for v in non_colliders[key]}
               for key in tmt_parents | tmt_children | to_be_oriented}),
      }