from typing import Union

import pandas as pd 
import numpy as np
import networkx as nx

import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils

PEN_COEFF = 1e4


class Client:

    def __init__(
            self,
            name: str,
            data: Union[pd.DataFrame, str],
            cd_function: str = 'pc',
            scoring_function: str = 'bic',
            masked: bool = True, # if False standard PERI
            linear: bool = True, # if False non-linear causal discovery
            pc_alpha: float = 0.1, # PC algorithm alpha threshold
            ebic_gamma: float = 1.0, # eBIC gamma parameter (only used if scoring_function='ebic')
            bic_coeff: float = 0.5, # BIC penalty coefficient (default 0.5)
        ):
            """ initializing client """
            self.name = name
            if isinstance(data, str):
                self.data = pd.read_csv(data)
            else:
                self.data = data

            self.masked = masked
            self.data = self.data.values
            self.variables = list(range(self.data.shape[1]))
            # cache for regrets and local scores
            self.cache = {i: {} for i in range(len(self.variables))} # regrets
            self.local_score = {} # score of the locally learned graph
            self.scoring_function = scoring_function

            # discovery of local graph and computing local scores
            self.cd_function = utils.get_cd_function(cd_function, linear=linear, alpha=pc_alpha)
            self.scoring_class = utils.get_scoring_class(scoring_function, lmbda=None, ebic_gamma=ebic_gamma, bic_coeff=bic_coeff)(self.data)
            self.graph = self.cd_function(self.data)
            self._compute_local_scores()
            # mapping of child to parents in local graph
            self.child_to_parents = {}
            for i, _ in enumerate(self.variables):
                self.child_to_parents[i] = list(map(int, np.where(self.graph[:, i] == 1)[0]))

            # undirected/bidirected edges
            self.undirected = set()
            for i in range(self.graph.shape[0]):
                for j in range(i+1, self.graph.shape[1]):
                    if self.graph[i, j] == 1 and self.graph[j, i] == 1:
                        edge = [i, j]
                        edge.sort()
                        self.undirected.add(tuple(edge))

    def _compute_local_scores(self):
        for i, var in enumerate(self.variables):
            parents = list(map(int, np.where(self.graph[:, i] == 1)[0]))
            self.local_score[var] = self.scoring_class.local_score(i, parents)
           
    def score(self, server_parents: np.ndarray, server_child: int, undirected: bool = False) -> float:
        """ compute local score """
        if not undirected:
            
            if self.masked:
                server_parents = list(set(server_parents) & 
                                    set(self.child_to_parents[server_child]))
                server_parents.sort()    
            
            if tuple(server_parents) not in self.cache[server_child]:       
                score = self.scoring_class.local_score(server_child, server_parents)
                score = score - self.local_score[server_child]
                self.cache[server_child][tuple(server_parents)] = score
            else:
                score = self.cache[server_child][tuple(server_parents)]

        if undirected:

            self.scoring_class = utils.get_scoring_class(
                'bic_pen',
                lmbda=PEN_COEFF * np.log(self.data.shape[0])
            )(self.data)
            for parent in server_parents.copy():
                if self.graph[int(parent), server_child] == 0 and self.graph[server_child, int(parent)] == 0:
                    server_parents.remove(parent)

            score = self.scoring_class.local_score(server_child, server_parents)
            score -= self.scoring_class.local_score(server_child, self.child_to_parents[server_child])

        return score   
    
    def clear_cache(self):
        self.cache = {i: {} for i in range(len(self.variables))} # regrets