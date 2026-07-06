from typing import List, Optional

import numpy as np

from pyciphod.causal_discovery.federated.regret_based.iperi.client import Client
from pyciphod.causal_discovery.federated.regret_based.iperi.score import IPeriScore
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
import pyciphod.causal_discovery.federated.regret_based.ges as ges

class IPeri:
    def __init__(
        self,
        n_variables: int, 
        clients: List[Client],
    ):
        self.n_variables = n_variables
        self.clients = clients
        self.server_graph = np.zeros((n_variables, n_variables))

    def _optimize(
        self, 
        max_iters: int = 10, 
        patience: int = 10,
        tol: float = 0,
        phases: List[str] = ['forward', 'backward'], 
        completion_algorithm: Optional[callable] = None,
        undirected: bool = False,
    ):  
        iteration, patience_counter = 0, 0
        best_score = np.inf
        while True:

            if iteration >= max_iters:
                print("Terminating after ", max_iters, " iterations")
                return self.server_graph
            
            graph, score = ges.fit(
                IPeriScore(
                    data = np.empty((0, self.n_variables)),  # Dummy data
                    clients = self.clients,
                    cache = True,
                    debug = 0,
                    undirected = undirected,
                ),
                completion_algorithm=completion_algorithm,
                A0=self.server_graph,
                phases=phases,
            )

            if best_score > score and score >= 0:
                best_score = score

            if best_score == 0:
                print("Score is 0, terminating")
                self.server_graph = graph
                return self.server_graph

            if best_score - score <= tol:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Terminating after ", patience, " iterations with no change")
                    return self.server_graph
            else:
                patience_counter = 0

            self.server_graph = graph
            iteration += 1

            print("Iteration ", iteration, " Score: ", score)

    def fit(
        self,
        max_iters: int = 1,
        patience: int = 5,
        tol: float = 0,
        cpdag_max_iters: int = 10,
    ):
        print("finding CPDAG...")
        cpdag = self._optimize(
            max_iters=cpdag_max_iters,
            patience=patience,
            tol=tol,
        )
        self.server_graph = cpdag
        
        # reset clients' caches
        for client in self.clients:
            client.clear_cache()
            # print(client.graph)

        print("orienting edges...")
        self.server_graph = self._optimize(
            max_iters=max_iters,
            patience=patience,
            tol=tol,
            phases=['orient'],
            completion_algorithm=utils.identity,
            undirected = True
        )

        return self.server_graph, cpdag


            
        