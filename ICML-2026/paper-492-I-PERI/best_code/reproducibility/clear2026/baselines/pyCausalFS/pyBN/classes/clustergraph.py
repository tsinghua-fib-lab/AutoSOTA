"""
******************
ClusterGraph Class
******************

This is a class for creating/manipulating Cluster Graphs,
and performing inference over them - currently the only
supported algorithm is Loopy Belief Propagation. Still,
the class structure is in place for easy addition of
any algorithms relying on the Cluster Graph framework.

NOTE: A cluster graph is a generalization of the clique tree
data structure - to generate a clique tree, you first generate
a cluster graph, then simply calculate a maximum spanning tree.
In other words, a clique tree can be considered as a special
type of cluster graph.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import pandas as pd
import networkx as nx

from pyBN.classes.cliquetree import Clique

class ClusterGraph(object):
    """
    ClusterGraph Class

    """

    def __init__(self, bn):
        """
        Initialize a ClusterGraph object

        """
        self.BN = BN
        self.V = {} # key = cluster index, value = Cluster objects
        self.E = []
        self.G = None
        self.initialize_graph(method)
        self.beliefs = {} # dict where key = cluster idx, value = belief cpt

    def initialize_graph(self):
        """
        Initialize the structure of the cluster graph.

        """
        # generate graph structure
        self.bethe()
        # initialize beliefs
        for clique in self.V.values():
            clique.compute_psi() 
        # initialize messages to 1
        self.initialize_messages()

    def bethe(self):
        """
        Generate Bethe cluster graph structure.
        """
        self.V = {}
        self.E = []

        factorization = Factorization(self.BN)
        prior_dict = {}
        for factor in factorization.f_list:
            # if factor is just a prior (i.e. already added as rv)
            if len(factor.scope) == 1:
                #self.V[len(self.V)] = Clique(scope=factor.scope)
                #self.V[len(self.V)-1].factors = [factor]
                prior_dict[factor.var] = factor
            if len(factor.scope) > 1:
                self.V[len(self.V)] = Clique(scope=factor.scope)
                self.V[len(self.V)-1].factors = [factor] # assign the factor
        sep_len = len(self.V)
        # First, add all individual random variables
        for rv in self.BN.V:
            # if rv is a prior, don't add it
            if rv in prior_dict.keys():
                factor = prior_dict[rv]
                self.V[len(self.V)] = Clique(scope=factor.scope)
                self.V[len(self.V)-1].factors = [factor]
            else:
                self.V[len(self.V)] = Clique(scope={rv})
                # create a new initial factor since it wont have one        
                new_factor = Factor(BN=self.BN, var=rv, init_to_one=True)
                self.V[len(self.V)-1].factors = [new_factor]        
        for i in range(sep_len):
            for j in range(sep_len,len(self.V)):
                if self.V[j].scope.issubset(self.V[i].scope):
                    self.E.append((i,j))

        new_G = nx.Graph()
        new_G.add_edges_from(self.E)        
        self.G = new_G

    def initialize_messages(self):
        """
        For each edge (i-j) in the ClusterGraph,
        set delta_(i-j) = 1 and
        set delta_(j-i) = 1.
        (i.e. send a message from each parent to every child where the 
            message is a df = 1)
        """
        for cluster in self.V:
            for neighbor in self.G.neighbors(cluster):
                self.V[cluster].send_initial_message(self.V[neighbor])


    def collect_beliefs(self):
        self.beliefs = {}
        for cluster in self.V:
            self.V[cluster].collect_beliefs()
            #print('Belief ' , cluster , ' : \n', self.V[cluster].belief.cpt)
            self.beliefs[cluster] = self.V[cluster].belief


    def loopy_belief_propagation(self, target, evidence, max_iter=100):
        """
        
        This is Message Passing (Loopy Belief Propagation) over a cluster graph.

        It is Sum-Product Belief Propagation in a cluster graph as shown in
        Koller p.397

        Notes:
            1. Definitely a problem due to normalization (prob vals way too small)
            2. Need to check the scope w.r.t. messages.. all clusters should not
               be accumulating rv's in their scope over the course of the algorithm.
        """
        # 1: Moralize the graph
        # 2: Triangluate
        # 3: Build a clique tree using max spanning
        # 4: Propagation of probabilities using message passing

        # creates clique tree and assigns factors, thus satisfying steps 1-3
        cgraph = copy.copy(self)
        G = cgraph.G

        edge_visit_dict = dict([(i,0) for i in cgraph.E])

        iteration = 0
        while not cgraph.is_calibrated():
            if iteration == max_iter:
                break
            if iteration % 50 == 0:
                print('Iteration: ' , iteration)
                for cluster in cgraph.V.values():
                    cluster.collect_beliefs()
            # select an edge
            e_idx = np.random.randint(0,len(cgraph.E))
            edge_select = cgraph.E[e_idx]
            p_idx = np.random.randint(0,2)
            parent_edge = edge_select[p_idx]
            child_edge = edge_select[np.abs(p_idx-1)]
            print(parent_edge , child_edge)

            # send a message along that edge
            cgraph.V[parent_edge].send_message(cgraph.V[child_edge])

            iteration += 1
        print('Now Collecting Beliefs..')
        self.collect_beliefs()
        self.BN.ctree = self
