"""
******************
CliqueTree Class 
&
Clique Class
******************

This is a class for creating/manipulating Junction (Clique) Trees,
and performing inference over them. The advantage of clique trees
over traditional variable elimination over the original Bayesian 
network is that clique trees allow you to compute marginal
probabilities of MULTIPLE variables without having to run the
entire algorithm over. 

Therefore, if you have to query the Bayesian network many times 
-- and you want the exact marginal values -- then it might be 
best to use the clique tree data structure for inference.
If you need to compute the marginal distribution for all variables,
but you dont mind an approximate, a sampling algorithm is probably
the way to go.

In general, the junction tree algorithms generalize 
Variable Elimination to the efficient, simultaneous execution 
of a large class of queries.

NOTE: A cluster graph is a generalization of the clique tree
data structure - to generate a clique tree, you first generate
a cluster graph, then simply calculate a maximum spanning tree.
In other words, a clique tree can be considered as a special
type of cluster graph.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""



import numpy as np
import networkx as nx
from copy import copy, deepcopy

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor
from pyBN.classes.factorization import Factorization

from pyBN.utils.graph import *



class CliqueTree(object):
    """
    CliqueTree Class

    Let G be a chordal graph (ie. a graph such that no cycle includes more
    than three nodes), then a CliqueTree is a tree H such that each
    maximal clique C in G is a node in H.

    Attributes
    ----------
    - bn 
        - BayesNet object

    - V
        - Vertices -> list

    - E
        - Edges -> dictionary

    - C
        - Cliques -> a dictionary where key = vertex idx,
                value = Clique object

    """

    def __init__(self, bn):
        """
        Instantiate a CliqueTree object.

        Arguments
        ---------
        *bn*: a BayesNet object

        Notes
        -----
        Ideally, the Factor class should be used as the
        cliques instead of the Clique class (because it's
        just a watered down version of the Factor class)        
        
        """
        ####
        self.V = None
        self.E = None
        self.C = None
        ###

        self.bn = bn
        self._F = Factorization(bn)
        self.initialize_tree()

        

    #def __repr__(self):
       # return self.C

    def __iter__(self):
        for clique in self.C.values():
            yield clique

    def __getitem__(self, rv):
        """
        Returns Clique of passed-in rv
        """
        return self.C[rv]

    def parents(self, v):
        p = []
        for rv in self.V:
            if v in self.E[rv]:
                p.append(v)
        return p

    def children(self, n):
        return self.E[n]

    def dfs_postorder(self, root):
        G = nx.Graph(self.E)
        tree_graph = nx.dfs_tree(G,root)
        clique_ordering = list(nx.dfs_postorder_nodes(tree_graph,root))
        return clique_ordering

    def initialize_tree(self):
        """
        Initialize the structure of a clique tree, using
        the following steps:
            - Moralize graph (i.e. marry parents)
            - Triangulate graph (i.e. make graph chordal)
            - Get max cliques (i.e. community/clique detection)
            - Max spanning tree over sepset cardinality (i.e. create tree)
        
        """
        ### MORALIZE GRAPH & MAKE IT CHORDAL ###
        chordal_G = make_chordal(self.bn) # must return a networkx object
        V = chordal_G.nodes()

        ### GET MAX CLIQUES FROM CHORDAL GRAPH ###
        C = {} # key = vertex, value = clique object
        max_cliques = reversed(list(nx.chordal_graph_cliques(chordal_G)))
        for v_idx,clique in enumerate(max_cliques):
            C[v_idx] = Clique(set(clique))

        ### MAXIMUM SPANNING TREE OVER COMPLETE GRAPH TO MAKE A TREE ###
        weighted_edge_dict = dict([(c_idx,{}) for c_idx in range(len(C))])
        for i in range(len(C)):
            for j in range(len(C)):
                if i!=j:
                    intersect_cardinality = len(C[i].sepset(C[j]))
                    weighted_edge_dict[i][j] = -1*intersect_cardinality
        mst_G = mst(weighted_edge_dict)
        ### SET V,E,C ###
        self.E = mst_G # dictionary
        self.V = mst_G.keys() # list
        self.C = C

        
        ### ASSIGN EACH FACTOR TO ONE CLIQUE ONLY ###
        v_a = dict([(rv, False) for rv in self.bn.nodes()])
        for clique in self.C.values():
            temp_scope = []
            for var in v_a:
                if v_a[var] == False and set(self.bn.scope(var)).issubset(clique.scope):
                    temp_scope.append(var)
                    v_a[var] = True
            clique._F = Factorization(self.bn, temp_scope)

        ### COMPUTE INITIAL POTENTIAL FOR EACH FACTOR ###
        # - i.e. multiply all of its assigned factors together
        for i, clique in self.C.items():
            if len(self.parents(i)) == 0:
                clique.is_ready = True
            clique.initialize_psi()


class Clique(object):
    """
    Clique Class

    *scope* : a set of variables in the clique's scope

    *_f* : a factorization object that contains only the
        factors of variables in the clique's scope

    *psi* : a factor
        The potential of the clique.

    *belief* : a factor
        The factor which holds the marginal/conditional
        probabilities of the relevant nodes after 
        belief propagation, etc

    *messages_received* : a list of Factors
        The messages the clique has received from its neighbors -
        stored so that the beliefs can be calculated after
        belief propagation, etc.

    *is_ready* : a boolean
        Whether the clique is ready to send a message -> must
        have RECEIVED messages from all of its neighbors first.

    """

    def __init__(self, scope):
        """
        Instantiate a clique object

        Arguments
        ---------
        *scope* : a python set
            The set of variables in the cliques scope,
            i.e. the main var and its parents


        """
        self.scope = scope
        self._F = None
        
        self.psi = None # Psi should never change -> Factor object
        self.belief = None
        
        self.messages_received = []
        self.is_ready = False

    def __repr__(self):
        return str(self.scope)

    def __rshift__(self, other_clique):
        """
        Send a message from self to other_clique
        """
        self.send_message(other_clique)

    def __lshift__(self, other_clique):
        """
        Send a message from other_clique to self
        """
        other_clique.send_message(self)

    def send_message(self, parent):
        """
        Send a message to the parent clique.

        To send a message from X to Y, you
        must first take the potential (self.psi) of
        X and sum out the variables NOT in the sepset
        of Y.

        THEN, if X has received any messages, the
        potential (self.psi) must be multiplied by
        all of the received messaged.

        Arguments
        ---------
        *parent* : a string
            The parent to which the message will
            be sent.

        """


        # First generate Belief = Original_Psi * all received messages
        if len(self.messages_received) > 0:
            # if there are messages received, mutliply them in to psi first
            if not self.belief:
                self.belief = copy(self.psi)
            for msg in self.messages_received:
                self.belief *= msg
            #self.belief.merge_multiply(self.messages_received)
        else:
            # if there are no messages received, simply move on with psi
            if not self.belief:
                self.belief = copy(self.psi)
        # generate message as belief with Ci - Sij vars summed out
        #vars_to_sumout = list(self.scope.difference(self.sepset(parent)))
        msg_to_send = copy(self.belief)
        for var_to_sumout in self.scope.difference(self.sepset(parent)):
            msg_to_send /= var_to_sumout
        #message_to_send = copy(self.belief)
        #message_to_send.sumout_var_list(vars_to_sumout)
        parent.messages_received.append(msg_to_send)

    def initialize_psi(self):
        """
        Compute a new psi (cpt) in order to 
        set the clique's belief. This involves
        multiplying the factors in the Clique together.
        """
        assert (len(self._F) != 0), 'No Factors assigned to this clique.'

        if len(self._F) == 1:
            self.psi = copy(self._F[0])
            self.belief = copy(self.psi)
        else:
            self.psi = max(self._F, key=lambda x: len(x.cpt))
            for f in self._F:
                self.psi *= f
            self.belief = copy(self.psi)

    def send_initial_message(self, other_clique):
        """
        NOT SURE IF THIS IS NEEDED.

        Send the first message to another clique.

        Arguments
        ---------
        *other_clique* : a different Clique object

        """
        psi_copy = copy(self.psi)
        sepset = self.sepset(other_clique)
        sumout_vars = self.scope.difference(sepset)
        # sum out variables not in the sepset of other_clique
        psi_copy.sumout_var_list(list(sumout_vars))


        #psi_copy.cpt = psi_copy.cpt.loc[:,[c for c in psi_copy.cpt.columns if 'Prob' not in c]]
        #psi_copy.cpt[str('Prob-Val-' + str(np.random.randint(0,1000000)))] = 1
        print('Init Msg: \n', psi_copy.cpt)
        other_clique.messages_received.append(psi_copy)

        self.belief = copy(self.psi)

    def collect_beliefs(self):
        """
        Collecting beliefs is done by taking the
        potential (self.psi) of X and multiplying by all of
        the messages which X has received.

        NOTE: once beliefs are collected, the messages
        which the Clique has received are cleared out.

        Notes
        -----
        A root node (i.e. one that doesn't send a message) must run collect_beliefs
        since they are only collected at send_message()

        Also, we collect beliefs at the end of loopy belief propagation (approx. inference)
        since the main algorithm is just sending messages for a while.
        """
        if len(self.messages_received) > 0:
            self.belief = copy(self.psi)
            for msg in self.messages_received:
                self.belief *= msg
            self.belief 
            self.messages_received = []
        else:
            self.belief = copy(self.belief)

    def sepset(self, other_clique):
        """
        The sepset of two cliques is the set of
        variables in the intersection of the two
        cliques' scopes.

        Arguments
        ---------
        *other_clique* : a Clique object

        """
        return self.scope.intersection(other_clique.scope)

    def marginalize_over(self, target):
        """
        Marginalize the cpt (belief) over a target variable.

        Arguments
        ---------
        *target* : a string
            The target random variable.

        """
        blf = copy(self.belief)
        blf.sumover_var(target)
        return blf















