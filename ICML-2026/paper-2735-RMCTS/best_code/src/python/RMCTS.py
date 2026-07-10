import numpy as np
import json
import matplotlib.pyplot as plt
from time import perf_counter
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path

from . import game, metaparm

c_int = ctypes.c_int
c_float = ctypes.c_float
array_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags='CONTIGUOUS')
array_float = npct.ndpointer(dtype=np.float32,ndim=1,flags='CONTIGUOUS')
array_double = npct.ndpointer(dtype=np.float64,ndim=1,flags='CONTIGUOUS')
ptr_int = ctypes.POINTER(c_int)
ptr_float = ctypes.POINTER(c_float)
ptr_void = ctypes.c_void_p # same as ctypes.c_voidp

libRMCTS_path = Path(__file__).parent / 'libRMCTS.so'
libRMCTS = npct.load_library(libRMCTS_path,".")

libRMCTS.MCTS_init.restype = ptr_void
libRMCTS.MCTS_init.argtypes = [c_int]*2 + [c_float] + [array_float]*7 + [array_int32]*10

libRMCTS.MCTS_free.restype = None
libRMCTS.MCTS_free.argtypes = [ptr_void]

libRMCTS.MCTS_flush_new_stack.restype = None
libRMCTS.MCTS_flush_new_stack.argtypes = [ptr_void]

libRMCTS.MCTS_propagate_all.restype = None
libRMCTS.MCTS_propagate_all.argtypes = [ptr_void]

def mcts_data(G, numSims, nnet, c_puct=None):
    if c_puct is None:
        c_puct = metaparm.c_puct
    mcts = MCTS(G,numSims,c_puct,nnet)
    pi1, v1 = mcts()
    m = G.shape[0]
    pi0 = mcts.policy[:m*mcts.n].reshape((m,mcts.n))
    v0 = mcts.value[:m]
    N = mcts.N[:m*mcts.n].reshape((m,mcts.n))
    Q = mcts.Q[:m*mcts.n].reshape((m,mcts.n))
    pi1 = pi1.reshape((m, mcts.n))
    return pi1, v1, pi0, v0, N, Q

def learn_pi_and_v(G, numSims, nnet, c_puct=None, return_mcts=False):
    if c_puct is None:
        c_puct = metaparm.c_puct
    assert numSims >= 1
    if numSims == 1:
        pi, v = nnet(G)
        v = v.flatten()
        # legalize policy
        for i in range(len(pi)):
            actions = game.getValidActions(G[i])
            pi_legal = np.zeros(pi.shape[1],dtype=np.float32)
            pi_legal[actions] = pi[i][actions]
            pi_legal /= np.sum(pi_legal)
            pi[i] = pi_legal
        if return_mcts:
            return pi, v, None
        else:
            return pi, v
    mcts = MCTS(G,numSims,c_puct,nnet)
    pi, v = mcts()
    pi = pi.reshape((mcts.num_lanes, mcts.n))
    if return_mcts:
        return pi, v, mcts
    else:
        return pi, v


def assign_simulations(S,pi):
    '''
    S is a positive integer
    pi is a probability distribution
    let n = len(pi)
    this code returns a random integer 
    array s, where the expectation 
    of s[i] = pi[i]*S, and where 
    the variance is minimal (I think)
    It uses Emma's x, x+1, x+2 method
    '''
    pi = np.array(pi,dtype=np.float64)
    pi /= np.sum(pi)
    n = len(pi)
    x = np.random.random()
    X = np.arange(S,dtype=np.float64) + x
    chunks = S*pi
    splits = np.zeros(n+1)
    splits[1:] = np.cumsum(chunks)
    s = np.histogram(X,bins=splits)[0]
    return s


def new_policy_common_ucb_Newton(Q,c,pi0,T,epsilon=1.0e-12):
    pi0 = pi0.astype(np.float64)
    Q = Q.astype(np.float64)
    assert np.min(pi0) > 0.0
    pi0 /= np.sum(pi0)
    c0 = np.float64(c) / np.sqrt(np.float64(T))
    epsilon = np.float64(epsilon)
    a_max = np.argmax(Q)
    Q_max = Q[a_max]
    delta = c0 * pi0[a_max]
    if delta < epsilon:
        delta = epsilon
    def f(delta):
        return c0*np.sum(pi0 / ((Q_max - Q) + delta)) - 1.0
    def fprime(delta):
        return -c0*np.sum(pi0 / ((Q_max - Q) + delta)**2)
    while f(delta) > epsilon:
        new_delta = delta - f(delta)/fprime(delta)
        if new_delta <= delta:
            break
        delta = new_delta
    pi1 = c0 *  pi0 / ((Q_max - Q) + delta)
    pi1 = np.maximum(pi1, 0.0) # just in case
    pi1 /= np.sum(pi1) # should already be close to 1
    pi1 = np.float32(pi1) # cast back to float32 again
    assert pi1.flags['C_CONTIGUOUS']
    return pi1

def new_policy_common_ucb_Simpleton(Q,c,pi0,T):
    c0 = c / np.sqrt(T)
    maxQ = np.max(Q)
    u = maxQ + c0
    pi1 = pi0 / (u - Q)
    pi1 /= np.sum(pi1)
    return pi1

def initial_Q(g, v):
    n = game.numActions()
    Q = np.zeros(n,dtype=np.float32)
    Q[:] = -np.inf
    actions = game.getValidActions(g)
    Q[actions] = v
    return Q

def legalize_policy(pi, g):
    '''
    pi is a policy
    g is a gamestate
    returns a policy that is supported on legal actions
    '''
    actions = game.getValidActions(g)
    pi_legal = np.zeros(len(pi),dtype=np.float32)
    pi_legal[actions] = pi[actions]
    pi_legal /= np.sum(pi_legal)
    return pi_legal, actions

class MCTS:
    def __init__(self, root_states, numSims, c_puct, nnet):
        assert numSims >= 2 #  TODO
        self.root_states = np.array(root_states, dtype=np.float32)

        # assert that root states are not terminal
        for i in range(self.root_states.shape[0]):
            ended, score = game.gameEnded(self.root_states[i])
            assert not ended

        self.numSims = numSims
        self.c_puct = c_puct
        self.nnet = nnet
        self.gamesize = game.gameLength()
        self.n = game.numActions()
        self.num_lanes = self.root_states.shape[0]

        # main part of the data
        # maximum possible number of rows to be used is 
        # the product of the number of root states with 
        # the number of simulations
        self.NUM_ALL = self.num_lanes * self.numSims
        
        # float32 parts
        self.G = np.zeros(self.NUM_ALL * self.gamesize, dtype=np.float32)
        self.G[:self.num_lanes * self.gamesize] = self.root_states.flatten()
        self.policy = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)
        self.value = np.zeros(self.NUM_ALL, dtype=np.float32)
        self.Q = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)
        self.N = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)

        # int32 parts
        self.parent = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.parent[:self.num_lanes] = -1
        self.a0 = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.a0[:self.num_lanes] = -1
        self.sims = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.sims[:self.num_lanes] = numSims
        self.sims_remaining = np.copy(self.sims)

        # number of occupied rows
        self.row_count = np.zeros(1, dtype=np.int32)
        self.row_count[0] = self.num_lanes

        self.inference_stack = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.inference_stack[:self.num_lanes] = np.arange(self.num_lanes, dtype=np.int32)
        self.inference_stack_size = np.zeros(1, dtype=np.int32)
        self.inference_stack_size[0] = self.num_lanes
        
        self.new_stack = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.new_stack_size = np.zeros(1, dtype=np.int32)

        self.new_policy = np.zeros(self.num_lanes * self.n, dtype=np.float32)
        self.new_value = np.zeros(self.num_lanes, dtype=np.float32)
        self.num_completed = np.zeros(1, dtype=np.int32)

        self._t = libRMCTS.MCTS_init(self.num_lanes, 
                                     self.numSims, 
                                     self.c_puct,
                                     self.new_policy,
                                     self.new_value, 
                                     self.G, 
                                     self.policy, 
                                     self.value, 
                                     self.Q, 
                                     self.N, 
                                     self.parent, 
                                     self.a0, 
                                     self.sims, 
                                     self.sims_remaining,
                                     self.inference_stack,
                                     self.inference_stack_size,
                                     self.new_stack,
                                     self.new_stack_size,
                                     self.num_completed,
                                     self.row_count)
        
        self.initialize()
        
        self.cpu_times = []
        self.gpu_times = []

    def __dealloc__(self):
        libRMCTS.MCTS_free(self._t)

    def local_engine(self, G):
        G = G.reshape(-1, self.gamesize)
        if self.nnet is None:
            m = G.shape[0]
            pi = np.zeros((m,self.n),dtype=np.float32)
            pi[:,:] = 1.0/self.n
            v = np.zeros(m,dtype=np.float32)
            return pi,v
        else:
            pi,v = self.nnet(G)
            v = v.flatten()
            return pi,v

    def check_data(self):
        attrs = ['G','policy','value','Q','N','parent','a0','sims','sims_remaining','inference_stack','new_stack']
        for attr in attrs:
            assert not np.any(np.isnan(getattr(self,attr)))

    def initialize(self):
        root_policy, root_values = self.local_engine(self.root_states)
        self.policy[:self.num_lanes*self.n] = root_policy.flatten()
        self.value[:self.num_lanes] = root_values.flatten()
        self.inference_stack_size[0] = 0
        self.new_stack[:self.num_lanes] = np.arange(self.num_lanes, dtype=np.int32)
        self.new_stack_size[0] = self.num_lanes

    def flush_new_stack(self):
        libRMCTS.MCTS_flush_new_stack(self._t)

    def propagate_all(self):
        libRMCTS.MCTS_propagate_all(self._t)

    def flush_inference_stack(self,batchsize_only=False):
        # flushes the inference stack
        if batchsize_only:
            if self.nnet is None:
                batchsize = self.inference_stack_size[0]
            else:
                batchsize = self.nnet.batchsize
        else:
            batchsize = self.inference_stack_size[0]
        G = self.G.reshape(self.NUM_ALL, self.gamesize)
        policy = self.policy.reshape(self.NUM_ALL, self.n) # should be in place
        value = self.value
        start = max(0, self.inference_stack_size[0] - batchsize)
        inference_indices = self.inference_stack[start : self.inference_stack_size[0]]
        G = G[inference_indices]
        pi, v = self.local_engine(G)
        policy[inference_indices] = pi
        value[inference_indices] = v.flatten()
        self.new_stack[self.new_stack_size[0] : self.new_stack_size[0] + len(inference_indices)] = inference_indices
        self.new_stack_size[0] += len(inference_indices)
        self.inference_stack_size[0] -= len(inference_indices)

    def main_mcts(self, batchsize_only=False):
        t0 = perf_counter()
        self.initialize()
        t1 = perf_counter()
        self.gpu_times.append(t1 - t0)
        t0 = t1

        # Stage 1: build the entire tree T
        while True:
            self.flush_new_stack()
            t1 = perf_counter()
            self.cpu_times.append(t1 - t0)
            if self.inference_stack_size[0] == 0:
                break
            self.flush_inference_stack(batchsize_only=batchsize_only)
            t2 = perf_counter()
            self.gpu_times.append(t2 - t1)
            t0 = t2

        # Stage 2: propagate values from leaves up to the roots
        t0 = perf_counter()
        self.propagate_all()
        t1 = perf_counter()
        self.cpu_times.append(t1 - t0)

        return self.new_policy, self.new_value

    def __call__(self, batchsize_only=False):
        return self.main_mcts(batchsize_only=batchsize_only)

    def plot_comparison(self, i):
        '''
        plot the comparison of the new policy and the old policy
        for the i-th lane
        '''
        n = self.n
        assert self.num_completed == self.num_lanes
        pi0 = self.policy[i*n:(i+1)*n]
        pi1 = self.new_policy[i*n:(i+1)*n]
        fig, ax = plt.subplots(2)
        ax[0].plot(pi0, label='old')
        ax[0].plot(pi1, label='new')
        ax[1].plot(self.Q[i*n:(i+1)*n], label='Q')
        ax[0].legend()
        ax[1].legend()
        plt.show()

class RMCTS_Tree:
    def __init__(self, root_state, engine):
        self.root_state = root_state
        self.engine = engine
        self.capacity = 256
        self.num_rows = 1
        ended, score = game.gameEnded(root_state)
        assert not ended
        self.P = np.zeros((self.capacity, game.numActions()), dtype=np.float32)
        self.v = np.zeros(self.capacity, dtype=np.float32)
        self.Q = np.zeros((self.capacity, game.numActions()), dtype=np.float32)
        self.N = np.zeros((self.capacity, game.numActions()), dtype=np.float32)
        self.parent = np.zeros(self.capacity, dtype=np.int32)
        self.a0 = np.zeros(self.capacity, dtype=np.int32)
        self.state = np.zeros((self.capacity, game.gameLength()), dtype=np.float32)
        self.total_sims = np.zeros(self.capacity, dtype=np.int32)
        self.new_sims = np.zeros(self.capacity, dtype=np.int32)
        self.new_sims_children = np.zeros(self.capacity, dtype=np.int32)
        self.node_type = np.zeros(self.capacity, dtype=np.int8)
        self.child_index = np.zeros((self.capacity, game.numActions()), dtype=np.int32)
        self.depth = np.zeros(self.capacity, dtype=np.int32)
        self.path_logit = np.zeros(self.capacity, dtype=np.float32)
        # node types:
        # 0 = needs inference, not terminal
        # 1 = does not need inference, not terminal (current policy and value v are known)
        # 2 = terminal
        self.state[0] = root_state
        self.parent[0] = -1
        self.a0[0] = -1
        self.node_type[0] = 0
        self.depth[0] = 0
        self.child_index[0,:] = -1

    @property
    def nrows(self):
        return self.num_rows

    def _expand(self):
        assert self.capacity < (1<<24) # just to be safe, since we use int32 for indices
        old_capacity = self.capacity
        self.capacity *= 2
        n = game.numActions()

        def _grow_2d(arr, rows, cols):
            new = np.zeros((rows, cols), dtype=arr.dtype)
            new[:old_capacity] = arr[:old_capacity]
            return new

        def _grow_1d(arr, rows):
            new = np.zeros(rows, dtype=arr.dtype)
            new[:old_capacity] = arr[:old_capacity]
            return new

        self.P           = _grow_2d(self.P,           self.capacity, n)
        self.Q           = _grow_2d(self.Q,           self.capacity, n)
        self.N           = _grow_2d(self.N,           self.capacity, n)
        self.child_index = _grow_2d(self.child_index, self.capacity, n)
        self.state       = _grow_2d(self.state,       self.capacity, game.gameLength())
        self.v                 = _grow_1d(self.v,                 self.capacity)
        self.parent            = _grow_1d(self.parent,            self.capacity)
        self.a0                = _grow_1d(self.a0,                self.capacity)
        self.total_sims        = _grow_1d(self.total_sims,        self.capacity)
        self.new_sims          = _grow_1d(self.new_sims,          self.capacity)
        self.new_sims_children = _grow_1d(self.new_sims_children, self.capacity)
        self.node_type         = _grow_1d(self.node_type,         self.capacity)
        self.depth             = _grow_1d(self.depth,             self.capacity)
        self.path_logit        = _grow_1d(self.path_logit,        self.capacity)

    def _add_child(self, parent_index, a):
        '''
        adds new node to tree if it isn't already present
        returns index of child node
        '''
        assert parent_index >= 0 and parent_index < self.num_rows
        g = self.state[parent_index]
        h = game.nextState(g, a)
        ended_h, score_h = game.gameEnded(h)
        if self.child_index[parent_index, a] == -1:
            # add new entry
            if self.capacity == self.num_rows:
                # make room for new entry
                self._expand()
            child_index = self.num_rows
            self.num_rows += 1
            self.state[child_index] = h
            self.child_index[parent_index,a] = child_index
            self.parent[child_index] = parent_index
            self.a0[child_index] = a
            self.depth[child_index] = self.depth[parent_index] + 1
            self.node_type[child_index] = 0 # new, needs inference if nonterminal
            self.child_index[child_index,:] = -1 # initialize child indices to -1
            self.path_logit[child_index] = self.path_logit[parent_index] + np.log2(self.P[parent_index,a])
        else:
            child_index = self.child_index[parent_index,a]
            self.node_type[child_index] = 1 # not new (might be terminal; if so later change to 2)
        if ended_h:
            self.node_type[child_index] = 2 # terminal
            self.v[child_index] = score_h * game.playerId(h)
        return child_index
    
    def _get_inferences(self, indices):
        '''
        gets network policy and value for indicated states
        and records P and v entries for these states
        changes node type to 1 for these states
        '''
        assert np.all(self.node_type[indices] == 0)
        states = self.state[indices]
        pi, v = self.engine(states)
        self.P[indices] = pi
        self.v[indices] = v.flatten()
        self.node_type[indices] = 1

    def _legalize_policies(self, indices):
        for i in indices:
            g = self.state[i]
            actions = game.getValidActions(g)
            if len(actions) == 0:
                assert self.node_type[i] == 2
                continue
            pi_legal = np.zeros(game.numActions(), dtype=np.float32)
            pi_legal[actions] = self.P[i,actions]
            if np.sum(pi_legal) > 0.0:
                pi_legal /= np.sum(pi_legal)
            else:
                pi_legal[actions] = 1.0 / len(actions)
            self.P[i] = pi_legal

    def _expand_node(self, i, temperature):
        '''
        adds (potentially) new children C to node at index i
        returns C_nonterminal, C_terminal
        '''
        C_nonterminal = []
        C_terminal = []
        n = self.new_sims_children[i]
        if n == 0:
            # no budget so returning empty lists
            return C_nonterminal, C_terminal
        pi = self.P[i]
        actions = np.argwhere(pi > 0.0).flatten()
        #print(f"_expand_node: {i} has {n} sims, legal actions {actions}, pi {pi[actions]}")
        pi_actions = np.power(pi[actions], 1.0/temperature)
        pi_actions /= np.sum(pi_actions)
        s = assign_simulations(n, pi_actions)
        #print(f"_expand_node: s = {s}")
        s_POS = np.argwhere(s > 0).flatten()
        A = actions[s_POS]
        #print(f"_expand_node: actions A = {A}")
        for p in s_POS:
            a = actions[p]
            child_index = self._add_child(i, a)
            self.new_sims[child_index] += s[p]
            if self.node_type[child_index] < 2:
                self.new_sims_children[child_index] += s[p]
            #print(f"_expand_node: {s[p]} sims assigned to child {child_index} for action {a}, node_type {self.node_type[child_index]}, new_sims_children {self.new_sims_children[child_index]}")
            if self.node_type[child_index] < 2:
                C_nonterminal.append(child_index)
            else:
                C_terminal.append(child_index)
        return C_nonterminal, C_terminal

    def explore(self, numSims, temperature=1.0):
        assert temperature > 0.0
        # run numSims simulations of RMCTS
        if numSims <= 0:
            return
        self.new_sims[0] += numSims
        self.new_sims_children[0] += numSims # might be reduced by 1 if we do inference at the root, but we will fix that later
        T = set([0]) # set of nodes (indices) in this new tree to explore, initialized with the root node
        S_nonterminal = [0] # stack of node (indices) at current depth in T
        S_terminal = []
        leaves = set() # indices of leaf nodes : either terminal state, or no sims for children.
        depth = 0
        while numSims > 0 and len(S_nonterminal) > 0:
            inference_stack = [i for i in S_nonterminal if self.node_type[i] == 0]
            if len(inference_stack) > 0:
                # run inference on the inference stack (expensive step, so we want to do it in batches)
                self._get_inferences(inference_stack)
                # legalize each new policy
                self._legalize_policies(inference_stack)
            # update sim counts for children budget
            for i in inference_stack:
                # decrement children budget by 1 because parent consumed 1 sim
                self.new_sims_children[i] -= 1
                numSims -= 1
            # assign sims to individual children of nodes in S_nonterminal
            C_nonterminal = []
            C_terminal = []
            for i in S_nonterminal:
                n_children = self.new_sims_children[i]
                if n_children == 0:
                    leaves.add(i)
                    continue
                Ci_nonterminal, Ci_terminal = self._expand_node(i, temperature)
                C_nonterminal.extend(Ci_nonterminal)
                C_terminal.extend(Ci_terminal)
            S_nonterminal = C_nonterminal
            S_terminal = C_terminal
            depth += 1
            T.update(S_nonterminal)
            T.update(S_terminal)
            leaves.update(S_terminal)
            print(f"explore: depth {depth}, num_nodes {len(T)}, numSims {numSims}")
        return T, leaves

    def export_tree_json(self, file_path, max_row=None):
        '''
        Export the currently explored tree to a JSON file.

        Node i gets label self.v[i].
        Edge parent(i) -> i gets label action self.a0[i].

        Parameters
        ----------
        file_path : str or Path
            Output JSON path.
        max_row : int or None
            Optional inclusive upper row bound. If None, exports through
            self.num_rows - 1.
        '''
        if max_row is None:
            last_row = self.num_rows - 1
        else:
            last_row = min(int(max_row), self.num_rows - 1)
        if last_row < 0:
            raise ValueError('tree is empty, nothing to export')

        nodes = []
        edges = []

        for i in range(last_row + 1):
            nodes.append({
                'id': int(i),
                'label': float(self.v[i]),
            })

            if i == 0:
                continue
            p = int(self.parent[i])
            if p < 0 or p > last_row:
                continue
            edges.append({
                'source': p,
                'target': int(i),
                'label': int(self.a0[i]),
            })

        payload = {
            'root': 0,
            'nrows': int(last_row + 1),
            'nodes': nodes,
            'edges': edges,
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        return payload

