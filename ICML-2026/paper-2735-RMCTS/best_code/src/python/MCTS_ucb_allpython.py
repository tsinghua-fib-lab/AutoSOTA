# this is an all-python implementation of MCTS_ucb 
import numpy as np
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path
import time
import matplotlib.pyplot as plt

from . import game, metaparm

UCB_EPSILON = 0.1

try:
    from taz.build import advice
    hint = advice.hint
except:
    n = game.numActions()
    uniform_prob = np.float32(1.0)/np.float32(n)
    def hint(G):
        assert len(G.shape) == 2
        assert G.shape[1] == game.gameLength()
        m = G.shape[0]
        pi = np.zeros((m,n), dtype=np.float32)
        pi[:] = uniform_prob
        v = np.zeros((m,1),dtype=np.float32)
        return pi, v


def policy_and_value(g, engine):
    if engine is None:
        pi,v = hint(g.reshape(1,-1))
        return pi[0], v[0]
    else:
        pi, v = engine(g.reshape(1,-1))
        return pi[0], v[0]

class MonteCarloTree:
    def __init__(self, root_state, engine=None):
        assert root_state.dtype == np.float32
        assert len(root_state) == game.gameLength()
        ended,terminal_score = game.gameEnded(root_state)
        assert not ended
        self.root_state = np.copy(root_state)
        self.P = {} # policy
        self.V = {} # value
        self.N = {} # frequency counts
        self.Q = {} # average value/quality
        self.c_puct = metaparm.c_puct
        self.engine = engine
        self.num_sims = 0
        self.n = game.numActions()
        self.initialize_root()
        self.root_key = self.root_state.tobytes()
        self.root_policy = self.P[self.root_key]
        self.root_value = self.V[self.root_key]
        self.Qall = np.zeros((0,self.n), dtype=np.float32)
        #self.Qall = self.root_value * np.ones((0,self.n), dtype=np.float32)
        self.Nall = np.zeros((0,self.n), dtype=np.float32)
        self.UCBall = np.zeros((0,self.n), dtype=np.float32)
        self.cpu_time = 0.0
        self.gpu_time = 0.0
               
    
    def initialize_root(self):
        root_key = self.root_state.tobytes()
        pi,v = policy_and_value(self.root_state, self.engine)
        self.P[root_key] = pi
        self.V[root_key] = v
        self.Q[root_key] = np.zeros(self.n, dtype=np.float32)
        #self.Q[root_key] = v*np.ones(self.n, dtype=np.float32)
        self.N[root_key] = np.zeros(self.n, dtype=np.float32)
        
    def propagate_value(self, game_path, action_path, value):
        for i,a in enumerate(action_path):
            g = game_path[i]
            gkey = g.tobytes()
            v = value * game.playerId(g)
            q_a = self.Q[gkey][a]
            n_a = self.N[gkey][a]
            self.Q[gkey][a] = (q_a * n_a + v) / (n_a + 1.0)
            self.N[gkey][a] = n_a + 1.0
            if gkey == self.root_key:
                self.Qall = np.r_[self.Qall, self.Q[gkey].reshape(1,self.n)]
                self.Nall = np.r_[self.Nall, self.N[gkey].reshape(1,self.n)]

    def compute_ucb(self,gkey):
        c_puct = self.c_puct
        ucb = self.Q[gkey] + c_puct * self.P[gkey] * np.sqrt(np.sum(self.N[gkey])+UCB_EPSILON)/(1.0 + self.N[gkey])        
        return ucb

    def run_simulation(self):
        t0 = time.time()
        g = self.root_state
        gkey = g.tobytes()
        game_path = [g]
        action_path = []
        while True:            
            ended,terminal_score=game.gameEnded(g)
            if ended:
                self.propagate_value(game_path, action_path, terminal_score)
                break
            elif gkey not in self.P:
                t1 = time.time()
                self.cpu_time += t1-t0
                t0 = t1
                pi,v = policy_and_value(g,self.engine)
                t1 = time.time()
                self.gpu_time += t1-t0
                t0 = t1
                self.P[gkey] = pi
                self.V[gkey] = v
                self.N[gkey] = np.zeros(self.n, dtype=np.float32)
                self.Q[gkey] = np.zeros(self.n, dtype=np.float32)
                #self.Q[gkey] = v*np.ones(self.n, dtype=np.float32)
                value = v * game.playerId(g)
                self.propagate_value(game_path, action_path, value)
                break
            else:
                ucb = self.compute_ucb(gkey)
                if gkey == self.root_key:
                    self.UCBall = np.r_[self.UCBall, ucb.reshape(1,self.n)]
                valid_actions = game.getValidActions(g)
                assert len(valid_actions) > 0
                ucb_valid = ucb[valid_actions]
                i_a = np.argmax(ucb_valid)
                a = valid_actions[i_a]
                g = game.nextState(g,a)
                gkey = g.tobytes()
                game_path.append(g)
                action_path.append(a)
        self.num_sims += 1
        t1 = time.time()
        self.cpu_time += t1-t0
    
    def run_mcts(self, numSims):
        for _ in range(numSims):
            self.run_simulation()
            
    def plot_mcts_data(self, actions, legend=True, cmap=plt.cm.Reds):
        fig,ax = plt.subplots(4)
        ax[0].set_title("value")
        ax[1].set_title("frequency")
        ax[2].set_title("policy")
        ax[3].set_title("UCB")
        c = np.linspace(0.2, 1.0, len(actions))
        pi = self.Nall / (1.0 + np.sum(self.Nall,axis=1)).reshape(-1,1)
        for i,a in enumerate(actions):
            ax[0].plot(self.Qall[:,a], color=cmap(c[i]), label=a)
            ax[1].plot(self.Nall[:,a], color=cmap(c[i]), label=a)
            m = self.Nall.shape[0]
            N0 = self.root_policy[a] * m
            ax[1].plot(m, N0, '.', color=cmap(c[i]), markersize=10)
            ax[2].plot(pi[:,a], color=cmap(c[i]), label=a)
            ax[2].plot(m, self.root_policy[a], '.', color=cmap(c[i]), markersize=10)
            ax[3].plot(self.UCBall[:,a], color=cmap(c[i]), label=a)
        if legend:
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
        plt.show()

def learn_N_and_Q(root_state, numSims, engine):
    t = MonteCarloTree(root_state, engine=engine)
    t.run_mcts(numSims)
    k = t.root_state.tobytes()
    N = t.N[k]
    Q = t.Q[k]
    return N,Q
