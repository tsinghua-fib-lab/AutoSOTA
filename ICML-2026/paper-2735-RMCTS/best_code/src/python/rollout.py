import numpy as np
import matplotlib.pyplot as plt

from . import game, MCTS_new

def rollout(g, nnet, numSims, c_puct, temperature=1.0):
    '''
    Perform MCTS rollouts for a given game state g using neural network nnet.
    Returns game states G, network values V1 wrt to current player.
    '''
    g = np.copy(g)
    G = [g]
    V1 = []
    P = [] # probability (post mcts, and with temperature)
    starting_player = game.playerId(g)
    ended, score = game.gameEnded(g)
    while not ended:
        pi, v = MCTS_new.learn_pi_and_v(g.reshape(1,-1), numSims, nnet, c_puct=c_puct)
        pi = pi[0]
        v = v[0]
        V1.append(v * game.playerId(g) * starting_player)  # value wrt starting player  
        pi = np.power(pi, 1.0 / temperature)
        pi = pi / np.sum(pi)
        while True:
            a = np.random.choice(len(pi), p=pi) 
            if game.isValidAction(g, a):
                P.append(pi[a])
                break
            # should never reach this point
        g = game.nextState(g, a)
        G.append(g)
        ended, score = game.gameEnded(g) # score is wrt player 1 (not necessarily current player)
    V1.append(score * starting_player)
    return G, V1, P

def plot_several_rollouts(num_rollouts, g, nnet, numSims, c_puct, temperature=1.0):
    '''
    Perform and plot several MCTS rollouts from the same starting state g.
    '''
    plt.figure(figsize=(12, 12))
    for i in range(num_rollouts):
        G, V1, P = rollout(g, nnet, numSims, c_puct, temperature)
        geometric_mean_P = np.exp(np.mean(np.log(np.array(P))))
        plt.plot(V1, alpha=geometric_mean_P, label=f'{geometric_mean_P:.3f}')
    plt.xlabel("move number")
    plt.ylabel("estimated value wrt starting player")
    plt.legend()
    plt.tight_layout()
    plt.show()
