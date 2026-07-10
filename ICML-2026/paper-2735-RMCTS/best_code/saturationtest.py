from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

from build.othello import game, inference, MCTS_ucb, RMCTS

def pit(g0, nnet1, nnet2, numgames, numSims1, c1, numSims2, c2, method1 = 'rmcts', method2 = 'mcts_ucb', temperature=0.2, verbose = False):
    scores_nnet1_first_player = []
    scores_nnet1_second_player = []
    time_first_player = 0.0
    time_second_player = 0.0
    first_player = game.playerId(g0)
    for i in range(numgames):
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            if game.playerId(g) == first_player:
                player = 1
                (nnet, numSims, c, method) = (nnet1, numSims1, c1, method1)
            else:
                player = 2
                (nnet, numSims, c, method) = (nnet2, numSims2, c2, method2)
            t0 = perf_counter()
            if method == 'rmcts':
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), numSims, nnet, c_puct = c)
            else:
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), numSims, nnet, c_puct = c)
            t1 = perf_counter()
            if player == 1:
                time_first_player += t1 - t0
            else:
                time_second_player += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0/temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        
        scores_nnet1_first_player.append(8 * score * first_player)

        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            if game.playerId(g) == first_player:
                player = 2
                (nnet, numSims, c, method) = (nnet2, numSims2, c2, method2)
            else:
                player = 1
                (nnet, numSims, c, method) = (nnet1, numSims1, c1, method1)
            t0 = perf_counter()
            if method == 'rmcts':
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), numSims, nnet, c_puct = c)
            else:
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), numSims, nnet, c_puct = c)
            t1 = perf_counter()
            if player == 1:
                time_first_player += t1 - t0
            else:
                time_second_player += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0/temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        
        scores_nnet1_second_player.append(-8 * score * first_player)
        
        if verbose:
            print(f"after game {i+1:3d}, player1 sum of scores: {sum(scores_nnet1_first_player) + sum(scores_nnet1_second_player):8.0f}.  times = ({time_first_player:8.2f}, {time_second_player:8.2f})")

    return scores_nnet1_first_player, scores_nnet1_second_player, time_first_player, time_second_player

if __name__ == '__main__':
    num_games = 32
    C = 1

    print(f"Using exploration constant C = {C}")
    C1 = C
    C2 = C
    method1 = 'rmcts'
    method2 = 'rmcts'
    temperature = 0.5
    print(f"Temperature used = {temperature}")

    print("Checking strength saturation of RMCTS by pitting it against itself,")
    print("but where one player gets twice as many MCTS simulations as the other.")
    print("We try this for larger and larger number of simulations.")

    print("loading network engine...")
    nnet1 = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    nnet2 = nnet1
    print("finishing loading network engine.")
    g0 = game.rootState()

    Nlist = np.power(2, np.arange(6, 12))  # From 64 to 2048

    score_table = np.zeros(len(Nlist))
    time_table = np.zeros(len(Nlist))

    for i,N1 in enumerate(Nlist):
        N2 = N1 // 2
        scores_as_first, scores_as_second, t1, t2 = pit(g0,
                                                        nnet1,     
                                                        nnet2, 
                                                        numgames=num_games, 
                                                        numSims1=N1, 
                                                        c1=C1, 
                                                        numSims2=N2, 
                                                        c2=C2, 
                                                        method1=method1, 
                                                        method2=method2, 
                                                        temperature=temperature,
                                                        verbose = True)
        overall_score = np.mean(scores_as_first + scores_as_second)
        print(f"N1 = {N1:6d}, N2 = {N2:6d}, average score of player1 = {overall_score:8.4f}")
        score_table[i] = overall_score
        time_table[i] = t1
        print()

    print("Recap of scores for player having twice as many simulations:")
    print(score_table)
    print("Total time taken by the stronger player:")
    print(time_table)

    
