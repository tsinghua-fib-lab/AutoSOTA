from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

from build.othello import game, inference, MCTS_ucb, RMCTS
#from build.connect4 import game, inference, MCTS_ucb, RMCTS
#from build.dotbox import game, inference, MCTS_ucb, RMCTS

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
        
        # othello: score should be multiplied by 8 to obtain checker difference
        # scores_nnet1_first_player.append(8 * score * first_player)
        scores_nnet1_first_player.append(score * first_player)

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
        
        # othello: score should be multiplied by 8 to obtain checker difference
        # scores_nnet1_second_player.append(-8 * score * first_player)
        scores_nnet1_second_player.append(-score * first_player)
        
        if verbose:
            print(f"after game {i+1:3d}, player1 sum of scores: {sum(scores_nnet1_first_player) + sum(scores_nnet1_second_player):8.0f}.  times = ({time_first_player:8.2f}, {time_second_player:8.2f})")
    if verbose:
        print()
    return scores_nnet1_first_player, scores_nnet1_second_player, time_first_player, time_second_player



if __name__ == '__main__':
    num_games = 32
    C1 = 1
    N1 = 256
    C2 = 1
    N2 = 256
    method1 = 'rmcts'
    method2 = 'mcts_ucb'
    temperature = 0.5

    p1 = f"({method1}, {N1} {C1})"
    p2 = f"({method2}, {N2} {C2})"

    print("Othello MCTS Pit!")
    #print("Connect4 MCTS Pit!")
    #print("DotBox MCTS Pit!")
    

    print(f"Playing {num_games} games.")
    print(f"Each game is really two games, with players alternating who goes first.")
    print(f"player1: {method1}, N = {N1}, C = {C1}")
    print(f"player2: {method2}, N = {N2}, C = {C2}")
    print(f"Temperature used = {temperature} (smaller/colder approaches argmax policy)")

    print("loading network engine...")
    nnet1 = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    #nnet1 = inference.Engine("./connect4/models/ResNet_8blocks_64channels.onnx")
    #nnet1 = inference.Engine("./dotbox/models/ResNet_8blocks_48channels.onnx")
    nnet2 = nnet1
    print("finishing loading network engine.")
    g0 = game.rootState()

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
    
    print(f"Scores as first player: {scores_as_first}")
    print(f"Scores as second player: {scores_as_second}")

    print(f"Average score of player1 = {np.mean(scores_as_first + scores_as_second)}")
    print(f"Total time taken for {p1}:", t1)
    print(f"Total time taken for {p2}:", t2)

    score1_256 = np.array(scores_as_first)
    score2_256 = np.array(scores_as_second)
    scores_256 = 0.5*(score1_256 + score2_256)

    print("mean score at N=256:", np.mean(scores_256))
    print("std dev of score at N=256:", np.std(scores_256))
    print("mean(time) for rmcts, N=256:", t1/(2*num_games))
    print("mean(time) for mcts_ucb, N=256:", t2/(2*num_games))
    print(f"speedup = {t2/t1:0.4f}")
    print()

    N1 = 512

    print("Starting second pit...")
    print(f"Playing {num_games} games.")
    print(f"player1: {method1}, N = {N1}, C = {C1}")
    print(f"player2: {method2}, N = {N2}, C = {C2}")
    print(f"Temperature used = {temperature} (smaller/colder approaches argmax policy)")

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

    print(f"Scores as first player: {scores_as_first}")
    print(f"Scores as second player: {scores_as_second}")

    print(f"Average score of player1 = {np.mean(scores_as_first + scores_as_second)}")
    print(f"Total time taken for {p1}:", t1)
    print(f"Total time taken for {p2}:", t2)

    score1_512 = np.array(scores_as_first)
    score2_512 = np.array(scores_as_second)
    scores_512 = 0.5*(score1_512 + score2_512)

    print("mean score at N=512:", np.mean(scores_512))
    print("std dev of score at N=512:", np.std(scores_512)) 
    print("mean(time) for rmcts, N=512:", t1/(2*num_games))
    print("mean(time) for mcts_ucb, N=512:", t2/(2*num_games))
    print(f"speedup = {t2/t1:0.4f}")

    
