import numpy as np
import sys
from time import perf_counter
import json

from build.othello import game, inference, MCTS_ucb, RMCTS

def main():
    num_games = 32   # 32 first + 32 second = 64 total
    C_rmcts = 1
    N_rmcts = 512
    C_ucb = 1
    N_ucb = 256
    temperature = 0.2
    
    print("=" * 70)
    print("RMCTS Reproduction: Othello Head-to-Head")
    print("RMCTS: N={}, C={}".format(N_rmcts, C_rmcts))
    print("MCTS-UCB: N={}, C={}".format(N_ucb, C_ucb))
    print("Games: {} as first + {} as second = {} total".format(num_games, num_games, 2*num_games))
    print("Temperature: {}".format(temperature))
    print("=" * 70)
    
    print("\nLoading ONNX model and building TensorRT engine...")
    onnx_path = "./othello/models/ResNet_8blocks_48channels.onnx"
    engine = inference.Engine(onnx_path)
    print("Engine ready.")
    
    g0 = game.rootState()
    first_player = game.playerId(g0)
    
    scores_rmcts_first = []
    scores_rmcts_second = []
    time_rmcts = 0.0
    time_ucb = 0.0
    
    print("\nStarting {} games...".format(2*num_games))
    
    for i in range(num_games):
        # Game 1: RMCTS as first player, MCTS-UCB as second
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player_on_move = game.playerId(g)
            if player_on_move == first_player:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1, -1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0
            
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        
        # Multiply by 8 for checker difference
        scores_rmcts_first.append(8.0 * score * first_player)
        
        # Game 2: RMCTS as second player, MCTS-UCB as first
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player_on_move = game.playerId(g)
            if player_on_move == first_player:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1, -1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0
            
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        
        # Multiply by 8 for checker difference
        scores_rmcts_second.append(-8.0 * score * first_player)
        
        cum_score = sum(scores_rmcts_first) + sum(scores_rmcts_second)
        print("  Game pair {:3d}/{}: cumulative score (checker diff) = {:8.1f}".format(i+1, num_games, cum_score))
        sys.stdout.flush()
    
    all_scores = scores_rmcts_first + scores_rmcts_second
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    total_games = 2 * num_games
    mean_time_rmcts = (time_rmcts / total_games) * 1000
    mean_time_ucb = (time_ucb / total_games) * 1000
    speedup = time_ucb / time_rmcts if time_rmcts > 0 else float('inf')
    
    print()
    print("=" * 70)
    print("REPRODUCTION RESULTS")
    print("=" * 70)
    print("Mean Score (RMCTS vs MCTS-UCB, checker diff): {:.2f}".format(mean_score))
    print("Std Dev of Score: {:.2f}".format(std_score))
    print("Mean Time per Game (RMCTS): {:.2f} ms".format(mean_time_rmcts))
    print("Mean Time per Game (MCTS-UCB): {:.2f} ms".format(mean_time_ucb))
    print("Speedup: {:.2f}x".format(speedup))
    print("Total RMCTS time: {:.3f} s".format(time_rmcts))
    print("Total MCTS-UCB time: {:.3f} s".format(time_ucb))
    print()
    
    print("RUBRIC COMPARISON")
    print("  Mean Score:       got {:.2f}, target 7.2  (CI: [0.0, 7.92])".format(mean_score))
    print("  Mean Time (RMCTS): got {:.2f} ms, target 178 ms (CI: [-34.2, 2300])".format(mean_time_rmcts))
    print("  Speedup:          got {:.2f}x, target 13x  (CI: [1.0, 14.2])".format(speedup))
    
    results = {
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "mean_time_rmcts_ms": float(mean_time_rmcts),
        "mean_time_ucb_ms": float(mean_time_ucb),
        "speedup": float(speedup),
        "total_games": total_games,
        "N_rmcts": N_rmcts,
        "N_ucb": N_ucb,
        "temperature": temperature,
        "scores_rmcts_first": [float(x) for x in scores_rmcts_first],
        "scores_rmcts_second": [float(x) for x in scores_rmcts_second],
    }
    
    with open("/repo/reproduction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /repo/reproduction_results.json")
    
    # Print final line for easy parsing
    print("FINAL_METRIC: mean_score={:.2f} mean_time_rmcts_ms={:.2f} speedup={:.2f}".format(
        mean_score, mean_time_rmcts, speedup))

if __name__ == "__main__":
    main()
