"""Parameterized version of reproduce_final.py for C and tau sweeps."""
import numpy as np
import sys
import argparse
from time import perf_counter
import json

from build.othello import game, inference, MCTS_ucb, RMCTS

def main():
    parser = argparse.ArgumentParser(description='RMCTS vs MCTS-UCB Head-to-Head')
    parser.add_argument('--num-games', type=int, default=32, help='Number of game pairs (default: 32)')
    parser.add_argument('--C-rmcts', type=float, default=1.0, help='RMCTS exploration constant (default: 1.0)')
    parser.add_argument('--N-rmcts', type=int, default=512, help='RMCTS simulation budget (default: 512)')
    parser.add_argument('--C-ucb', type=float, default=1.0, help='MCTS-UCB exploration constant (default: 1.0)')
    parser.add_argument('--N-ucb', type=int, default=256, help='MCTS-UCB simulation budget (default: 256)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Action selection temperature (default: 0.2)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--onnx-path', type=str, default='./othello/models/ResNet_8blocks_48channels.onnx',
                        help='Path to ONNX model')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 70)
    print("RMCTS Reproduction: Othello Head-to-Head [PARAM SWEEP]")
    print("RMCTS: N={}, C={}".format(args.N_rmcts, args.C_rmcts))
    print("MCTS-UCB: N={}, C={}".format(args.N_ucb, args.C_ucb))
    print("Games: {} as first + {} as second = {} total".format(
        args.num_games, args.num_games, 2*args.num_games))
    print("Temperature: {}".format(args.temperature))
    print("Seed: {}".format(args.seed))
    print("=" * 70)

    print("\nLoading ONNX model and building TensorRT engine...")
    engine = inference.Engine(args.onnx_path)
    print("Engine ready.")

    g0 = game.rootState()
    first_player = game.playerId(g0)

    scores_rmcts_first = []
    scores_rmcts_second = []
    time_rmcts = 0.0
    time_ucb = 0.0

    print("\nStarting {} games...".format(2*args.num_games))

    for i in range(args.num_games):
        # Game 1: RMCTS as first player, MCTS-UCB as second
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player_on_move = game.playerId(g)
            if player_on_move == first_player:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), args.N_rmcts, engine, c_puct=args.C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1, -1), args.N_ucb, engine, c_puct=args.C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0

            pi = pi.flatten()
            pi = np.power(pi, 1.0 / args.temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)

        scores_rmcts_first.append(8.0 * score * first_player)

        # Game 2: RMCTS as second player
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player_on_move = game.playerId(g)
            if player_on_move == first_player:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1, -1), args.N_ucb, engine, c_puct=args.C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), args.N_rmcts, engine, c_puct=args.C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0

            pi = pi.flatten()
            pi = np.power(pi, 1.0 / args.temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)

        scores_rmcts_second.append(-8.0 * score * first_player)

        cum_score = sum(scores_rmcts_first) + sum(scores_rmcts_second)
        print("  Game pair {:3d}/{}: cumulative score = {:8.1f}".format(
            i+1, args.num_games, cum_score))
        sys.stdout.flush()

    all_scores = scores_rmcts_first + scores_rmcts_second
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)

    total_games = 2 * args.num_games
    mean_time_rmcts = (time_rmcts / total_games) * 1000
    mean_time_ucb = (time_ucb / total_games) * 1000
    speedup = time_ucb / time_rmcts if time_rmcts > 0 else float('inf')

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Mean Score (checker diff): {:.2f}".format(mean_score))
    print("Std Dev of Score: {:.2f}".format(std_score))
    print("Mean Time per Game (RMCTS): {:.2f} ms".format(mean_time_rmcts))
    print("Mean Time per Game (MCTS-UCB): {:.2f} ms".format(mean_time_ucb))
    print("Speedup: {:.2f}x".format(speedup))

    print("FINAL_METRIC: mean_score={:.2f} mean_time_rmcts_ms={:.2f} speedup={:.2f}".format(
        mean_score, mean_time_rmcts, speedup))

    # Write results
    results = {
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "mean_time_rmcts_ms": float(mean_time_rmcts),
        "mean_time_ucb_ms": float(mean_time_ucb),
        "speedup": float(speedup),
        "total_games": total_games,
        "N_rmcts": args.N_rmcts,
        "C_rmcts": args.C_rmcts,
        "N_ucb": args.N_ucb,
        "C_ucb": args.C_ucb,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    with open("/repo/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
