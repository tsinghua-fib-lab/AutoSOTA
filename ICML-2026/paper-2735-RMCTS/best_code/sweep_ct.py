#!/usr/bin/env python3
"""Batch C×τ sweep runner for RMCTS optimization."""
import numpy as np
import sys
import json
from time import perf_counter

from build.othello import game, inference, MCTS_ucb, RMCTS


def run_match(args, engine, g0, first_player):
    """Run one match configuration. Returns (scores, time_rmcts, time_ucb)."""
    scores_rmcts_first = []
    scores_rmcts_second = []
    time_rmcts = 0.0
    time_ucb = 0.0

    for i in range(args["num_games"]):
        # Game 1: RMCTS first
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player = game.playerId(g)
            if player == first_player:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), args["N_rmcts"], engine, c_puct=args["C_rmcts"])
                t1 = perf_counter()
                time_rmcts += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), args["N_ucb"], engine, c_puct=args["C_ucb"])
                t1 = perf_counter()
                time_ucb += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / args["temperature"])
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        scores_rmcts_first.append(8.0 * score * first_player)

        # Game 2: RMCTS second
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player = game.playerId(g)
            if player == first_player:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), args["N_ucb"], engine, c_puct=args["C_ucb"])
                t1 = perf_counter()
                time_ucb += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), args["N_rmcts"], engine, c_puct=args["C_rmcts"])
                t1 = perf_counter()
                time_rmcts += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / args["temperature"])
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        scores_rmcts_second.append(-8.0 * score * first_player)

        cum = sum(scores_rmcts_first) + sum(scores_rmcts_second)
        if i % 4 == 3 or i == args["num_games"] - 1:
            print("    Game pair {:3d}/{}: cum = {:8.1f}".format(i+1, args["num_games"], cum))
            sys.stdout.flush()

    all_scores = scores_rmcts_first + scores_rmcts_second
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    total = 2 * args["num_games"]
    mean_time = (time_rmcts / total) * 1000
    speedup_ratio = time_ucb / time_rmcts if time_rmcts > 0 else 0

    return {
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "mean_time_rmcts_ms": float(mean_time),
        "speedup": float(speedup_ratio),
    }


def main():
    # Common args
    base = {
        "num_games": 32,
        "N_ucb": 256,
        "C_ucb": 1.0,
    }

    # C × τ sweep grid
    C_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tau_values = [0.1, 0.2, 0.3, 0.5]

    print("=" * 70)
    print("RMCTS C×τ Systematic Sweep")
    print("C ∈ {}, τ ∈ {}".format(C_values, tau_values))
    print("N_rmcts=512, N_ucb=256, 32 game pairs per config")
    print("=" * 70)

    print("\nLoading ONNX model and building TensorRT engine (shared)...")
    t0 = perf_counter()
    engine = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    t1 = perf_counter()
    print("Engine ready ({:.1f}s).".format(t1 - t0))

    g0 = game.rootState()
    first_player = game.playerId(g0)

    results = []
    total_configs = len(C_values) * len(tau_values)
    config_idx = 0

    for C in C_values:
        for tau in tau_values:
            config_idx += 1
            args = {**base, "C_rmcts": C, "temperature": tau}
            label = "C={:.2f} τ={:.2f}".format(C, tau)

            print("\n[{}/{}] Testing {}...".format(config_idx, total_configs, label))
            sys.stdout.flush()

            t_start = perf_counter()
            result = run_match(args, engine, g0, first_player)
            t_end = perf_counter()

            result["C_rmcts"] = C
            result["temperature"] = tau
            result["wall_clock_s"] = float(t_end - t_start)

            results.append(result)
            print("  RESULT: Mean Score={:.2f}, Time={:.1f}ms, Speedup={:.2f}x (wall {:.0f}s)".format(
                result["mean_score"], result["mean_time_rmcts_ms"],
                result["speedup"], result["wall_clock_s"]))

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (sorted by Mean Score)")
    print("=" * 70)
    results.sort(key=lambda r: r["mean_score"], reverse=True)
    print("{:<20s} {:>10s} {:>10s} {:>10s}".format("Config", "Score", "Time(ms)", "Speedup"))
    print("-" * 50)
    for r in results:
        label = "C={:.2f} τ={:.2f}".format(r["C_rmcts"], r["temperature"])
        print("{:<20s} {:>10.2f} {:>10.1f} {:>10.2f}x".format(
            label, r["mean_score"], r["mean_time_rmcts_ms"], r["speedup"]))

    # Save
    with open("/repo/sweep_all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to /repo/sweep_all_results.json")

    # Print best for parsing
    best = results[0]
    print("SWEEP_BEST: C={:.2f} tau={:.2f} mean_score={:.2f} mean_time_rmcts_ms={:.1f} speedup={:.2f}".format(
        best["C_rmcts"], best["temperature"], best["mean_score"],
        best["mean_time_rmcts_ms"], best["speedup"]))


if __name__ == "__main__":
    main()
