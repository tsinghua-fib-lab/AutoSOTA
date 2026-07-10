#!/usr/bin/env python3
"""Phase 1: C sweep at tau=0.2 for RMCTS."""
import numpy as np
import sys
import json
from time import perf_counter

from build.othello import game, inference, MCTS_ucb, RMCTS

def run_match(num_games, N_rmcts, C_rmcts, N_ucb, C_ucb, temperature, engine, g0, first_player):
    scores_first = []
    scores_second = []
    time_rmcts = 0.0
    time_ucb = 0.0

    for i in range(num_games):
        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player = game.playerId(g)
            if player == first_player:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        scores_first.append(8.0 * score * first_player)

        g = g0.copy()
        ended, score = game.gameEnded(g)
        while not ended:
            player = game.playerId(g)
            if player == first_player:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                time_ucb += t1 - t0
            else:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                time_rmcts += t1 - t0
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            g = game.nextState(g, a)
            ended, score = game.gameEnded(g)
        scores_second.append(-8.0 * score * first_player)

        if i % 8 == 7:
            cum = sum(scores_first) + sum(scores_second)
            print("    {:3d}/{} cum={:.0f}".format(i+1, num_games, cum))
            sys.stdout.flush()

    all_scores = scores_first + scores_second
    return {
        "mean_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
        "mean_time_rmcts_ms": float((time_rmcts / (2*num_games)) * 1000),
        "speedup": float(time_ucb / time_rmcts) if time_rmcts > 0 else 0,
    }

def main():
    num_games = 32
    C_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tau = 0.2
    N_rmcts = 512
    N_ucb = 256
    C_ucb = 1.0

    print("=" * 60)
    print("Phase 1: C sweep at tau={}".format(tau))
    print("C values: {}".format(C_values))
    print("N_rmcts={}, N_ucb={}, {} game pairs".format(N_rmcts, N_ucb, num_games))
    print("=" * 60)

    engine = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    g0 = game.rootState()
    first = game.playerId(g0)

    results = []
    for C in C_values:
        label = "C={:.2f} τ={:.2f}".format(C, tau)
        print("\nTesting {}...".format(label))
        sys.stdout.flush()
        t0 = perf_counter()
        r = run_match(num_games, N_rmcts, C, N_ucb, C_ucb, tau, engine, g0, first)
        t1 = perf_counter()
        r["C"] = C; r["tau"] = tau; r["wall_s"] = float(t1-t0)
        results.append(r)
        print("  Score={:.2f} Time={:.1f}ms Speedup={:.2f}x ({}s)".format(
            r["mean_score"], r["mean_time_rmcts_ms"], r["speedup"], int(t1-t0)))

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY (sorted by Mean Score)")
    print("=" * 60)
    results.sort(key=lambda r: r["mean_score"], reverse=True)
    for r in results:
        print("  C={:.2f} | Score={:.2f} | Time={:.1f}ms | Speedup={:.2f}x".format(
            r["C"], r["mean_score"], r["mean_time_rmcts_ms"], r["speedup"]))

    best = results[0]
    print("\nBEST: C={:.2f} Score={:.2f} Time={:.1f}ms".format(
        best["C"], best["mean_score"], best["mean_time_rmcts_ms"]))
    print("PHASE1_BEST: C={:.2f} mean_score={:.2f} mean_time_rmcts_ms={:.1f} speedup={:.2f}".format(
        best["C"], best["mean_score"], best["mean_time_rmcts_ms"], best["speedup"]))

    with open("/repo/sweep_c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to /repo/sweep_c_results.json")

if __name__ == "__main__":
    main()
