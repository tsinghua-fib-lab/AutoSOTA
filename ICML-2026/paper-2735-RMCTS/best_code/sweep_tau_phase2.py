#!/usr/bin/env python3
"""Phase 2: tau sweep at best C from Phase 1."""
import numpy as np, sys, json
from time import perf_counter
from build.othello import game, inference, MCTS_ucb, RMCTS
import argparse

def run_match(num_games, N_rmcts, C_rmcts, N_ucb, C_ucb, temperature, engine, g0, first):
    s1, s2, t_rmcts, t_ucb = [], [], 0.0, 0.0
    for i in range(num_games):
        for flip in [0, 1]:
            g = g0.copy()
            ended, score = game.gameEnded(g)
            while not ended:
                player = game.playerId(g)
                is_rmcts = (player == first) ^ (flip == 1)
                t0 = perf_counter()
                if is_rmcts:
                    pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), N_rmcts, engine, c_puct=C_rmcts)
                else:
                    pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                if is_rmcts: t_rmcts += t1-t0
                else: t_ucb += t1-t0
                pi = pi.flatten()
                pi = np.power(pi, 1.0/temperature); pi /= np.sum(pi)
                a = np.random.choice(len(pi), p=pi)
                g = game.nextState(g, a)
                ended, score = game.gameEnded(g)
            if flip == 0: s1.append(8.0*score*first)
            else: s2.append(-8.0*score*first)
        if i % 8 == 7:
            print("    {:3d}/{} cum={:.0f}".format(i+1, num_games, sum(s1)+sum(s2)))
            sys.stdout.flush()
    scores = s1 + s2
    return {"mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
            "mean_time_rmcts_ms": float(t_rmcts/(2*num_games)*1000),
            "speedup": float(t_ucb/t_rmcts) if t_rmcts>0 else 0}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--best-c", type=float, required=True)
    p.add_argument("--num-games", type=int, default=32)
    args = p.parse_args()

    tau_vals = [0.1, 0.2, 0.3, 0.5, 0.05, 0.15, 0.25]
    print("Phase 2: tau sweep at C={:.2f}".format(args.best_c))
    engine = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    g0, first = game.rootState(), game.playerId(game.rootState())

    results = []
    for tau in tau_vals:
        print("\nTesting tau={:.2f}...".format(tau)); sys.stdout.flush()
        t0 = perf_counter()
        r = run_match(args.num_games, 512, args.best_c, 256, 1.0, tau, engine, g0, first)
        r["C"]=args.best_c; r["tau"]=tau; r["wall_s"]=float(perf_counter()-t0)
        results.append(r)
        print("  Score={:.2f} Time={:.1f}ms Speedup={:.2f}x".format(
            r["mean_score"], r["mean_time_rmcts_ms"], r["speedup"]))

    results.sort(key=lambda r: r["mean_score"], reverse=True)
    print("\nTAU SWEEP SUMMARY:")
    for r in results:
        print("  tau={:.2f} | Score={:.2f} | Time={:.1f}ms | Speedup={:.2f}x".format(
            r["tau"], r["mean_score"], r["mean_time_rmcts_ms"], r["speedup"]))
    best = results[0]
    print("BEST: C={:.2f} tau={:.2f} Score={:.2f} Time={:.1f}ms".format(
        best["C"], best["tau"], best["mean_score"], best["mean_time_rmcts_ms"]))
    with open("/repo/sweep_tau_results.json", "w") as f: json.dump(results, f, indent=2)
