"""Test Dirichlet noise at root for RMCTS."""
import numpy as np, sys, json
from time import perf_counter
from build.othello import game, inference, MCTS_ucb, RMCTS

def main():
    num_games = 32
    C_rmcts, N_rmcts = 0.75, 512
    C_ucb, N_ucb = 1.0, 256
    tau = 0.2
    # Test configurations
    configs = [
        ("no_noise", 0.0, 0.0),
        ("alpha=0.3_frac=0.25", 0.3, 0.25),
        ("alpha=0.03_frac=0.25", 0.03, 0.25),
        ("alpha=0.3_frac=0.10", 0.3, 0.10),
    ]

    print("=" * 60)
    print("Dirichlet Noise Test")
    print("C={}, tau={}, N_rmcts={}".format(C_rmcts, tau, N_rmcts))
    print("=" * 60, flush=True)

    engine = inference.Engine("./othello/models/ResNet_8blocks_48channels.onnx")
    g0 = game.rootState()
    first = game.playerId(g0)

    for label, alpha, frac in configs:
        print("\n--- Testing {} ---".format(label), flush=True)
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
                        pi, v = RMCTS.learn_pi_and_v(g.reshape(1,-1), N_rmcts, engine,
                                                      c_puct=C_rmcts,
                                                      dirichlet_alpha=alpha if flip==0 else alpha,
                                                      dirichlet_frac=frac if flip==0 else frac)
                    else:
                        pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1,-1), N_ucb, engine, c_puct=C_ucb)
                    t1 = perf_counter()
                    if is_rmcts: t_rmcts += t1-t0
                    else: t_ucb += t1-t0
                    pi = pi.flatten()
                    pi = np.power(pi, 1.0/tau); pi /= np.sum(pi)
                    a = np.random.choice(len(pi), p=pi)
                    g = game.nextState(g, a)
                    ended, score = game.gameEnded(g)
                if flip == 0: s1.append(8.0*score*first)
                else: s2.append(-8.0*score*first)
            if i % 8 == 7:
                cum = sum(s1) + sum(s2)
                print("  {:3d}/{} cum={:.0f}".format(i+1, num_games, cum), flush=True)

        scores = s1 + s2
        ms = float(np.mean(scores))
        mt = float(t_rmcts/(2*num_games)*1000)
        sp = float(t_ucb/t_rmcts) if t_rmcts>0 else 0
        print("  RESULT: Score={:.2f} Time={:.1f}ms Speedup={:.2f}x".format(ms, mt, sp), flush=True)
        print("DIR_TEST: label={} score={:.2f} time={:.1f} speedup={:.2f}".format(label, ms, mt, sp), flush=True)

if __name__ == "__main__":
    main()
