import numpy as np
import sys
from time import perf_counter
import json
import collections

from build.othello import game, inference, MCTS_ucb, RMCTS

def main():
    num_games = 32   # 32 first + 32 second = 64 total
    C_rmcts = 1
    N_rmcts = 512
    C_ucb = 1
    N_ucb = 256
    temperature = 0.2

    # Profiling accumulators
    profile = {
        "rmcts_calls": 0, "rmcts_total_time": 0.0, "rmcts_times": [],
        "ucb_calls": 0, "ucb_total_time": 0.0, "ucb_times": [],
        "temperature_calls": 0, "temperature_total_time": 0.0,
        "game_step_calls": 0, "game_step_total_time": 0.0,
        "rmcts_inference_calls": 0, "rmcts_inference_total": 0.0,
        "ucb_inference_calls": 0, "ucb_inference_total": 0.0,
    }

    print("=" * 70)
    print("RMCTS Reproduction: Othello Head-to-Head [PROFILED]")
    print("RMCTS: N={}, C={}".format(N_rmcts, C_rmcts))
    print("MCTS-UCB: N={}, C={}".format(N_ucb, C_ucb))
    print("Games: {} as first + {} as second = {} total".format(num_games, num_games, 2*num_games))
    print("Temperature: {}".format(temperature))
    print("=" * 70)

    print("\nLoading ONNX model and building TensorRT engine...")
    onnx_path = "./othello/models/ResNet_8blocks_48channels.onnx"
    t0 = perf_counter()
    engine = inference.Engine(onnx_path)
    t1 = perf_counter()
    print("Engine ready ({:.1f}s build time). max_batchsize={}, opt_batchsize={}".format(
        t1-t0, engine.max_batchsize, engine.opt_batchsize))

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
        game_moves = 0
        while not ended:
            player_on_move = game.playerId(g)
            if player_on_move == first_player:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                elapsed = t1 - t0
                time_rmcts += elapsed
                profile["rmcts_calls"] += 1
                profile["rmcts_total_time"] += elapsed
                profile["rmcts_times"].append(elapsed)
            else:
                t0 = perf_counter()
                pi, v = MCTS_ucb.learn_pi_and_v(g.reshape(1, -1), N_ucb, engine, c_puct=C_ucb)
                t1 = perf_counter()
                elapsed = t1 - t0
                time_ucb += elapsed
                profile["ucb_calls"] += 1
                profile["ucb_total_time"] += elapsed
                profile["ucb_times"].append(elapsed)

            tt0 = perf_counter()
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            tt1 = perf_counter()
            profile["temperature_calls"] += 1
            profile["temperature_total_time"] += tt1 - tt0

            gs0 = perf_counter()
            g = game.nextState(g, a)
            gs1 = perf_counter()
            profile["game_step_calls"] += 1
            profile["game_step_total_time"] += gs1 - gs0

            ended, score = game.gameEnded(g)
            game_moves += 1

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
                elapsed = t1 - t0
                time_ucb += elapsed
                profile["ucb_calls"] += 1
                profile["ucb_total_time"] += elapsed
                profile["ucb_times"].append(elapsed)
            else:
                t0 = perf_counter()
                pi, v = RMCTS.learn_pi_and_v(g.reshape(1, -1), N_rmcts, engine, c_puct=C_rmcts)
                t1 = perf_counter()
                elapsed = t1 - t0
                time_rmcts += elapsed
                profile["rmcts_calls"] += 1
                profile["rmcts_total_time"] += elapsed
                profile["rmcts_times"].append(elapsed)

            tt0 = perf_counter()
            pi = pi.flatten()
            pi = np.power(pi, 1.0 / temperature)
            pi = pi / np.sum(pi)
            a = np.random.choice(len(pi), p=pi)
            tt1 = perf_counter()
            profile["temperature_calls"] += 1
            profile["temperature_total_time"] += tt1 - tt0

            gs0 = perf_counter()
            g = game.nextState(g, a)
            gs1 = perf_counter()
            profile["game_step_calls"] += 1
            profile["game_step_total_time"] += gs1 - gs0

            ended, score = game.gameEnded(g)

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
    print("=" * 70)
    print("PROFILING BREAKDOWN")
    print("=" * 70)

    rmcts_times = np.array(profile["rmcts_times"])
    ucb_times = np.array(profile["ucb_times"])

    print("RMCTS calls: {} | Mean: {:.1f}ms | Median: {:.1f}ms | P95: {:.1f}ms | Max: {:.1f}ms".format(
        profile["rmcts_calls"],
        np.mean(rmcts_times)*1000 if len(rmcts_times) > 0 else 0,
        np.median(rmcts_times)*1000 if len(rmcts_times) > 0 else 0,
        np.percentile(rmcts_times, 95)*1000 if len(rmcts_times) > 0 else 0,
        np.max(rmcts_times)*1000 if len(rmcts_times) > 0 else 0))

    print("MCTS-UCB calls: {} | Mean: {:.1f}ms | Median: {:.1f}ms | P95: {:.1f}ms | Max: {:.1f}ms".format(
        profile["ucb_calls"],
        np.mean(ucb_times)*1000 if len(ucb_times) > 0 else 0,
        np.median(ucb_times)*1000 if len(ucb_times) > 0 else 0,
        np.percentile(ucb_times, 95)*1000 if len(ucb_times) > 0 else 0,
        np.max(ucb_times)*1000 if len(ucb_times) > 0 else 0))

    total_profile_time = (profile["rmcts_total_time"] + profile["ucb_total_time"] +
                          profile["temperature_total_time"] + profile["game_step_total_time"])
    pct_rmcts = 100.0 * profile["rmcts_total_time"] / total_profile_time if total_profile_time > 0 else 0
    pct_ucb = 100.0 * profile["ucb_total_time"] / total_profile_time if total_profile_time > 0 else 0
    pct_temp = 100.0 * profile["temperature_total_time"] / total_profile_time if total_profile_time > 0 else 0
    pct_step = 100.0 * profile["game_step_total_time"] / total_profile_time if total_profile_time > 0 else 0

    print()
    print("Time breakdown (of tracked operations):")
    print("  RMCTS search:        {:6.1f}s ({:5.1f}%)".format(profile["rmcts_total_time"], pct_rmcts))
    print("  MCTS-UCB search:     {:6.1f}s ({:5.1f}%)".format(profile["ucb_total_time"], pct_ucb))
    print("  Temperature/choice:  {:6.3f}s ({:5.1f}%)".format(profile["temperature_total_time"], pct_temp))
    print("  Game state advance:  {:6.3f}s ({:5.1f}%)".format(profile["game_step_total_time"], pct_step))
    print()
    print("Per-RMCTS-call breakdown:")
    print("  Total time: {:.2f}s for {} calls".format(profile["rmcts_total_time"], profile["rmcts_calls"]))
    print("  Mean per call: {:.1f}ms".format(profile["rmcts_total_time"] * 1000 / profile["rmcts_calls"] if profile["rmcts_calls"] > 0 else 0))
    print("  Calls per game (avg): {:.1f}".format(profile["rmcts_calls"] / total_games if total_games > 0 else 0))

    # Engine info
    print()
    print("Engine configuration:")
    print("  max_batchsize: {}".format(engine.max_batchsize))
    print("  opt_batchsize: {}".format(engine.opt_batchsize))
    print("  input_shape: {}".format(engine.input_shape))
    print("  num_actions: {}".format(engine.num_actions))

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
        "profiling": {
            "rmcts_calls": profile["rmcts_calls"],
            "rmcts_mean_ms": float(np.mean(rmcts_times)*1000) if len(rmcts_times) > 0 else 0,
            "rmcts_median_ms": float(np.median(rmcts_times)*1000) if len(rmcts_times) > 0 else 0,
            "rmcts_p95_ms": float(np.percentile(rmcts_times, 95)*1000) if len(rmcts_times) > 0 else 0,
            "ucb_mean_ms": float(np.mean(ucb_times)*1000) if len(ucb_times) > 0 else 0,
            "ucb_median_ms": float(np.median(ucb_times)*1000) if len(ucb_times) > 0 else 0,
        }
    }

    with open("/repo/reproduction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /repo/reproduction_results.json")

    print("FINAL_METRIC: mean_score={:.2f} mean_time_rmcts_ms={:.2f} speedup={:.2f}".format(
        mean_score, mean_time_rmcts, speedup))

if __name__ == "__main__":
    main()
