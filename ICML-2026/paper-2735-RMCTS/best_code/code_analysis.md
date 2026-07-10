# Code Analysis for Paper 2735: Recursive Monte-Carlo Tree Search (RMCTS)

## Evaluation Path

- **Entry:** `reproduce_final.py` (standalone eval script)
- **Flow:** Load ONNX model → Build TensorRT engine → Run 32 game pairs (64 total) of RMCTS (N=512, C=1.0) vs MCTS-UCB (N=256, C=1.0) → Report Mean Score (checker diff), Mean Time, Speedup
- **Key functions:** `RMCTS.learn_pi_and_v()`, `MCTS_ucb.learn_pi_and_v()`, `pit.pit()`
- **Output:** stdout `FINAL_METRIC:` line + `/repo/reproduction_results.json`

## Config/Parameter Path

- **Eval config (defaults):** `reproduce_final.py` lines 13-17 (N_rmcts=512, C_rmcts=1, N_ucb=256, C_ucb=1, temperature=0.2, num_games=32)
- **MCTS metaparams:** `build/othello/metaparm.py` (numLanes=32, engine_batchsize=256, c_puct=4.0)
- **C hardcoded:** `src/c/RMCTS.c:13-14` (SOFTPOWER=16.0, UCB_EPSILON=0.1)
- **TensorRT config:** `build/othello/inference.py:77` (max_batchsize=1024, opt_batchsize=256)

## Metric Parser

- Mean Score: np.mean(all_scores) where all_scores = scores_rmcts_first + scores_rmcts_second
- Mean Time per Game (RMCTS): time_rmcts / (2*num_games) * 1000 [ms]
- Speedup: time_ucb / time_rmcts

## Build System

- `GAME=othello make all` compiles C code with `-O3 -fPIC -march=native`
- Copies Python from `src/python/` and `othello/src/python/` to `build/othello/`
- C files compiled to `.o` → linked to `.so` shared libraries
- `build/othello/` is the runtime directory used by eval

## Reusable Resources

- **Pre-trained model:** `othello/models/ResNet_8blocks_48channels.onnx` (included in repo)
- No external datasets required
- No paper_data mount needed per manifest

## Safe Modification Targets

- `reproduce_final.py` — eval script (add profiling, parameterization)
- `build/othello/metaparm.py` — runtime params (numLanes, batchsize)
- `src/python/inference.py` — TensorRT engine builder (FP16, batchsize)
- `src/c/RMCTS.c` — C search algorithm (SOFTPOWER, UCB_EPSILON)
- `build/othello/RMCTS.py` — Python RMCTS wrapper (new params)

## Risky Files (do not modify)

- `src/c/game.h`, `othello/src/c/game.c` — game logic/rules
- `src/c/aes_hash.*`, `src/c/dict32.*` — hashing/transposition table
- `othello/models/ResNet_8blocks_48channels.onnx` — model weights (can prune, not randomize)

## Modification → Rebuild Chain

- C source changed → `GAME=othello make all` rebuilds
- Python source (`src/python/`) changed → `make all` copies to `build/othello/`
- Build Python files changed directly → no rebuild needed (but gets overwritten by make)

## Baseline Commit

- d64dc44 "iter-0: Reproduced baseline [success]" — Mean Score 3.88, Time 274.37ms, Speedup 16.6x
