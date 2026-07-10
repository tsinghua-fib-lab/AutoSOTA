import numpy as np
from time import perf_counter

from build.connect4 import game, inference, RMCTS, MCTS_ucb

batchsizes = np.power(2, np.arange(5, 12))
num_lanes = 64

onnx_path = "./connect4/models/ResNet_8blocks_64channels.onnx"

print(f"Timing tests for Connect4:")

# generate random game states
G = []
while len(G) < num_lanes:
    G.extend(game.randomRollout()[:-3])
G = np.array(G[:num_lanes])
g = game.rootState().reshape(1, -1)

print(f"Creating TensorRT engine...")
engine = inference.Engine(onnx_path, max_batchsize=1<<11, opt_batchsize=1<<11)
print("Finished creating engine.")

print("Timings in milliseconds for a single root state:")

labels = ['batchsize', 'rmcts', 'ucb', 'speedup']
for label in labels:
    print(f"{label:>10}", end='')
print()

rep1 = 10
rep64 = 5

for batchsize in batchsizes:
    # warmup
    pi, v = RMCTS.learn_pi_and_v(g, 32, engine, c_puct = 1)

    t0 = perf_counter()
    for i in range(rep1):
        pi, v = RMCTS.learn_pi_and_v(g, batchsize, engine, c_puct = 1)
    t1 = perf_counter()
    trt_rmcts = (t1 - t0)*1000/rep1

    t0 = perf_counter()
    for i in range(rep1):
        pi, v = MCTS_ucb.learn_pi_and_v(g, batchsize, engine, c_puct = 1)
    t1 = perf_counter()
    trt_ucb = (t1 - t0)*1000/rep1

    trt_ratio = trt_ucb / trt_rmcts
    print(f"{batchsize:10d}{trt_rmcts:10.4f}{trt_ucb:10.4f}{trt_ratio:10.4f}") 

print("Timings in milliseconds for 64 root states (avg time per root state):")

labels = ['batchsize', 'rmcts', 'ucb', 'speedup']
for label in labels:
    print(f"{label:>10}", end='')
print()

for batchsize in batchsizes:
    # warmup
    pi, v = RMCTS.learn_pi_and_v(G, 32, engine, c_puct = 1)

    t0 = perf_counter()
    for i in range(rep64):
        pi, v = RMCTS.learn_pi_and_v(G, batchsize, engine, c_puct = 1)
    t1 = perf_counter()
    trt_rmcts = (t1 - t0)*1000/(64*rep64)

    t0 = perf_counter()
    for i in range(rep64):
        pi, v = MCTS_ucb.learn_pi_and_v(G, batchsize, engine, c_puct = 1)
    t1 = perf_counter()
    trt_ucb = (t1 - t0)*1000/(64*rep64)

    trt_ratio = trt_ucb / trt_rmcts

    print(f"{batchsize:10d}{trt_rmcts:10.4f}{trt_ucb:10.4f}{trt_ratio:10.4f}")