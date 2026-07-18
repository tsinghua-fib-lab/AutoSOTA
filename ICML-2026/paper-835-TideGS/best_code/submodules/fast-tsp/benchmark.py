import fast_tsp
import time

dists1 = [
    [ 0, 63, 72, 70],
    [63,  0, 57, 53],
    [72, 57,  0,  4],
    [70, 53,  4,  0],
]

dists2 = [
    [abs(i-j) for j in range(16)] for i in range(16)
]

dists3 = [
    [abs(i-j) for j in range(32)] for i in range(32)
]

def benchmark(dists):
    start_time = time.time()
    tour = fast_tsp.find_tour(dists)
    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)

import json
import random
import numpy as np

def tasks_to_dists(tasks):
    max_in_all_dist = 0
    dists_per_task = []
    setsize_per_task = []
    for task in tasks:
        filters = task["filters"]
        filters = [np.array(f) for f in filters]
        max_value = int(max([f.max() for f in filters])) + 1
        filters_bool = []
        for f in filters:
            f_bool = np.zeros(max_value, dtype=np.uint8)
            f_bool[f] = 1
            filters_bool.append(f_bool)
        
        setsize = 0
        for f in filters:
            setsize += f.sum()
        setsize_per_task.append(setsize)

        n = len(filters)
        dist = []
        for i in range(n):
            this_dists = []
            for j in range(n):
                # |filter_i xor filter_j|
                this_dists.append(int(np.sum(filters_bool[i] != filters_bool[j])))
                max_in_all_dist = max(max_in_all_dist, this_dists[-1])
            dist.append(this_dists)
        dists_per_task.append(dist)

    downsample_ratio = max_in_all_dist // 30000 + 1
    for dist in dists_per_task:
        for i in range(len(dist)):
            for j in range(len(dist[i])):
                dist[i][j] //= downsample_ratio

    # import pdb; pdb.set_trace()
    return dists_per_task, setsize_per_task

def load_all_tasks(file_path):
    lines = open(file_path, "r").readlines()
    tasks = []
    for line in lines:
        tasks.append(json.loads(line))
    return tasks_to_dists(tasks)

def create_larger_tasks(file_path, replicated_num):
    lines = open(file_path, "r").readlines()
    tasks = []
    i = 0
    for line in lines[:10]:
        print("load " + str(i))
        tasks.append(json.loads(line))
    lines = None

    # import pdb; pdb.set_trace()
    concated_tasks = []
    for i in range(0, len(tasks)):
        filters = []
        for j in range(replicated_num):
            filters.extend(tasks[(i+j) % len(tasks)]["filters"])
        concated_tasks.append({"filters": filters})

    return tasks_to_dists(concated_tasks)

def compare_on_tsp_task(dists, output_file, timeout_threshold, average_stats):
    # nxn matrix
    print(f"Benchmarking on{len(dists)}x{len(dists)} matrix: {json.dumps(dists)}")
    output_file.write(f"Benchmarking on{len(dists)}x{len(dists)} matrix: {json.dumps(dists)}\n")

    all_evaluted_methods = [
        "random",
        "greedy_nearest_neighbor",
        "greedy_sls",
        "optimal"
    ]
    for method in all_evaluted_methods:
        start_time = time.time()
        if method == "random":
            # a random permutation of n
            tour = random.sample(range(len(dists)), len(dists))
        elif method == "greedy_nearest_neighbor":
            tour = fast_tsp.greedy_nearest_neighbor(dists)
        elif method == "greedy_sls":
            tour = fast_tsp.find_tour(dists, timeout_threshold * 0.001)
        elif method == "optimal":
            if len(dists) > 20:
                continue
            tour = fast_tsp.solve_tsp_exact(dists)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()

        elapsed_time = time.time() - start_time
        cost = fast_tsp.compute_cost(tour, dists)
        print(f"Method: {method}, Cost: {cost}, Elapsed time: {elapsed_time}")
        output_file.write(f"Method: {method}, Cost: {cost}, Elapsed time: {elapsed_time}\n")
        average_stats[method].append(cost)


# file_path = "/home/hexu/Grendel-XS/output/bigcity/stat/20250310_183252_4090__4_bigcity_w=1_100mpcd_finalv7/sampled_filters.log"
# file_path = "/home/hexu/Grendel-XS/output/alameda/stat/20250310_185315_4090__4_alameda_42.8mpcd_finalv7/sampled_filters.log"
# file_path = "/home/hexu/Grendel-XS/output/ithaca/loc0/stat/20250310_183750_4090__4_ithacaloc0_76mpcd_finalv7/sampled_filters.log"

file_path = "/home/hexu/Grendel-XS/output/rubble4k/stat/20250310_182202_4090__4_rubble4k_40mpcd_finalv7/sampled_filters.log"
timeout_threshold = 5
replicated = 1
dists_per_task, setsize_per_task = create_larger_tasks(file_path, replicated)
average_stats = {
    "random": [],
    "greedy_nearest_neighbor": [],
    "greedy_sls": [],
    "optimal": []
}
save_path = file_path.replace(".log", f"results_rep{replicated}_time{timeout_threshold}.log")
save_file = open(save_path, "w")
for i in range(len(dists_per_task)):
    print(f"Processing Task {i}")
    compare_on_tsp_task(dists_per_task[i],
        save_file,
        timeout_threshold,
        average_stats
    )
average_stats = {k: np.mean(v) for k, v in average_stats.items()}
save_file.write(f"Average stats: {json.dumps(average_stats)}")
save_file.close()

# dists_per_task, setsize_per_task = create_larger_tasks(file_path, 4)

# python 

# conda install -c conda-forge mpich
# Ensure pkg-config can locate mpich.pc: --- export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
# conda install -c conda-forge cmake ninja compilers
# pip install --no-cache-dir .



















# for i in range(10):
#     print("Benchmarking...")
#     benchmark(dists1)
#     benchmark(dists2)
#     benchmark(dists3)

# def create_tsp_task(n, sparsity, random_seed):
#     random.seed(random_seed)
#     np.random.seed(random_seed)

#     total_set_size = 50000
#     each_set_size = int(total_set_size * sparsity)

#     filters = []
#     for i in range(n):
#         l = random.randint(0, total_set_size - each_set_size)
#         r = l + each_set_size

#         this_filter = np.zeros(total_set_size, dtype=np.bool)
#         # random positions within [l,r) are True
#         this_filter[np.random.randint(l, r, each_set_size)] = True
#         filters.append(this_filter)

#     dists = []
#     for i in range(n):
#         this_dists = []
#         for j in range(n):
#             this_dists.append(np.sum(filters[i] != filters[j]))
#         dists.append(this_dists)
#     return dists






# print(tour)  # [0, 1, 3, 2]


