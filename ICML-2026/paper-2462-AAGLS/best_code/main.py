import argparse
import os
import subprocess
import sys
import time


algos = ["ours", "cb", "gb", "ca"]
bisectalgos = ["spectral", "spectralbalanced_10", "spectralbalanced_1", "metismulti_1", "metismulti_10", "metismulti_100"]
ks = [10, 50, 100]
num_itr = 10
experiment_name = str(time.time_ns())

parser = argparse.ArgumentParser()
parser.add_argument("graph", help="Path to graph file")
parser.add_argument("--algos", nargs="+", default=algos, help="Algorithms to run")
parser.add_argument("--bisectalgos", nargs="+", default=bisectalgos, help="Bisection algorithms")
parser.add_argument("--ks", nargs="+", type=int, default=ks, help="Values of k")
parser.add_argument("--num_itr", type=int, default=num_itr, help="Number of iterations")
parser.add_argument("--experiment_name", type=int, default=experiment_name, help="Name of Experiment")
args = parser.parse_args()

graph = args.graph
algos = args.algos
bisectalgos = args.bisectalgos
ks = args.ks
num_itr = args.num_itr
experiment_name = args.experiment_name


processes = []

if "ours" in algos: 
    # Check if enough sparsifiers / make more
    sparsifier_processes = []
    sparsifier_files = []
    for bisectalgo in bisectalgos:
        sparsifier_dir = f"sparsifiers/{graph.replace("/","_")}/{bisectalgo}"
        os.makedirs(sparsifier_dir, exist_ok=True)
        sparsifiers = os.listdir(sparsifier_dir)

        # local_num_itr = num_itr
        local_num_itr = 1
        if (bisectalgo[0]=='s'): local_num_itr = 1


        if len(sparsifiers) < local_num_itr:
            for i in range(local_num_itr - len(sparsifiers)):
                timestamp = str(time.time_ns())
                time.sleep(0.001)
                f = open(f"{sparsifier_dir}/{timestamp}.pickle", "w")
                p = subprocess.Popen([sys.executable, "-u", "sparsifier.py", graph, bisectalgo], stdout=f, close_fds=True)
                sparsifier_processes.append(p)
                processes.append(p)
                sparsifier_files.append(f)

    # WAIT FOR SPARSIFIERS TO BE DONE
    for p in sparsifier_processes: p.wait()
    for f in sparsifier_files: f.close()

    # Run experiments on sparsifiers
    for bisectalgo in bisectalgos:
        sparsifier_dir = f"sparsifiers/{graph.replace("/","_")}/{bisectalgo}"
        sparsifiers = os.listdir(sparsifier_dir)

        # local_num_itr = num_itr
        local_num_itr = 1
        if (bisectalgo[0]=='s'): local_num_itr = 1

        for k in ks:
            for itr, current_sparsifier in enumerate(sparsifiers[:local_num_itr]):
                dirname = f"results/{experiment_name}/{graph.replace("/","_")}/ours-{bisectalgo}/k-{k}"
                os.makedirs(dirname, exist_ok=True)
                path = f"{dirname}/itr-{itr}"
                p = subprocess.Popen([sys.executable, "-u", "treeDP.py", graph, f"{sparsifier_dir}/{current_sparsifier}", str(k)], stdout=open(path, "wb"), close_fds=True)
                processes.append(p)


if "cb" in algos: 
    for k in ks:
        for itr in range(num_itr):
            dirname = f"results/{experiment_name}/{graph.replace("/","_")}/cb/k-{k}"
            os.makedirs(dirname, exist_ok=True)
            path = f"{dirname}/itr-{itr}"
            p = subprocess.Popen([sys.executable, "-u", "cesa_bianchi_et_al.py", graph, str(k)], stdout=open(path, "w"), close_fds=True)
            processes.append(p)


if "gb" in algos: 
    for k in ks:
        for itr in range(num_itr):
            dirname = f"results/{experiment_name}/{graph.replace("/","_")}/gb/k-{k}"
            os.makedirs(dirname, exist_ok=True)
            path = f"{dirname}/itr-{itr}"
            p = subprocess.Popen([sys.executable, "-u", "guillory_bilmes.py", graph, str(k)], stdout=open(path, "w"), close_fds=True)
            processes.append(p)


if "ca" in algos: 
    for k in ks:
        for itr in range(1):
            dirname = f"results/{experiment_name}/{graph.replace("/","_")}/ca/k-{k}"
            os.makedirs(dirname, exist_ok=True)
            path = f"{dirname}/itr-{itr}"
            p = subprocess.Popen([sys.executable, "-u", "greedy_via_flow.py", graph, str(k)], stdout=open(path, "w"), close_fds=True)
            processes.append(p)



# WAIT FOR ALL PROCESSES TO FINISH
for p in processes: p.wait()