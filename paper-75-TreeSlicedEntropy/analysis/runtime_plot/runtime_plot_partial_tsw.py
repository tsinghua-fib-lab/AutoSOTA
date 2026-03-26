import torch
import numpy as np
import matplotlib.pyplot as pl
import os
import time
from tqdm import tqdm

from tsw import PartialTSW, generate_trees_frames

def warmup_run(TW_obj, N, d, ntree, nline):
    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode="gaussian_raw")
    X = torch.rand(N, d).to("cuda")
    Y = torch.rand(N, d).to("cuda")
    TW_obj(X, Y, theta, intercept)
    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode="gaussian_raw")
    X = torch.rand(N, d).to("cuda")
    Y = torch.rand(N, d).to("cuda")
    TW_obj(X, Y, theta, intercept)

def run(TW_obj, N, d, ntree, nline):
    torch.cuda.reset_peak_memory_stats(device=None)
    X = torch.rand(N, d).to("cuda")
    Y = torch.rand(N, d).to("cuda")
    start = time.time()
    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode="gaussian_raw")
    tw = TW_obj(X, Y, theta, intercept)
    end = time.time()
    mem = torch.cuda.max_memory_allocated(device=None)
    return end - start, mem

if __name__ == "__main__":
    nrun = 100
    Ns = [100, 1000, 5000, 10000, 50000]
    ds = [50, 100, 500, 1000]
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']
    
    # ds = [500,1000,3000,5000,7000,8000,10000]
    # defaul_N = 1000
    
    ntree = 25
    nline = 4

    runtimes = np.zeros((len(ds), len(Ns), nrun))
    mems = np.zeros((len(ds), len(Ns), nrun))
    for i, d in enumerate(ds):
        for j, N in enumerate(Ns):
            TW_obj = torch.compile(PartialTSW(ntrees=ntree, mass_division='distance_based'))
            warmup_run(TW_obj, N, d, ntree, nline)
            for k in range(nrun):
                runtime, mem = run(TW_obj, N, d, ntree, nline)
                runtimes[i,j,k] = runtime
                mems[i,j,k] = mem
    
    # average over runs
    # runtimes = np.log10(runtimes)
    runtimes = runtimes * 1000
    avg_runtimes = np.mean(runtimes, axis=2)
    std_runtimes = np.std(runtimes, axis=2)
    
    pl.figure(figsize=(10, 6))

    # Plot SW with mean and shaded standard deviation (log scale)
    for i, d in enumerate(ds):
        pl.plot(Ns, avg_runtimes[i], label=f'd = {d}', color=colors[i])
        pl.fill_between(range(len(Ns)), 
                        avg_runtimes[i] - std_runtimes[i], 
                        avg_runtimes[i] + std_runtimes[i], 
                        color=colors[i], alpha=0.2)

    # Add text box with argument information
    # Prepare the text box content without dataset_name
    args_dict = {'L': ntree, 'k': nline}
    args_info = [f'{key.replace("_", " ")}: {value}' for key, value in args_dict.items()]

    # Join the list into a single string with newline separation
    textstr = '\n'.join(args_info)

    # Place a text box with argument information
    # pl.gca().text(0.05, 0.95, textstr, transform=pl.gca().transAxes, fontsize=17,
    #             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Finalize the plot with dataset name in the title
    # pl.title(r'Runtime of Db-TSW with different dimensions', fontsize=20)
    pl.xlabel('Number of supports $(n)$', fontsize=20)
    pl.ylabel(r'Runtime (ms)', fontsize=20)
    pl.legend(fontsize=17, loc='upper left')
    pl.grid(True)
    pl.tick_params(axis='both', which='major', labelsize=17)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_filename = os.path.join(results_dir, f'db_N_d_variation_runtime')
    pl.savefig(plot_filename + ".png")
    pl.savefig(plot_filename + ".pdf")
    pl.clf() 

    # average over runs
    # runtimes = np.log10(runtimes)
    mems = mems / 1024 / 1024
    avg_mems = np.mean(mems, axis=2)
    std_mems = np.std(mems, axis=2)
    
    pl.figure(figsize=(10, 6))

    # Plot SW with mean and shaded standard deviation (log scale)
    for i, d in enumerate(ds):
        pl.plot(Ns, avg_mems[i], label=f'd = {d}', color=colors[i])
        pl.fill_between(range(len(Ns)), 
                        avg_mems[i] - std_mems[i], 
                        avg_mems[i] + std_mems[i], 
                        color=colors[i], alpha=0.2)

    # Add text box with argument information
    # Prepare the text box content without dataset_name
    args_dict = {'L': ntree, 'k': nline}
    args_info = [f'{key.replace("_", " ")}: {value}' for key, value in args_dict.items()]

    # Join the list into a single string with newline separation
    textstr = '\n'.join(args_info)

    # Place a text box with argument information
    # pl.gca().text(0.05, 0.95, textstr, transform=pl.gca().transAxes, fontsize=17,
    #             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Finalize the plot with dataset name in the title
    # pl.title(r'Memory usage of Db-TWD with different dimensions', fontsize=20)
    pl.xlabel('Number of supports $(n)$', fontsize=20)
    pl.ylabel(r'Memory usage (MB)', fontsize=20)
    pl.legend(fontsize=17, loc='upper left')
    pl.grid(True)
    pl.tick_params(axis='both', which='major', labelsize=17)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_filename = os.path.join(results_dir, f'db_N_d_variation_memory')
    pl.savefig(plot_filename + ".png")
    pl.savefig(plot_filename + ".pdf")
    pl.clf() 
