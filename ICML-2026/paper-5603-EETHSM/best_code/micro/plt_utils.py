import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# colors = {"TF": {"TF": "blue", "SSM": "orange"}, "SSM": {"TF": "green", "SSM": "red"}, "TF-nC": {"TF-nC": "brown"}}
colors = {"TF_TF": "blue", "TF_SSM": "orange", "SSM_TF": "green", "SSM_SSM": "red", "TF-nC_TF-nC": "brown"}



def Int(s): return int("".join([c for c in s if c.isnumeric()]))
def Empty(): return []



def get_val_and_bounds(data):
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    
    # return mean, np.min(data, axis=0), np.max(data, axis=0)
    # return mean, mean-np.std(data, axis=0), mean+np.std(data, axis=0)
    # return median, np.min(data, axis=0), np.max(data, axis=0)
    return median, np.quantile(data, 0.10, axis=0), np.quantile(data, 0.90, axis=0)


def savefig(taskname, filename):
    if "fig" not in os.listdir("results/" + taskname):
        os.mkdir("results/" + taskname + "/fig")
        
    plt.savefig("results/" + taskname + "/fig/" + filename + ".png")



split_array = ['_', 'task_name', 'layer1', 'layer2', 'window', 'dim', 'num_heads', 'state_dim']
def plot(data, params, ind_var, diff_lines="layers", param_counts=None, x_axis=None):
    fig, ax = plt.subplots()
    if x_axis == 'epochs':
        xs = defaultdict(Empty)
        ys = defaultdict(Empty)
        ys_lower = defaultdict(Empty)
        ys_upper = defaultdict(Empty)
        
        # Get the relevant data for these params
        for k in data.keys():
            d = dict(zip(split_array, k.split('_')))
            if diff_lines != 'layer1'    and params['layer1']    != d['layer1']:         continue
            if diff_lines != 'layer2'    and params['layer2']    != d['layer2']:         continue
            if diff_lines != 'window'    and params['window']    != Int(d['window']):    continue
            if diff_lines != 'dim'       and params['dim']       != Int(d['dim']):       continue
            if diff_lines != 'num_heads' and params['num_heads'] != Int(d['num_heads']): continue
            if diff_lines != 'state_dim' and params['state_dim'] != Int(d['state_dim']): continue

            key = Int(d[diff_lines])

            xs[key] = np.arange(0, data[k].shape[1])
            ys[key], ys_lower[key], ys_upper[key] = get_val_and_bounds(data[k])

        legend = []
        keys = sorted(list(ys.keys()))
        for key in keys:
            plt.plot(xs[key], ys[key])
            legend.append(key)
        
        plt.legend(legend)

        for key in ys.keys():
            plt.fill_between(xs[key], ys_lower[key], ys_upper[key], color='lightblue', alpha=0.08)
                

    elif diff_lines == 'layers':
        xs = defaultdict(Empty)
        ys = defaultdict(Empty)
        ys_lower = defaultdict(Empty)
        ys_upper = defaultdict(Empty)

        # Get the relevant data for these params
        for k in data.keys():
            d = dict(zip(split_array, k.split('_')))
            if ind_var != 'window'    and params['window']    != Int(d['window']):    continue
            if ind_var != 'dim'       and params['dim']       != Int(d['dim']):       continue
            if ind_var != 'num_heads' and params['num_heads'] != Int(d['num_heads']): continue
            if ind_var != 'state_dim' and params['state_dim'] != Int(d['state_dim']): continue

            key = d['layer1'] + '_' + d['layer2']

            if d['layer2'].isnumeric():
                print("Ignoring", key)
                continue

            if x_axis == 'params':
                xs[key].append(param_counts[k])
            else:
                xs[key].append(Int(d[ind_var]))
                
            r1, r2, r3 = get_val_and_bounds(data[k][:, -1])
            ys[key].append(r1)
            ys_lower[key].append(r2)
            ys_upper[key].append(r3)

        # Sort the data so it is in order on the x axis
        for key in ys.keys():
            ys[key] = [a[1] for a in sorted(zip(xs[key], ys[key]))]
            ys_lower[key] = [a[1] for a in sorted(zip(xs[key], ys_lower[key]))]
            ys_upper[key] = [a[1] for a in sorted(zip(xs[key], ys_upper[key]))]
            xs[key].sort()

        # Plot the lines
        legend = []
        for key in ys.keys():
            if key == "SSM_SSM" and ind_var == "num_heads" or key == "TF_TF" and ind_var == "state_dim":
                plt.axhline(y=ys[key][0], color=colors[key], linestyle='dashed')
            else:
                plt.plot(xs[key], ys[key], c=colors[key])
            legend.append(key)
        plt.legend(legend)

        # Plot the error bars
        for key in ys.keys():
            if key == "SSM_SSM" and ind_var == "num_heads" or key == "TF_TF" and ind_var == "state_dim":
                plt.fill_between(ax.get_xlim(), ys_lower[key][0], ys_upper[key][0], color=colors[key], alpha=0.08)
            else:
                plt.fill_between(xs[key], ys_lower[key], ys_upper[key], color=colors[key], alpha=0.08)


    elif diff_lines == 'depths':
        assert False # TODO: Doesn't plot across depth
        xs = defaultdict(Empty)
        ys = defaultdict(Empty)
        ys_lower = defaultdict(Empty)
        ys_upper = defaultdict(Empty)

        # Get the relevant data for these params
        for k in data.keys():
            d = dict(zip(split_array, k.split('_')))
            if ind_var != 'window'    and params['window']    != Int(d['window']):    continue
            if ind_var != 'dim'       and params['dim']       != Int(d['dim']):       continue
            if ind_var != 'num_heads' and params['num_heads'] != Int(d['num_heads']): continue
            if ind_var != 'state_dim' and params['state_dim'] != Int(d['state_dim']): continue

            key = d['layer1'] + '_' + d['layer2']

            if x_axis == 'params':
                xs[key].append(param_counts[key])
            else:
                xs[key].append(Int(d[ind_var]))
                
            r1, r2, r3 = get_val_and_bounds(data[k][:, -1])
            ys[key].append(r1)
            ys_lower[key].append(r2)
            ys_upper[key].append(r3)

        # Sort the data so it is in order on the x axis
        for key in ys.keys():
            ys[key] = [a[1] for a in sorted(zip(xs[key], ys[key]))]
            ys_lower[key] = [a[1] for a in sorted(zip(xs[key], ys_lower[key]))]
            ys_upper[key] = [a[1] for a in sorted(zip(xs[key], ys_upper[key]))]
            xs[key].sort()

        # Plot the lines
        legend = []
        for key in ys.keys():
            plt.plot(xs[key], ys[key], c=colors[key.split("_")[0] + "_" + key.split("_")[0]])
            legend.append(key)
        plt.legend(legend)

        # Plot the error bars
        for key in ys.keys():
            plt.fill_between(xs[key], ys_lower[key], ys_upper[key], color=colors[key.split("_")[0] + "_" + key.split("_")[0]], alpha=0.08)

    return diff_lines + "_" + ind_var