import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# colors = {"TF": {"TF": "blue", "SSM": "orange"}, "SSM": {"TF": "green", "SSM": "red"}, "TF-nC": {"TF-nC": "brown"}}
colors = {"TF-TF": "blue", "TF-SSM": "orange", "SSM-TF": "green", "SSM-SSM": "red", "TF~nC-TF~nC": "brown", "TF-TF-TF": "blue", "SSM-SSM-SSM": "red", "SSM-SSM-TF": "green"}

colors.update({"hybrid": "green", "TF": "blue", "SSM": "red"})


def Int(s): return int("".join([c for c in s if c.isnumeric()]))
def Empty(): return []



def get_val_and_bounds(data):
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    
    # return mean, np.min(data, axis=0), np.max(data, axis=0)
    # return mean, mean-np.std(data, axis=0), mean+np.std(data, axis=0)
    # return median, np.min(data, axis=0), np.max(data, axis=0)
    return mean, np.quantile(data, 0.10, axis=0), np.quantile(data, 0.90, axis=0)
    # return median, np.quantile(data, 0.10, axis=0), np.quantile(data, 0.90, axis=0)


def savefig(taskname, filename):
    if "fig" not in os.listdir("results/" + taskname):
        os.mkdir("results/" + taskname + "/fig")
        
    plt.savefig("results/" + taskname + "/fig/" + filename + ".png")



split_array = ['_', 'task_name', 'layers', 'window', 'dim', 'num_heads', 'state_dim']
def plot(data, params, ind_var, diff_lines="layers", param_counts=None, x_axis=None, num_layers=2):
    fig, ax = plt.subplots()
    if x_axis == 'epochs':
        xs = defaultdict(Empty)
        ys = defaultdict(Empty)
        ys_lower = defaultdict(Empty)
        ys_upper = defaultdict(Empty)
        
        # Get the relevant data for these params
        for k in data.keys():
            d = dict(zip(split_array, k.split('_')))
            # if diff_lines != 'layer1'    and params['layer1']    != d['layer1']:         continue
            # if diff_lines != 'layer2'    and params['layer2']    != d['layer2']:         continue
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
                

    if diff_lines == 'layers':
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

            key = d['layers']

            # print(k)
            # print(d['layers'])
            if d['layers'].split("-")[1].isnumeric() or len(d['layers'].split("-")) != num_layers:
                # print("Ignoring", key)
                continue

            if x_axis == 'params':
                xs[key].append(param_counts[k])
            else:
                xs[key].append(Int(d[ind_var]))
                
            r1, r2, r3 = get_val_and_bounds(data[k])
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
            if key == "SSM-SSM" and (ind_var in ["num_heads", "window"]) or key == "TF-TF" and ind_var == "state_dim":
                plt.axhline(y=np.mean(ys[key]), color=colors[key], linestyle='dashed')
                # plt.axhline(y=ys[key][0], color=colors[key], linestyle='dashed')
            else:
                plt.plot(xs[key], ys[key], c=colors[key])
            legend.append(key)
        plt.legend(legend)

        # Plot the error bars
        for key in ys.keys():
            if key == "SSM-SSM" and (ind_var in ["num_heads", "window"]) or key == "TF-TF" and ind_var == "state_dim":
                plt.fill_between(ax.get_xlim(), np.mean(ys_lower[key]), np.mean(ys_upper[key]), color=colors[key], alpha=0.08)
                # plt.fill_between(ax.get_xlim(), ys_lower[key][0], ys_upper[key][0], color=colors[key], alpha=0.08)
            else:
                plt.fill_between(xs[key], ys_lower[key], ys_upper[key], color=colors[key], alpha=0.08)


    elif diff_lines == 'depths':
        # assert False # TODO: Doesn't plot across depth
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
            # print(d['layers'])
            if not d['layers'].split("-")[1].isnumeric(): continue
            # print("Here")

            key = d['layers'].split("-")[0] #+ "-" + d['layers'].split("-")[-1]

            if x_axis == 'params':
                xs[key].append(param_counts[key])
            else:
                xs[key].append(Int(d[ind_var]))
                
            r1, r2, r3 = get_val_and_bounds(data[k])
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
            plt.plot(xs[key], ys[key], c=colors[key.split("-")[0]])
            legend.append(key.split("-")[0])
        plt.legend(legend)

        # Plot the error bars
        for key in ys.keys():
            plt.fill_between(xs[key], ys_lower[key], ys_upper[key], color=colors[key.split("-")[0]], alpha=0.08)

    if num_layers == 2:
        return diff_lines + "_" + ind_var
    if num_layers == 3:
        return diff_lines + "_" + ind_var + "_3"