from environment import *


def mark(CFG):
    return f"{CFG.model}_{CFG.data_path.split('/')[-1].split('.')[0]}_{CFG.pred_len}"


def update_result(CFG, result=None):
    results = {}
    try:
        with open(os.path.join(CFG.root_path, 'result.txt'), 'r') as file:
            for line in file:
                line = line.strip().split('  -  ')
                key = line[0].split()
                value = line[1]
                results[(key[0], key[1], int(key[2]))] = value
    except FileNotFoundError:
        pass
    if result:
        results[(CFG.model, CFG.data_path.split('/')[-1].split('.')[0], CFG.pred_len)] = result
    results = OrderedDict(sorted(results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])))
    with open(os.path.join(CFG.root_path, 'result.txt'), 'w') as file:
        for key, value in results.items():
            file.write(f"{key[0]} {key[1]} {key[2]:>3}  -  {value}\n")


def update_table(CFG):
    results = {}
    try:
        with open(os.path.join(CFG.root_path, 'result.txt'), 'r') as file:
            for line in file:
                line = line.strip().split('  -  ')
                key = line[0].split()
                value = line[1]
                results[(key[0], key[1], int(key[2]))] = value
    except FileNotFoundError:
        pass
    models = CFG.models.split()
    pred_lens = {
        0: [96, 192, 336, 720],
        1: [24, 36, 48, 60],
        2: [12, 24, 48, 96]
    }
    with open(os.path.join(CFG.root_path, 'table.txt'), 'w') as file:
        file.write("|         Models         |" + ''.join([' ' * int((16 - len(model)) / 2) + model + ' ' * int((16 - len(model)) / 2 + 0.5) + '|' for model in models]) + '\n')
        file.write("|------------------------|" + ''.join(['-' * 16 + '|' for _ in models]) + '\n')
        file.write("| Data / Length / Metric |" + ''.join(["  MSE  --  MAE  " + '|' for _ in models]) + '\n')
        file.write("|------------------------|" + ''.join(['-' * 16 + '|' for _ in models]) + '\n')
        for name, dataset in {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2', 'ECL': 'electricity', 'Traffic': 'traffic', 'Weather': 'weather', 'Exchange-Rate': 'exchange-rate', 'ILI': 'illness', 'PEMS-03': 'PEMS03', 'PEMS-04': 'PEMS04', 'PEMS-07': 'PEMS07', 'PEMS-08': 'PEMS08', 'Solar': 'solar'}.items():
            r1s, r2s = {}, {}
            for model in models:
                r1s[model], r2s[model] = [], []
            if dataset == 'illness':
                i = 1
            elif dataset == 'PEMS03' or dataset == 'PEMS04' or dataset == 'PEMS07' or dataset == 'PEMS08':
                i = 2
            else:
                i = 0
            for i, pred_len in enumerate(pred_lens[i]):
                mid = ' ' * int((15 - len(name)) / 2) + name + ' ' * int((15 - len(name)) / 2 + 0.5) if i == 2 else ' ' * 15
                write = '|' + mid + '|' + f"  {pred_len:>3}   |"
                for model in models:
                    key = (model, dataset, pred_len)
                    if key in results:
                        result = results[key].split(',  ')
                        r1, r2 = float(result[0][-5:]), float(result[1][-5:])
                        r1s[model].append(r1)
                        r2s[model].append(r2)
                        write += f" {r1:.3f}    {r2:.3f} |"
                    else:
                        write += ' ' * 16 + '|'
                file.write(write + '\n')
            write = "|               |  Avg   |"
            for model in models:
                if r1s[model]:
                    write += f" {np.mean(r1s[model]):.3f}    {np.mean(r2s[model]):.3f} |"
                else:
                    write += ' ' * 16 + '|'
            file.write(write + '\n')
            file.write("|------------------------|" + ''.join(['-' * 16 + '|' for _ in models]) + '\n')
