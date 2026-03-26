import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.preprocessing import SplineTransformer
from tueplots import bundles, axes

from causa.het_ridge import convex_fgls
import torch.functional as F
import matplotlib.pyplot as plt
from tueplots import bundles, axes
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer

from causa.utils import TensorDataLoader
from causa.datasets import MNU
from causa.het_ridge import convex_fgls
from causa.ml import map_optimization
from causa.iterative_fgls import iterative_fgls


# front page causal identification example (MNU-55)
cpal = sns.color_palette()
blue, orange, grey = cpal[0], cpal[1], cpal[-3]
purple, red = cpal[4], cpal[3]
class HetSpindlyHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1, 1, bias=False)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight.data.fill_(0.0)

    def forward(self, input):
        out1 = self.lin1(input[:, 0].unsqueeze(-1))
        out2 = torch.exp(self.lin2(input[:, 1].unsqueeze(-1)))
        return torch.cat([out1, out2], 1)


def build_het_network(in_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 500),
        torch.nn.Tanh(),
        torch.nn.Linear(500, 2),
        HetSpindlyHead()
    )


def build_network(in_dim=1, out_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 500),
        torch.nn.Tanh(),
        torch.nn.Linear(500, out_dim),
    )

device = 'cpu'
dataset = MNU(55, preprocessor=StandardScaler(), double=True)
torch.set_default_dtype(torch.float64)
test_data_true = torch.linspace(dataset.cause.min()-1, dataset.cause.max()+1, 500).reshape(-1, 1)
y_true = dataset.effect.numpy().flatten()
test_data_false = torch.linspace(dataset.effect.min()-1, dataset.effect.max()+1, 500).reshape(-1, 1)
y_false = dataset.cause.numpy().flatten()

def compute_loc_scale(x, model):
    with torch.no_grad():
        eta = model(x)
        eta_1 = eta[:, 0]
        eta_2 = -0.5 * eta[:, 1]
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * eta_1 / eta_2
        return loc.numpy(), scale.numpy()

map_kwargs = dict(
    scheduler='cos',
    lr=1e-2,
    lr_min=1e-6,
    n_epochs=5000,
    likelihood='heteroscedastic_regression',
    param_1='natural',
    param_2='multiplicative',
    prior_prec=0.0,
    nu_noise_init=0.5
)

# True order
loader = TensorDataLoader(
    dataset.cause.to(device), dataset.effect.to(device).flatten(), batch_size=len(dataset.effect)
)
model, losses = map_optimization(
    build_het_network(1),
    loader,
    **map_kwargs
)[:2]
print('NN LSNM LOGLIK true', -losses[-1] / len(dataset.effect))
loc_true_nn, scale_true_nn = compute_loc_scale(test_data_true, model)
loc_true_nn_scat, scale_true_nn_scat = compute_loc_scale(dataset.cause.to(device), model)

# False order
loader = TensorDataLoader(
    dataset.effect.to(device), dataset.cause.to(device).flatten(), batch_size=len(dataset.effect)
)
model, losses = map_optimization(
    build_het_network(1),
    loader,
    **map_kwargs
)[:2]
print('NN LSNM LOGLIK wrong', -losses[-1] / len(dataset.effect))
loc_false_nn, scale_false_nn = compute_loc_scale(test_data_false, model)
loc_false_nn_scat, scale_false_nn_scat = compute_loc_scale(dataset.effect.to(device), model)

map_kwargs = {**map_kwargs, **dict(likelihood='regression')}

loader = TensorDataLoader(
    dataset.cause.to(device), dataset.effect.to(device), batch_size=len(dataset.effect)
)
model, losses = map_optimization(
    build_network(),
    loader,
    **map_kwargs
)[:2]
print('NN AN LOGLIK true:', -losses[-1] / len(dataset.effect))
with torch.no_grad():
    mean_true_ml = model(test_data_true).numpy().flatten()

loader = TensorDataLoader(
    dataset.effect.to(device), dataset.cause.to(device), batch_size=len(dataset.effect)
)
model, losses = map_optimization(
    build_network(),
    loader,
    **map_kwargs
)[:2]
print('NN AN LOGLIK false:', -losses[-1] / len(dataset.effect))
with torch.no_grad():
    mean_false_ml = model(test_data_false).numpy().flatten()
    
with plt.rc_context({**bundles.aistats2022(column='half'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3.25, 1.8))
    axs[0].scatter(dataset.cause.flatten().numpy(), dataset.effect.flatten().numpy(), alpha=0.3, color='black',
                   lw=0.5, s=8)
    axs[0].plot(test_data_true.flatten(), loc_true_nn, label='LSNM', color=blue, lw=1.5)
    axs[0].fill_between(test_data_true.flatten(), loc_true_nn-2*scale_true_nn, 
                        loc_true_nn+2*scale_true_nn, color=blue, alpha=0.3)
    axs[0].plot(test_data_true.flatten(), mean_true_ml, label='ANM', color=orange, lw=1.5)
    axs[0].set_xlim(test_data_true.min()+0.4, test_data_true.max())
    axs[0].set_ylim(y_true.min()-1, y_true.max()+1)
    axs[0].grid()
    axs[0].set_xlabel('Cause $X$')
    axs[0].set_ylabel('Effect $Y$')
    axs[0].set_title('Causal Direction')

    axs[1].scatter(dataset.effect.flatten().numpy(), dataset.cause.flatten().numpy(), alpha=0.3, color='black',
                   lw=0.5, s=8)
    axs[1].plot(test_data_false.flatten(), loc_false_nn, label='LSNM', color=blue, lw=1.5)
    axs[1].fill_between(test_data_false.flatten(), loc_false_nn-2*scale_false_nn, 
                        loc_false_nn+2*scale_false_nn, color=blue, alpha=0.3)
    axs[1].plot(test_data_false.flatten(), mean_false_ml, label='ANM', color=orange, lw=1.5)
    axs[1].legend()
    axs[1].set_xlim(test_data_false.min(), test_data_false.max())
    axs[1].set_ylim(y_false.min()-1, y_false.max()+1)
    axs[1].set_ylabel('Cause $X$')
    axs[1].set_xlabel('Effect $Y$')
    axs[1].set_title('Anticausal Direction')
    axs[1].grid()
plt.savefig('paper_figures/MNU_55_example.pdf', transparent=True)
plt.show()


# NN LSNM LOGLIK true -0.7655911970636645
# NN LSNM LOGLIK wrong -0.9233285179923271
with plt.rc_context({**bundles.aistats2022(column='half'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3.25, 1.5))
    axs[0].scatter(dataset.cause.flatten().numpy(), dataset.effect.flatten().numpy(), alpha=0.3, color='black',
                   lw=0.5, s=8)
    axs[0].plot(test_data_true.flatten(), loc_true_nn, label='LSNM', color=blue, lw=1.5)
    axs[0].fill_between(test_data_true.flatten(), loc_true_nn-2*scale_true_nn, 
                        loc_true_nn+2*scale_true_nn, color=blue, alpha=0.3)
    axs[0].set_xlim(test_data_true.min()+0.4, test_data_true.max())
    axs[0].set_ylim(y_true.min()-1, y_true.max()+1)
    axs[0].grid()
    axs[0].set_xlabel('Cause $X$')
    axs[0].set_ylabel('Effect $Y$')
    axs[0].set_title('Causal')

    axs[1].scatter(dataset.effect.flatten().numpy(), dataset.cause.flatten().numpy(), alpha=0.3, color='black',
                   lw=0.5, s=8)
    axs[1].plot(test_data_false.flatten(), loc_false_nn, label='LSNM', color=blue, lw=1.5)
    axs[1].fill_between(test_data_false.flatten(), loc_false_nn-2*scale_false_nn, 
                        loc_false_nn+2*scale_false_nn, color=blue, alpha=0.3)
    axs[1].legend()
    axs[1].set_xlim(test_data_false.min(), test_data_false.max())
    axs[1].set_ylim(y_false.min()-1, y_false.max()+1)
    axs[1].set_ylabel('Cause $X$')
    axs[1].set_xlabel('Effect $Y$')
    axs[1].set_title('Anticausal')
    axs[1].grid()
    plt.savefig('paper_figures/MNU_55_example_lsnm_only_top.pdf', transparent=True)
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3.25, 1.5))
    axs[0].set_xlim(test_data_true.min()+0.4, test_data_true.max())
    axs[0].scatter(dataset.cause.flatten().numpy(), (dataset.effect.flatten().numpy() - loc_true_nn_scat) / scale_true_nn_scat, 
                   lw=0.5, s=8, color='black', alpha=0.3)
    axs[0].grid()
    axs[0].set_ylabel('Residual $\\frac{Y - f(X)}{g(X)}$')
    axs[0].set_xlabel('Cause $X$')
    axs[0].set_title('LogLik $=-0.77$')

    axs[1].set_xlim(test_data_false.min(), test_data_false.max())
    axs[1].scatter(dataset.effect.flatten().numpy(), (dataset.cause.flatten().numpy() - loc_false_nn_scat) / scale_false_nn_scat, 
                   lw=0.5, s=8, color='black', alpha=0.3)
    axs[1].set_xlabel('Effect $Y$')
    axs[1].set_ylabel('Residual $\\frac{X - h(Y)}{k(Y)}$')
    axs[1].grid()
    axs[1].set_title('LogLik $=-0.92$')
    plt.savefig('paper_figures/MNU_55_example_lsnm_only_bot.pdf', transparent=True)
    plt.show()

# Causality pair benchmark results
result_dir = 'results/'
benchmarks = ['SIM', 'SIMc', 'SIMln', 'SIMG', 'AN', 'ANs', 'LS', 'LSs', 'MNU', 'Tuebingen', 'Cha', 'Net', 'Multi']
n_datasets = [100, 100, 100, 100, 100, 100, 100, 100, 100, 108, 300, 300, 300]

# Tuebingen data  weights and additional information
tuebingen_meta = pd.read_csv(
    'data/Tuebingen/pairmeta.txt', delim_whitespace=True, 
    header=None, 
    names=['id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
    index_col=0).astype(float)
discrete_pairs = [47, 70, 107]
multivariate_pairs = [52, 53, 54, 55, 71, 105]
tue_blacklist = discrete_pairs + multivariate_pairs
tuebingen_meta.loc[tue_blacklist, 'weight'] = 0.0

def build_result_table(benchmark):
    base_frame = pd.read_csv(result_dir + benchmark + '_1.csv', index_col=0)
    for pair_id in range(2, n_datasets[benchmarks.index(benchmark)]+1):
        try:
            add_frame = pd.read_csv(result_dir + benchmark + f'_{pair_id}.csv', index_col=0)
            base_frame.loc[pair_id] = add_frame.loc[pair_id]
        except FileNotFoundError:
            if 'Tuebingen' in benchmark and pair_id in tue_blacklist:
                continue
            print(pair_id, 'unavailable for', benchmark)
            base_frame.loc[pair_id] = 0.0
    
    # combine with baseline results
    baselines = pd.read_csv(f'baseline_results/{benchmark}.tab', sep='\t', index_col=0)
    return base_frame.join(baselines)

def compute_performance(frame, benchmark):
    # create new frame
    acc_frame = pd.DataFrame(index=frame.index)
    conf_frame = pd.DataFrame(index=frame.index)
    
    def assess_method(key):
        # assessses method and adds to res_frame
        if 'hsic' in key:  # returns p-value and not score so needs to be reversed
            acc_frame[key] = frame[key + '_true'] < frame[key + '_false']
        else:
            acc_frame[key] = frame[key + '_true'] > frame[key + '_false']
        conf_frame[key] = (frame[key + '_true'] - frame[key + '_false']).abs()
    
    # call assess on all methods
    def filter_fn(string):
        return ('lin' in string or 'nn' in string or 'heci' in string) and 'true' in string
    method_names = [e.replace('_true', '') for e in filter(filter_fn, list(frame.columns))]
    for method in method_names:
        if method == 'nn_het' or method == 'nn_het_hsic':
            # skip nn_mix which uses a bad marglik estimator
            continue
        assess_method(method)
    
    # handle external methods
    # QCCD, GRCI, CAM, IGCI
    acc_frame['QCCD'] = (frame['QCCD'] * frame['GroundTruth']) > 0
    acc_frame['GRCI'] = (frame['GRCI'] * frame['GroundTruth']) > 0
    acc_frame['CAM'] = (frame['CAM'] * frame['GroundTruth']) > 0
    acc_frame['IGCI'] = (frame['IGCI'] * frame['GroundTruth']) > 0
    acc_frame['RESIT'] = (frame['RESIT_std'] * frame['GroundTruth']) > 0
    conf_frame['QCCD'] = frame['QCCD'].abs()
    conf_frame['GRCI'] = frame['GRCI'].abs()
    conf_frame['CAM'] = frame['CAM'].abs()
    conf_frame['IGCI'] = frame['IGCI'].abs()
    conf_frame['RESIT'] = frame['RESIT_std'].abs()
    
    return acc_frame.sort_index(axis='columns'), conf_frame.sort_index(axis='columns')

results = {k: dict() for k in benchmarks}
for benchmark in benchmarks:
    frame = build_result_table(benchmark)
    results[benchmark]['accs'], results[benchmark]['confs'] = compute_performance(frame, benchmark)

baselines = ['CAM', 'GRCI', 'QCCD', 'RESIT', 'IGCI', 'cgnn', 'heci', 'lin_ml', 'lin_ml_hsic', 'nn_ml', 'nn_ml_hsic']
our_methods = ['lin_ml_hetconv', 'lin_ml_hetconv_hsic', 'nn_ml_het', 'nn_ml_het_hsic']
method_filter = baselines + our_methods
columns = results['Tuebingen']['accs'].columns
accuracy_table = pd.DataFrame(index=benchmarks, columns=columns)
for benchmark in benchmarks:
    accs = results[benchmark]['accs']
    if 'Tuebingen' in benchmark:
        keep_ixs = tuebingen_meta.index[tuebingen_meta.weight>0.0]
        accs = accs.loc[keep_ixs]
        tbm = tuebingen_meta.loc[keep_ixs]
        accuracy_table.loc[benchmark, columns] = (accs * tbm.weight.values.reshape(-1, 1)).sum(0) / tbm.weight.sum(0)
    else:
        accuracy_table.loc[benchmark, columns] = accs.sum(0) / len(accs)
accuracy_table = accuracy_table[method_filter]

def compute_retention(method, accs, confs, weight):
    mdf = pd.DataFrame(index=accs.index)
    mdf['acc'] = accs[method]
    mdf['conf'] = confs[method]
    mdf['weight'] = weight
    sorted_mdf = mdf.sort_values(by='conf', ascending=False)
    accum = (sorted_mdf['acc'] * sorted_mdf['weight']).astype(float).cumsum() / sorted_mdf['weight'].cumsum()
    return accum.mean()

methods = results['AN']['accs'].columns
retention_table = pd.DataFrame(index=benchmarks, columns=methods)
for benchmark in benchmarks:
    accs = results[benchmark]['accs']
    confs = results[benchmark]['confs']
    if 'Tuebingen' in benchmark:
        keep_ixs = tuebingen_meta.index[tuebingen_meta.weight>0.0]
        accs = accs.loc[keep_ixs]
        confs = confs.loc[keep_ixs]
        tbm = tuebingen_meta.loc[keep_ixs]
        weight = tbm.weight.values
    else:
        weight = np.ones(len(accs))
    for method in columns:
        retention_table.loc[benchmark, method] = compute_retention(method, accs, confs, weight)
retention_table = retention_table[method_filter]

cpal = sns.color_palette()
blue, orange, grey = cpal[0], cpal[1], cpal[-3]
purple, red = cpal[4], cpal[3]

# Barplot for the five Tagasovska datasets (AN, ANs, LS, LSs, MNU)
methods = ['nn_ml_het', 'nn_ml_het_hsic', 'lin_ml_hetconv', 'lin_ml_hetconv_hsic', 'GRCI', 'QCCD', 'heci', 'CAM', 'RESIT']
method_names = [r'NN-$\textsc{Loci}_\textrm{M}$', 
                r'NN-$\textsc{Loci}_\textrm{H}$',
                r'$\textsc{Loci}_\textrm{M}$', 
                r'$\textsc{Loci}_\textrm{H}$',  
                'GRCI', 'QCCD', 'HECI', 'CAM', 'RESIT']
colors = [blue, blue, orange, orange] + 6 * [grey]
with plt.rc_context({**bundles.aistats2022(column='full'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(6.75, 4.1717/2.4), sharey=True)
    xs = np.linspace(len(methods)-1, 0, len(methods))
    
    benchmarks = ['AN', 'ANs', 'LS', 'LSs', 'MNU']
    for ax, benchmark in zip(axs, benchmarks):
        ax.grid(axis='x')
        accs = accuracy_table.loc[benchmark]
        ax.barh(xs, accuracy_table.loc[benchmark, methods].values, 
                height=0.7, align='center', tick_label=method_names,
                color=colors, alpha=.9)
        ax.set_xlim([0.0, 1.02])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([0, 25, 50, 75, 100])
        ax.set_title(benchmark)
    
    axs[2].set_xlabel('Accuracy [\%]', labelpad=8)
plt.savefig('paper_figures/anlsmnu.pdf', transparent=True)

# Comparison with Homoscedastic Estimators
methods = ['nn_ml_het', 'nn_ml_het_hsic', 'nn_ml', 'nn_ml_hsic', 
           'lin_ml_hetconv', 'lin_ml_hetconv_hsic', 'lin_ml', 'lin_ml_hsic']
method_names = methods
benchmarks = ['LS', 'LSs', 'MNU']
colors = [blue, blue, grey, grey, orange, orange, grey, grey]
sub_table = accuracy_table.loc[benchmarks, methods].copy()
sub_table = sub_table.rename(columns=lambda x: x.replace('hetconv', 'het')).astype(float)
keys_hom = ['nn_ml', 'nn_ml_hsic', 'lin_ml', 'lin_ml_hsic']
keys_het = ['nn_ml_het', 'nn_ml_het_hsic', 'lin_ml_het', 'lin_ml_het_hsic']
names = [r'NN-$\textrm{ANM}_\textrm{M}$', 
         r'NN-$\textrm{ANM}_\textrm{H}$',
         r'$\textrm{ANM}_\textrm{M}$', 
         r'$\textrm{ANM}_\textrm{H}$']
colors = [blue, purple, orange, red]
with plt.rc_context({**bundles.aistats2022(column='half'), **axes.lines()}):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 4.1717/2.9))
    ax.bar(xs[0], 0, width=0, label=r'LSNM', hatch='/////', color='white')
    xs = np.linspace(0, 2.2, 3)
    offset = 0.2
    ax.set_ylabel(r'Accuracy')
    for i, (method_hom, method_het) in enumerate(zip(keys_hom, keys_het)):
        het, hom = sub_table[method_het], sub_table[method_hom]
        ax.bar(xs+i*offset, hom * 100, color=colors[i], 
               alpha=0.8, width=offset, label=names[i])
        top = np.where(hom >= het, 0, het - hom)
        ax.bar(xs+i*offset, bottom=hom*100, height=top*100,
               color=colors[i], alpha=0.8, width=offset, hatch='/////',
               fill=True)
        
    ax.set_xticks(xs+1.5*offset)
    ax.set_xticklabels(benchmarks)
    
    ax.legend(loc='lower right', ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(1.08, -0.042))
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set_ylim([0, 100.5])
    ax.set_xlim([-0.1, 4.3])
plt.savefig('paper_figures/an_to_lsnm.pdf', transparent=True)

# Overall performance comparison
methods = ['nn_ml_het', 'nn_ml_het_hsic', 'lin_ml_hetconv', 'lin_ml_hetconv_hsic', 'GRCI', 'QCCD', 'heci', 'CAM', 'RESIT']
method_names = [r'NN-$\textsc{Loci}_\textrm{M}$', 
                r'NN-$\textsc{Loci}_\textrm{H}$',
                r'$\textsc{Loci}_\textrm{M}$', 
                r'$\textsc{Loci}_\textrm{H}$',  
                'GRCI', 'QCCD', 'HECI', 'CAM', 'RESIT']
colors = [blue, blue, orange, orange] + 5 * [grey]
benchmarks = ['Tuebingen', 'Cha', 'Net', 'Multi', 'SIM', 'SIMc', 'SIMln', 'SIMG']
benchmarks = accuracy_table.index

with plt.rc_context({**bundles.aistats2022(column='half'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3.25, 4.1717/2.4), sharey=True)
    xs = np.linspace(len(methods)-1, 0, len(methods))
    
    axs[0].grid(axis='x')
    axs[0].barh(xs, accuracy_table.loc[benchmarks, methods].mean().values, 
                height=0.7, align='center', tick_label=method_names,
                color=colors, alpha=0.9)
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axs[0].set_xticklabels([0, 25, 50, 75, 100])
    axs[0].set_xlabel('Accuracy [\%]')
    
    axs[1].grid(axis='x')
    axs[1].barh(xs, retention_table.loc[benchmarks, methods].mean().values, 
                height=0.7, align='center', tick_label=method_names,
                color=colors, alpha=0.9)
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axs[1].set_xticklabels([0, 25, 50, 75, 100])
    axs[1].set_xlabel('AUDRC [\%]')
plt.savefig('paper_figures/overall.pdf', transparent=True)

# Tables
methods = ['nn_ml_het', 'nn_ml_het_hsic', 'lin_ml_hetconv', 'lin_ml_hetconv_hsic', 'GRCI', 'QCCD', 'heci', 'cgnn', 'IGCI', 'CAM', 'RESIT']
method_names = [r'NN-$\textsc{Loci}_\textrm{M}$', 
                r'NN-$\textsc{Loci}_\textrm{H}$', 
                r'$\textsc{Loci}_\textrm{M}$', 
                r'$\textsc{Loci}_\textrm{H}$',
                'GRCI', 'QCCD', 'HECI', 'CGNN', 'IGCI', 'CAM', 'RESIT']
datasets = ['AN', 'ANs', 'LS', 'LSs', 'MNU', 'SIM', 'SIMc', 'SIMln', 'SIMG', 'Tuebingen', 'Cha', 'Net', 'Multi']
# Accuracy Table
res_table = accuracy_table.loc[datasets, methods].rename({k: v for k, v in zip(methods, method_names)}, axis=1) * 100
res_table = res_table.rename(index=lambda x: 'Tue' if x == 'Tuebingen' else x)
res_table = res_table.astype(float).round().astype(int).T
for c in res_table.columns:
    def bolding(x):
        if x == res_table[c].max():
            return r'\textbf{' + str(x) + '}'
        return str(x)
    res_table[c] = res_table[c].apply(bolding)
print(res_table.to_latex(escape=False, column_format='l|rrrrr|rrrrr|rrr'))

res_table = retention_table.loc[datasets, methods].rename({k: v for k, v in zip(methods, method_names)}, axis=1) * 100
res_table = res_table.rename(index=lambda x: 'Tue' if x == 'Tuebingen' else x)
res_table = res_table.astype(float).round().astype(int).T
for c in res_table.columns:
    def bolding(x):
        if x == res_table[c].max():
            return r'\textbf{' + str(x) + '}'
        return str(x)
    res_table[c] = res_table[c].apply(bolding)
print(res_table.to_latex(escape=False, column_format='l|rrrrr|rrrrr|rrr'))

# Estimator Benchmark
np.random.seed(711)
data = np.zeros((100, 11, 4))
for i, seed in enumerate(np.random.randint(1, np.iinfo(np.int32).max, 100)):
    perfs = pd.read_csv(f'results/estimator_benchmark_sine_sqrt_{seed}.csv', index_col=0)
    try:
        data[i] = perfs.values
    except:
        print(seed, i)
perfs_med_sqrt = pd.DataFrame(np.median(data, axis=0), columns=perfs.columns)

np.random.seed(711)
data = np.zeros((100, 11, 4))
for i, seed in enumerate(np.random.randint(1, np.iinfo(np.int32).max, 100)):
    perfs = pd.read_csv(f'results/estimator_benchmark_sine_lin_{seed}.csv', index_col=0)
    try:
        data[i] = perfs.values
    except:
        print(seed, i)
perfs_med_lin = pd.DataFrame(np.median(data, axis=0), columns=perfs.columns)

samples = 1000
np.random.seed(711)
x = (np.random.rand(samples) - 0.5) * 8.0 * np.pi
n = np.random.randn(samples)
y = np.sin(x) + n * ((4.0 * np.pi - np.abs(x)) * 0.1 + 0.2)

# define ground truth
x_test = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 1000)
gt_mean = np.sin(x_test)
## Knot heuristic `='sqrt'`
spline_trans = SplineTransformer(n_knots=int(np.sqrt(samples)), degree=5)
Phi_test = spline_trans.fit_transform(x_test.reshape(-1, 1))
Phi = spline_trans.transform(x.reshape(-1, 1))
y = y.flatten()
w1, _, weight_2, _, _ = iterative_fgls(y, Phi, Psi=Phi, n_steps=100, takeLog=False)
loc_fgls_sqrt = Phi_test @ w1
scale_fgls_sqrt = np.sqrt(np.clip(Phi_test @ weight_2, a_min=1e-7, a_max=1e7))

w11, w12, _, _ = convex_fgls(torch.from_numpy(Phi), torch.from_numpy(np.abs(Phi)), 
                             torch.from_numpy(y), delta_Phi=1e-6, delta_Psi=1e-6, n_steps=100) 
eta_1 = Phi_test @ w11.numpy()
eta_2 = - np.abs(Phi_test) @ w12.numpy()
scale_mlconv_sqrt = np.sqrt(- 0.5 / eta_2)
loc_mlconv_sqrt = - 0.5 * eta_1 / eta_2

## Knot heuristic `='lin'`
spline_trans = SplineTransformer(n_knots=int(samples / 10), degree=5)
Phi_test = spline_trans.fit_transform(x_test.reshape(-1, 1))
Phi = spline_trans.transform(x.reshape(-1, 1))
y = y.flatten()
w1, _, weight_2, _, _ = iterative_fgls(y, Phi, Psi=Phi, n_steps=100, takeLog=False)
loc_fgls_lin = Phi_test @ w1
scale_fgls_lin = np.sqrt(np.clip(Phi_test @ weight_2, a_min=1e-7, a_max=1e7))

w11, w12, _, _ = convex_fgls(torch.from_numpy(Phi), torch.from_numpy(np.abs(Phi)), 
                             torch.from_numpy(y), delta_Phi=1e-6, delta_Psi=1e-6, n_steps=100) 
eta_1 = Phi_test @ w11.numpy()
eta_2 = - np.abs(Phi_test) @ w12.numpy()
scale_mlconv_lin = np.sqrt(- 0.5 / eta_2)
loc_mlconv_lin = - 0.5 * eta_1 / eta_2

with plt.rc_context({**bundles.aistats2022(column='full'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.75, 4.1717/2.2))
    axs[0].plot(perfs_med_sqrt.Samples, perfs_med_sqrt.mlconv, label='Convex (ours)',lw=2, alpha=0.8, c=blue)
    axs[0].plot(perfs_med_sqrt.Samples, perfs_med_sqrt.ifgls, label='IFGLS', lw=2, alpha=0.8, c=orange)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$D_{\mathrm{KL}}[p || q_{\mathrm{est}}]$')
    axs[0].set_xlabel('Samples $N$')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].scatter(x, y, alpha=0.3, color='black', lw=0.5, s=4)
    axs[1].plot(x_test, loc_mlconv_sqrt, color=blue, lw=1.5)
    axs[1].fill_between(x_test, loc_mlconv_sqrt-2*scale_mlconv_sqrt, loc_mlconv_sqrt+2*scale_mlconv_sqrt, 
                        color=blue, alpha=0.3)
    axs[1].grid()
    axs[1].set_ylim([-5, 5])
    
    axs[2].scatter(x, y, alpha=0.3, color='black', lw=0.5, s=4)
    axs[2].plot(x_test, loc_fgls_sqrt, color=orange, lw=1.5)
    axs[2].fill_between(x_test, loc_fgls_sqrt-2*scale_fgls_sqrt, loc_fgls_sqrt+2*scale_fgls_sqrt, 
                        color=orange, alpha=0.3)
    axs[2].grid()
    axs[2].set_ylim([-6, 5])
    axs[1].set_ylim([-6, 5])
    axs[2].set_xlabel(r'$x$')
    axs[1].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')
    axs[1].set_ylabel(r'$y$')
    
    axs[0].scatter(1000, perfs_med_sqrt.mlconv[perfs_med_sqrt.Samples==1000], 
                   color=blue, alpha=0.8, marker='*', s=100, zorder=100, lw=0.2, edgecolor='black')
    axs[1].set_title('Convex (ours) for $N=1000$')
    axs[2].set_title('IFGLS for $N=1000$')
    axs[0].scatter(1000, perfs_med_sqrt.ifgls[perfs_med_sqrt.Samples==1000], 
                   color=orange, alpha=0.8, marker='*', s=100, zorder=99, lw=0.2, edgecolor='black')
plt.savefig('paper_figures/estimator_benchmark_sqrt.pdf', transparent=True)

with plt.rc_context({**bundles.aistats2022(column='full'), **axes.lines()}):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.75, 4.1717/2.2))
    axs[0].plot(perfs_med_lin.Samples, perfs_med_lin.mlconv, label='Convex (ours)',lw=2, alpha=0.8, c=blue)
    axs[0].plot(perfs_med_lin.Samples, perfs_med_lin.ifgls, label='IFGLS', lw=2, alpha=0.8, c=orange)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$D_{\mathrm{KL}}[p || q_{\mathrm{est}}]$')
    axs[0].set_xlabel('Samples')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].scatter(x, y, alpha=0.3, color='black', lw=0.5, s=4)
    axs[1].plot(x_test, loc_mlconv_sqrt, color=blue, lw=1.5)
    axs[1].fill_between(x_test, loc_mlconv_lin-2*scale_mlconv_lin, loc_mlconv_lin+2*scale_mlconv_lin, 
                        color=blue, alpha=0.3)
    axs[1].grid()
    axs[1].set_ylim([-5, 5])
    
    axs[2].scatter(x, y, alpha=0.3, color='black', lw=0.5, s=4)
    axs[2].plot(x_test, loc_fgls_lin, color=orange, lw=1.5)
    axs[2].fill_between(x_test, loc_fgls_lin-2*scale_fgls_lin, loc_fgls_lin+2*scale_fgls_lin, 
                        color=orange, alpha=0.3)
    axs[2].grid()
    axs[2].set_ylim([-5, 5])
    axs[2].set_xlabel(r'$x$')
    axs[1].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')
    axs[1].set_ylabel(r'$y$')
    axs[2].set_ylim([-6, 5])
    axs[1].set_ylim([-6, 5])
    axs[2].set_xlabel(r'$x$')
    axs[1].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')
    axs[1].set_ylabel(r'$y$')
    
    axs[0].scatter(1000, perfs_med_lin.mlconv[perfs_med_sqrt.Samples==1000], 
                   color=blue, alpha=0.8, marker='*', s=100, zorder=100, lw=0.2, edgecolor='black')
    axs[1].set_title('Convex (ours) for $N=1000$')
    axs[2].set_title('IFGLS for $N=1000$')
    axs[0].scatter(1000, perfs_med_lin.ifgls[perfs_med_sqrt.Samples==1000], 
                   color=orange, alpha=0.8, marker='*', s=100, zorder=99, lw=0.2, edgecolor='black')
plt.savefig('paper_figures/estimator_benchmark_lin.pdf', transparent=True)
