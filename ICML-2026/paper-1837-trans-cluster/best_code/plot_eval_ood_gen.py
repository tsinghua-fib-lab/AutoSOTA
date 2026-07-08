#!/usr/bin/env python

import argparse
import os
import time
import sys
import random
import glob
from itertools import product
from typing import cast

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd
pd.set_option('display.precision', 4)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import torch
torch.set_printoptions(precision=4)
from torch import nn
from torch.distributions.categorical import Categorical

print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()

from utils import lloyds_iters_batched
from losses import kmeans_obj_batched, ADICT
from ctasks import DMAP, ClusteringTasks
from kmt import ATTDICT, KMeansTransformer, KModel

from plot_train_id_gen import label_gen


INIT_SCHEMES = [
  'random',
  'gonzalez',
  'kmeans++',
]


def gnzkpp(X, c, k, sample):
  clist = [c]
  for i in range(k - 1):
    C = torch.stack(clist)
    sqdist = torch.square(torch.cdist(X, C))
    minsqdist, _ = sqdist.min(dim=1)
    weights = minsqdist / minsqdist.sum()
    new_cidx = (
      Categorical(probs=weights).sample() if sample
      else minsqdist.argmax()
    )
    clist += [X[new_cidx]]
  C = torch.stack(clist)
  return C


def seed_lloyds(tasks, scheme):
  assert scheme in INIT_SCHEMES
  if scheme == 'random':
    return tasks
  ret = []
  print(f"{len(tasks)} tasks")
  for XX, CC in tasks:
    print(XX.shape, CC.shape)
    bsz = XX.shape[0]
    GCC = torch.stack([gnzkpp(
      XX[i], CC[i][0], CC.shape[1],
      sample=(scheme=='kmeans++')
    ) for i in range(bsz) ])
    assert CC.shape == GCC.shape
    print(XX.shape, CC.shape, GCC.shape)
    ret += [(XX, GCC)]
  return ret


def process_model(
    batch_samples: torch.Tensor,
    batch_inits: torch.Tensor,
    chkpt_file: str,
    reset: bool,
    niters: int,
):
  bsz, nclusters, ndims = batch_inits.shape
  assert batch_samples.shape[0] == bsz
  assert batch_samples.shape[2] == ndims

  # load up model from checkpoint file
  checkpoint = torch.load(chkpt_file)
  cli_args = checkpoint['cli_args']
  assert cli_args['ndims'] == ndims
  # load model
  ssize = {
    'onehot': nclusters,
    'none': 0,
  }
  demb = ndims + ssize[cli_args['scratch']]
  # initializing the kmeans transformer model
  model = KModel(
    demb, cli_args['dqkv']*demb,
    inv_temp=cli_args['attn_itemp'],
    dropout_p=cli_args['dropout'],
    act=cli_args['attn_act'],
  )
  model.load_state_dict(checkpoint['model_state_dict'])
  if GPUE:
    model = model.to('cuda')
  model.eval()
  kmeans_obj = [kmeans_obj_batched(batch_samples, batch_inits)]
  with torch.no_grad():
    BXX, BCC = get_scratch(
      batch_samples, batch_inits,
      cli_args['scratch'], ssize[cli_args['scratch']]
    )
    for i in range(niters):
      BXX, BCC = model(BXX, BCC)
      kmeans_obj += [kmeans_obj_batched(batch_samples, BCC[:, :, :ndims])]
      if reset:
        BXX, BCC = get_scratch(
          batch_samples, BCC[:, :, :ndims],
          cli_args['scratch'], ssize[cli_args['scratch']]
        )
  return torch.stack(kmeans_obj)


def get_scratch(
    samples: torch.Tensor,
    centers: torch.Tensor,
    scratch: str,
    scratch_size: int,
):
  b, n, d = samples.shape
  _, nclusters, _ = centers.shape
  assert centers.shape[0] == b
  assert centers.shape[2] == d
  def get_cluster_id_enc():
    if scratch == 'onehot':
      # one-hot encoding equivalent to an identity matrix
      return torch.eye(scratch_size)
    assert False

  if scratch == 'none':
    XX = samples.clone().detach()
    CC = centers.clone().detach()
  else:
    YY = torch.zeros(b, n, scratch_size)
    MM = get_cluster_id_enc().repeat(b, 1, 1)
    if GPUE:
      YY = YY.to('cuda')
      MM = MM.to('cuda')
    XX = torch.cat((samples.clone().detach(), YY), dim=2)
    CC = torch.cat((centers.clone().detach(), MM), dim=2)
  return XX, CC


def process_ood_task(
    task_dict: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    niters: int,
    ax: Axes,
    optdf: pd.DataFrame,
    color_map: dict[str, str],
    vhp: str,
    lossact: str,
    seedcolname: str,
    reset: bool,
    quantile: float,
    lloyds_res_cached=None,
):
  xx = np.arange(0, niters+1).astype(int)
  # run Lloyd's and plot result
  if lloyds_res_cached is None:
    lloyds_obj_dict = {seed: torch.hstack([
      lloyds_iters_batched(BXX, BCC, niters, use_cuda=GPUE)
      for (BXX, BCC) in tasks
    ]) for seed, tasks in task_dict.items()}
    print(f"Lloyd's iteration objectives computed")
    #   and report avg log-scaled-kmeans-objective
    yyy = torch.stack([
      torch.mean(torch.log(lloyds_obj), dim=1)
      for seed, lloyds_obj in lloyds_obj_dict.items()
    ])
    yy = torch.quantile(yyy, q=0.5, dim=0)
    yyub = torch.quantile(yyy, q=(1.0-quantile), dim=0)
    yylb = torch.quantile(yyy, q=quantile, dim=0)
  else:
    print(f"Using cached results for Lloyd's iteration")
    yy, yylb, yyub = lloyds_res_cached
  assert xx.shape == yy.shape
  ax.plot(
    xx, yy, color='xkcd:almost black', linestyle='solid',
    label="Lloyd's", linewidth=1
  )
  ax.fill_between(xx, yylb, yyub, color='xkcd:almost black', alpha=0.2)
  print(f"Lloyd's iteration objectives plotted:")


  # for all models in optdf, run model and plot results
  ret_list = []
  for v1, vhpdf in optdf.groupby(vhp):
    label = label_gen(v1, lossact, alt=args.alt_plot)
    color = color_map[label]
    print(v1, lossact, reset, color, label)
    model_obj_dict = {int(row[seedcolname]): torch.hstack([
      process_model(BXX, BCC, str(row['fname']), reset=reset, niters=niters)
      for BXX, BCC in task_dict[int(row[seedcolname])]
    ]) for idx, row in vhpdf.iterrows()}
    myyy = torch.stack([
      torch.mean(torch.log(model_objs), dim=1).to('cpu')
      for seed, model_objs in model_obj_dict.items()
    ])
    myy = torch.quantile(myyy, q=0.5, dim=0)
    myyub = torch.quantile(myyy, q=(1.0-quantile), dim=0)
    myylb = torch.quantile(myyy, q=quantile, dim=0)
    ax.plot(
      xx, myy, color=color, linestyle='solid',
      label=label, linewidth=1
    )
    ax.fill_between(xx, myylb, myyub, color=color, alpha=0.1)
    ret_list += [[
      label, yy[0].item(), yy[-1].item(),
      myy[0].item(), myy[-1].item()
    ]]

  return (yy, yylb, yyub), ret_list

def add_grids_and_save(figobj, axes, out_file):
  print('adding grids to figure')
  for ax in axes.reshape(-1):
    ax.minorticks_on()
    ax.grid(which='major', axis='both', alpha=0.9)
    ax.grid(which='minor', axis='both', alpha=0.4)
  figobj.tight_layout()
  print(f"saving figure in {out_file} ...")
  figobj.savefig(out_file)
  plt.close()


def set_seed(SEED):
  RNG = np.random.RandomState(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)


if __name__ == '__main__':

  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-I', '--indir', help='Input directory', type=str, required=True,
  )
  parser.add_argument(
    '-R', '--regex', help='Regular expression for results', type=str, required=True,
  )
  parser.add_argument(
    '-V', '--var_hps',
    help='HPs to vary on in the same plot; colon-separated',
    type=str, required=True,
  )
  parser.add_argument(
    '--seed', help='Argument name for expt seed', type=str, required=True,
  )
  parser.add_argument(
    '-O', '--outdir', help='Output directory', type=str, required=True,
  )
  parser.add_argument(
    '-P', '--prefix', help='Output file name prefix', type=str, required=True,
  )
  parser.add_argument(
    '-M', '--size_multiplier', help='Multiplier for plot size',
    type=float, default=1.0,
  )
  parser.add_argument(
    '-T', '--niters', help='Number of clustering iterations',
    type=int, required=True,
  )
  parser.add_argument(
    '--logy', help="Use logscale for y-axis", action="store_true",
  )
  parser.add_argument(
    '--sharey', help="Share y-axis within row", action="store_true",
  )
  parser.add_argument(
    '--logx', help="Use logscale for x-axis", action="store_true",
  )
  opmodes = ['mstep', 'lengen', 'varfam']
  parser.add_argument(
    '--opmode', choices=opmodes, required=True,
  )
  parser.add_argument(
    '--quantile', help="Quantile for multi-trial aggregation",
    type=float, default=0.25
  )
  parser.add_argument(
    '--nlegendcols', help="Number of columns in the legend",
    type=int, default=1,
  )
  parser.add_argument(
    '--skip_rembed_false', help="Should we skip rembed=false", action="store_true",
  )
  parser.add_argument(
    '--alt_plot', help="Should we color and label alt?", action="store_true",
  )
  parser.add_argument(
    '--init_scheme', help="initialization scheme for lloyd's",
    choices=INIT_SCHEMES, default=INIT_SCHEMES[0],
  )
  args = parser.parse_args()
  assert os.path.isdir(args.indir)
  assert os.path.isdir(args.outdir)
  assert 0. <= args.quantile <= 0.5

  tstr = time.ctime().replace('  ', ' ').split(' ')
  dstr = f"{tstr[4]}{tstr[1]}{tstr[2]}"

  rfiles = glob.glob(os.path.join(args.indir, args.regex))
  print(f"Found {len(rfiles)} files ...")

  def split_fname(f):
    return [
      val.split('=')
      for val in f.replace(args.indir, "")
      .replace('/', '').replace('_last.pt', '')
      .split('_')
    ]

  snames = [split_fname(rf) for rf in rfiles]
  sdict = {k: [] for k, _ in snames[0]}
  sdict['fname'] = []
  vhps = args.var_hps.split(':')
  assert len(vhps) == 2
  for vv in vhps:
    assert vv in sdict.keys(), f"HP: {vv} not found in\n{sdict.keys()}"
  assert args.seed in sdict.keys(), f"Seed column not found in\n{sdict.keys()}"
  for sn, fn in zip(snames, rfiles):
    for k, v in sn:
      sdict[k] += [v]
    sdict['fname'] += [fn]
  sdf = pd.DataFrame.from_dict(sdict)
  for vv in vhps:
    print(f"Unique values for var HP {vv}:{sdf[vv].unique()}")

  sdf[args.seed] = pd.to_numeric(sdf[args.seed])
  print(sdf[args.seed].unique())
  # Check if all have the same set of validation tasks
  task_opts = [
    'nlb', 'nub', 'nclusters', 'ndims',
    'train_dists', 'scale', 'em', 'es', 'sdb',
    'bsz', 'val_nbatches',
  ]
  id_val_task_dict = {}
  task_args = None
  for seed, seeddf in sdf.groupby(args.seed):
    sfiles = seeddf['fname'].values
    chkpt = torch.load(sfiles[0])
    id_val_tasks = chkpt['val_tasks']
    cli_args = chkpt['cli_args']
    if task_args is None:
      task_args = {k: cli_args[k] for k in task_opts}
      print(f"Task args: {task_args}")
    print(f"Found {len(id_val_tasks)} batches of validation tasks ...")
    print(f"... checking consistency across HPs for seed {seed} ...")
    for rf in sfiles:
      tmp = torch.load(rf)
      cvts = tmp['val_tasks']
      assert len(cvts) == len(id_val_tasks)
      if args.opmode != opmodes[2] or task_args['train_dists'] == tmp['cli_args']['train_dists']:
        assert all([
          torch.all(cvt[0] == vt[0])
          for (cvt, vt) in zip(cvts, id_val_tasks)
        ]), f"File {rf} does not match"
        assert all([
          torch.all(cvt[1] == vt[1])
          for (cvt, vt) in zip(cvts, id_val_tasks)
        ]), f"File {rf} does not match"
      else:
        pass
      for k, v in task_args.items():
        if k == 'train_dists' and args.opmode == opmodes[2]:
          continue
        assert tmp['cli_args'][k] == v, (
          f"file: {rf}, cli-arg: {k}, value {v} vs {tmp['cli_args'][k]}"
        )
    print(f"[Seed {seed}] All validation tasks and task configurations match!!")
    id_val_task_dict[seed] = id_val_tasks

  # Ensure task_args was initialized (i.e., at least one seed was processed)
  if task_args is None:
    raise ValueError(
      f"No valid checkpoint files found matching the seed column '{args.seed}'. "
      "Ensure the input directory contains checkpoint files with the expected format."
    )

  tmp_dict = {
    k: seed_lloyds(v, args.init_scheme)
    for k, v in id_val_task_dict.items()
  }
  id_val_task_dict = tmp_dict
  lossact = 'softmax'
  ofile = os.path.join(
    args.outdir,
    f"{args.prefix}_iters{args.niters}_{lossact}_{args.opmode}_{dstr}.pdf")
  print(f"saving file in {ofile}")

  mult = args.size_multiplier

  # multi-step evaluation
  if args.opmode == opmodes[0]:
    nrows = 1
    ncols = 1 if args.skip_rembed_false else 2
    fig, msaxs = plt.subplots(
      nrows, ncols, figsize=(ncols * mult, nrows * mult),
      sharex=True,
      sharey=("row" if args.sharey else False),
    )
    if ncols == 1 and nrows == 1:
      msaxs = np.array([msaxs])
    # eval on in-distribution validation tasks vs Lloyd's
    print('='*60)
    print('Processing in-distribution tasks')
    print('-'*60)
    # - run Lloyd's on all validation tasks across all seeds
    id_lloyds_obj_dict = {seed: torch.hstack([
      lloyds_iters_batched(BXX, BCC, args.niters, use_cuda=GPUE)
      for (BXX, BCC) in id_val_tasks
    ]) for seed, id_val_tasks in id_val_task_dict.items()}
    print(f"Lloyd's iteration objectives computed ...")
    xx = np.arange(0, args.niters+1).astype(int)
    #   and report avg log-scaled-kmeans-objective across all seeds
    yyy = torch.stack([
      torch.mean(torch.log(id_lloyds_obj), dim=1)
      for seed, id_lloyds_obj in id_lloyds_obj_dict.items()
    ])
    yy = torch.quantile(yyy, q=0.5, dim=0)
    yyub = torch.quantile(yyy, q=(1.0-args.quantile), dim=0)
    yylb = torch.quantile(yyy, q=args.quantile, dim=0)
    assert xx.shape == yy.shape
    for ax in msaxs:
      ax.plot(
        xx, yy, color='xkcd:almost black',
        linestyle='solid',
        label="Lloyd's", linewidth=1
      )
      ax.fill_between(xx, yylb, yyub, color='xkcd:almost black', alpha=0.2)
    print(f"Lloyd's iteration objectives plotted ...")

    reset = [True] if args.skip_rembed_false else [False, True]
    mcolor_list = [
      'xkcd:salmon',
      'xkcd:peacock blue',
      'xkcd:dark orange',
    ] if args.alt_plot else [
      'xkcd:apple green',
      'xkcd:olive green',
      'xkcd:aquamarine',
      'xkcd:peacock blue',
      'xkcd:salmon',
      'xkcd:dark orange',
    ]
    v1vals = sorted(sdf[vhps[0]].unique())
    v2vals = sorted(sdf[vhps[1]].unique())
    color_dict = {
      label_gen(e, r, alt=args.alt_plot): mcolor_list[idx]
      for idx, (e, r) in enumerate(product(v1vals, v2vals))
    }
    print(color_dict)
    for cidx, r in enumerate(reset):
      # - run all HP models on all validation tasks without scratch reset
      # - run all HP models on all validation tasks with scratch reset
      for group_key, vhpdf in sdf.groupby(vhps):
        v1, v2 = cast(tuple, group_key)
        label = label_gen(v1, v2, alt=args.alt_plot)
        color = color_dict[label]
        print(r, v1, v2, label, color)
        model_obj_dict = {int(row[args.seed]): torch.hstack([
          process_model(BXX, BCC, str(row['fname']), reset=r, niters=args.niters)
          for BXX, BCC in id_val_task_dict[int(row[args.seed])]
        ]) for idx, row in vhpdf.iterrows()}
        myyy = torch.stack([
          torch.mean(torch.log(model_objs), dim=1).to('cpu')
          for seed, model_objs in model_obj_dict.items()
        ])
        myy = torch.quantile(myyy, q=0.5, dim=0)
        myyub = torch.quantile(myyy, q=(1.0-args.quantile), dim=0)
        myylb = torch.quantile(myyy, q=args.quantile, dim=0)
        msaxs[cidx].plot(
          xx, myy, color=color, linestyle='solid',
          label=label, linewidth=1
        )
        msaxs[cidx].fill_between(xx, myylb, myyub, color=color, alpha=0.2)
      msaxs[cidx].set_xlabel("# clustering steps", fontsize=10)
      if not args.alt_plot:
        msaxs[cidx].set_title(f"Re-embed: {r}", fontsize=10)
    msaxs[0].set_ylabel(f"avg log-kmeans-obj", fontsize=10)
    msaxs[-1].legend(
      ncol=args.nlegendcols, loc='best', fontsize=8,
      framealpha=1.0, fancybox=True, handlelength=0.7
    )
    add_grids_and_save(fig, msaxs, ofile)

  # eval on out-of-distribution validation tasks vs Lloyd's
  optdf = pd.DataFrame(sdf[sdf[vhps[1]] == lossact])
  print(f"Found configs of {lossact}: {optdf.shape}")
  reset = True
  mcolor_list = [
    'xkcd:salmon',
    'xkcd:peacock blue',
    'xkcd:dark orange',
  ] if args.alt_plot else [
    'xkcd:olive green',
    'xkcd:peacock blue',
    'xkcd:dark orange',
  ]
  v1vals = sorted(optdf[vhps[0]].unique())  # type: ignore[attr-defined]
  color_dict = {
    label_gen(e, lossact, alt=args.alt_plot): mcolor_list[idx]
    for idx, e in enumerate(v1vals)
  }
  print(color_dict)
  SEEDS = id_val_task_dict.keys()

  # - out of distribution in terms of the number of samples to cluster
  if args.opmode == opmodes[1]:
    fig, axs = plt.subplots(
      1, 4, figsize=(4 * mult, mult),
      sharex=True,
      sharey=("row" if args.sharey else False),
    )
    print('='*60)
    print('Processing out-of-distribution tasks -- number of samples')
    print('-'*60)
    nvals = [
      task_args['nub'] // 4,
      task_args['nub'] // 2,
      task_args['nub'] * 2,
      task_args['nub'] * 4,
    ]
    for cidx, nval in enumerate(nvals):
      # Instantiate task generator and obtain validation set for each seed
      ood_val_task_dict = {}
      for seed in SEEDS:
        set_seed(seed)
        task_gen = ClusteringTasks(
          task_args['nclusters'], task_args['nclusters'], task_args['ndims'],
          nval, nval, [d for d in task_args['train_dists'].split('+')],
          scale=task_args['scale'],
          equal_mix=task_args['em'], equal_scales=task_args['es'],
        )
        ood_val_task_dict[seed] = [
          task_gen.sample_batch(task_args['bsz'], same_dist_batch=task_args['sdb'])
          for _ in range(task_args['val_nbatches'])
        ]
      axs[cidx].set_title(f"#samples: {nval}")
      _ = process_ood_task(
        ood_val_task_dict, args.niters,
        axs[cidx], optdf,
        color_dict,
        vhps[0], lossact, args.seed,
        reset, args.quantile
      )
      axs[cidx].set_xlabel("# clustering steps", fontsize=10)
    axs[0].set_ylabel(f"avg log-kmeans-obj", fontsize=10)
    axs[-1].legend(ncol=args.nlegendcols, loc='best', fontsize=8)
    add_grids_and_save(fig, axs, ofile)

  # - Distribution changes cauchy/gumbel/laplace/lognormal
  if args.opmode == opmodes[2]:
    dists = DMAP.keys()
    fig, axs = plt.subplots(
      1, len(dists), figsize=(len(dists) * mult, mult),
      sharex=True,
      sharey=("row" if args.sharey else False),
    )
    print('='*60)
    print('Processing out-of-distribution tasks -- distribution family')
    print('-'*60)
    drows = []
    for cidx, dist in enumerate(dists):
      print(f"Processing distribution: {dist}")
      # Instantiate task generator and obtain validation set for each seed
      ood_val_task_dict = {}
      for seed in SEEDS:
        set_seed(seed)
        task_gen = ClusteringTasks(
          task_args['nclusters'], task_args['nclusters'],
          task_args['ndims'],
          task_args['nlb'], task_args['nub'],
          [dist],
          scale=task_args['scale'],
          equal_mix=task_args['em'], equal_scales=task_args['es'],
        )
        ood_val_task_dict[seed] = [task_gen.sample_batch(
          task_args['bsz'], same_dist_batch=task_args['sdb']
        ) for _ in range(task_args['val_nbatches'])]
      axs[cidx].set_title(f"Dist:{dist}")
      _, drow = process_ood_task(
        ood_val_task_dict, args.niters,
        axs[cidx], optdf,
        color_dict,
        vhps[0], lossact, args.seed,
        reset, args.quantile
      )
      drows += [(dist, *dr) for dr in drow]
      axs[cidx].set_xlabel("# clustering steps", fontsize=10)
      print(f"Dist: {dist} completed")
    axs[0].set_ylabel(f"avg log-kmeans-obj", fontsize=10)
    axs[-1].legend(ncol=args.nlegendcols, loc='best', fontsize=8)
    add_grids_and_save(fig, axs, ofile)
    print(pd.DataFrame(drows))
