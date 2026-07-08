#!/usr/bin/env python

import argparse
import os
import time
import sys
import random
import glob
from itertools import product

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd
pd.set_option('display.precision', 4)
import matplotlib as mpl
import matplotlib.pyplot as plt


def label_gen(ename, rname, alt=False):
  edict = {
    'onehot': 'demb=d+k' if alt else 'OH',
    'none': 'demb=d' if alt else 'NA'
  }
  rdict = {
    'softmax': 'NE',
    'sparsemax': 'L2',
  }
  assert ename in edict.keys()
  assert rname in rdict.keys()
  return edict[ename] if alt else f"{edict[ename]}:{rdict[rname]}"


if __name__ == '__main__':

  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-I', '--indir', help='Input directory', type=str, required=True,
  )
  parser.add_argument(
    '-R', '--regex', help='Regular expression for results',
    type=str, required=True,
  )
  parser.add_argument(
    '-S', '--split_hp', help='HP to split on', type=str, required=True,
  )
  parser.add_argument(
    '--numeric', help="HP to split is numeric", action="store_true",
  )
  parser.add_argument(
    '-V', '--var_hps',
    help='HPs to vary on in the same plot; colon-separated',
    type=str, required=True,
  )
  parser.add_argument(
    '-T', '--hpname', help='HP name', type=str, required=True,
  )
  parser.add_argument(
    '-C', '--config', help='Config string to add to filename',
    type=str, required=True,
  )
  parser.add_argument(
    '-O', '--outdir', help='Output directory', type=str, required=True,
  )
  parser.add_argument(
    '-M', '--size_multiplier', help='Multiplier for plot size',
    type=float, default=1.0,
  )
  parser.add_argument(
    '--sharey', help="Share y-axis within row", action="store_true",
  )
  parser.add_argument(
    '--logy', help="Use logscale for y-axis", action="store_true",
  )
  parser.add_argument(
    '--logx', help="Use logscale for x-axis", action="store_true",
  )
  parser.add_argument(
    '--quantile', help="Quantile for multi-trial aggregation",
    type=float, default=25.0
  )
  parser.add_argument(
    '--drop_step_zero',
    help="Whether to drop losses at step = 0", action="store_true",
  )
  parser.add_argument(
    '--inv_hp_val',
    help="Whether to invert HP value (use 1/value)", action="store_true",
  )
  parser.add_argument(
    '--hp_val_int', help="Is HP an integer?", action="store_true",
  )
  parser.add_argument(
    '--nlegendcols', help="Number of columns in the legend",
    type=int, default=1,
  )
  parser.add_argument(
    '--skip_embs', help="colon-separated list of embeddings to skip",
    type=str, default="",
  )
  parser.add_argument(
    '--alt_plot', help="Is this alternate plot?", action="store_true",
  )
  parser.add_argument(
    '--xticks_steps', help="Steps for x-axis steps",
    type=int, default=500,
  )

  args = parser.parse_args()
  assert os.path.isdir(args.indir)
  assert os.path.isdir(args.outdir)
  assert 0. <= args.quantile <= 50.0

  # set timestamp for file name
  tstr = time.ctime().replace('  ', ' ').split(' ')
  dstr = f"{tstr[4]}{tstr[1]}{tstr[2]}"

  rfiles = glob.glob(os.path.join(args.indir, args.regex))
  print(f"Found {len(rfiles)} files ...")

  def split_fname(f):
    return [
      val.split('=')
      for val in f.replace(args.indir, "")
      .replace('/', '').replace('.csv', '')
      .split('_')
    ]

  skip_embs = args.skip_embs.split(":")

  snames = [split_fname(rf) for rf in rfiles]
  sdict = {k: [] for k, _ in snames[0]}
  sdict['fname'] = []
  assert args.split_hp in sdict.keys()
  vhps = args.var_hps.split(':')
  assert len(vhps) == 2
  for vv in vhps:
    assert vv in sdict.keys(), f"HP: {vv} not found in\n{sdict.keys()}"
  for sn, fn in zip(snames, rfiles):
    skip = False
    for k, v in sn:
      if k == "e":
        if v in skip_embs:
          skip = True
    if skip:
      continue
    for k, v in sn:
      sdict[k] += [v]
    sdict['fname'] += [fn]
  sdf = pd.DataFrame.from_dict(sdict)
  if args.numeric:
    sdf[args.split_hp] = pd.to_numeric(sdf[args.split_hp])
  print(
    f"Unique values for split HP {args.split_hp}: "
    f"{np.sort(sdf[args.split_hp].unique())}"
  )
  for vv in vhps:
    print(f"Unique values for var HP {vv}: {sdf[vv].unique()}")

  hist_lens = pd.to_numeric(sdf['t']).unique()
  print(f"History lengths: {hist_lens}")
  assert len(hist_lens) == 1
  hist_len = hist_lens[0]

  mcolor_list = [
    'xkcd:apple green',
    'xkcd:olive green',
    'xkcd:aquamarine',
    'xkcd:peacock blue',
    'xkcd:salmon',
    'xkcd:dark orange',
    'xkcd:cadet blue',
    'xkcd:navy blue',
    'xkcd:almost black',
    'xkcd:forest green',
    'xkcd:raspberry',
    'xkcd:cadet blue',
    'xkcd:bright purple',
    'xkcd:brick red',
    'xkcd:brownish pink',
  ]
  if args.alt_plot:
    mcolor_list = mpl.colormaps['Dark2'].colors

  mstyle_list = [
    'solid',
    'dashed',
    'dashdot',
    'dotted',
  ]
  combos = product(
    np.sort(sdf[vhps[0]].unique()),
    np.sort(sdf[vhps[1]].unique()),
  )
  color_map = {
    f"{vv[0]}__{vv[1]}": mcolor_list[idx]
    for idx, vv in enumerate(combos)
  }
  print(color_map)
  tname = args.hpname.replace(' ', '_').replace('$', '').replace("\\", "")
  ofile = os.path.join(
    args.outdir,
    f"{tname.lower()}_{args.config}_{dstr}.pdf"
  )
  print(f"saving file in {ofile}")

  mult = args.size_multiplier
  n_cols = len(sdf[args.split_hp].unique())
  fig, axs = plt.subplots(
    2, n_cols,
    figsize=(1.25 * mult * n_cols, 1.75 * mult * 2),
    sharex=True,
    sharey=("row" if args.sharey else False),
  )
  if len(axs.shape) == 1:
    axs = np.expand_dims(axs, axis=1)
  axs[0, 0].set_ylabel("Training loss")
  axs[1, 0].set_ylabel("Relative validation loss")

  cidx = 0
  ee = None  # Initialize ee to avoid unbound variable warning
  for g, gdf in sdf.groupby(args.split_hp):
    hpval = 1.0/float(g) if args.inv_hp_val else float(g) if args.numeric else g
    hpstr = (f"{int(hpval):d}" if args.hp_val_int else f"{hpval:.2f}") if args.numeric else str(hpval)
    axs[0, cidx].set_title(f"{args.hpname}: {hpstr}")
    axs[1, cidx].set_xlabel("Train steps", loc='left')
    for (v1, v2), vdf in gdf.groupby([vhps[0], vhps[1]]):
      lcolor = color_map[f"{v1}__{v2}"]
      print(v1, v2, vdf.shape)
      lstyle = "solid"
      xxx, yyy, zzz, eee = [], [], [], []
      set_names = ['train-batch-loss', 'val-kmeans-rel-obj']
      for fn in vdf['fname'].values:
        df = pd.read_csv(fn)
        df1 = df[df['set'] == set_names[0]]
        fxx = np.array(df1['step'].values)[int(args.drop_step_zero):]
        fyy = np.array(df1['loss'].values)[int(args.drop_step_zero):]
        assert len(fxx) == len(fyy)
        xx, yy = [], []
        for i in range(0, len(fxx), int(hist_len)):
          xx += [fxx[min(i+int(hist_len)-1, len(fxx)-1)]]
          yy += [np.mean(fyy[i: min(i+int(hist_len), len(fyy))])]
        xx = np.array(xx)
        yy = np.array(yy)
        xxx += [xx]
        yyy += [yy]
        df2 = df[df['set'] == set_names[1]]
        eee += [np.array(df2['step'].values)[int(args.drop_step_zero):]]
        zzz += [np.array(df2['loss'].values)[int(args.drop_step_zero):]]
      # aggregate training stats
      xx = xxx[0]
      yyy = np.array(yyy)
      yy = np.percentile(yyy, q=50, axis=0)
      ylb = np.percentile(yyy, q=args.quantile, axis=0)
      yub = np.percentile(yyy, q=100.0-args.quantile, axis=0)
      assert len(yy) == len(ylb) == len(yub) == len(xx), (
        f"x-len: {len(xx)}, y-shape: {yy.shape}, {ylb.shape}, {yub.shape}"
      )
      # aggregate validation stats
      ee = eee[0]
      zzz = np.array(zzz)
      zz = np.percentile(zzz, q=50, axis=0)
      zlb = np.percentile(zzz, q=args.quantile, axis=0)
      zub = np.percentile(zzz, q=100.0-args.quantile, axis=0)
      assert len(zz) == len(zlb) == len(zub) == len(ee), (
        f"e-len: {len(ee)}, z-shape: {zz.shape}, {zlb.shape}, {zub.shape}"
      )
      axs[0, cidx].plot(
        xx, yy, color=lcolor, linestyle=lstyle,
        label=label_gen(v1, v2, alt=args.alt_plot), linewidth=1.0,
      )
      axs[0, cidx].fill_between(xx, ylb, yub, color=lcolor, alpha=0.2)
      axs[1, cidx].plot(
        ee, zz, color=lcolor, linestyle=lstyle,
        label=label_gen(v1, v2, alt=args.alt_plot), linewidth=1.0,
      )
      axs[1, cidx].fill_between(ee, zlb, zub, color=lcolor, alpha=0.2)
      axs[1, cidx].plot([0, xx[-1]], [1, 1], color='black', linestyle='dotted')
    if args.logy:
      axs[0, cidx].set_yscale('log', base=2)
      axs[1, cidx].set_yscale('log', base=2)
    if args.logx:
      axs[0, cidx].set_xscale('log', base=2)
      axs[1, cidx].set_xscale('log', base=2)
    cidx += 1

  for ax in axs.reshape(-1):
    ax.minorticks_on()
    ax.grid(which='major', axis='both', alpha=0.6)
    ax.grid(which='minor', axis='both', alpha=0.3)
    if ee is not None:
      ax.set_xticks(np.arange(0, ee.max(), step=args.xticks_steps))
  legend = axs[1, -1].legend(
    ncol=args.nlegendcols, loc='best', fontsize=7.5, framealpha=0.6,
    fancybox=True, handlelength=0.7
  )
  # change the line width for the legend
  for line in legend.get_lines():
    line.set_linewidth(2.0)
  fig.tight_layout()
  fig.savefig(ofile)
  plt.close()
