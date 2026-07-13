# Reproducing the experiments

This file reconstructs the commands used for the simulation figures in
`The_Cost_of_Learning_ICML.pdf`, based on the PDF text, the generated figure
files, and the scripts in this repository.

Run all commands from the repository root:

```bash
cd /path/to/the_cost_of_learning
mkdir -p .mplconfig
```

On macOS/headless environments, use:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=.mplconfig
```

The main dependencies used by the scripts are `numpy`, `matplotlib`, and `Pillow`
for the illustration scripts.

## Figure map

| Paper figure | Output file(s) | Script |
| --- | --- | --- |
| Fig. 1, endogenous-confounding illustration | `illustrations/figures/mixture_illustration_3x2.png` | `illustrations/mixture_illustration.py` |
| Fig. 2, ATC schematic run | `illustrations/figures_illustration/atc_illustration.png` | `illustrations/illustration.py` |
| Fig. 3(a,b), synthetic environment and cumulative regret | `icml/fig_new/alarms_T1200_exact_md.png`, `icml/fig_main/cumregret_T1200_md.png` | `icml/ATC.py` |
| Fig. 3(c), regret vs log horizon | `icml/fig_main/regret_vs_logT_md.png` | `icml/ATC.py` |
| Fig. 4, NAB CPU data | `icml/fig_nab_algo/*.png` | `icml/atc_nab_run.py` |
| Fig. 5(a), dense changes | `icml/figures_dense_new/regret_vs_logT_scenarios-dense_adver_gap160_dense_sqrt_adver_gap160.png` | `icml/atc_scaling_scenarios_dense_adver.py` |
| Fig. 5(b), passive baselines | `icml/fig-passive/regret_vs_logT_md_gap160.png` | `icml/atc_scaling_scenarios_dense_adver.py` |
| Fig. 6, constant thresholds | `icml/fig_const/regret_vs_horizon.png`, `icml/fig_const/regret_vs_horizon_fa.png` | `icml/consants.py` |
| Fig. 7, NAB sigma misspecification | same NAB runner, with `--sigma-list` | `icml/atc_nab_run.py` |
| Fig. 8, adversarial environment | `icml/fig_adver/regret_vs_logT_scenarios-adver_gap160_adver_gap1000.png` | `icml/atc_scaling_scenarios_dense_adver.py` |
| Fig. 9, regret vs number of changes | `icml/atc_vs_S_experiment/regret_vs_S.png` | `icml/atc_vary_S_experiment.py` |

## Illustrations

```bash
python3 illustrations/mixture_illustration.py
python3 illustrations/illustration.py
```

These produce Figure 1 and Figure 2 style schematic plots. They are deterministic
illustrations, not Monte Carlo experiments.

## Main synthetic experiments, Fig. 3

The PDF specifies the `md` environment:

- `S=5`
- change points at `floor(0.2T)+1`, `floor(0.4T)+1`, `tau2+10`,
  `floor(0.75T)+1`, `floor(0.9T)+1`
- segment means `(0, 2, 0.5, 2.5, -1.5, 1.5)`
- `sigma=1`, `alpha=0.05`
- horizons `600 1200 2400 4800 7000 9000`
- Fig. 3(c) reports 5000 MC replications
- efficient ATC uses geometric endpoint scans with base `2`

Representative environment/alarm plot:

```bash
python3 icml/ATC.py \
  --algos exact \
  --n-mc 10 \
  --T-list 1200 \
  --scenario md \
  --plot-alarms \
  --alarms-T 1200 \
  --alarms-algo exact \
  --alarms-seed 7 \
  --out-dir icml/fig_new \
  --skip-logT
```

Cumulative regret at `T=1200`:

```bash
python3 icml/ATC.py \
  --algos exact \
  --n-mc 5000 \
  --T-list 1200 \
  --scenario md \
  --plot-cumregret \
  --cumregret-T 1200 \
  --cumregret-annotate \
  --cumregret-alarms-algo exact \
  --cumregret-annotate-seed 7 \
  --out-dir icml/fig_main \
  --skip-logT
```

Regret scaling against `log(T)`:

```bash
python3 icml/ATC.py \
  --algos exact geom_ends \
  --geom-base 2.0 \
  --n-mc 5000 \
  --T-list 600 1200 2400 4800 7000 9000 \
  --scenario md \
  --out-dir icml/fig_main
```

For a faster check, replace `--n-mc 5000` with `--n-mc 10` or `--n-mc 100`.

## NAB CPU benchmark, Fig. 4

The PDF identifies `ec2_cpu_utilization_ac20cd.csv` and fixed reference change
points `377 420 592 3575`. The plotted baselines use window length `30` and
discount `0.98`.

```bash
python3 icml/atc_nab_run.py \
  --csv-path NAB/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv \
  --algos exact sliding discount \
  --window 30 --window-formula fixed \
  --discount 0.98 --discount-formula fixed \
  --cp-list 377 420 592 3575 \
  --plot-regret --plot-cp --plot-means \
  --out-dir icml/fig_nab_algo \
  --no-alarms
```

## Dense-change experiment, Fig. 5(a)

The PDF specifies `T in {1200,2400,4800,7000,9000,12000,15000,18000,20000,22000,25000}`,
`sigma=1`, `alpha=0.05`, `gap_c=160`, and 1000 MC replications. The two scenarios are
`dense_adver` with `S=floor(T/log T)` and `dense_sqrt_adver` with
`S=floor(T^0.95/log T)`.

```bash
python3 icml/atc_scaling_scenarios_dense_adver.py \
  --scenario dense_adver dense_sqrt_adver \
  --algos exact \
  --gap-c 160 \
  --sigma 1.0 --alpha 0.05 \
  --T-list 1200 2400 4800 7000 9000 12000 15000 18000 20000 22000 25000 \
  --n-mc 1000 \
  --out-dir icml/figures_dense_new \
  --x-axis T \
  --no-fit \
  --dense-delta 0.95
```

## Passive baselines, Fig. 5(b)

```bash
python3 icml/atc_scaling_scenarios_dense_adver.py \
  --algos exact sliding discount \
  --window-formula sqrt_TlogT_over_S \
  --discount-formula one_minus_sqrt_S_over_T \
  --n-mc 1000 \
  --T-list 600 1200 2400 3000 3800 4800 5500 6300 7000 8000 9000 \
  --scenario md \
  --no-fit \
  --out-dir icml/fig-passive
```

## Constant thresholds, Fig. 6

The figure currently in the repo is produced from hard-coded arrays in
`icml/consants.py`; the raw MC generator for those exact arrays is not present as
a separate script. The PDF describes a single-change environment with
`tau1=50`, jump `0.75`, `sigma=1`, `alpha=0.05`, horizons
`5000 10000 15000 20000 30000`, and 1000 MC replications.

To reproduce the stored plot:

```bash
python3 icml/consants.py --out-dir icml/fig_const
```

## NAB sigma misspecification, Fig. 7

```bash
python3 icml/atc_nab_run.py \
  --csv-path NAB/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv \
  --sigma-list 0.5 1 4 \
  --cp-list 377 420 592 3575 \
  --plot-regret --plot-cp --plot-means \
  --out-dir icml/fig_nab_sigma \
  --no-alarms
```

## Adversarial environment, Fig. 8

The PDF specifies fixed `S=5`, equal spacing, alternating levels, horizons
`600 1200 2400 4800 7000 9000`, `sigma=1`, `alpha=0.05`, 1000 MC replications,
and compares `gap_c=160` against the easier `gap_c=1000`.

```bash
python3 icml/atc_scaling_scenarios_dense_adver.py \
  --scenario adver \
  --algos exact \
  --gap-c 160 1000 \
  --sigma 1.0 --alpha 0.05 \
  --T-list 600 1200 2400 4800 7000 9000 \
  --n-mc 1000 \
  --out-dir icml/fig_adver \
  --no-fit
```

Optional environment examples:

```bash
python3 icml/atc_scaling_scenarios_dense_adver.py \
  --scenario adver \
  --algos exact \
  --gap-c 160 \
  --sigma 1.0 --alpha 0.05 \
  --T-list 1200 2400 \
  --n-mc 1 \
  --plot-env --env-T 1200 \
  --out-dir icml/figures_adver \
  --skip-logT
```

## Regret versus number of changes, Fig. 9

The PDF says `T=1000`, `S=2,4,...,20`, `sigma=1`, `alpha=0.05`, and 1000 MC
runs. The script varies `S` by `--S_max`/`--S_step`.

```bash
python3 icml/atc_vary_S_experiment.py \
  --T 1000 \
  --S_max 20 \
  --S_step 2 \
  --n_mc 1000 \
  --sigma 1.0 \
  --alpha 0.05 \
  --delta 2.0 \
  --seed 0 \
  --max_splits 0 \
  --fit_S_max 20 \
  --outdir icml/atc_vs_S_experiment
```

`--max_splits 0` is the exact scan and can be slow. Use `--max_splits 250` for a
faster approximate reproduction.

## Notes on runtime

The commands above use the MC counts stated in the PDF where available. Some are
heavy because exact ATC scans all split points. For GitHub readers, it is useful
to provide a `quick` variant with `--n-mc 10` and the paper command with
`--n-mc 1000` or `5000`.

The generated file names may differ slightly from the final manuscript folders,
because the repo contains several historical output directories. The command
arguments above target the folders whose timestamps and names most closely match
the final PDF.
