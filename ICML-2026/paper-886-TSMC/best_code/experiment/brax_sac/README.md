# Run Baseline SAC on brax environments

We fork the `learn` entry-point for the brax module within `main.py` with some simplifications for efficiency (see https://github.com/google/brax/blob/main/brax/training/learner.py). At the time of writing, the `PyPi` release of `brax` did not work.


This script can be called with:
```bash
python main.py --env ant --logdir $HOME/out
```
this will run `sac` by default.

Or one can simply run `sac` on all environments we tested by running:
```bash
./run.sh 0 30 $HOME/out/brax
```
This will run 4 environments
 - ant
 - hopper
 - walker2d
 - halfcheetah

For 30 seeds (from 0 to 29 inclusively), and generate log-output folders (tensorboard/ tf-events files) according to the naming convention: `$HOME/out/brax/{env}_{seed}/`.
