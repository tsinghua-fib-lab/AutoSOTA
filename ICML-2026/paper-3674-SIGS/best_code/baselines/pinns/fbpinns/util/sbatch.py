"""
Generic helper functions for running parallel sbatch jobs
"""

import subprocess
import sys
import pickle

from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.constants import Constants


def submit(runs):
    "submits parallel sbatch jobs"

    # remove any duplicate runs
    names_, runs_ = [], []
    for run in runs:
        name = run[0].run
        if name not in names_:
            runs_.append(run)
            names_.append(name)
        else:
            print(f"WARNING: removing duplicated run: {name}")
    runs = runs_

    for i,(c,_) in enumerate(runs): print(i,c.run)
    for i,(c,trainer) in enumerate(runs):
        print(f"submitting job {i+1} of {len(runs)}..")
        # save constants and submit sbatch job
        c.get_outdirs()
        c.save_constants_file()
        o = subprocess.run(["bash", "run.sh", f"{i}", c.constants_file, trainer],
                           check=False, capture_output=True)
        print(o.stdout.decode())
        print(o.stderr.decode())
        if o.returncode != 0:
            raise Exception(f"process returned non-zero exit code {o.returncode}")
    print("all jobs submitted")

def run():
    "target function for sbatch job"

    # parse arguments
    n = len(sys.argv)
    if n-1 != 2: raise Exception(f"ERROR: expected 2 input arguments: {sys.argv}")
    constants_file, trainer = sys.argv[1:]

    # get trainer
    if trainer not in ["FBPINN", "PINN"]:
        raise Exception(f"ERRROR: unexpected trainer: {trainer}")
    trainer = {"FBPINN":FBPINNTrainer, "PINN": PINNTrainer}[trainer]

    # get constants
    with open(constants_file, "rb") as f:
        kwargs = pickle.load(f)

    # create constants object
    c = Constants(**kwargs)
    c.show_figures = c.clear_output = False# make sure plots are not shown

    # create trainer, start training
    run = trainer(c)
    run.train()