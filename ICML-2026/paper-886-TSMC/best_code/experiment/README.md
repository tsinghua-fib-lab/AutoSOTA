
# Scripts for experiment deployment

## Command to run the main pipeline.

To simply run our experiment pipeline with a particular configuration:
```bash
bash run.sh deploy/task.sh \
  -M offline \
  -P configs/combined/debug.txt
```

Or to test this under a W&B project:
```bash
bash run.sh deploy/task.sh \
  -M offline \
  -P configs/combined/debug.txt \
  --wandb DEBUG  # DEBUG = wandb project name
```

## Command to run a W&B sweep for parameter ablations

To directly create and eagerly run a W&B sweep:
```bash
bash run.sh deploy/sweep.sh \
  -M offline \
  -P configs/combined/debug.txt \
  -O MY/OUTPUT/DIR \
  -E ENTITY \  # ENTITY = wandb entity (user/ team name)
  -N DEBUG_SWEEP \  # DEBUG_SWEEP = wandb project name
  -S configs/sweeps/repetitions.yaml configs/sweeps/ablations/smc.yaml
```

To create a W&B sweep without directly running the sweep agent:
```bash
bash run.sh deploy/sweep/compile_sweep.sh \
  -M offline \
  -P configs/combined/debug.txt \
  -O MY/OUTPUT/DIR \
  -E ENTITY \  # ENTITY = wandb entity (user/ team name)
  -p PROJECT \  # DEBUG_SWEEP = wandb project name
  -N DEBUG_SWEEP \  # DEBUG_SWEEP = wandb sweep name
  -S configs/sweeps/repetitions.yaml configs/sweeps/ablations/smc.yaml

# W&B output
# > wandb: Creating sweep from: /var/folders/tv/c20sghvj2hs50lbv63_2c51r0000gn/T/tmp.kZaYAeDduX
# > wandb: Creating sweep with ID: SWEEP_ID
# > wandb: View sweep at: https://wandb.ai/ENTITY/DEBUG_SWEEP/sweeps/SWEEP_ID
# > wandb: Run sweep agent with: wandb agent ENTITY/DEBUG_SWEEP/SWEEP_ID

# Script output
# > Generated Agent Script -- Sweep ID: SWEEP_ID
# > Run the wandb Agent with:
# > ./run.sh deploy/sweep/run/SWEEP_ID.sh $NUM_AGENTS $CHUNK_SIZE
```

This generates a bash script that calls the just generated agent, this agent can run on multiple devices asynchronously as long as it can synchronize to the W&B service (so it can communicate which config it has to run execute). The arguments to this script indicate 1) `$NUM_AGENTS` the number of parallel agents (best to keep at 1 when using GPUs), and 2) `CHUNK_SIZE` the number of configs this script should run before stopping (e.g., to limit the time it takes for a script to terminate) 

For example, if we compile a sweep with 10 configs that we need to run. Say we use a cluster with 2 GPU nodes, we can submit the following job to both nodes to split up the work evenly:
```bash
bash run.sh deploy/sweep/run/SWEEP_ID.sh 1 5

# Starts a parameter sweep Experiment ...
```

## Command to instantiate all experiment-sweeps
```bash
pwd
>> .../SMZ/experiment

# Make sure that a 'EXPERIMENT_OUTDIR=...' variable is set
./run.sh deploy/setup/generate_sweeps.sh -E ... -O '$EXPERIMENT_OUTDIR'
>> ...

# Instantiates and generates many W&B sweep runs
# To run the experiments, submit all run-files in `deploy/sweep/run`
```
