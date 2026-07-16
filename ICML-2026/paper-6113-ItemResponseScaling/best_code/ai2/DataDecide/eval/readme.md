# Eval

## Evaluating processed checkpoints
We make of Ai2's evaluation framework to run evals over checkpoints. There are many tasks implemented in the framework that we have not yet evaluated DataDecide. You can try these out by just specifying task names in `all_olmes_rc_tasks.txt`.

Currently the code works around an internal version of this framework, but we are porting this to use public [OLMES framework](https://github.com/allenai/olmes) which is the same thing without Ai2 infrastructure code.


The following kicks off separate jobs for each combination of checkpoint and task with `eval_checkpoints_x_tasks.py`. The inputs to this are file like the `model_checkpoints.jsonl` we obtained in the checkpoint processing step (below) that lists what revisions are at what locations, and also a file with line separated task names.
```
python eval_checkpoints_x_tasks.py --checkpoint_data ../checkpoints/weka_paths.jsonl --tasks all_olmes_rc_tasks.txt --remote_output_dir_prefix s3://<> -- --cluster ai2/<>  --beaker-priority low --beaker-workspace ai2/cheap_decisions --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY --limit 10000000000000000000
```

## What to do if you haven't already processed (unsharded and converted) your checkpoints

We use a script in a branch of the OLMo repo to handle this.

Setup
```
git clone git@github.com:allenai/OLMo.git
cd OLMo
git checkout ianm/consistent-ranking-conversion
conda create -n olmo-batched-checkpoint-processing python=3.10
conda activate olmo-batched-checkpoint-processing
pip install -e .
pip install beaker-gantry
```

Now, from the OLMo root, you can process checkpoints like this:
```
sh scripts/convert_checkpoints.sh "s3://ai2-llm/checkpoints/OLMo-ladder/benb/c4-*/"
```

The beaker results dataset of this job will contain a `model_checkpoints.jsonl` file which maps the names and revisions of the models to various paths. This is the input you need for `eval_checkpoints_x_tasks.py`.

