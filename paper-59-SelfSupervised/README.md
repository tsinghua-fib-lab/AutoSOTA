******Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting******

The template we use is https://github.com/ashleve/lightning-hydra-template.

To run the code, first install with requirements.txt.

```
conda create --name your_env_name --file environment.yml.
```

Then you should specify your data paths in **.env** file.

```
PROJECT_ROOT="your working directory"

DATA_DIR="your data directory"

HYDRA_FULL_ERROR=1 # 1 for complete error information for hydra
```

Finally run with: 

```
python run/src/train.py task=scam_{mlp/cyclenet/itransformer/patchtst}
# specify the name of the config file under configs/task/ for the model you would like to run with

```

For a specific configs setting, you can change the .yaml files in **configs** directories.
Or you can run with the scripts provided under the **scripts/** directory. 

The config structure will look like

```
configs
----data
----model
    ----components
    ----scam.yaml
    ...
----task
...
```

You should use the following to specify the backbone model you would like to implement with SCAM:

```
defaults:
 ...
 - {model_name}@predictor.pred_model # you also must have a {model_name}.yaml under the same directory
```

in the {model_name}.yaml file, you specify the nn.Module which is your implmentation of backbone model, which looks like:
```
_target_: src.models.nets.mlp.Model
_partial_: true
name: mlp
input_size: ${data.input_size}
output_size: ${data.output_size}
dim: 128
```
with the "\_target\_" keyword as the location of the nn.Module.
