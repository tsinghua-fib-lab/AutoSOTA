
## How to evaluate the models in ProbTS using the GIFT-EVAL benchmark

Link to the GIFT-EVAL benchmark: [Github Repo](https://github.com/SalesforceAIResearch/gift-eval) [Paper](https://openreview.net/forum?id=9EBSEkFSje)

1. Follow installation instructions in the GIFT-EVAL repository to **download the dataset** from its huggingface dataset repository.
2. Also, set the environment variable `GIFT_EVAL` to the path where the dataset is downloaded.
``` bash
echo "GIFT_EVAL=/path/to/gift-eval" >> .env
```
3. Quick start example:
``` bash
python run.py --config config/default/mean.yaml \
              --seed_everything 0 \
              --model.forecaster.init_args.mode batch \
              --data.data_manager.init_args.dataset gift/ett1/H/long \
              --data.data_manager.init_args.path ./datasets \
              --trainer.default_root_dir ./exps
```

> [!NOTE]  
> The dataset name for the GIFT-EVAL format should be specified as follows: `"gift/" + "dataset_name (main_name/freq)" + "short/medium/long"`. For example, `gift/ett1/H/long`. More dataset names can be found in the GIFT-EVAL repository (for example [naive.ipynb](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/naive.ipynb)).
