## Setup:
```
conda create -n DataDecide-single-scale python=3.11
conda activate DataDecide-single-scale
pip install -r requirements.txt
```

## Process all results given S3 dir path:
```
python process_s3_to_csv.py \
    --s3 s3://<> \
    --csv_name "results_ladder_5xC.csv"
```

## Clean data

run `data_exploration_and_cleaning.ipynb` to output `results_ladder_5xC_seeds_cleaned_correct_params.csv`

You also need to run it with the variable `DIRTY_OUT = True` to generate `results_ladder_5xC_seeds_dirty_correct_params.csv ` for some visualizations

## run single scale predictions

```
python main.py
```

## run per task analysis

Generate configs 
```
sh create_per_task_configs.sh config.yaml per_task_configs
```

Run configs
```
# ls per_task_configs/*.yaml | xargs -I {} python main.py --config {}
ls per_task_configs/*.yaml | parallel --progress --eta --tee --results logs_dir python main.py --config {}
```
