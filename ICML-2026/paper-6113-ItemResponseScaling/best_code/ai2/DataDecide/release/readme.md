
## Setup
```
sudo apt-get install git-lfs
pip install -r requirements.txt
```


## upload final checkpoints with default seed to main branches
```
python make_final_checkpoint_script.py
parallel --joblog progress.log --resume --progress -j1 < final_checkpoint_upload_script.sh
```

## reupload the model card

```
parallel --joblog progress_model_card.log --resume --progress -j10 python upload_model_card.py --repo_id allenai/{} < repo_names.txt
```

## upload checkpoints
```
parallel --joblog progress_checkpoint.log --resume --progress -j10 python upload_checkpoints.py --repo_name {} < repo_names.txt
```


## upload eval metrics results
```
python upload_eval.py --private
```

## change privacy

```
parallel --joblog progress_public.log --resume --progress python make_repo_public.py --repo_id allenai/{} < repo_names.txt
```