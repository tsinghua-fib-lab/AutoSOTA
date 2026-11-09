All of these are executed from the `src/` directory.

1. `python pipeline_pre.py 1`, which 

- Reads training data from `RAW_DATA_DIR`
- Adds enhanced information to new csv files in `PROCESSED_DATA_DIR`
- This can take about an hour to run, mostly due to external libraries (`rdkit` and `openbabel`).

2. `python pipeline_pre.py 2`, which

- Reads enhanced information from new csv files in `PROCESSED_DATA_DIR`
- Creates loaders, which are pickled into `PROCESSED_DATA_DIR`
- This can take 60-90 minutes to run, mostly creating the data loaders

3. `python train.py --cuda [--multi-gpu] [--batch_size 60] [--d_model 700] [...]`, which

- Trains a graph transformer model with specified parameters and for designated number of epochs
- More options can be found via the `-h` flag. Some commonly used flags are `--d_model` (the embedding dimension), `--dropout` (fully connected layer dropout rate), and `--n_layer` (number of transformer layers). You can also specify on whether to use 80% or 100% of the training data with `--mode` (if 80%, the rest of the data will be used as a validation set).

4. `python predictor.py`, which would

- Read list of models from `CONFIG_DIR/models.json`
- Read those models from `MODEL_DIR`
- Use them to make predictions, using `test.csv` in `RAW_DATA_DIR` (can take ~15 min each for the Kaggle test set)
- Write the predictions of each model to `SUBMISSION_DIR`
- Ensemble those into a new prediction csv in `SUBMISSION_DIR`

5. `python predictor.py fast`, which would only

- Read the predictions of each model from `SUBMISSION_DIR`
- Ensemble those into a new prediction csv in `SUBMISSION_DIR`
