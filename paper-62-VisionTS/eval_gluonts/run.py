import math
import sys
import os
import argparse
sys.path.append("../")

import einops
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gluonts.model.forecast import SampleForecast
from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE
from gluonts.model.evaluation import evaluate_forecasts

from dataset import get_gluonts_test_dataset
from visionts import VisionTS, freq_to_seasonality_list

POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4], # 1 year or 1 month
    "M": [12, 6, 3], # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2], # 6 months or 1 year
}


def imputation_nan(array):
    """
    Impute missing value using Naive forecasting.
    """
    not_nan_mask = ~np.isnan(array)
    if not_nan_mask.all():
        return array
    if not not_nan_mask.any():
        return np.zeros_like(array)

    array_imputed = np.copy(array)
    for i in range(len(array)):
        if not not_nan_mask[i]:
            array_imputed[i] = array_imputed[i - 1]
    return array_imputed


def forecast(model: VisionTS, train_list: list, test_list: list, batch_size, device, periodicity):
    # We combine testing data with the context lengths
    seq_len_to_group_data = {}
    for i in range(len(train_list)):
        train_len = len(train_list[i])
        if train_len not in seq_len_to_group_data:
            seq_len_to_group_data[train_len] = [[], [], []] # index, input, output
        seq_len_to_group_data[train_len][0].append(i)
        seq_len_to_group_data[train_len][1].append(train_list[i])
        seq_len_to_group_data[train_len][2].append(test_list[i])
    
    forecast_np = {} # raw index -> forecast
    for train_len in seq_len_to_group_data:
        cur_idx_list, cur_train, cur_test = seq_len_to_group_data[train_len]
        convert = lambda array: torch.FloatTensor(
            einops.rearrange(np.array(array), 'b t -> b t 1')
        ).to(device)
        cur_train = convert(cur_train)
        cur_test = convert(cur_test)
        context_len = cur_train.shape[1]
        pred_len = cur_test.shape[1]
        model.update_config(context_len=context_len, pred_len=pred_len, periodicity=periodicity)

        for batch_i in range(int(math.ceil(len(cur_train) / batch_size))):
            batch_start = batch_i * batch_size
            if batch_start >= len(cur_train):
                continue
            batch_end = batch_start + batch_size
            if batch_end > len(cur_train):
                batch_end = len(cur_train)

            cur_batch_train = cur_train[batch_start:batch_end]
            cur_batch_pred = model(cur_batch_train, fp64=True) # [b t 1]
            for i in range(len(cur_batch_pred)):
                cur_idx = cur_idx_list[batch_start + i]
                assert cur_idx not in forecast_np
                forecast_np[cur_idx] = cur_batch_pred[i, :, 0].detach().cpu().numpy()
    return np.stack([forecast_np[i] for i in range(len(train_list))])


def convert_context_len(context_len, no_periodicity_context_len, periodicity):
    if periodicity == 1:
        context_len = no_periodicity_context_len
    # Round context length to the integer multiples of the period
    context_len = int(round(context_len / periodicity)) * periodicity
    return context_len


def evaluate(
    dataset,
    save_path,
    context_len,
    no_periodicity_context_len,
    device="cuda:0",
    checkpoint_dir="./ckpt",
    mae_arch="mae_base",
    batch_size=512,
    periodicity="autotune",
):
    model = VisionTS(mae_arch, ckpt_dir=checkpoint_dir).to(device)

    test_data, metadata = get_gluonts_test_dataset(dataset)
    data_train = [imputation_nan(x['target']) for x in test_data.input]
    data_test = [x['target'] for x in test_data.label]
    pred_len = len(data_test[0])
    
    if periodicity == "autotune":
        seasonality_list = freq_to_seasonality_list(metadata.freq, POSSIBLE_SEASONALITIES)
        best_valid_mae = float('inf')
        best_valid_p = 1
        for periodicity in tqdm(seasonality_list, desc='validate seasonality'):
            cur_context_len = convert_context_len(context_len, no_periodicity_context_len, periodicity)

            val_train = [x[-cur_context_len-pred_len:-pred_len] for x in data_train]
            val_test = [x[-pred_len:] for x in data_train]
            val_forecast = forecast(model, val_train, val_test, batch_size, device, periodicity)
            val_mae = np.abs(np.asarray(val_test) - val_forecast).mean()
            if val_mae < best_valid_mae:
                best_valid_p = periodicity
                best_valid_mae = val_mae
                tqdm.write(f"autotune: P = {periodicity} | valid mae = {val_mae}, accept!")
            else:
                tqdm.write(f"autotune: P = {periodicity} | valid mae = {val_mae}, reject!")
        periodicity = best_valid_p
    elif periodicity == "freq":
        periodicity = freq_to_seasonality_list(metadata.freq, POSSIBLE_SEASONALITIES)[0]
    else:
        periodicity = int(periodicity)

    cur_context_len = convert_context_len(context_len, no_periodicity_context_len, periodicity)
    train = [x[-cur_context_len:] for x in data_train]
    forecast_values = forecast(model, train, data_test, batch_size, device, periodicity)

    sample_forecasts = []
    for item, ts in zip(forecast_values, test_data.input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(
                samples=np.reshape(item, (1, -1)), start_date=forecast_start_date
            )
        )
    metrics_df = evaluate_forecasts(
        sample_forecasts,
        test_data=test_data,
        metrics=[
            MSE(),
            MAE(),
            SMAPE(),
            MASE(),
            ND(),
            NRMSE(),
        ],
    )
    metrics_df.insert(loc=0, column='dataset', value=[dataset])
    if os.path.exists(save_path):
        old_metrics_df = pd.read_csv(save_path)
        metrics_df = pd.concat([old_metrics_df, metrics_df], ignore_index=True)
    print(metrics_df)
    metrics_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    print("-" * 5, f"Evaluation of {dataset} complete", "-" * 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VisionTS on Monash or PF"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        required=True,
        help=(
            "Time series periodicity length. Can be the following param: "
            + "(1) 'autotune': find the best periodicity on the validation set based on frequency "
            + "(2) 'freq': use the pre-defined periodicity based on frequency "
            + "(3) An integer: use the given periodicity."
        ),
    )
    parser.add_argument(
        "--save_name", type=str, default="result.csv", help="Directory to save the results"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../ckpt/",
        help="Path to load the model. Auto download if not exists.",
    )
    parser.add_argument("--context_len", type=int, default=1000, help="Context length.")
    parser.add_argument(
        "--no_periodicity_context_len",
        type=int,
        default=1000,
        help="Context length for data with periodicity = 1.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for generating samples"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device. cuda or cpu"
    )

    args = parser.parse_args()

    with torch.no_grad():
        evaluate(
            args.dataset,
            args.save_name,
            context_len=args.context_len,
            no_periodicity_context_len=args.no_periodicity_context_len,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            periodicity=args.periodicity,
        )