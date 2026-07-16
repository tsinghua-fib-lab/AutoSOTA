import json
import logging
import os
from functools import partial

import hydra
import numpy as np
import torch
import transformers
from scaleformer.scaleformer.dataset import TimeSeriesDataset
from scaleformer.scaleformer.evaluation import (
    evaluate_forecasting_model,
    evaluate_mlm_model,
)
from scaleformer.scaleformer.pipeline import PatchTSTPipeline
from scaleformer.utils import (
    get_dim_from_dataset,
    get_eval_data_dict,
    log_on_main,
    process_trajs,
    save_evaluation_results,
)

import torch.distributed as dist
from scaleformer.utils import is_main_process
import torch.nn as nn

from collections import defaultdict

from torch.utils.data import DataLoader 

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    checkpoint_path = cfg.eval.checkpoint_path
    log(f"Using checkpoint: {checkpoint_path}")
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        log(f"Training info file found at: {training_info_path}")
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
            train_config = training_info.get("train_config", None)
            if train_config is None:  # for backwards compatibility
                train_config = training_info.get("training_config", None)
    else:
        log(f"No training info file found at: {training_info_path}")
        train_config = None

    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    pipeline = PatchTSTPipeline.from_pretrained(
        mode=cfg.eval.mode,
        pretrain_path=checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )

    model_config = dict(vars(pipeline.model.config))
    train_config = train_config or dict(cfg.train)
    # set floating point precision
    use_tf32 = train_config.get("tf32", False)
    log(f"use tf32: {use_tf32}")
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
        )
        use_tf32 = False

    rseed = train_config.get("seed", cfg.train.seed)
    log(f"Using SEED: {rseed}")
    transformers.set_seed(seed=rseed)

    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    channel_attention = model_config["channel_attention"]
    use_dynamics_embedding = model_config["use_dynamics_embedding"]

    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"eval prediction_length: {cfg.eval.prediction_length}")
    log(f"channel_attention: {channel_attention}")
    log(f"use_dynamics_embedding: {use_dynamics_embedding}")

    if channel_attention:
        # check use of channel rope
        channel_rope = model_config["channel_rope"]
        log(f"channel_rope: {channel_rope}")

    if use_dynamics_embedding:
        # check dynamics embedding parameters
        dynamics_embedding_config = {
            k: model_config[k]
            for k in [
                "num_poly_feats",
                "poly_degrees",
                "rff_trainable",
                "rff_scale",
                "num_rff",
            ]
        }
        log(f"dynamics embedding config: {dynamics_embedding_config}")

    pipeline.model.eval()
    
    system_dims = {
        system_name: get_dim_from_dataset(test_data_dict[system_name][0])
        for system_name in test_data_dict
    }
    n_system_samples = {
        system_name: len(test_data_dict[system_name]) for system_name in test_data_dict
    }

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: TimeSeriesDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.eval.mode,
            mode="test",
        )
        for system_name in test_data_dict
    }

    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
    )
    process_trajs_fn = partial(
        process_trajs,
        split_coords=cfg.eval.split_coords,
        overwrite=cfg.eval.overwrite,
        verbose=cfg.eval.verbose,
    )
    log(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    if cfg.eval.mode == "predict":
        parallel_sample_reduction_fn = {
            "mean": lambda x: np.mean(x, axis=0),
            "median": lambda x: np.median(x, axis=0),
        }.get(cfg.eval.parallel_sample_reduction, lambda x: x)

        predictions, contexts, labels, metrics = evaluate_forecasting_model(
            pipeline,
            test_datasets,
            batch_size=cfg.eval.batch_size,
            prediction_length=cfg.eval.prediction_length,
            metric_names=cfg.eval.metric_names,
            return_predictions=cfg.eval.save_predictions,
            return_contexts=cfg.eval.save_contexts,
            return_labels=cfg.eval.save_labels,
            parallel_sample_reduction_fn=parallel_sample_reduction_fn,
            redo_normalization=True,
            prediction_kwargs=dict(
                sliding_context=cfg.eval.sliding_context,
                limit_prediction_length=cfg.eval.limit_prediction_length,
                verbose=cfg.eval.verbose,
            ),
            eval_subintervals=[
                (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
            ],
        )
        save_eval_results_fn(metrics)
        
        # ==================== 【修改后的代码】: 保存 trues(包含context) 和 preds 为 .pt 文件 ====================
        # 要实现此功能, 必须在配置文件中同时设置 save_labels=True 和 save_contexts=True
        if cfg.eval.save_labels and cfg.eval.save_contexts:
            log("Saving trues (context + labels) and preds (predictions) as .pt files...")
            
            # 确保 contexts 和 labels 字典都已成功生成
            if contexts is None or labels is None:
                log("Error: 'contexts' or 'labels' is None. Cannot save concatenated files. Please check evaluation settings.")
            else:
                save_dir = cfg.eval.metrics_save_dir
                os.makedirs(save_dir, exist_ok=True)

                # 遍历每个被评估的系统/数据集
                for system_name in labels.keys():
                    # 定义文件名
                    preds_filename = os.path.join(save_dir, f"predictions_{system_name}.pt")
                    trues_filename = os.path.join(save_dir, f"labels_{system_name}.pt")

                    # 【核心修改】: 将 context 和 labels 在时间维度(axis=2)上拼接
                    # 假设数据维度为 (num_samples, num_channels, sequence_length)
                    combined_trues = np.concatenate(
                        [contexts[system_name], labels[system_name]], axis=2
                    )

                    # 从 numpy 数组转换为 torch 张量
                    preds_tensor = torch.from_numpy(predictions[system_name])
                    # 使用拼接后的'combined_trues'来创建张量
                    trues_tensor = torch.from_numpy(combined_trues)

                    # 保存张量
                    torch.save(preds_tensor, preds_filename)
                    torch.save(trues_tensor, trues_filename)
                    log(f"  - 已保存 '{system_name}' 的预测值到: {preds_filename}")
                    log(f"  - 已保存 '{system_name}' 的真实值 (context+labels) 到: {trues_filename}")
        else:
            log("Skipped saving since eval.save is not enabled.")
        # =======================================================================================================

        if cfg.eval.save_predictions and cfg.eval.save_contexts:
            assert predictions is not None and contexts is not None
            process_trajs_fn(
                cfg.eval.forecast_save_dir,
                {  # concatenate contexts and predictions
                    system: np.concatenate(
                        [contexts[system], predictions[system]], axis=2
                    )
                    for system in predictions
                },
            )

        if cfg.eval.save_labels and cfg.eval.save_contexts:
            assert labels is not None and contexts is not None
            process_trajs_fn(
                cfg.eval.labels_save_dir,
                {  # concatenate contexts and labels
                    system: np.concatenate([contexts[system], labels[system]], axis=2)
                    for system in labels
                },
            )

    elif cfg.eval.mode == "pretrain":
        completions, processed_past_values, timestep_masks, metrics = (
            evaluate_mlm_model(
                pipeline,
                test_datasets,
                metric_names=cfg.eval.metric_names,
                batch_size=cfg.eval.batch_size,
                undo_normalization=False,
                return_completions=cfg.eval.save_completions,
                return_processed_past_values=cfg.eval.save_contexts,
                return_masks=cfg.eval.save_masks,
            )
        )
        save_eval_results_fn(metrics)

        if cfg.eval.save_completions and completions is not None:
            process_trajs_fn(cfg.eval.completions_save_dir, completions)
        if cfg.eval.save_contexts and processed_past_values is not None:
            process_trajs_fn(cfg.eval.patch_input_save_dir, processed_past_values)
        if cfg.eval.save_masks and timestep_masks is not None:
            process_trajs_fn(cfg.eval.timestep_masks_save_dir, timestep_masks)
    else:
        raise ValueError(f"Invalid eval mode: {cfg.eval.mode}")
    
if __name__ == "__main__":
    main()
