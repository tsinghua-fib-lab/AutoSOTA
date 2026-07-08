import sys
import logging
import copy
import torch
import json
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from datetime import datetime
import time


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
        timestamp,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    fc_curve = {"top1": [], "top5": []}
    knn_curve = {"top1": [], "top5": []}
    ncm_curve = {"top1": [], "top5": []}
    hc_soinn_curve = {"top1": [], "top5": []}
    fc_matrix, knn_matrix, ncm_matrix, hc_soinn_matrix = [], [], [], []

    start_task = 0
    if args.get("load_all_checkpoints", False):
        logging.info("Auto-loading all checkpoints mode enabled: will load checkpoints for all tasks and skip training")
        start_task = 0

    for task in range(start_task, data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        is_last_task = (task == data_manager.nb_tasks - 1)
        _apply_task_epoch_override(args, task, is_last_task)
        train_time = None
        eval_time = None
        
        if is_last_task:
            train_start = time.time()
            model.incremental_train(data_manager)
            train_end = time.time()
            train_time = train_end - train_start
            if args.get("training_time_profile_last_task", False):
                timing_breakdown = {}
                if hasattr(model, "get_last_training_time_breakdown"):
                    timing_breakdown = model.get_last_training_time_breakdown()
                _write_training_time_profile_result(
                    args,
                    {
                        "classifier": args.get("training_time_classifier", args.get("prefix", "")),
                        "seed": args.get("seed", ""),
                        "task": task,
                        "tuned_epoch": args.get("tuned_epoch", ""),
                        "total_incremental_train_sec": train_time,
                        "total_incremental_train_min": train_time / 60,
                        "pure_train_sec": float(timing_breakdown.get("pure_train_sec", 0.0)),
                        "periodic_eval_sec": float(timing_breakdown.get("periodic_eval_sec", 0.0)),
                        "classifier_build_sec": float(timing_breakdown.get("classifier_build_sec", 0.0)),
                    },
                    logfilename + ".log",
                )

            if args.get("training_time_profile_last_task", False) and args.get("skip_eval_for_training_time_profile", True):
                eval_start = time.time()
                eval_results = {}
                eval_end = time.time()
                eval_time = eval_end - eval_start
                logging.info("Skipping evaluation for last-task training time profile")
            elif args.get("speed_profile_last_task", False):
                eval_start = time.time()
                profile_results = model.profile_last_task_classifiers()
                eval_results = _speed_profile_results_to_eval_results(profile_results)
                _write_speed_profile_results(args, profile_results, logfilename + ".log")
                eval_end = time.time()
                eval_time = eval_end - eval_start
            else:
                eval_start = time.time()
                eval_results = model.eval_task()
                eval_end = time.time()
                eval_time = eval_end - eval_start
            
            logging.info("=" * 60)
            logging.info("Last Task (Task {}) Time Statistics:".format(task))
            logging.info("  Training time: {:.2f} seconds ({:.2f} minutes)".format(train_time, train_time / 60))
            logging.info("  Evaluation time: {:.2f} seconds ({:.2f} minutes)".format(eval_time, eval_time / 60))
            logging.info("  Total time: {:.2f} seconds ({:.2f} minutes)".format(train_time + eval_time, (train_time + eval_time) / 60))
            logging.info("=" * 60)
        else:
            model.incremental_train(data_manager)
            if args.get("training_time_profile_last_task", False) and args.get("skip_eval_for_training_time_profile", True):
                eval_results = {}
                logging.info("Skipping evaluation before last task for training time profile")
            else:
                eval_results = model.eval_task()
        
        model.after_task()

        if "fc" in eval_results:
            fc_accy = eval_results["fc"]
            logging.info("FC: {}".format(fc_accy["grouped"]))
            fc_keys = [key for key in fc_accy["grouped"].keys() if '-' in key]
            fc_values = [fc_accy["grouped"][key] for key in fc_keys]
            fc_matrix.append(fc_values)
            fc_curve["top1"].append(fc_accy["top1"])
            fc_curve["top5"].append(fc_accy["top5"])

        if "knn" in eval_results:
            knn_accy = eval_results["knn"]
            logging.info("KNN: {}".format(knn_accy["grouped"]))
            knn_keys = [key for key in knn_accy["grouped"].keys() if '-' in key]
            knn_values = [knn_accy["grouped"][key] for key in knn_keys]
            knn_matrix.append(knn_values)
            knn_curve["top1"].append(knn_accy["top1"])
            knn_curve["top5"].append(knn_accy["top5"])

        if "ncm" in eval_results:
            ncm_accy = eval_results["ncm"]
            logging.info("NCM: {}".format(ncm_accy["grouped"]))
            ncm_keys = [key for key in ncm_accy["grouped"].keys() if '-' in key]
            ncm_values = [ncm_accy["grouped"][key] for key in ncm_keys]
            ncm_matrix.append(ncm_values)
            ncm_curve["top1"].append(ncm_accy["top1"])
            ncm_curve["top5"].append(ncm_accy["top5"])

        if "hc_soinn" in eval_results:
            hc_soinn_accy = eval_results["hc_soinn"]
            logging.info("HC-SOINN: {}".format(hc_soinn_accy["grouped"]))
            hc_soinn_keys = [key for key in hc_soinn_accy["grouped"].keys() if '-' in key]
            hc_soinn_values = [hc_soinn_accy["grouped"][key] for key in hc_soinn_keys]
            hc_soinn_matrix.append(hc_soinn_values)
            hc_soinn_curve["top1"].append(hc_soinn_accy["top1"])
            hc_soinn_curve["top5"].append(hc_soinn_accy["top5"])

        curves_to_log = []
        if "fc" in eval_results:
            curves_to_log.append(("FC", fc_curve))
        if "knn" in eval_results:
            curves_to_log.append(("KNN", knn_curve))
        if "ncm" in eval_results:
            curves_to_log.append(("NCM", ncm_curve))
        if "hc_soinn" in eval_results:
            curves_to_log.append(("HC-SOINN", hc_soinn_curve))

        for name, curve in curves_to_log:
            logging.info("{} top1 curve: {}".format(name, curve["top1"]))
            logging.info("{} top5 curve: {}".format(name, curve["top5"]))

        avg_accs = []
        if "fc" in eval_results and len(fc_curve["top1"]) > 0:
            avg_fc = sum(fc_curve["top1"]) / len(fc_curve["top1"])
            avg_accs.append(("FC", avg_fc))
            print('Average Accuracy (FC):', avg_fc)
            logging.info("Average Accuracy (FC): {}".format(avg_fc))

        if "knn" in eval_results and len(knn_curve["top1"]) > 0:
            avg_knn = sum(knn_curve["top1"]) / len(knn_curve["top1"])
            avg_accs.append(("KNN", avg_knn))
            print('Average Accuracy (KNN):', avg_knn)
            logging.info("Average Accuracy (KNN): {}".format(avg_knn))

        if "ncm" in eval_results and len(ncm_curve["top1"]) > 0:
            avg_ncm = sum(ncm_curve["top1"]) / len(ncm_curve["top1"])
            avg_accs.append(("NCM", avg_ncm))
            print('Average Accuracy (NCM):', avg_ncm)
            logging.info("Average Accuracy (NCM): {}".format(avg_ncm))

        if "hc_soinn" in eval_results and len(hc_soinn_curve["top1"]) > 0:
            avg_hc_soinn = sum(hc_soinn_curve["top1"]) / len(hc_soinn_curve["top1"])
            avg_accs.append(("HC-SOINN", avg_hc_soinn))
            print('Average Accuracy (HC-SOINN):', avg_hc_soinn)
            logging.info("Average Accuracy (HC-SOINN): {}".format(avg_hc_soinn))

        logging.info("")

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(fc_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(fc_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (FC):')
            print(np_acctable)
            logging.info('Forgetting (FC): {}'.format(forgetting))
        if len(knn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(knn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (KNN):')
            print(np_acctable)
            logging.info('Forgetting (KNN): {}'.format(forgetting))
        if len(ncm_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(ncm_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NCM):')
            print(np_acctable)
            logging.info('Forgetting (NCM): {}'.format(forgetting))
        if len(hc_soinn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(hc_soinn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (HC-SOINN):')
            print(np_acctable)
            logging.info('Forgetting (HC-SOINN): {}'.format(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _apply_task_epoch_override(args, task, is_last_task):
    """Allow cheap early tasks and a full-epoch final task in timing experiments."""
    if "_base_tuned_epoch" not in args:
        args["_base_tuned_epoch"] = args.get("tuned_epoch")

    if is_last_task and "last_task_tuned_epoch" in args:
        args["tuned_epoch"] = args["last_task_tuned_epoch"]
    elif (not is_last_task) and "pre_last_task_tuned_epoch" in args:
        args["tuned_epoch"] = args["pre_last_task_tuned_epoch"]
    else:
        args["tuned_epoch"] = args["_base_tuned_epoch"]

    logging.info(
        "Task {} tuned_epoch set to {}{}".format(
            task,
            args["tuned_epoch"],
            " (last task)" if is_last_task else "",
        )
    )


def _write_training_time_profile_result(args, record, log_path):
    """Persist last-task training time records plus aggregate summary files."""
    output_prefix = args.get(
        "training_time_output_prefix",
        "resources/coda_prompt_cifar10task_training_time",
    )
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    row = dict(record)
    row["config_prefix"] = args.get("prefix", "")
    row["dataset"] = args.get("dataset", "")
    row["log_path"] = log_path

    json_path = output_prefix + ".json"
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            logging.warning(f"Could not parse existing training time JSON: {json_path}")
    existing.append(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    csv_path = output_prefix + ".csv"
    fieldnames = [
        "classifier", "seed", "task", "tuned_epoch", "total_incremental_train_sec",
        "total_incremental_train_min", "pure_train_sec", "periodic_eval_sec",
        "classifier_build_sec", "config_prefix", "dataset", "log_path",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for item in existing:
            values = []
            for field in fieldnames:
                value = item.get(field, "")
                if isinstance(value, str):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(str(value))
            f.write(",".join(values) + "\n")

    summary = _summarize_training_time_results(existing)
    md_path = output_prefix + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CODA-Prompt CIFAR 10-task Last-Task Training Time\n\n")
        f.write("| classifier | repeats | tuned_epoch | total mean | total var | pure train mean | pure train var | periodic eval mean | build mean | build var |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for item in summary:
            f.write(
                f"| {item['classifier']} | {item['repeats']} | {item['tuned_epoch']} | "
                f"{item['total_incremental_train_sec_mean']:.6f} | "
                f"{item['total_incremental_train_sec_var']:.6f} | "
                f"{item['pure_train_sec_mean']:.6f} | {item['pure_train_sec_var']:.6f} | "
                f"{item['periodic_eval_sec_mean']:.6f} | "
                f"{item['classifier_build_sec_mean']:.6f} | "
                f"{item['classifier_build_sec_var']:.6f} |\n"
            )

    logging.info(f"Training time profile results written to {json_path}, {csv_path}, {md_path}")


def _summarize_training_time_results(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["classifier"], []).append(row)

    summary = []
    metrics = [
        "total_incremental_train_sec",
        "pure_train_sec",
        "periodic_eval_sec",
        "classifier_build_sec",
    ]
    for classifier, items in grouped.items():
        tuned_epochs = sorted({str(item.get("tuned_epoch", "")) for item in items})
        summary_row = {
            "classifier": classifier,
            "repeats": len(items),
            "tuned_epoch": "/".join(tuned_epochs),
        }
        for metric in metrics:
            values = np.array([float(item.get(metric, 0.0)) for item in items], dtype=np.float64)
            summary_row[f"{metric}_mean"] = float(values.mean()) if len(values) > 0 else 0.0
            summary_row[f"{metric}_var"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
            summary_row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_row["total_incremental_train_min_mean"] = summary_row["total_incremental_train_sec_mean"] / 60
        summary.append(summary_row)
    return summary


def _speed_profile_results_to_eval_results(profile_results):
    """Convert profiler accuracy records into trainer-compatible eval results."""
    eval_results = {}
    for item in profile_results:
        classifier = item["classifier"]
        if classifier in eval_results:
            continue
        eval_results[classifier] = {
            "grouped": {"total": item["top1"]},
            "top1": item["top1"],
            "top5": item["top5"],
        }
        if classifier == "kac":
            eval_results["fc"] = eval_results[classifier]
    return eval_results


def _write_speed_profile_results(args, profile_results, log_path):
    """Persist speed profile repeats plus aggregate summary files."""
    output_prefix = args.get(
        "speed_profile_output_prefix",
        "resources/coda_prompt_cifar10task_speed_profile",
    )
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    enriched = []
    for item in profile_results:
        row = dict(item)
        row["config_prefix"] = args.get("prefix", "")
        row["dataset"] = args.get("dataset", "")
        row["seed"] = args.get("seed", "")
        row["log_path"] = log_path
        enriched.append(row)

    json_path = output_prefix + ".json"
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            logging.warning(f"Could not parse existing speed profile JSON: {json_path}")
    existing.extend(enriched)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    csv_path = output_prefix + ".csv"
    fieldnames = [
        "classifier", "repeat", "samples", "batches", "total_sec",
        "backbone_sec", "backbone_pct", "classifier_sec", "classifier_pct",
        "top1", "top5", "config_prefix", "dataset", "seed", "log_path",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in existing:
            values = []
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, str):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(str(value))
            f.write(",".join(values) + "\n")

    summary = _summarize_speed_profile_results(existing)
    md_path = output_prefix + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CODA-Prompt CIFAR 10-task Classifier Speed Profile\n\n")
        f.write("| classifier | repeats | samples | total_sec mean±std | backbone_sec mean±std | backbone_pct | classifier_sec mean±std | classifier_pct | top1 | top5 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary:
            f.write(
                f"| {row['classifier']} | {row['repeats']} | {row['samples']} | "
                f"{row['total_sec_mean']:.6f}±{row['total_sec_std']:.6f} | "
                f"{row['backbone_sec_mean']:.6f}±{row['backbone_sec_std']:.6f} | "
                f"{row['backbone_pct_mean']:.2f}% | "
                f"{row['classifier_sec_mean']:.6f}±{row['classifier_sec_std']:.6f} | "
                f"{row['classifier_pct_mean']:.2f}% | "
                f"{row['top1_mean']:.2f} | {row['top5_mean']:.2f} |\n"
            )

    logging.info(f"Speed profile results written to {json_path}, {csv_path}, {md_path}")


def _summarize_speed_profile_results(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["classifier"], []).append(row)

    summary = []
    metric_names = [
        "total_sec", "backbone_sec", "backbone_pct", "classifier_sec",
        "classifier_pct", "top1", "top5",
    ]
    for classifier, items in grouped.items():
        summary_row = {
            "classifier": classifier,
            "repeats": len(items),
            "samples": int(items[0].get("samples", 0)),
        }
        for metric in metric_names:
            values = np.array([float(item.get(metric, 0.0)) for item in items], dtype=np.float64)
            summary_row[f"{metric}_mean"] = float(values.mean()) if len(values) > 0 else 0.0
            summary_row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(summary_row)
    return summary